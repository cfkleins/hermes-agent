"""Execute contributor CI's range selection and mapping loop in disposable Git histories."""

import os
from pathlib import Path
import subprocess

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/contributor-check.yml"


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def history(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "CI fixture")
    _git(tmp_path, "config", "user.email", "mapped@example.test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    _git(tmp_path, "commit", "--allow-empty", "-qm", "old main")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(tmp_path, "commit", "--allow-empty", "-qm", "historical unmapped",
         "--author=Historical <historical@example.test>")
    base = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "contributors/emails").mkdir(parents=True)
    (tmp_path / "contributors/emails/mapped@example.test").write_text("fixture\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/release.py").write_text("AUTHOR_MAP = {}\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "new mapped contributor")
    return tmp_path, base


def _check(repo, base, event="pull_request"):
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    step = next(s for s in workflow["jobs"]["check-attribution"]["steps"]
                if s.get("id") == "check-emails")
    # Run the actual range selection and mapping loop, stopping before jq's
    # presentation-only formatting so this regression also runs on Git Bash.
    script = step["run"].split('if [ -n "$MISSING" ]; then')[0]
    script += '\nprintf "EMAILS=%s\\nMISSING=%s\\n" "$NEW_EMAILS" "$MISSING"\n'
    env = dict(os.environ, GITHUB_EVENT_NAME=event, PR_BASE_SHA=base,
               GITHUB_OUTPUT=str(repo / "output"))
    return subprocess.run(["bash", "-eo", "pipefail", "-c", script], cwd=repo,
                          env=env, capture_output=True, text=True)


def test_actual_pr_base_excludes_history_but_checks_new_authors(history):
    repo, base = history
    result = _check(repo, base)
    assert result.returncode == 0, result.stderr
    assert "historical@example.test" not in result.stdout
    assert "mapped@example.test" in result.stdout
    assert result.stdout.rstrip().endswith("MISSING=")
    _git(repo, "commit", "--allow-empty", "-qm", "new unmapped contributor",
         "--author=New <new@example.test>")
    result = _check(repo, base)
    assert result.returncode == 0, result.stderr
    emails, missing = result.stdout.split("MISSING=", 1)
    assert "new@example.test" in emails and "mapped@example.test" in emails
    assert "new@example.test" in missing and "mapped@example.test" not in missing
    assert "historical@example.test" not in result.stdout
    legacy = _check(repo, "", event="push")
    assert legacy.returncode == 0, legacy.stderr
    assert "historical@example.test" in legacy.stdout.split("MISSING=", 1)[1]


@pytest.mark.parametrize("bad_base", ["", "main", "a" * 39, "g" * 40,
                                      "0" * 40, "$(touch injected)", "blob"])
def test_pr_base_fails_closed(history, bad_base):
    repo, _ = history
    if bad_base == "blob":
        bad_base = _git(repo, "rev-parse", "HEAD:scripts/release.py")
    result = _check(repo, bad_base)
    assert result.returncode != 0
    assert "EMAILS=" not in result.stdout
    assert not (repo / "injected").exists()
