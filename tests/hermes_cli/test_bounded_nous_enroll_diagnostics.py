"""Offline regression for bounded, secret-safe persistence failure codes."""
from pathlib import Path

import httpx
import pytest

from hermes_cli import bounded_nous_enroll as cli
from tests.hermes_cli.test_bounded_nous_enroll import device, tokens, protect_directory

pytestmark = pytest.mark.windows_only


@pytest.mark.parametrize("reason,expected", [
    ("invalid_jwt", "persistence_invalid_jwt"),
    ("unsupported_payload_schema", "persistence_unsupported_payload_schema"),
    ("scope_mismatch", "persistence_scope_mismatch"),
    ("lease_timeout", "persistence_lease_timeout"),
    ("persistence_readback_failed", "persistence_persistence_readback_failed"),
    ("DEMO secret must never be printed", "persistence_failed"),
])
def test_known_failure_codes_are_diagnostic_without_secrets(tmp_path, monkeypatch, capsys, reason, expected):
    private = tmp_path / "private"
    private.mkdir()
    protect_directory(private)
    source = private / "nous-login.dpapi"
    requests = []

    def respond(request):
        requests.append(request.url.path)
        return httpx.Response(200, json=device() if len(requests) == 1 else tokens())

    def refuse(**kwargs):
        raise RuntimeError(reason)

    original = cli.run_enrollment
    monkeypatch.setattr(cli, "run_enrollment", lambda **kwargs: original(
        **kwargs, transport=httpx.MockTransport(respond), opener=lambda uri: None,
        persist=refuse, sleep=lambda seconds: None))
    assert cli.main(["--source", str(source)]) == 1
    captured = capsys.readouterr()
    import json
    assert json.loads(captured.err) == {"status": "error", "code": expected}
    assert "DEMO secret" not in captured.err
    assert "synthetic-access" not in captured.err
    assert "synthetic-refresh" not in captured.err
    assert "enrolled" not in captured.out
    assert not source.exists()
    assert len(requests) == 2
