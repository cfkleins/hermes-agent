"""Configured bounded CLI: fail closed before constructing state or a socket."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name('bounded_service_fixture')
TOKEN = 'fixture-bounded-token-0123456789'
KEY = 'fixture-nous-key-0123456789'
PROMPT = b'OFFLINE FIXTURE ONLY. Ask one concise interview question.\r\n'


def configuration(tmp_path):
    prompt = tmp_path / 'approved.txt'
    prompt.write_bytes(PROMPT)
    policy = dict(version=1, principal_id='vcc', owner_id='fixture-owner',
                  agent_id='vector', case_id='new-persistent-case', source='vcc-bounded',
                  case_selector='vector_today', prompt_file=str(prompt),
                  prompt_sha256=hashlib.sha256(PROMPT).hexdigest(), provider='nous',
                  model='openai/gpt-6-astra', api_mode='chat_completions',
                  base_url='https://inference-api.nousresearch.com/v1', context_length=1050000,
                  required_skills=['core-interview', 'project-issue-interview'],
                  token_env='VCC_BOUNDED_HERMES_TOKEN', key_env='VCC_BOUNDED_NOUS_KEY')
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy), encoding='utf-8')
    return path, policy, tmp_path / 'persistent'


def environment(tmp_path, home):
    # No real secrets, home/profile or network environment is inherited.
    env = {k: v for k, v in os.environ.items() if k.upper() in {
        'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATH', 'PATHEXT', 'TEMP', 'TMP'}}
    ambient = tmp_path / 'ambient'
    ambient.mkdir(exist_ok=True)
    env.update(HOME=str(ambient), USERPROFILE=str(ambient), LOCALAPPDATA=str(ambient),
               APPDATA=str(ambient), HERMES_HOME=str(ambient / 'default-hermes'),
               VCC_BOUNDED_HERMES_TOKEN=TOKEN, VCC_BOUNDED_NOUS_KEY=KEY,
               PYTHONPATH=os.pathsep.join((str(FIXTURE), str(ROOT))),
               BOUNDED_FIXTURE_HOME=str(home), BOUNDED_FIXTURE_LOG=str(tmp_path / 'fixture.jsonl'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    return env


def command(path, home, port=38653):
    return [sys.executable, '-m', 'gateway.bounded_service', '--policy', str(path),
            '--home', str(home), '--port', str(port)]


def invoke(tmp_path, path, home, *, port=38653, env=None):
    return subprocess.run(command(path, home, port) + ['--check'], cwd=ROOT,
                          env=env or environment(tmp_path, home), capture_output=True,
                          text=True, timeout=45)


@pytest.mark.parametrize('bad', [
    'unknown', 'duplicate', 'duplicate-escaped', 'oversize', 'malformed', 'nonfinite',
    'float', 'bool', 'missing', 'hash', 'prompt-oversize', 'prompt-utf8', 'secret',
    'ref', 'provider', 'model-auto', 'url', 'skills', 'identity', 'relative-prompt',
])
def test_invalid_policy_never_creates_home(tmp_path, bad):
    path, policy, home = configuration(tmp_path)
    changes = {
        'unknown': {'unknown': True}, 'float': {'context_length': 65536.0},
        'bool': {'context_length': True}, 'hash': {'prompt_sha256': '0' * 64},
        'secret': {'api_key': KEY}, 'ref': {'key_env': 'NOUS_API_KEY'},
        'provider': {'provider': 'auto'}, 'model-auto': {'model': 'auto'},
        'url': {'base_url': 'https://example.invalid/v1'},
        'skills': {'required_skills': ['core-interview']}, 'identity': {'case_id': ' x'},
        'relative-prompt': {'prompt_file': 'approved.txt'},
    }
    policy.update(changes.get(bad, {}))
    if bad == 'missing':
        del policy['owner_id']
    if bad.startswith('prompt-'):
        data = b'x' * 131073 if bad == 'prompt-oversize' else b'\xff'
        Path(policy['prompt_file']).write_bytes(data)
        policy['prompt_sha256'] = hashlib.sha256(data).hexdigest()
    raw = json.dumps(policy)
    raw = {'duplicate': raw[:-1] + ',"version":1}',
           'duplicate-escaped': raw[:-1] + ',"\\u0076ersion":1}',
           'oversize': raw + ' ' * 16385, 'malformed': '{',
           'nonfinite': raw.replace('1050000', 'NaN')}.get(bad, raw)
    path.write_text(raw, encoding='utf-8')
    result = invoke(tmp_path, path, home)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'bounded_service_invalid' in result.stderr
    assert not home.exists()
    assert KEY not in result.stderr and TOKEN not in result.stderr


@pytest.mark.parametrize('port', [0, 1024, 65536, -1])
def test_bad_port_before_writes(tmp_path, port):
    path, _, home = configuration(tmp_path)
    result = invoke(tmp_path, path, home, port=port)
    assert result.returncode == 2
    assert not home.exists()


@pytest.mark.parametrize('name', ['VCC_BOUNDED_HERMES_TOKEN', 'VCC_BOUNDED_NOUS_KEY'])
def test_service_secret_required_no_generic_fallback(tmp_path, name):
    path, _, home = configuration(tmp_path)
    env = environment(tmp_path, home)
    del env[name]
    env.update(API_SERVER_KEY=TOKEN, NOUS_API_KEY=KEY, OPENAI_API_KEY=KEY)
    result = invoke(tmp_path, path, home, env=env)
    assert result.returncode == 2
    assert not home.exists()


def test_check_validates_native_stores_and_rejects_policy_drift(tmp_path):
    path, policy, home = configuration(tmp_path)
    result = invoke(tmp_path, path, home)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['status'] == 'checked'
    assert (home / 'state.db').is_file()
    assert (home / 'runs_idempotency.db').is_file()
    again = invoke(tmp_path, path, home)
    assert again.returncode == 0, again.stderr
    assert json.loads(again.stdout)['session_id'] == json.loads(result.stdout)['session_id']
    policy['source'] = 'changed-source'
    path.write_text(json.dumps(policy), encoding='utf-8')
    drift = invoke(tmp_path, path, home)
    assert drift.returncode == 2
    log = tmp_path / 'fixture.jsonl'
    events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    assert not events, events  # preflight dispatches no SDK call / ambient I/O / network


@pytest.mark.parametrize('db', ['state.db', 'runs_idempotency.db'])
def test_corrupt_durable_store_cannot_check_successfully(tmp_path, db):
    path, _, home = configuration(tmp_path)
    first = invoke(tmp_path, path, home)
    assert first.returncode == 0, first.stderr
    (home / db).write_bytes(b'not a sqlite database')
    failed = invoke(tmp_path, path, home)
    assert failed.returncode == 2


def test_unmarked_existing_home_is_not_adopted(tmp_path):
    path, _, home = configuration(tmp_path)
    home.mkdir()
    sentinel = home / 'leave-alone'
    sentinel.write_bytes(b'unchanged')
    failed = invoke(tmp_path, path, home)
    assert failed.returncode == 2
    assert list(home.iterdir()) == [sentinel]


def test_package_import_has_no_runtime_side_effects(tmp_path):
    _, _, home = configuration(tmp_path)
    result = subprocess.run([sys.executable, '-c',
        "import gateway,sys; assert 'hermes_cli.config' not in sys.modules; "
        "assert 'gateway.session' not in sys.modules"], cwd=ROOT,
        env=environment(tmp_path, home), capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_provider_jwt_8192_preserved_through_native_preflight(tmp_path, monkeypatch):
    from gateway.bounded_service import load_policy
    path, _, home = configuration(tmp_path)
    env = environment(tmp_path, home)
    key = 'eyJ.' + 'A' * 8186 + '.Z'
    env['VCC_BOUNDED_NOUS_KEY'] = key
    monkeypatch.setenv('VCC_BOUNDED_HERMES_TOKEN', TOKEN)
    monkeypatch.setenv('VCC_BOUNDED_NOUS_KEY', key)
    assert load_policy(str(path)).key == key
    result = invoke(tmp_path, path, home, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['status'] == 'checked'


@pytest.mark.parametrize('name,value', [
    ('VCC_BOUNDED_HERMES_TOKEN', 'A' * 513),
    ('VCC_BOUNDED_NOUS_KEY', 'A' * 8193),
    ('VCC_BOUNDED_NOUS_KEY', 'A B'),
    ('VCC_BOUNDED_NOUS_KEY', 'A,B'),
    ('VCC_BOUNDED_NOUS_KEY', '${TOKEN}'),
    ('VCC_BOUNDED_NOUS_KEY', 'A\u007f'),
    ('VCC_BOUNDED_NOUS_KEY', 'A\u00e9'),
])
def test_secret_caps_fail_before_home_writes(tmp_path, name, value):
    path, _, home = configuration(tmp_path)
    env = environment(tmp_path, home)
    env[name] = value
    result = invoke(tmp_path, path, home, env=env)
    assert result.returncode == 2
    assert not home.exists()
