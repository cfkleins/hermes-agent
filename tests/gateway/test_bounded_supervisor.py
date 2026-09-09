"""Offline generation ownership and portable stop contracts."""
import asyncio
import json
import signal
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from gateway import bounded_service


def module():
    from gateway import bounded_supervisor
    return bounded_supervisor


def config(tmp_path):
    m = module()
    return m.Config(policy=tmp_path/'policy.json', home=tmp_path/'home',
        control=tmp_path/'control', connection=tmp_path/'connection.json',
        source=tmp_path/'private'/'nous-login.dpapi',
        marker=tmp_path/'uncertain.json')


def test_environment_is_allowlist(tmp_path):
    m = module()
    env = m.child_environment(tmp_path, 'DEMO-bearer', 'DEMO-key', {
        'SYSTEMROOT': 'C:/Windows', 'HTTP_PROXY': 'secret', 'OPENAI_API_KEY': 'secret',
        'PYTHONPATH': 'evil', 'NOUS_REFRESH_TOKEN': 'secret', 'PATH': 'paths'})
    assert env == {'SYSTEMROOT': 'C:/Windows', 'PATH': 'paths',
        'HERMES_HOME': str(tmp_path), 'VCC_BOUNDED_HERMES_TOKEN': 'DEMO-bearer',
        'VCC_BOUNDED_NOUS_KEY': 'DEMO-key'}


@pytest.mark.parametrize('code,receipt,auth_count', [(75,True,2),(2,False,1),(75,False,1),(0,False,1)])
def test_only_closed_expiry_can_renew(tmp_path, monkeypatch, code, receipt, auth_count):
    m = module(); cfg = config(tmp_path); events=[]
    monkeypatch.setattr(m, 'load_token', lambda p: 'DEMO-bearer')
    monkeypatch.setattr(m, 'port_free', lambda p: events.append('port'))
    def auth(*, source_path, marker_path, min_ttl_seconds):
        assert source_path == cfg.source and marker_path == cfg.marker
        assert min_ttl_seconds == 180
        assert not events or events[-1] == 'port'
        events.append('auth')
        return SimpleNamespace(access_token='DEMO-key', expires_at=time.time()+500)
    def child(cfg, credential, token):
        events.append('exit')
        if events.count('auth') == 2:
            cfg.stop_file.write_text('stop')
            return 0, False
        return code, receipt
    monkeypatch.setattr(m, 'run_child', child)
    result = m.supervise(cfg, credentials=auth)
    assert events.count('auth') == auth_count
    assert result == (0 if auth_count == 2 else 2)
    if result == 2:
        assert m.supervise(cfg, credentials=auth) == 2
        assert events.count('auth') == auth_count


def test_auth_failure_and_persistent_stop_never_spawn(tmp_path, monkeypatch):
    m=module(); cfg=config(tmp_path); calls=[]
    monkeypatch.setattr(m, 'port_free', lambda p: None)
    monkeypatch.setattr(m, 'load_token', lambda p: 'DEMO-bearer')
    monkeypatch.setattr(m, 'run_child', lambda *a: pytest.fail('spawn'))
    def auth(**kw):
        calls.append(1)
        raise ValueError('DEMO-secret-never-log')
    assert m.supervise(cfg, credentials=auth) == 2
    assert m.supervise(cfg, credentials=auth) == 2
    assert calls == [1]


def test_demo_dpapi_schema_and_no_fallback(tmp_path):
    import base64
    m=module(); path=tmp_path/'connection.json'
    value=dict(version=1,base_url='http://127.0.0.1:8643',case_selector='vector-primary-v1',
               token_dpapi=base64.b64encode(b'DEMO-encrypted').decode())
    path.write_text(json.dumps(value))
    assert m.load_token(path,unprotect=lambda b: b'DEMO-bearer') == 'DEMO-bearer'
    value['base_url']='https://example.invalid'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        m.load_token(path,unprotect=lambda b: pytest.fail('must not decrypt'))


def test_signal_writes_stop_and_waits_natural_child_return(tmp_path, monkeypatch):
    m=module(); cfg=config(tmp_path)
    monkeypatch.setattr(m,'port_free',lambda p: None)
    monkeypatch.setattr(m,'load_token',lambda p: 'DEMO-bearer')
    calls=[]
    def auth(**kw):
        calls.append('auth')
        return SimpleNamespace(access_token='DEMO-key',expires_at=time.time()+500)
    def child(*args):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        assert cfg.stop_file.exists()
        calls.append('drained')
        return 0,False
    monkeypatch.setattr(m,'run_child',child)
    assert m.supervise(cfg,credentials=auth) == 0
    assert calls == ['auth','drained']
    assert m.supervise(cfg,credentials=lambda **kw: pytest.fail('stopped')) == 0


def test_invalid_stop_path_before_home_write(tmp_path):
    assert bounded_service.main(['--policy',str(tmp_path/'missing'),'--home',str(tmp_path/'home'),
        '--port','8643','--stop-file','relative.stop']) == 2
    assert not (tmp_path/'home').exists()


def test_isolated_pre_enrollment_and_provider_length(tmp_path, monkeypatch):
    m = module(); cfg = config(tmp_path)
    cfg.source.parent.mkdir()
    cfg.source.write_bytes(b'DEMO-only-envelope')
    key = 'eyJ.' + 'A' * 8186 + '.Z'
    monkeypatch.setattr(m, 'port_free', lambda p: None)
    monkeypatch.setattr(m, 'load_token', lambda p: 'DEMO-bearer')
    def auth(*, source_path, marker_path, min_ttl_seconds):
        assert source_path == cfg.source and marker_path == cfg.marker
        assert min_ttl_seconds == 180
        return SimpleNamespace(access_token=key, expires_at=time.time()+500)
    def child(cfg, credential, token):
        assert credential.access_token == key
        assert m.child_environment(cfg.home, token, credential.access_token)['VCC_BOUNDED_NOUS_KEY'] == key
        cfg.stop_file.write_text('stop')
        return 0, False
    monkeypatch.setattr(m, 'run_child', child)
    assert m.supervise(cfg, credentials=auth) == 0
    assert cfg.source.read_bytes() == b'DEMO-only-envelope'
    with pytest.raises(ValueError):
        m._secret('A' * 513)


@pytest.mark.parametrize('source', ['auth.json', 'shared/nous_auth.json', 'private/other.dpapi', 'control/private/nous-login.dpapi'])
def test_nonisolated_source_rejected_before_control_write(tmp_path, source):
    from dataclasses import replace
    m = module(); cfg = replace(config(tmp_path), source=tmp_path/source)
    assert m.supervise(cfg, credentials=lambda **kw: pytest.fail('auth')) == 2
    assert not cfg.control.exists()


def test_existing_unbound_control_never_reseeded(tmp_path):
    m = module(); cfg = config(tmp_path)
    cfg.control.mkdir()
    pending = cfg.control/'generation.uncertain'
    pending.write_text('uncertain')
    assert m.supervise(cfg, credentials=lambda **kw: pytest.fail('auth')) == 2
    assert not (cfg.control/'supervisor.json').exists()
    assert pending.read_text() == 'uncertain'


def test_default_credential_api_and_control_binding(tmp_path, monkeypatch):
    import sys
    from dataclasses import replace
    from types import ModuleType
    m = module(); cfg = config(tmp_path); calls = []
    module_stub = ModuleType('hermes_cli.bounded_nous_credentials')
    def isolated(*, source_path, marker_path, min_ttl_seconds=180):
        calls.append((source_path, marker_path, min_ttl_seconds))
        return SimpleNamespace(access_token='DEMO-key', expires_at=time.time()+500)
    module_stub.isolated_nous_credentials = isolated
    monkeypatch.setitem(sys.modules, module_stub.__name__, module_stub)
    monkeypatch.setattr(m, 'load_token', lambda p: 'DEMO-bearer')
    monkeypatch.setattr(m, 'port_free', lambda p: None)
    def child(*args):
        cfg.stop_file.write_text('stop')
        return 0, False
    monkeypatch.setattr(m, 'run_child', child)
    assert m.supervise(cfg) == 0
    assert calls == [(cfg.source, cfg.marker, 180)]
    assert m.supervise(replace(cfg, marker=tmp_path/'other-marker')) == 2
    assert calls == [(cfg.source, cfg.marker, 180)]


@pytest.mark.asyncio
async def test_portable_stop_drains_before_return(tmp_path, monkeypatch):
    from aiohttp import web
    stop=tmp_path/'stop'; entered=asyncio.Event(); release=asyncio.Event()
    class Site:
        def __init__(self,*a,**kw): pass
        async def start(self): stop.write_text('stop')
        async def stop(self): entered.set()
    monkeypatch.setattr(web,'TCPSite',Site)
    from gateway.platforms import api_server_bounded_runs as runs
    monkeypatch.setattr(runs,'_http_routes',lambda s: [])
    async def drain(): await release.wait()
    service=SimpleNamespace(_expires_at=None,_draining=False,drain=drain)
    task=asyncio.create_task(bounded_service.serve(service,8643,stop_file=stop))
    await asyncio.wait_for(entered.wait(),2)
    assert service._draining and not task.done()
    release.set()
    assert await asyncio.wait_for(task,2) == 'stopped'
    assert stop.exists()
