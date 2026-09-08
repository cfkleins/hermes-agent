"""Native two-generation proof; fixture injection exists only in this test.

Popen wrapper explicitly loads the existing SDK/audit fixture (no production
PYTHONPATH). Only bounded_service.time is advanced 500s after a file signal;
the production 180s cutoff, listener, drain, stores and CLI are unchanged.
"""
import base64
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

from tests.gateway.test_bounded_service import ROOT, FIXTURE, TOKEN, PROMPT, configuration, environment
from tests.gateway.test_bounded_service_process import request, terminal, body


@pytest.mark.windows_only
def test_native_dpapi_demo(tmp_path):
    import win32crypt
    from gateway.bounded_supervisor import load_token
    blob = win32crypt.CryptProtectData(b'DEMO-only-bearer', 'DEMO', None, None, None, 0)
    path = tmp_path / 'connection.json'
    path.write_text(json.dumps(dict(version=1, base_url='http://127.0.0.1:8643',
        case_selector='vector-primary-v1', token_dpapi=base64.b64encode(blob).decode())))
    assert load_token(path) == 'DEMO-only-bearer'


@pytest.mark.windows_only
def test_native_two_generations(tmp_path, monkeypatch):
    from gateway import bounded_supervisor as m
    path, policy, home = configuration(tmp_path)
    cfg = m.Config(path, home, tmp_path/'control', tmp_path/'connection.json',
                   tmp_path/'private'/'nous-login.dpapi', tmp_path/'uncertain')
    expire = tmp_path/'expire'
    entered = tmp_path/'entered'
    release = tmp_path/'release'
    exited = tmp_path/'worker-exit'
    calls, generations, errors, workers = [], [], [], []
    real_popen = subprocess.Popen
    real_run = m.run_child
    script = '''import os, sys, runpy, time, atexit
from pathlib import Path
from types import SimpleNamespace
runpy.run_path(sys.argv.pop(1))
from gateway import bounded_service as service
from openai.resources.chat.completions import Completions
expire, entered, release, exited = map(Path, sys.argv[1:5]); del sys.argv[1:5]
real_time = time.time
service.time = SimpleNamespace(time=lambda: real_time() + (500 if expire.exists() else 0))
original = Completions.create

def create(resource, **kw):
    if kw['messages'][-1]['content'] == 'held-native':
        entered.write_text(str(os.getpid()))
        expire.write_text('expire')
        deadline = time.monotonic()+30
        while not release.exists():
            if time.monotonic()>deadline: raise RuntimeError('fixture release timeout')
            time.sleep(.02)
    return original(resource, **kw)
Completions.create = create
atexit.register(lambda: exited.write_text(str(os.getpid())))
raise SystemExit(service.main(sys.argv[1:]))
'''
    def popen(command, **kw):
        env = environment(tmp_path, home)
        env.update(kw['env'])
        env.pop('PYTHONPATH', None)
        kw['env'] = env
        # -c retains the checkout cwd on sys.path, then main owns home.
        cmd = [command[0], '-c', script, str(FIXTURE/'sitecustomize.py'),
               str(expire), str(entered), str(release), str(exited), *command[3:]]
        process = real_popen(cmd, **kw)
        generations.append(process)
        return process
    monkeypatch.setattr(m.subprocess, 'Popen', popen)
    monkeypatch.setattr(m, 'load_token', lambda p: TOKEN)
    def credentials(*, source_path, marker_path, min_ttl_seconds):
        assert source_path == cfg.source and marker_path == cfg.marker
        assert min_ttl_seconds == 180
        if calls:
            assert exited.exists(), 'refresh before worker atexit'
            for worker in workers:
                assert not worker.is_running(), 'refresh before actual Windows worker exit'
            assert outcomes == [(75, True)]
            with m.own_home(home, fresh=False):
                with sqlite3.connect(home/'state.db') as db:
                    assert db.execute('pragma integrity_check').fetchone() == ('ok',)
            expire.unlink()
        calls.append(time.monotonic())
        return SimpleNamespace(access_token='DEMO-generation-'+str(len(calls)), expires_at=time.time()+600)
    outcomes = []
    def run(*args):
        result = real_run(*args)
        outcomes.append(result)
        return result
    monkeypatch.setattr(m, 'run_child', run)
    def wait_for(predicate, timeout=40):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            if predicate(): return
            time.sleep(.03)
        raise AssertionError('fixture condition timeout')
    def ready():
        try: return request(8643, 'GET', '/v1/bounded-runs/brun_probe')[0] == 404
        except OSError: return False
    results = {}
    def drive():
        try:
            wait_for(ready)
            workers.extend(psutil.Process(generations[0].pid).children(recursive=True))
            assert workers, 'native Windows venv launcher topology not exercised'
            code, first = request(8643, 'POST', '/v1/bounded-runs', body('first', 'one'))
            assert code == 202, first
            first_result = terminal(8643, first['run_id'])
            assert first_result['status'] == 'completed'
            code, held = request(8643, 'POST', '/v1/bounded-runs', body('held-native', 'held'))
            assert code == 202, held
            wait_for(entered.exists)
            # Expiry closes admission while the native SDK thread remains held.
            time.sleep(2)
            assert len(calls) == 1 and not exited.exists()
            assert all(w.is_running() for w in workers)
            release.write_text('release')
            wait_for(lambda: len(generations)==2)
            wait_for(ready)
            code, replay = request(8643, 'POST', '/v1/bounded-runs', body('first', 'one'))
            assert code == 202 and replay['replayed'] and replay['run_id'] == first['run_id']
            assert request(8643, 'POST', '/v1/bounded-runs', body('changed', 'one'))[0] == 409
            assert request(8643, 'GET', '/v1/bounded-runs/'+first['run_id'], token='wrong')[0] == 401
            code, second = request(8643, 'POST', '/v1/bounded-runs', body('second', 'two'))
            assert code == 202, second
            result = terminal(8643, second['run_id'])
            assert result['status'] == 'completed' and result['session_id'] == first_result['session_id']
            results.update(session_id=result['session_id'], first_run=first['run_id'])
        except BaseException as exc:
            errors.append(exc)
        finally:
            release.write_text('release')
            m.write_stop(cfg.stop_file)
    driver = threading.Thread(target=drive)
    driver.start()
    started = time.monotonic()
    code = m.supervise(cfg, credentials=credentials)
    driver.join(45)
    assert not driver.is_alive()
    if errors: raise errors[0]
    assert code == 0 and outcomes == [(75, True), (0, False)]
    assert len(calls) == 2
    events = [json.loads(line) for line in (tmp_path/'fixture.jsonl').read_text().splitlines()]
    assert len(events) == 3 and all(e['event']=='sdk_request' for e in events), events
    for event in events:
        req = event['request']
        assert req['messages'][0] == {'role':'system', 'content':PROMPT.decode()}
        assert not req.get('tools') and req['model'] == policy['model']
    history = events[-1]['request']['messages']
    assert any(v.get('content') == 'fixture answer: first' for v in history)
    with sqlite3.connect(home/'state.db') as db:
        assert db.execute('select session_id from session_bindings').fetchall() == [(results['session_id'],)]
    print(json.dumps(dict(native_generations=2, outcomes=outcomes,
        worker_pids=[w.pid for w in workers], seconds=time.monotonic()-started,
        sdk_requests=len(events), continuity=True)))
