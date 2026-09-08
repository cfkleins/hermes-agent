"""Actual configured subprocess cold start/restart, replacing only SDK I/O."""
import json
from contextlib import ExitStack
import os
import signal
import socket
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request

import pytest
import psutil

from tests.gateway.test_bounded_service import (
    ROOT, TOKEN, PROMPT, command, configuration, environment,
)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def request(port, method, path, body=None, token=TOKEN):
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (port, path),
                                 data=None if body is None else json.dumps(body).encode(),
                                 method=method, headers={'Authorization': 'Bearer ' + token,
                                                       'Content-Type': 'application/json'})
    try:
        response = urllib.request.urlopen(req, timeout=2)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        raw = response.read()
        return response.status, json.loads(raw) if raw.startswith(b'{') else raw.decode()


def start(tmp_path, path, home, port, index):
    log = (tmp_path / ('process-%s.log' % index)).open('w+', encoding='utf-8')
    process = subprocess.Popen(command(path, home, port), cwd=ROOT,
                               env=environment(tmp_path, home), stdout=log, stderr=log)
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.seek(0)
            pytest.fail('configured service exited: ' + log.read())
        try:
            code, data = request(port, 'GET', '/v1/bounded-runs/brun_probe')
            if code == 404 and isinstance(data, dict):
                assert data['error']['code'] == 'bounded_run_not_found'
                return process, log
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.05)
    stop_process(process)
    log.seek(0)
    diagnostic = log.read()
    log.close()
    pytest.fail('configured service never ready: ' + diagnostic)


def terminal(port, run_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        code, data = request(port, 'GET', '/v1/bounded-runs/' + run_id)
        assert code == 200, data
        if data['status'] in {'completed', 'failed', 'cancelled', 'interrupted'}:
            return data
        time.sleep(0.05)
    pytest.fail('native bounded run never terminal')


def body(message, key):
    return dict(message=message, case_selector='vector_today', idempotency_key=key)


def stop_process(process):
    # A Windows venv launcher can exit before its interpreter releases SQLite
    # mappings. Retain real descendant handles before killing that launcher.
    with ExitStack() as resources:
        handles = []
        if os.name == 'nt' and process.poll() is None:
            import _winapi
            for child in psutil.Process(process.pid).children(recursive=True):
                try:
                    handle = _winapi.OpenProcess(0x00100000, False, child.pid)
                except OSError:
                    if child.is_running():
                        raise
                    continue
                resources.callback(_winapi.CloseHandle, handle)
                handles.append(handle)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=15)
        for handle in handles:
            assert _winapi.WaitForSingleObject(handle, 15000) == _winapi.WAIT_OBJECT_0


def kill_if_live(process, log):
    try:
        stop_process(process)
    finally:
        log.close()


def test_configured_process_cold_start_restart_and_sigterm_drain(tmp_path):
    path, _, home = configuration(tmp_path)
    port = free_port()
    first, log = start(tmp_path, path, home, port, 1)
    try:
        for method, route in [('POST', '/v1/runs'), ('GET', '/health'),
                              ('GET', '/api/sessions'), ('GET', '/v1/skills'),
                              ('GET', '/p/default/v1/bounded-runs/brun_probe')]:
            assert request(port, method, route)[0] == 404
        assert request(port, 'GET', '/v1/bounded-runs/brun_probe', token='wrong')[0] == 401
        code, accepted = request(port, 'POST', '/v1/bounded-runs', body('first', 'one'))
        assert code == 202, accepted
        result = terminal(port, accepted['run_id'])
        assert result['status'] == 'completed', result
        assert result['output'] == 'fixture answer: first'
        session_id = result['session_id']
        # Trigger registered SIGTERM during real executor work and await cleanup.
        code, draining = request(port, 'POST', '/v1/bounded-runs', body('fixture-sigterm', 'drain'))
        assert code == 202, draining
        assert first.wait(timeout=20) == 0
    finally:
        kill_if_live(first, log)

    second, log = start(tmp_path, path, home, port, 2)
    try:
        code, replay = request(port, 'POST', '/v1/bounded-runs', body('first', 'one'))
        assert code == 202 and replay['run_id'] == accepted['run_id']
        assert replay['replayed'] is True
        drained = terminal(port, draining['run_id'])
        assert drained['status'] in {'completed', 'cancelled', 'failed'}, drained
        code, resumed = request(port, 'POST', '/v1/bounded-runs', body('second', 'two'))
        assert code == 202, resumed
        result = terminal(port, resumed['run_id'])
        assert result['session_id'] == session_id
        assert result['status'] == 'completed', result
        # Terminal Stop is an idempotent read of the completed status (HTTP 200).
        code, stopped = request(port, 'POST', '/v1/bounded-runs/' + resumed['run_id'] + '/stop')
        assert code == 200 and stopped == result
    finally:
        kill_if_live(second, log)

    events = [json.loads(line) for line in (tmp_path / 'fixture.jsonl').read_text().splitlines()]
    assert all(event['event'] == 'sdk_request' for event in events), events
    calls = [event['request'] for event in events]
    assert len(calls) == 3
    for call in calls:
        assert call['messages'][0] == {'role': 'system', 'content': PROMPT.decode()}
        assert not call.get('tools')
        assert set(call['timeout']) == {'connect', 'read', 'write', 'pool'}
        assert all(value > 0 for value in call['timeout'].values())
    prior_messages = calls[2]['messages'][1:3]
    # Native history carries persistence metadata as well as exact role/content.
    for message in prior_messages:
        assert set(message) <= {'role', 'content', '_db_persisted', 'timestamp'}
        assert message.get('_db_persisted') is True
        assert isinstance(message.get('timestamp'), (int, float))
    assert [{key: message[key] for key in ('role', 'content')} for message in prior_messages] == [
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'content': 'fixture answer: first'},
    ]
    with sqlite3.connect(home / 'state.db') as db:
        assert db.execute('select session_id from session_bindings').fetchall() == [(session_id,)]
        messages = db.execute('select role, content from messages where session_id=? order by id',
                              (session_id,)).fetchall()
    assert messages[:2] == [('user', 'first'), ('assistant', 'fixture answer: first')]
    assert messages[-2:] == [('user', 'second'), ('assistant', 'fixture answer: second')]
    assert not list((tmp_path / 'ambient').rglob('*'))


@pytest.mark.linux_only
def test_external_sigterm_stops_configured_listener(tmp_path):
    path, _, home = configuration(tmp_path)
    process, log = start(tmp_path, path, home, free_port(), 'posix')
    try:
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=20) == 0
    finally:
        kill_if_live(process, log)
