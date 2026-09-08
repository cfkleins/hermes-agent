"""Offline expiry fencing and natural generation drain contracts."""
import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from gateway import bounded_service as service_module
from gateway.platforms import api_server_bounded_runs as runs
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_state import SessionDB
from tests.gateway.test_api_server_bounded_runs import make_registry, request_body, completed_agent, GATEWAY_KEY


class Request:
    def __init__(self, body=None, authorization=None):
        self.headers = {'Authorization': authorization or 'Bearer ' + GATEWAY_KEY}
        self.body = request_body() if body is None else body
        self.content = self
        self.match_info = {}

    async def iter_chunked(self, size):
        yield self.body


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(service_module, 'time', SimpleNamespace(time=lambda: 1000), raising=False)
    db = SessionDB(tmp_path / 'state.db')
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'), require_durable=True)
    instance = service_module.BoundedService(make_registry(), db, store)
    # Also exercises late boundary crossing rather than constructor-only checking.
    instance._expires_at = 1181.0
    yield instance
    store.close()
    db.close()


@pytest.mark.asyncio
async def test_expiry_fences_new_work_but_not_replay_or_auth(service, monkeypatch):
    monkeypatch.setattr(runs, 'create_bound_agent', lambda *a, **k: completed_agent())
    accepted = await service._handle_bounded_runs(Request())
    assert accepted.status == 202
    await asyncio.gather(*service._background_tasks)
    monkeypatch.setattr(service_module.time, 'time', lambda: 1001)
    for name in ('reserve',):
        monkeypatch.setattr(service._run_idempotency_store, name, Mock(side_effect=AssertionError(name)))
    monkeypatch.setattr(service, '_ensure_session_db', Mock(side_effect=AssertionError('session')))
    monkeypatch.setattr(runs, 'create_bound_agent', Mock(side_effect=AssertionError('agent')))
    response = await service._handle_bounded_runs(Request(request_body(key='new')))
    assert response.status == 503
    assert json.loads(response.text)['error']['code'] == 'bounded_generation_expired'
    replay = await service._handle_bounded_runs(Request())
    assert replay.status == 202 and json.loads(replay.text)['replayed'] is True
    conflict = await service._handle_bounded_runs(Request(request_body(message='different')))
    assert conflict.status == 409
    assert (await service._handle_bounded_runs(Request(authorization='Bearer wrong'))).status == 401
    assert (await service._handle_bounded_runs(Request(b'{'))).status == 400
    status_request = Request()
    status_request.match_info = {'run_id': json.loads(accepted.text)['run_id']}
    assert (await service._handle_get_bounded_run(status_request)).status == 200
    service._run_idempotency_store.reserve.assert_not_called()
    service._ensure_session_db.assert_not_called()
    runs.create_bound_agent.assert_not_called()


@pytest.mark.parametrize('expiry', ['NaN', 'Infinity', '-1', '0', 'true', '', ' 1181', '1_181', '1180', '1179.99'])
def test_cli_invalid_expiry_before_policy_or_home_writes(tmp_path, monkeypatch, expiry):
    monkeypatch.setattr(service_module, 'time', SimpleNamespace(time=lambda: 1000), raising=False)
    policy = Mock(side_effect=AssertionError('policy must not be loaded'))
    monkeypatch.setattr(service_module, 'load_policy', policy)
    home = tmp_path / 'new-home'
    assert service_module.main(['--policy', str(tmp_path / 'missing'), '--home', str(home),
                                '--port', '38653', '--expires-at=' + expiry]) == 2
    assert not home.exists()
    policy.assert_not_called()


@pytest.mark.asyncio
async def test_expiry_serve_waits_for_native_work_without_cancellation(service, monkeypatch):
    from aiohttp import web
    entered = asyncio.Event()
    released = threading.Event()
    listener_stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    agent = completed_agent()

    def native(**kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert released.wait(5), 'test failed to release native work'
        return {'final_response': 'naturally completed'}

    agent.run_conversation.side_effect = native
    monkeypatch.setattr(runs, 'create_bound_agent', lambda *a, **k: agent)
    assert (await service._handle_bounded_runs(Request())).status == 202
    await entered.wait()
    tasks = tuple(service._background_tasks)

    class Site:
        def __init__(self, *args, **kwargs):
            pass
        async def start(self):
            monkeypatch.setattr(service_module.time, 'time', lambda: 1001)
        async def stop(self):
            listener_stopped.set()

    monkeypatch.setattr(web, 'TCPSite', Site)
    serving = asyncio.create_task(service_module.serve(service, 38653))
    try:
        await asyncio.wait_for(listener_stopped.wait(), 2)
        assert not serving.done()
        assert all(not task.done() for task in tasks)
        assert service._draining
    finally:
        released.set()
    assert await asyncio.wait_for(serving, 3) == 'expired'
    assert all(task.done() and not task.cancelled() for task in tasks)
    assert not service._bounded_pending_terminal_statuses
    assert not service._bounded_active_run_tasks
    agent.interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_drain_awaits_active_tasks_even_outside_background_set(service):
    started, release = asyncio.Event(), asyncio.Event()
    async def active():
        started.set()
        await release.wait()
    task = asyncio.create_task(active())
    service._bounded_active_run_tasks['owned'] = task
    await started.wait()
    drain = asyncio.create_task(service.drain())
    await asyncio.sleep(0)
    try:
        assert not drain.done()
        assert not task.cancelled()
    finally:
        release.set()
        await task
        await drain


@pytest.mark.parametrize('unresolved', [False, True])
def test_main_receipt_only_after_stores_close_under_lock(tmp_path, monkeypatch, capsys, unresolved):
    from contextlib import contextmanager
    import hermes_bounded_bootstrap
    from tests.gateway.test_bounded_service import configuration, TOKEN, KEY
    path, _, home = configuration(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('VCC_BOUNDED_HERMES_TOKEN', TOKEN)
    monkeypatch.setenv('VCC_BOUNDED_NOUS_KEY', KEY)
    monkeypatch.setattr(hermes_bounded_bootstrap, 'activate', lambda: None)
    monkeypatch.setattr(service_module, 'time', SimpleNamespace(time=lambda: 1000))
    events = []
    own_home = service_module.own_home
    db_close, store_close = SessionDB.close, RunIdempotencyStore.close

    @contextmanager
    def locked(*args, **kwargs):
        with own_home(*args, **kwargs):
            events.append('locked')
            yield
            assert events[-2:] == ['store_closed', 'db_closed']
        events.append('unlocked')

    def close_db(self):
        db_close(self)
        events.append('db_closed')

    def close_store(self):
        store_close(self)
        events.append('store_closed')

    async def expired(instance, port):
        assert instance._expires_at == 1181
        if unresolved:
            instance._bounded_pending_terminal_statuses['missing'] = ('scope', {'status': 'completed'})
        await instance.drain()
        return 'expired'

    monkeypatch.setattr(service_module, 'own_home', locked)
    monkeypatch.setattr(SessionDB, 'close', close_db)
    monkeypatch.setattr(RunIdempotencyStore, 'close', close_store)
    monkeypatch.setattr(service_module, 'serve', expired)
    code = service_module.main(['--policy', str(path), '--home', str(home), '--port', '38653',
                                '--expires-at', '1181'])
    output = capsys.readouterr()
    assert events[:3] == ['locked', 'store_closed', 'db_closed']
    if unresolved:
        assert code == 2
        assert 'generation_expired' not in output.out
    else:
        assert events[-1] == 'unlocked'
        assert code == service_module.EXPIRED_EXIT_CODE
        assert json.loads(output.out) == {'status': 'generation_expired', 'storage_closed': True}
