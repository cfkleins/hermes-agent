"""Dedicated, explicitly configured loopback service for bounded native runs.

Run in a fresh interpreter with --policy, --home and --port. The home is owned
by this immutable policy; it cannot adopt an ordinary Hermes conversation.
No generic gateway, profile routes, dotenv, or credential recovery is started.
"""
import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import dataclass, field
from hashlib import sha256
import hmac
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
from types import SimpleNamespace, MethodType

from gateway.bounded_service_storage import (
    own_home, safe_components, safe_node, safe_tree, seal_stores, validate_stores,
)


_POLICY_CAP = 16_384
_PROMPT_CAP = 131_072
MIN_NEW_RUN_LIFETIME = 180.0
EXPIRED_EXIT_CODE = 75
_MARKER = 'bounded-service.json'
_IDENTITIES = ('principal_id', 'owner_id', 'agent_id', 'case_id', 'source', 'case_selector')
_LANE = {
    'provider': 'nous',
    'model': 'openai/gpt-6-astra',
    'api_mode': 'chat_completions',
    'base_url': 'https://inference-api.nousresearch.com/v1',
    'token_env': 'VCC_BOUNDED_HERMES_TOKEN',
    'key_env': 'VCC_BOUNDED_NOUS_KEY',
}
_FIELDS = frozenset((*_IDENTITIES, *_LANE, 'version', 'prompt_file',
                     'prompt_sha256', 'context_length', 'required_skills'))
_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', re.ASCII)


class InvalidService(ValueError):
    """Never carry untrusted policy text or secrets in public diagnostics."""


def _validate_expiry(value):
    if value is None:
        return None
    if type(value) is str:
        if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', value, re.ASCII) is None:
            raise InvalidService()
        value = float(value)
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value <= 0 or value - time.time() <= MIN_NEW_RUN_LIFETIME):
        raise InvalidService()
    return value


@dataclass(frozen=True)
class LoadedPolicy:
    values: dict = field(repr=False)
    prompt: bytes = field(repr=False)
    token: str = field(repr=False)
    key: str = field(repr=False)
    digest: str


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidService()
        result[key] = value
    return result


def _reject_constant(_value):
    raise InvalidService()


def _absolute_path(value):
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise InvalidService()
    path = Path(value)
    if '..' in path.parts:
        raise InvalidService()
    safe_components(path)
    return path.resolve()


def _read_capped(path, cap):
    with path.open('rb') as stream:
        data = stream.read(cap + 1)
    if not data or len(data) > cap:
        raise InvalidService()
    return data


def _json_bytes(data):
    return json.loads(data.decode('utf-8'), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def _provider_key(value):
    # Keep this pure: importing OAuth before policy/home isolation is forbidden.
    if (type(value) is not str or not value or len(value) > 8192
            or any(ord(c) < 33 or ord(c) > 126 for c in value)
            or ',' in value or '${' in value):
        raise InvalidService()
    return value


def load_policy(path):
    """Validate every field and exact prompt bytes before any runtime imports."""
    values = _json_bytes(_read_capped(_absolute_path(path), _POLICY_CAP))
    if type(values) is not dict or set(values) != _FIELDS:
        raise InvalidService()
    if type(values['version']) is not int or values['version'] != 1:
        raise InvalidService()
    if any(type(values[name]) is not str or not _TOKEN.fullmatch(values[name])
           for name in _IDENTITIES):
        raise InvalidService()
    if any(values[name] != expected for name, expected in _LANE.items()):
        raise InvalidService()
    if (type(values['context_length']) is not int
            or not 65_536 <= values['context_length'] <= 16_777_216
            or values['required_skills'] != ['core-interview', 'project-issue-interview']):
        raise InvalidService()
    digest = values['prompt_sha256']
    if type(digest) is not str or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise InvalidService()
    prompt = _read_capped(_absolute_path(values['prompt_file']), _PROMPT_CAP)
    prompt.decode('utf-8', errors='strict')
    if not hmac.compare_digest(sha256(prompt).hexdigest(), digest):
        raise InvalidService()
    token, key = (os.environ.get(values[name], '') for name in ('token_env', 'key_env'))
    _provider_key(key)
    for secret in (token,):
        if (not secret or len(secret) > 512 or not secret.isascii()
                or any(ord(ch) < 33 or ord(ch) == 127 for ch in secret)
                or ',' in secret or '${' in secret):
            raise InvalidService()
    canonical = json.dumps(values, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return LoadedPolicy(values, prompt, token, key, sha256(canonical.encode()).hexdigest())


def _home_marker(home, policy):
    if not home.is_dir():
        raise InvalidService()
    path = home / _MARKER
    safe_node(path)
    marker = _json_bytes(_read_capped(path, _POLICY_CAP))
    if type(marker) is not dict or marker.get('policy_sha256') != policy.digest:
        raise InvalidService()
    return marker


def _prepare_policy(policy):
    from gateway.platforms.api_server_bound_admission import (
        BoundAdmissionRegistry, BoundCase, prepare_bound_run,
    )
    values = policy.values
    case = BoundCase(
        **{name: values[name] for name in _IDENTITIES if name != 'case_selector'},
        context_digest=values['prompt_sha256'], context_bytes=policy.prompt,
        skills=tuple(values['required_skills']), tool_policy='deny_all',
        **{name: values[name] for name in ('provider', 'model', 'api_mode',
                                         'base_url', 'context_length')},
        api_key=policy.key,
    )
    registry = BoundAdmissionRegistry(
        gateway_principals={policy.token: values['principal_id']},
        cases={(values['principal_id'], values['case_selector']): case},
    )
    # Exercise production policy/runtime validation, without dispatch or run reservation.
    prepared = prepare_bound_run(
        raw_body=json.dumps({'message': 'Bounded service preflight',
                             'case_selector': values['case_selector'],
                             'idempotency_key': 'bounded-service-preflight'}).encode(),
        authorization='Bearer ' + policy.token, idempotency_key=None, registry=registry,
    )
    return registry, prepared


class BoundedService:
    """Only the state and lifecycle required by the production bounded handlers."""

    def __init__(self, registry, session_db, run_store, *, expires_at=None):
        from gateway.platforms import api_server_bounded_runs as runs
        from gateway.status import get_process_start_time

        self._expires_at = _validate_expiry(expires_at)
        self.config = SimpleNamespace(extra={'bounded_admission_registry': registry})
        self._session_db = session_db
        self._run_idempotency_store = run_store
        self._run_owner_pid = os.getpid()
        self._run_owner_started = int(get_process_start_time(self._run_owner_pid) or 0)
        self._background_tasks = set()
        self._pending_agent_requests = 0
        self._draining = False
        runs.initialize_bounded_run_state(self)
        self._handle_bounded_runs = MethodType(runs.handle_bounded_runs, self)
        self._handle_get_bounded_run = MethodType(runs.handle_get_bounded_run, self)
        self._handle_stop_bounded_run = MethodType(runs.handle_stop_bounded_run, self)

    def _ensure_session_db(self):
        return self._session_db

    def _new_bounded_work_response(self):
        if (self._expires_at is not None
                and self._expires_at - time.time() <= MIN_NEW_RUN_LIFETIME):
            from aiohttp import web
            return web.json_response(
                {'error': {'code': 'bounded_generation_expired',
                           'message': 'Bounded generation requires renewal'}}, status=503)
        # Recheck after an HTTP body read that may have crossed shutdown.
        return self._draining_response()

    def _draining_response(self):
        if self._draining:
            from aiohttp import web
            return web.json_response({'error': {'code': 'gateway_draining',
                                      'message': 'Gateway is draining existing work'}},
                                     status=503, headers={'Retry-After': '1'})
        return None

    def _concurrency_limited_response(self, *, exclude_current_pending=False):
        # One immutable case: never permit two concurrent writers to its history.
        pending = self._pending_agent_requests - int(exclude_current_pending)
        if pending or self._bounded_active_run_tasks or self._bounded_pending_terminal_statuses:
            from aiohttp import web
            return web.json_response({'error': {'code': 'rate_limit_exceeded',
                                      'message': 'A bounded run is already active'}},
                                     status=429, headers={'Retry-After': '1'})
        return None

    async def drain(self):
        from gateway.platforms import api_server_bounded_runs as runs

        self._draining = True
        # Await executor-backed work rather than cancel its asyncio wrapper and
        # close SQLite while its native conversation thread is still writing.
        tasks = tuple(set(self._background_tasks) | set(self._bounded_active_run_tasks.values()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for run_id, (scope, _status) in tuple(self._bounded_pending_terminal_statuses.items()):
            runs._reconcile_pending_terminal_status(self, scope, run_id)


async def serve(service, port, *, stop_file=None):
    from aiohttp import web
    from gateway.platforms.api_server_bounded_runs import _http_routes

    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()

    def stop(_signum, _frame):
        service._draining = True
        loop.call_soon_threadsafe(stopped.set)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    app = web.Application(client_max_size=65_536)
    for method, path, handler in _http_routes(service):
        app.router.add_route(method, path, handler)
    runner = web.AppRunner(app, access_log=None, handle_signals=False)
    site = None
    expiry_timer = None
    stop_timer = None
    expired = False

    def check_stop():
        nonlocal stop_timer
        if stop_file.exists():
            service._draining = True
            stopped.set()
        else:
            stop_timer = loop.call_later(0.25, check_stop)

    def check_expiry():
        nonlocal expiry_timer, expired
        remaining = service._expires_at - time.time() - MIN_NEW_RUN_LIFETIME
        if remaining <= 0:
            expired = True
            service._draining = True
            stopped.set()
        else:
            # Recheck wall time periodically, including forward clock corrections.
            expiry_timer = loop.call_later(min(remaining, 1.0), check_expiry)

    try:
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', port, reuse_address=False)
        await site.start()
        if stop_file is not None:
            check_stop()
        if service._expires_at is not None:
            check_expiry()
        await stopped.wait()
    finally:
        if stop_timer is not None:
            stop_timer.cancel()
        if expiry_timer is not None:
            expiry_timer.cancel()
        service._draining = True
        try:
            if site is not None:
                await site.stop()
            # Finish admitted HTTP requests before collecting their executor tasks.
            await runner.cleanup()
            await service.drain()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    return 'expired' if expired else 'stopped'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--home', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--expires-at')
    parser.add_argument('--stop-file')
    args = parser.parse_args(argv)
    try:
        stop_file = _absolute_path(args.stop_file) if args.stop_file is not None else None
        expires_at = _validate_expiry(args.expires_at)
        outcome = None
        if not 1025 <= args.port <= 65535:
            raise InvalidService()
        home = _absolute_path(args.home)
        policy = load_policy(args.policy)
        fresh = not home.exists()
        # Do not mutate an unmarked existing directory, even to create a lock.
        marker = None if fresh else _home_marker(home, policy)
        if fresh:
            home.mkdir(mode=0o700, parents=True, exist_ok=False)
        with ExitStack() as resources:
            # Released last, after listener drain, background writers and DB close.
            resources.enter_context(own_home(home, fresh=fresh))
            safe_tree(home)
            if not fresh:
                marker = _home_marker(home, policy)
                validate_stores(home, marker, policy)
            os.chdir(home)
            os.environ['HERMES_HOME'] = str(home)
            from hermes_bounded_bootstrap import activate
            activate()
            registry, prepared = _prepare_policy(policy)
            from gateway.platforms.api_server_bound_admission import complete_bound_run
            from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
            from hermes_state import SessionDB

            session_db = SessionDB(db_path=home / 'state.db')
            resources.callback(session_db.close)
            run_store = RunIdempotencyStore(str(home / 'runs_idempotency.db'), require_durable=True)
            resources.callback(run_store.close)
            admission = complete_bound_run(prepared, session_db)
            if fresh:
                marker = seal_stores(home, policy.digest, admission.session_id)
                with (home / _MARKER).open('x', encoding='utf-8') as stream:
                    json.dump(marker, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
            elif admission.session_id != marker['session_id']:
                raise InvalidService()
            if args.check:
                print(json.dumps({'status': 'checked', 'session_id': admission.session_id}))
            else:
                service = BoundedService(registry, session_db, run_store, expires_at=expires_at)
                options = {'stop_file': stop_file} if stop_file is not None else {}
                outcome = asyncio.run(serve(service, args.port, **options))
        if outcome == 'expired':
            # ExitStack has closed both stores and released the owned-home lock.
            print(json.dumps({'status': 'generation_expired', 'storage_closed': True}))
            return EXPIRED_EXIT_CODE
        return 0
    except Exception as exc:
        # Type only: exception text/tracebacks can embed policy or credentials.
        print('bounded_service_invalid: ' + type(exc).__name__, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
