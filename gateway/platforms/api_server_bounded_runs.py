"""Isolated durable HTTP execution for server-owned bounded runs."""

import asyncio
from contextlib import suppress
from hashlib import sha256
import logging
import time
import uuid
from types import MappingProxyType
from typing import Any

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

from gateway.platforms.api_server_bound_admission import (
    MAX_RAW_BODY_BYTES,
    BoundAdmission,
    BoundAdmissionError,
    BoundAdmissionRegistry,
    PreparedBoundAdmission,
    authenticate_bound_authorization,
    complete_bound_run,
    prepare_bound_run,
)
from gateway.platforms.api_server_run_idempotency import TERMINAL_STATUSES
from gateway.platforms.api_server_runs import _owner_alive
from run_agent import AIAgent


logger = logging.getLogger("gateway.platforms.api_server")

BOUNDED_MAX_ITERATIONS = 1
BOUNDED_API_MAX_RETRIES = 1
BOUNDED_RUN_BUDGET_SECONDS = 120.0
BOUNDED_MAX_TOKENS = 4096
BOUNDED_REASONING_POLICY = MappingProxyType({"enabled": False})


def initialize_bounded_run_state(adapter) -> None:
    """Keep bounded ownership and controls isolated from generic ``/v1/runs``."""
    adapter._bounded_run_owners = {}
    adapter._bounded_run_statuses = {}
    adapter._bounded_active_run_agents = {}
    adapter._bounded_active_run_tasks = {}
    adapter._bounded_stopping_run_ids = set()
    adapter._bounded_pending_terminal_statuses = {}


def _http_routes(adapter) -> list[tuple[str, str, Any]]:
    return [
        ("POST", "/v1/bounded-runs", adapter._handle_bounded_runs),
        ("GET", "/v1/bounded-runs/{run_id}", adapter._handle_get_bounded_run),
        ("POST", "/v1/bounded-runs/{run_id}/stop", adapter._handle_stop_bounded_run),
    ]


class BoundedHTTPError(ValueError):
    """Sanitized bounded-route failure."""

    def __init__(self, status: int, code: str, message: str):
        self.status = status
        self.code = code
        self.message = message
        super().__init__(message)


def _error(error: BoundedHTTPError) -> "web.Response":
    return web.json_response(
        {"error": {"code": error.code, "message": error.message}}, status=error.status
    )


def _admission_error(error: BoundAdmissionError) -> BoundedHTTPError:
    status_by_code = {
        "unauthorized": 401,
        "invalid_request": 400,
        "invalid_idempotency_key": 400,
        "not_found": 404,
        "binding_conflict": 409,
        "invalid_server_policy": 503,
        "runtime_unavailable": 503,
        "state_unavailable": 503,
    }
    return BoundedHTTPError(
        status_by_code.get(error.code, 503), f"bounded_{error.code}", str(error)
    )


def _resolve_registry(adapter) -> BoundAdmissionRegistry:
    registry = (adapter.config.extra or {}).get("bounded_admission_registry")
    if not isinstance(registry, BoundAdmissionRegistry):
        raise BoundedHTTPError(
            503, "bounded_policy_unavailable", "Bounded admission policy is unavailable"
        )
    return registry


def _authorization(request) -> str:
    headers = request.headers
    getall = getattr(headers, "getall", None)
    if callable(getall):
        values = list(getall("Authorization", []))
    else:
        value = headers.get("Authorization")
        values = [] if value is None else [value]
    if len(values) != 1 or type(values[0]) is not str:
        raise BoundedHTTPError(
            401, "bounded_unauthorized", "Bounded admission unauthorized"
        )
    return values[0]


def _authenticate(request, registry: BoundAdmissionRegistry) -> str:
    try:
        return authenticate_bound_authorization(_authorization(request), registry)
    except BoundAdmissionError as exc:
        raise _admission_error(exc) from None


def _require_durable_store(adapter) -> None:
    store = getattr(adapter, "_run_idempotency_store", None)
    if store is None or store.durable is not True:
        raise BoundedHTTPError(
            503,
            "bounded_idempotency_unavailable",
            "Durable bounded idempotency is unavailable",
        )


def _reject_declared_oversize(request) -> None:
    value = request.headers.get("Content-Length")
    if value is None:
        return
    try:
        declared = int(value)
    except (TypeError, ValueError):
        raise BoundedHTTPError(400, "bounded_invalid_request", "Invalid bounded request") from None
    if declared < 0:
        raise BoundedHTTPError(400, "bounded_invalid_request", "Invalid bounded request")
    if declared > MAX_RAW_BODY_BYTES:
        raise BoundedHTTPError(413, "bounded_request_too_large", "Bounded request is too large")


async def read_bounded_raw_body(request) -> bytes:
    """Read the immutable request bytes once, stopping at the bounded cap."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.content.iter_chunked(MAX_RAW_BODY_BYTES + 1):
        total += len(chunk)
        if total > MAX_RAW_BODY_BYTES:
            raise BoundedHTTPError(
                413, "bounded_request_too_large", "Bounded request is too large"
            )
        chunks.append(bytes(chunk))
    return b"".join(chunks)


def _digest_fields(namespace: str, fields: list[tuple[str, object]]) -> str:
    """Hash typed, length-prefixed fields without ambiguous string joining."""
    digest = sha256()
    for label, value in [("namespace", namespace), *fields]:
        label_bytes = label.encode("utf-8")
        value_bytes = value if isinstance(value, bytes) else str(value).encode("utf-8")
        digest.update(len(label_bytes).to_bytes(4, "big"))
        digest.update(label_bytes)
        digest.update(len(value_bytes).to_bytes(8, "big"))
        digest.update(value_bytes)
    return digest.hexdigest()


def _request_profile() -> str:
    from gateway.platforms.api_server import _api_request_profile

    return str(_api_request_profile.get() or "default")


def _idempotency_scope(prepared: PreparedBoundAdmission, *, profile: str) -> str:
    identity = prepared.binding_identity
    return _digest_fields("bounded-runs-scope-v1", [
        ("profile", profile),
        ("principal_id", prepared.principal_id),
        ("owner_id", identity["owner_id"]),
        ("agent_id", identity["agent_id"]),
        ("case_id", identity["case_id"]),
    ])


def _idempotency_fingerprint(prepared: PreparedBoundAdmission, raw_body: bytes) -> str:
    identity = prepared.binding_identity
    runtime = prepared.runtime
    fields: list[tuple[str, object]] = [
        ("raw_body_sha256", sha256(raw_body).hexdigest()),
        ("principal_id", prepared.principal_id),
        *((name, identity[name]) for name in (
            "owner_id", "agent_id", "case_id", "source", "context_digest"
        )),
        ("provider", runtime.provider),
        ("model", runtime.model),
        ("api_mode", runtime.api_mode),
        ("base_url", runtime.base_url),
        ("context_length", runtime.context_length),
        ("tool_policy", prepared.tool_policy),
        ("max_iterations", BOUNDED_MAX_ITERATIONS),
        ("api_max_retries", BOUNDED_API_MAX_RETRIES),
        ("run_budget_seconds", BOUNDED_RUN_BUDGET_SECONDS),
        ("max_tokens", BOUNDED_MAX_TOKENS),
        ("reasoning_enabled", BOUNDED_REASONING_POLICY["enabled"]),
        ("save_trajectories", False),
        ("verbose_logging", False),
        ("quiet_mode", True),
        ("skills_count", len(prepared.skills)),
        *((f"skill_{index}", skill) for index, skill in enumerate(prepared.skills)),
    ]
    return _digest_fields("bounded-runs-fingerprint-v1", fields)


def _case_scope(principal: str, case, *, profile: str) -> str:
    return _digest_fields("bounded-runs-scope-v1", [
        ("profile", profile),
        ("principal_id", principal),
        ("owner_id", case.owner_id),
        ("agent_id", case.agent_id),
        ("case_id", case.case_id),
    ])


def _initial_status(run_id: str, prepared: PreparedBoundAdmission) -> dict[str, Any]:
    now = time.time()
    return {
        "object": "hermes.bounded_run",
        "run_id": run_id,
        "status": "queued",
        "admission_state": "reserved",
        "created_at": now,
        "updated_at": now,
        "model": prepared.runtime.model,
    }


def _accepted(run_id: str, status: str, *, replayed: bool) -> "web.Response":
    headers = {"Idempotency-Replayed": "true"} if replayed else None
    return web.json_response(
        {"run_id": run_id, "status": status, "replayed": replayed},
        status=202,
        headers=headers,
    )


def _is_unpublished(record: dict | None) -> bool:
    return (record or {}).get("status", {}).get("admission_state") == "reserved"


def _reject_unpublished_replay(record: dict | None) -> None:
    if _is_unpublished(record):
        raise BoundedHTTPError(
            503,
            "bounded_run_initializing",
            "Bounded run admission is still initializing",
        )


def _recover_unpublished_reservation(
    adapter,
    *,
    scope: str,
    key: str,
    fingerprint: str,
    record: dict | None,
) -> bool:
    if not _is_unpublished(record):
        return False
    if _owner_alive(
        int(record.get("owner_pid") or 0), int(record.get("owner_started") or 0)
    ):
        _reject_unpublished_replay(record)
    try:
        recovered = adapter._run_idempotency_store.rollback_reservation(
            scope, key, fingerprint, record["run_id"]
        )
    except Exception:
        recovered = False
    if not recovered:
        raise BoundedHTTPError(
            503,
            "bounded_state_unavailable",
            "Bounded runtime state is unavailable",
        )
    return True


def _status_payload(adapter, run_id: str, status: str, **fields) -> dict[str, Any]:
    current = dict(adapter._bounded_run_statuses.get(run_id, {}))
    now = time.time()
    current.update(
        object="hermes.bounded_run", run_id=run_id, status=status, updated_at=now
    )
    current.setdefault("created_at", now)
    current.update(fields)
    return current


def _set_status(adapter, scope: str, run_id: str, status: str, **fields) -> dict[str, Any]:
    current = _status_payload(adapter, run_id, status, **fields)
    adapter._run_idempotency_store.update_status_strict(scope, run_id, current)
    adapter._bounded_run_statuses[run_id] = current
    return current


def _release_terminal_references(adapter, run_id: str) -> None:
    adapter._bounded_active_run_agents.pop(run_id, None)
    adapter._bounded_active_run_tasks.pop(run_id, None)
    adapter._bounded_stopping_run_ids.discard(run_id)
    adapter._bounded_pending_terminal_statuses.pop(run_id, None)


def _retain_pending_terminal_status(
    adapter, scope: str, run_id: str, status: str, **fields
) -> None:
    adapter._bounded_pending_terminal_statuses[run_id] = (
        scope,
        _status_payload(adapter, run_id, status, **fields),
    )


def create_bound_agent(admission: BoundAdmission, *, session_db) -> AIAgent:
    """Construct only the exact, tool-free runtime authorized by admission."""
    runtime = admission.runtime
    agent = AIAgent(
        base_url=runtime.base_url,
        api_key=runtime.api_key,
        provider=runtime.provider,
        requested_provider=runtime.provider,
        api_mode=runtime.api_mode,
        model=runtime.model,
        enabled_toolsets=[],
        deny_all_tools=True,
        require_durable_history=True,
        credential_pool=None,
        fallback_model=None,
        request_overrides=None,
        max_iterations=BOUNDED_MAX_ITERATIONS,
        run_budget_seconds=BOUNDED_RUN_BUDGET_SECONDS,
        max_tokens=BOUNDED_MAX_TOKENS,
        reasoning_config=dict(BOUNDED_REASONING_POLICY),
        save_trajectories=False,
        verbose_logging=False,
        quiet_mode=True,
        exact_system_prompt_bytes=admission.exact_system_prompt_bytes,
        exact_context_length=runtime.context_length,
        session_db=session_db,
        session_id=admission.session_id,
        platform="api_server",
        skip_context_files=True,
        load_soul_identity=False,
        skip_memory=True,
        skip_background_review=True,
    )
    agent._api_max_retries = BOUNDED_API_MAX_RETRIES
    return agent


def _validate_bound_agent(admission: BoundAdmission, agent) -> None:
    """Fail closed if generic initialization changed admitted runtime intent."""
    runtime = admission.runtime
    try:
        valid = (
            agent.provider == runtime.provider
            and agent.requested_provider == runtime.provider
            and agent.model == runtime.model
            and agent.api_mode == runtime.api_mode
            and agent.base_url == runtime.base_url
            and agent._credential_pool is None
            and agent._fallback_model is None
            and agent.request_overrides in (None, {})
            and agent.max_iterations == BOUNDED_MAX_ITERATIONS
            and agent.iteration_budget.max_total == BOUNDED_MAX_ITERATIONS
            and agent._api_max_retries == BOUNDED_API_MAX_RETRIES
            and agent.run_budget_seconds == BOUNDED_RUN_BUDGET_SECONDS
            and agent.max_tokens == BOUNDED_MAX_TOKENS
            and agent.reasoning_config == BOUNDED_REASONING_POLICY
            and agent.save_trajectories is False
            and agent.verbose_logging is False
            and agent.quiet_mode is True
            and agent._exact_system_prompt
            == admission.exact_system_prompt_bytes.decode("utf-8")
            and agent._exact_context_length == runtime.context_length
            and agent.deny_all_tools is True
            and agent.require_durable_history is True
            and agent.tools == []
        )
    except Exception:
        valid = False
    if not valid:
        raise BoundedHTTPError(
            503,
            "bounded_runtime_unavailable",
            "Bounded runtime is unavailable",
        )


async def _execute_bounded_run(
    adapter, *, run_id: str, scope: str, admission: BoundAdmission, agent
) -> None:
    loop = asyncio.get_running_loop()
    terminal_persisted = False
    try:
        _set_status(adapter, scope, run_id, "running")
        result = await loop.run_in_executor(
            None,
            lambda: agent.run_conversation(
                user_message=admission.input_text,
                conversation_history=None,
                task_id=admission.session_id,
                binding_identity=dict(admission.binding_identity),
            ),
        )
        result = result if isinstance(result, dict) else {}
        usage = {
            "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
            "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
            "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
        }
        if run_id in adapter._bounded_stopping_run_ids and result.get("interrupted") is True:
            _set_status(adapter, scope, run_id, "cancelled")
        elif result.get("failed"):
            _set_status(adapter, scope, run_id, "failed", error="Bounded run failed")
        else:
            _set_status(
                adapter,
                scope,
                run_id,
                "completed",
                output=result.get("final_response", ""),
                usage=usage,
            )
        terminal_persisted = True
    except asyncio.CancelledError:
        try:
            _set_status(adapter, scope, run_id, "cancelled")
            terminal_persisted = True
        except Exception as status_exc:
            _retain_pending_terminal_status(adapter, scope, run_id, "cancelled")
            logger.error(
                "[api_server] bounded terminal status persistence pending "
                "(run_id=%s error_type=%s)",
                run_id,
                type(status_exc).__name__,
            )
        raise
    except Exception as exc:
        logger.error(
            "[api_server] bounded run failed (run_id=%s error_type=%s)",
            run_id,
            type(exc).__name__,
        )
        try:
            _set_status(adapter, scope, run_id, "failed", error="Bounded run failed")
            terminal_persisted = True
        except Exception as status_exc:
            _retain_pending_terminal_status(
                adapter, scope, run_id, "failed", error="Bounded run failed"
            )
            logger.error(
                "[api_server] bounded terminal status persistence pending "
                "(run_id=%s error_type=%s)",
                run_id,
                type(status_exc).__name__,
            )
    finally:
        if terminal_persisted:
            _release_terminal_references(adapter, run_id)


def _remove_prelaunch_state(adapter, run_id: str) -> None:
    adapter._bounded_run_owners.pop(run_id, None)
    adapter._bounded_run_statuses.pop(run_id, None)
    adapter._bounded_active_run_agents.pop(run_id, None)
    task = adapter._bounded_active_run_tasks.pop(run_id, None)
    if task is not None and not task.done():
        task.cancel()
    adapter._bounded_stopping_run_ids.discard(run_id)
    adapter._bounded_pending_terminal_statuses.pop(run_id, None)


def _rollback_prelaunch_reservation(
    adapter, reservation: tuple[str, str, str, str]
) -> bool:
    scope, key, fingerprint, run_id = reservation
    _remove_prelaunch_state(adapter, run_id)
    try:
        rolled_back = adapter._run_idempotency_store.rollback_reservation(
            scope, key, fingerprint, run_id
        )
    except Exception as exc:
        logger.error(
            "[api_server] bounded prelaunch rollback failed "
            "(run_id=%s error_type=%s)",
            run_id,
            type(exc).__name__,
        )
        return False
    if not rolled_back:
        logger.error(
            "[api_server] bounded prelaunch rollback failed "
            "(run_id=%s error_type=reservation_mismatch)",
            run_id,
        )
        return False
    return True


async def handle_bounded_runs(adapter, request) -> "web.Response":
    """POST /v1/bounded-runs."""
    reservation_active = False
    created_reservation: tuple[str, str, str, str] | None = None
    try:
        registry = _resolve_registry(adapter)
        _authenticate(request, registry)
        draining = adapter._draining_response()
        if draining is not None:
            return draining
        adapter._pending_agent_requests += 1
        reservation_active = True
        _require_durable_store(adapter)
        _reject_declared_oversize(request)

        raw_body = await read_bounded_raw_body(request)
        try:
            prepared = prepare_bound_run(
                raw_body=raw_body,
                authorization=_authorization(request),
                idempotency_key=None,
                registry=registry,
            )
        except BoundAdmissionError as exc:
            raise _admission_error(exc) from None

        profile = _request_profile()
        scope = _idempotency_scope(prepared, profile=profile)
        fingerprint = _idempotency_fingerprint(prepared, raw_body)
        lookup_outcome, lookup_record = adapter._run_idempotency_store.lookup(
            scope, prepared.idempotency_key, fingerprint
        )
        if lookup_outcome == "conflict":
            raise BoundedHTTPError(
                409,
                "bounded_idempotency_conflict",
                "Bounded idempotency key conflicts with an existing request",
            )
        if lookup_outcome == "reused":
            recovered = _recover_unpublished_reservation(
                adapter,
                scope=scope,
                key=prepared.idempotency_key,
                fingerprint=fingerprint,
                record=lookup_record,
            )
            if not recovered:
                replay_run_id = lookup_record["run_id"]
                stored = _reconcile_pending_terminal_status(
                    adapter, scope, replay_run_id
                )
                if stored is None:
                    stored = _reconcile_durable_record(
                        adapter, scope, replay_run_id, lookup_record
                    )
                return _accepted(
                    replay_run_id, stored.get("status", "queued"), replayed=True
                )
        limited = adapter._concurrency_limited_response(exclude_current_pending=True)
        if limited is not None:
            return limited
        run_id = f"brun_{uuid.uuid4().hex}"
        initial_status = _initial_status(run_id, prepared)
        try:
            outcome, record = adapter._run_idempotency_store.reserve(
                scope,
                prepared.idempotency_key,
                fingerprint,
                run_id,
                initial_status,
                owner_pid=adapter._run_owner_pid,
                owner_started=adapter._run_owner_started,
            )
        except Exception:
            raise BoundedHTTPError(
                503,
                "bounded_idempotency_unavailable",
                "Durable bounded idempotency is unavailable",
            ) from None
        if outcome == "conflict":
            raise BoundedHTTPError(
                409,
                "bounded_idempotency_conflict",
                "Bounded idempotency key conflicts with an existing request",
            )
        if outcome == "reused":
            _reject_unpublished_replay(record)
            stored = _reconcile_durable_record(
                adapter, scope, record["run_id"], record
            )
            return _accepted(record["run_id"], stored.get("status", "queued"), replayed=True)
        created_reservation = (scope, prepared.idempotency_key, fingerprint, run_id)

        session_db = adapter._ensure_session_db()
        if session_db is None:
            raise BoundedHTTPError(
                503, "bounded_state_unavailable", "Bounded session state is unavailable"
            )
        try:
            admission = complete_bound_run(prepared, session_db)
        except BoundAdmissionError as exc:
            raise _admission_error(exc) from None
        agent = create_bound_agent(admission, session_db=session_db)
        _validate_bound_agent(admission, agent)

        status = dict(initial_status)
        status.update(session_id=admission.session_id, updated_at=time.time())
        adapter._bounded_run_owners[run_id] = (prepared.principal_id, scope)
        adapter._bounded_run_statuses[run_id] = status
        adapter._bounded_active_run_agents[run_id] = agent
        task = asyncio.create_task(
            _execute_bounded_run(
                adapter, run_id=run_id, scope=scope, admission=admission, agent=agent
            )
        )
        adapter._bounded_active_run_tasks[run_id] = task
        with suppress(TypeError):
            adapter._background_tasks.add(task)
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(adapter._background_tasks.discard)
        published_status = dict(status)
        published_status.update(admission_state="published", updated_at=time.time())
        if not adapter._run_idempotency_store.publish_reservation(
            scope,
            prepared.idempotency_key,
            fingerprint,
            run_id,
            published_status,
        ):
            raise BoundedHTTPError(
                503,
                "bounded_state_unavailable",
                "Bounded runtime state is unavailable",
            )
        adapter._bounded_run_statuses[run_id] = published_status
        created_reservation = None
        return _accepted(run_id, "started", replayed=False)
    except BoundedHTTPError as exc:
        if created_reservation is not None:
            _rollback_prelaunch_reservation(adapter, created_reservation)
        return _error(exc)
    except Exception as exc:
        logger.error(
            "[api_server] bounded run admission failed (error_type=%s)",
            type(exc).__name__,
        )
        if created_reservation is not None:
            _rollback_prelaunch_reservation(adapter, created_reservation)
        return _error(
            BoundedHTTPError(500, "bounded_launch_failed", "Bounded run launch failed")
        )
    finally:
        if reservation_active:
            adapter._pending_agent_requests = max(0, adapter._pending_agent_requests - 1)


def _profile_case_scopes(registry, principal: str, profile: str) -> set[str]:
    return {
        _case_scope(principal, case, profile=profile)
        for (case_principal, _selector), case in registry.cases.items()
        if case_principal == principal
    }


def _reconcile_pending_terminal_status(adapter, scope: str, run_id: str):
    pending = adapter._bounded_pending_terminal_statuses.get(run_id)
    if pending is None or pending[0] != scope:
        return None
    status = dict(pending[1])
    try:
        adapter._run_idempotency_store.update_status_strict(scope, run_id, status)
    except Exception as exc:
        logger.error(
            "[api_server] bounded terminal status reconciliation failed "
            "(run_id=%s error_type=%s)",
            run_id,
            type(exc).__name__,
        )
        raise BoundedHTTPError(
            503,
            "bounded_status_unavailable",
            "Bounded run status is temporarily unavailable",
        ) from None
    adapter._bounded_run_statuses[run_id] = status
    _release_terminal_references(adapter, run_id)
    return status


def _owned_record(adapter, registry, principal: str, run_id: str, *, profile: str):
    allowed_scopes = _profile_case_scopes(registry, principal, profile)
    owner = adapter._bounded_run_owners.get(run_id)
    if owner is not None:
        if owner[0] != principal or owner[1] not in allowed_scopes:
            return None
        pending = _reconcile_pending_terminal_status(adapter, owner[1], run_id)
        if pending is not None:
            return owner[1], pending
        status = adapter._bounded_run_statuses.get(run_id)
        if status is not None and status.get("admission_state") == "reserved":
            return None
        return (owner[1], status) if status is not None else None

    for scope in allowed_scopes:
        record = adapter._run_idempotency_store.status_for_run(scope, run_id)
        if record is not None:
            if _is_unpublished(record):
                return None
            return scope, _reconcile_durable_record(adapter, scope, run_id, record)
    return None


def _reconcile_durable_record(adapter, scope: str, run_id: str, record: dict) -> dict:
    status = dict(record.get("status") or {})
    if status.get("status") not in TERMINAL_STATUSES and not _owner_alive(
        int(record.get("owner_pid") or 0), int(record.get("owner_started") or 0)
    ):
        status.update(
            status="interrupted",
            error="The gateway restarted before this bounded run settled.",
            updated_at=time.time(),
        )
        adapter._run_idempotency_store.update_status_strict(scope, run_id, status)
    return status


def _not_found() -> "web.Response":
    return _error(BoundedHTTPError(404, "bounded_run_not_found", "Bounded run not found"))


async def handle_get_bounded_run(adapter, request) -> "web.Response":
    """GET /v1/bounded-runs/{run_id}."""
    try:
        registry = _resolve_registry(adapter)
        principal = _authenticate(request, registry)
        _require_durable_store(adapter)
        owned = _owned_record(
            adapter, registry, principal, request.match_info["run_id"], profile=_request_profile()
        )
        if owned is None:
            return _not_found()
        scope, status = owned
        adapter._bounded_run_owners.setdefault(request.match_info["run_id"], (principal, scope))
        adapter._bounded_run_statuses.setdefault(request.match_info["run_id"], status)
        return web.json_response(status)
    except BoundedHTTPError as exc:
        return _error(exc)


async def handle_stop_bounded_run(adapter, request) -> "web.Response":
    """POST /v1/bounded-runs/{run_id}/stop."""
    try:
        registry = _resolve_registry(adapter)
        principal = _authenticate(request, registry)
        _require_durable_store(adapter)
        run_id = request.match_info["run_id"]
        owned = _owned_record(
            adapter, registry, principal, run_id, profile=_request_profile()
        )
        if owned is None:
            return _not_found()
        scope, status = owned
        if status.get("status") in TERMINAL_STATUSES:
            return web.json_response(status)
        agent = adapter._bounded_active_run_agents.get(run_id)
        task = adapter._bounded_active_run_tasks.get(run_id)
        if agent is None and task is None:
            return _error(
                BoundedHTTPError(
                    409, "bounded_run_not_active", "Bounded run is not active"
                )
            )
        adapter._bounded_stopping_run_ids.add(run_id)
        _set_status(adapter, scope, run_id, "stopping")
        if agent is not None:
            with suppress(Exception):
                agent.interrupt("Stop requested via bounded Runs API")
        return web.json_response({"run_id": run_id, "status": "stopping"})
    except BoundedHTTPError as exc:
        return _error(exc)
