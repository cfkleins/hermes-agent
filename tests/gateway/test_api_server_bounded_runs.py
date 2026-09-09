"""Offline end-to-end contract for the isolated bounded Runs surface."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.platforms.api_server_bound_admission import (
    BoundAdmissionRegistry,
    BoundCase,
)
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_state import SessionDB


GATEWAY_KEY = "synthetic-bounded-gateway-key"
OTHER_KEY = "synthetic-other-bounded-key"
MODEL_KEY = "synthetic-bounded-model-key"
PRINCIPAL = "bounded-principal"
OTHER_PRINCIPAL = "other-principal"
CASE_SELECTOR = "case_fixture"
MODEL = "openai/gpt-6-astra"
BASE_URL = "https://bounded-fixture.invalid/v1"
CONTEXT_LENGTH = 1_050_000
APPROVED_PROMPT = (
    b"  BOUND-RUN-PROMPT-V1\n"
    b"[reviewed core-interview copy]\nAsk one question at a time.\n"
    b"[reviewed project-issue-interview copy]\nCapture outcome, owner, and next action.  \n"
)
PROMPT_DIGEST = sha256(APPROVED_PROMPT).hexdigest()
IDENTITY = {
    "owner_id": "bounded-owner",
    "agent_id": "bounded-advisor",
    "case_id": "bounded-case",
    "source": "bounded-api",
    "context_digest": PROMPT_DIGEST,
}


def bounded_module():
    name = "gateway.platforms.api_server_bounded_runs"
    assert importlib.util.find_spec(name) is not None, "bounded Runs integration is missing"
    return importlib.import_module(name)


def make_case(principal=PRINCIPAL, **changes):
    values = dict(
        principal_id=principal,
        **IDENTITY,
        context_bytes=APPROVED_PROMPT,
        provider="nous",
        model=MODEL,
        api_mode="chat_completions",
        base_url=BASE_URL,
        api_key=MODEL_KEY,
        context_length=CONTEXT_LENGTH,
    )
    values.update(changes)
    return BoundCase(**values)


def make_registry(**primary_changes):
    return BoundAdmissionRegistry(
        gateway_principals={GATEWAY_KEY: PRINCIPAL, OTHER_KEY: OTHER_PRINCIPAL},
        cases={
            (PRINCIPAL, CASE_SELECTOR): make_case(**primary_changes),
            (OTHER_PRINCIPAL, CASE_SELECTOR): make_case(OTHER_PRINCIPAL, owner_id="other-owner"),
        },
    )


def request_body(message="Begin the approved interview.", *, selector=CASE_SELECTOR, key="idem_fixture"):
    return json.dumps(
        {"message": message, "case_selector": selector, "idempotency_key": key},
        separators=(",", ":"),
    ).encode()


def make_adapter(tmp_path, *, registry=None, durable=True):
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"bounded_admission_registry": registry or make_registry()})
    )
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(
        str(tmp_path / "runs.db") if durable else ":memory:"
    )
    adapter._session_db = SessionDB(tmp_path / "state.db")
    return adapter


def completed_agent(output="done", **changes):
    agent = MagicMock()
    agent.model, agent.provider, agent.requested_provider = MODEL, "nous", "nous"
    agent.api_mode, agent.base_url = "chat_completions", BASE_URL
    agent._credential_pool, agent._fallback_model, agent.request_overrides = None, None, {}
    agent._exact_system_prompt = APPROVED_PROMPT.decode("utf-8")
    agent._exact_context_length = CONTEXT_LENGTH
    agent.deny_all_tools, agent.require_durable_history = True, True
    agent.tools = []
    agent.max_iterations = 1
    agent.iteration_budget = SimpleNamespace(max_total=1)
    agent._api_max_retries = 1
    agent.run_budget_seconds = 120.0
    agent.max_tokens = 4096
    agent.reasoning_config = {"enabled": False}
    agent.save_trajectories = False
    agent.verbose_logging = False
    agent.quiet_mode = True
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    agent.run_conversation.return_value = {"final_response": output}
    for name, value in changes.items():
        setattr(agent, name, value)
    return agent


def make_app(adapter):
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/bounded-runs") or path == "/v1/runs":
            app.router.add_route(method, path, handler)
    return app


async def wait_terminal(client, run_id, key=GATEWAY_KEY):
    for _ in range(100):
        response = await client.get(
            f"/v1/bounded-runs/{run_id}", headers={"Authorization": f"Bearer {key}"}
        )
        payload = await response.json()
        if payload.get("status") in {"completed", "failed", "cancelled", "interrupted"}:
            return response, payload
        await asyncio.sleep(0.01)
    raise AssertionError("bounded run did not become terminal")


class _NeverReadContent:
    def __getattribute__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        raise AssertionError("request body was touched")


class _FakeRequest:
    def __init__(self, *, authorization="", content_length=None, content=None, path="/v1/bounded-runs"):
        self.headers = {"Authorization": authorization}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.content = content if content is not None else _NeverReadContent()
        self.path = path
        self.method = "POST"
        self.match_info = {}


@pytest.mark.asyncio
async def test_draining_gateway_rejects_new_bounded_admission_before_state_change(
    tmp_path,
):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    runner = SimpleNamespace(_draining=True, _external_drain_active=False)
    app = make_app(adapter)

    with patch("gateway.run._gateway_runner_ref", lambda: runner), patch.object(
        module, "create_bound_agent", return_value=completed_agent()
    ):
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=request_body(),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            payload = await response.json()

    assert response.status == 503
    assert response.headers["Retry-After"] == "1"
    assert payload["error"]["code"] == "gateway_draining"
    assert adapter._pending_agent_requests == 0
    assert adapter._bounded_run_statuses == {}
    assert adapter._bounded_active_run_tasks == {}
    stored = adapter._run_idempotency_store._conn.execute(
        "SELECT COUNT(*) FROM run_idempotency"
    ).fetchone()[0]
    assert stored == 0


@pytest.mark.asyncio
async def test_invalid_auth_and_nondurable_store_precede_body_access(tmp_path):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    bad = _FakeRequest(authorization="Bearer wrong")
    response = await module.handle_bounded_runs(adapter, bad)
    assert response.status == 401
    assert json.loads(response.text)["error"]["code"] == "bounded_unauthorized"
    assert adapter._pending_agent_requests == 0

    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(":memory:")
    good = _FakeRequest(authorization=f"Bearer {GATEWAY_KEY}")
    response = await module.handle_bounded_runs(adapter, good)
    assert response.status == 503
    assert json.loads(response.text)["error"]["code"] == "bounded_idempotency_unavailable"


@pytest.mark.asyncio
async def test_duplicate_authorization_values_fail_before_body_access(tmp_path):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    app = make_app(adapter)
    with patch.object(
        module,
        "read_bounded_raw_body",
        side_effect=AssertionError("duplicate authorization must fail before body access"),
    ) as body_reader:
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=request_body(),
                headers=[
                    ("Authorization", f"Bearer {GATEWAY_KEY}"),
                    ("Authorization", "Bearer invalid-gateway-token"),
                ],
            )
            error = await response.json()
    assert response.status == 401
    assert error["error"]["code"] == "bounded_unauthorized"
    body_reader.assert_not_called()
    assert adapter._pending_agent_requests == 0


@pytest.mark.asyncio
async def test_declared_and_streamed_raw_body_caps(tmp_path):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    declared = _FakeRequest(
        authorization=f"Bearer {GATEWAY_KEY}", content_length=65_537
    )
    response = await module.handle_bounded_runs(adapter, declared)
    assert response.status == 413

    class Chunks:
        def __init__(self, chunks):
            self._chunks = chunks
            self.reads = 0

        async def iter_chunked(self, _size):
            self.reads += 1
            for chunk in self._chunks:
                yield chunk

    exact = Chunks([b"x" * 32_768, b"x" * 32_768])
    assert await module.read_bounded_raw_body(_FakeRequest(content=exact)) == b"x" * 65_536
    assert exact.reads == 1
    over = Chunks([b"x" * 65_536, b"x"])
    with pytest.raises(module.BoundedHTTPError) as caught:
        await module.read_bounded_raw_body(_FakeRequest(content=over))
    assert caught.value.status == 413
    assert over.reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        b'{"message":"x","case_selector":"case_fixture","idempotency_key":"k","message":"y"}',
        b'{"message":NaN,"case_selector":"case_fixture","idempotency_key":"k"}',
        b'{"message":"x","case_selector":"case_fixture","idempotency_key":"k","model":"other"}',
        b"[]",
        b"\xff",
    ],
)
async def test_duplicate_extra_and_invalid_json_never_reserve_or_launch(tmp_path, raw):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    app = make_app(adapter)
    with patch.object(module, "create_bound_agent") as factory:
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=raw,
                headers={"Authorization": f"Bearer {GATEWAY_KEY}", "Content-Type": "application/json"},
            )
    assert response.status == 400
    assert adapter._run_statuses == {}
    assert adapter._active_run_tasks == {}
    factory.assert_not_called()


def test_distinct_routes_and_generic_route_are_registered_without_collision(tmp_path):
    adapter = make_adapter(tmp_path)
    routes = {(method, path) for method, path, _handler in adapter._http_route_table()}
    assert ("POST", "/v1/runs") in routes
    assert ("POST", "/v1/bounded-runs") in routes
    assert ("GET", "/v1/bounded-runs/{run_id}") in routes
    assert ("POST", "/v1/bounded-runs/{run_id}/stop") in routes
    assert len(routes) == len(adapter._http_route_table())


@pytest.mark.asyncio
async def test_reservation_precedes_factory_and_exact_launch_contract(tmp_path):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    app = make_app(adapter)
    observed = {}
    real_reserve = adapter._run_idempotency_store.reserve

    def reserve(*args, **kwargs):
        observed["reserved"] = True
        return real_reserve(*args, **kwargs)

    agent = completed_agent(output="bounded done")

    def factory(admission, **kwargs):
        assert observed.get("reserved") is True
        observed["admission"] = admission
        observed["factory_kwargs"] = kwargs
        return agent

    adapter._run_idempotency_store.reserve = reserve
    with patch.object(module, "create_bound_agent", side_effect=factory), patch.object(
        adapter, "_create_agent", side_effect=AssertionError("generic factory used")
    ):
        async with TestClient(TestServer(app)) as client:
            accepted = await client.post(
                "/v1/bounded-runs",
                data=request_body(),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            payload = await accepted.json()
            terminal_response, terminal = await wait_terminal(client, payload["run_id"])
    assert accepted.status == 202
    assert terminal_response.status == 200
    assert terminal["status"] == "completed"
    call = agent.run_conversation.call_args.kwargs
    assert call == {
        "user_message": "Begin the approved interview.",
        "conversation_history": None,
        "task_id": observed["admission"].session_id,
        "binding_identity": dict(observed["admission"].binding_identity),
    }
    assert observed["factory_kwargs"]["session_db"] is adapter._session_db


@pytest.mark.asyncio
async def test_simultaneous_adapters_reserve_once_before_downstream_work(tmp_path):
    module = bounded_module()
    first = make_adapter(tmp_path)
    second = make_adapter(tmp_path)
    agent = completed_agent(output="bounded concurrency done")

    with patch.object(module, "create_bound_agent", return_value=agent) as factory:
        async with TestClient(TestServer(make_app(first))) as first_client, TestClient(
            TestServer(make_app(second))
        ) as second_client:
            responses = await asyncio.gather(
                first_client.post(
                    "/v1/bounded-runs",
                    data=request_body(),
                    headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
                ),
                second_client.post(
                    "/v1/bounded-runs",
                    data=request_body(),
                    headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
                ),
            )
            payloads = await asyncio.gather(*(response.json() for response in responses))
            await wait_terminal(first_client, payloads[0]["run_id"])

    assert [response.status for response in responses] == [202, 202]
    assert payloads[0]["run_id"] == payloads[1]["run_id"]
    assert sorted(payload["replayed"] for payload in payloads) == [False, True]
    factory.assert_called_once()
    assert agent.run_conversation.call_count == 1


def test_concurrent_replay_never_acknowledges_rolled_back_prelaunch_reservation(
    tmp_path,
):
    first = make_adapter(tmp_path)
    second = make_adapter(tmp_path)
    reservation_visible = threading.Event()
    release_original = threading.Event()

    def fail_after_reservation():
        reservation_visible.set()
        assert release_original.wait(timeout=10)
        return None

    first._ensure_session_db = fail_after_reservation

    async def submit(adapter):
        async with TestClient(TestServer(make_app(adapter))) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=request_body(key="prelaunch_replay_key"),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            return response.status, await response.json()

    with ThreadPoolExecutor(max_workers=2) as executor:
        original = executor.submit(asyncio.run, submit(first))
        assert reservation_visible.wait(timeout=5)
        replay = executor.submit(asyncio.run, submit(second))
        replay_status, replay_payload = replay.result(timeout=5)
        release_original.set()
        original_status, _original_payload = original.result(timeout=5)

    prepared = importlib.import_module(
        "gateway.platforms.api_server_bound_admission"
    ).prepare_bound_run(
        raw_body=request_body(key="prelaunch_replay_key"),
        authorization=f"Bearer {GATEWAY_KEY}",
        idempotency_key=None,
        registry=make_registry(),
    )
    module = bounded_module()
    scope = module._idempotency_scope(prepared, profile="default")
    fingerprint = module._idempotency_fingerprint(
        prepared, request_body(key="prelaunch_replay_key")
    )
    outcome, record = second._run_idempotency_store.lookup(
        scope, "prelaunch_replay_key", fingerprint
    )

    assert original_status == 503
    if replay_status == 202:
        assert outcome == "reused"
        assert record["run_id"] == replay_payload["run_id"]
    else:
        assert replay_status == 503
        assert replay_payload["error"]["code"] == "bounded_run_initializing"


@pytest.mark.asyncio
async def test_dead_owner_unpublished_reservation_is_recovered_before_readmission(
    tmp_path,
):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    raw_body = request_body(key="dead_prelaunch_key")
    prepared = importlib.import_module(
        "gateway.platforms.api_server_bound_admission"
    ).prepare_bound_run(
        raw_body=raw_body,
        authorization=f"Bearer {GATEWAY_KEY}",
        idempotency_key=None,
        registry=make_registry(),
    )
    scope = module._idempotency_scope(prepared, profile="default")
    fingerprint = module._idempotency_fingerprint(prepared, raw_body)
    stale_run_id = "brun_dead_prelaunch"
    stale_status = module._initial_status(stale_run_id, prepared)
    assert adapter._run_idempotency_store.reserve(
        scope,
        "dead_prelaunch_key",
        fingerprint,
        stale_run_id,
        stale_status,
        owner_pid=2_147_483_647,
        owner_started=1,
    )[0] == "created"

    agent = completed_agent(output="recovered admission")
    with patch.object(module, "create_bound_agent", return_value=agent):
        async with TestClient(TestServer(make_app(adapter))) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=raw_body,
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            payload = await response.json()
            assert response.status == 202
            assert payload["replayed"] is False
            assert payload["run_id"] != stale_run_id
            _, terminal = await wait_terminal(client, payload["run_id"])

    assert terminal["status"] == "completed"
    assert agent.run_conversation.call_count == 1


@pytest.mark.asyncio
async def test_publication_failure_cancels_task_and_rolls_back_before_execution(
    tmp_path,
):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    agent = completed_agent(output="must not execute")

    with patch.object(module, "create_bound_agent", return_value=agent), patch.object(
        adapter._run_idempotency_store,
        "publish_reservation",
        return_value=False,
    ):
        async with TestClient(TestServer(make_app(adapter))) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=request_body(key="publication_failure_key"),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            payload = await response.json()
            await asyncio.sleep(0)

    assert response.status == 503
    assert payload["error"]["code"] == "bounded_state_unavailable"
    assert agent.run_conversation.call_count == 0
    assert adapter._bounded_run_owners == {}
    assert adapter._bounded_run_statuses == {}
    assert adapter._bounded_active_run_agents == {}
    assert adapter._bounded_active_run_tasks == {}


@pytest.mark.asyncio
async def test_authenticated_status_does_not_expose_unpublished_reservation(tmp_path):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    raw_body = request_body(key="hidden_prelaunch_key")
    prepared = importlib.import_module(
        "gateway.platforms.api_server_bound_admission"
    ).prepare_bound_run(
        raw_body=raw_body,
        authorization=f"Bearer {GATEWAY_KEY}",
        idempotency_key=None,
        registry=make_registry(),
    )
    scope = module._idempotency_scope(prepared, profile="default")
    fingerprint = module._idempotency_fingerprint(prepared, raw_body)
    run_id = "brun_hidden_prelaunch"
    assert adapter._run_idempotency_store.reserve(
        scope,
        "hidden_prelaunch_key",
        fingerprint,
        run_id,
        module._initial_status(run_id, prepared),
        owner_pid=adapter._run_owner_pid,
        owner_started=adapter._run_owner_started,
    )[0] == "created"

    async with TestClient(TestServer(make_app(adapter))) as client:
        response = await client.get(
            f"/v1/bounded-runs/{run_id}",
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        payload = await response.json()

    assert response.status == 404
    assert payload["error"]["code"] == "bounded_run_not_found"


@pytest.mark.asyncio
async def test_reserve_failure_has_zero_downstream_mutation(tmp_path):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    app = make_app(adapter)
    adapter._run_idempotency_store.reserve = MagicMock(side_effect=OSError("synthetic sqlite failure"))
    with patch.object(module, "create_bound_agent") as factory:
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/bounded-runs", data=request_body(),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
    assert response.status == 503
    assert adapter._run_owners == {}
    assert adapter._run_statuses == {}
    assert adapter._run_streams == {}
    assert adapter._active_run_tasks == {}
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_prelaunch_rollback_failure_retains_reservation_and_logs_only_safe_identity(
    tmp_path, caplog
):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    raw_body = request_body(key="rollback_failure_key")
    prepared = importlib.import_module(
        "gateway.platforms.api_server_bound_admission"
    ).prepare_bound_run(
        raw_body=raw_body,
        authorization=f"Bearer {GATEWAY_KEY}",
        idempotency_key=None,
        registry=make_registry(),
    )
    scope = module._idempotency_scope(prepared, profile="default")
    fingerprint = module._idempotency_fingerprint(prepared, raw_body)
    adapter._ensure_session_db = MagicMock(return_value=None)
    adapter._run_idempotency_store.rollback_reservation = MagicMock(
        side_effect=OSError("synthetic private rollback detail")
    )

    with caplog.at_level("ERROR", logger="gateway.platforms.api_server"):
        async with TestClient(TestServer(make_app(adapter))) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=raw_body,
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            payload = await response.json()

    outcome, record = adapter._run_idempotency_store.lookup(
        scope, "rollback_failure_key", fingerprint
    )
    assert response.status == 503
    assert payload["error"]["code"] == "bounded_state_unavailable"
    assert outcome == "reused"
    assert record["status"]["status"] == "queued"
    assert adapter._bounded_run_owners == {}
    assert adapter._bounded_run_statuses == {}
    assert adapter._bounded_active_run_agents == {}
    assert adapter._bounded_active_run_tasks == {}
    assert "bounded prelaunch rollback failed" in caplog.text
    assert record["run_id"] in caplog.text
    assert "error_type=OSError" in caplog.text
    assert "synthetic private rollback detail" not in caplog.text


@pytest.mark.asyncio
async def test_launch_exception_log_omits_exception_text(tmp_path, caplog):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    private_detail = "synthetic private launch detail"

    with caplog.at_level("ERROR", logger="gateway.platforms.api_server"), patch.object(
        module, "create_bound_agent", side_effect=RuntimeError(private_detail)
    ):
        async with TestClient(TestServer(make_app(adapter))) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=request_body(key="launch_log_key"),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            payload = await response.json()

    assert response.status == 500
    assert payload["error"] == {
        "code": "bounded_launch_failed",
        "message": "Bounded run launch failed",
    }
    assert "bounded run admission failed (error_type=RuntimeError)" in caplog.text
    assert private_detail not in caplog.text
    assert adapter._bounded_active_run_tasks == {}


@pytest.mark.asyncio
async def test_execution_exception_log_and_durable_status_omit_exception_text(
    tmp_path, caplog
):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    private_detail = "synthetic private execution detail"
    agent = completed_agent()
    agent.run_conversation.side_effect = RuntimeError(private_detail)

    with caplog.at_level("ERROR", logger="gateway.platforms.api_server"), patch.object(
        module, "create_bound_agent", return_value=agent
    ):
        async with TestClient(TestServer(make_app(adapter))) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=request_body(key="execution_log_key"),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            accepted = await response.json()
            _, terminal = await wait_terminal(client, accepted["run_id"])

    assert response.status == 202
    assert terminal["status"] == "failed"
    assert terminal["error"] == "Bounded run failed"
    assert (
        f"bounded run failed (run_id={accepted['run_id']} error_type=RuntimeError)"
        in caplog.text
    )
    assert private_detail not in caplog.text


@pytest.mark.asyncio
async def test_terminal_status_write_failure_stays_visible_until_authenticated_reconciliation(
    tmp_path, caplog
):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    agent = completed_agent()
    original_publish = adapter._run_idempotency_store.publish_reservation
    original_update = adapter._run_idempotency_store.update_status_strict
    publication_attempts = []
    status_write_attempts = []

    def publish(scope, key, fingerprint, run_id, status):
        publication_attempts.append(status["status"])
        return original_publish(scope, key, fingerprint, run_id, status)

    def fail_terminal_writes(scope, run_id, status):
        status_write_attempts.append(status["status"])
        if status["status"] in {"queued", "running"}:
            return original_update(scope, run_id, status)
        raise OSError("synthetic private durable-store detail")

    with caplog.at_level("ERROR", logger="gateway.platforms.api_server"), patch.object(
        module, "create_bound_agent", return_value=agent
    ), patch.object(
        adapter._run_idempotency_store,
        "publish_reservation",
        side_effect=publish,
    ), patch.object(
        adapter._run_idempotency_store,
        "update_status_strict",
        side_effect=fail_terminal_writes,
    ):
        async with TestClient(TestServer(make_app(adapter))) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=request_body(key="terminal_reconciliation_key"),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            accepted = await response.json()
            run_id = accepted["run_id"]
            task = adapter._bounded_active_run_tasks[run_id]
            await task

            durable = adapter._run_idempotency_store.status_for_run(
                adapter._bounded_run_owners[run_id][1], run_id
            )
            assert durable["status"]["status"] == "running"
            assert publication_attempts == ["queued"]
            assert status_write_attempts == ["running", "completed", "failed"]
            assert run_id in adapter._bounded_pending_terminal_statuses
            assert run_id in adapter._bounded_active_run_tasks
            assert run_id in adapter._bounded_active_run_agents
            assert adapter.active_agent_work_count() == 1

            attempts_before_wrong_principal = list(status_write_attempts)
            wrong_principal = await client.get(
                f"/v1/bounded-runs/{run_id}",
                headers={"Authorization": f"Bearer {OTHER_KEY}"},
            )
            assert wrong_principal.status == 404
            assert status_write_attempts == attempts_before_wrong_principal

            unavailable_response = await client.get(
                f"/v1/bounded-runs/{run_id}",
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            unavailable = await unavailable_response.json()
            assert unavailable_response.status == 503
            assert unavailable == {
                "error": {
                    "code": "bounded_status_unavailable",
                    "message": "Bounded run status is temporarily unavailable",
                }
            }
            assert run_id in adapter._bounded_pending_terminal_statuses
            assert adapter.active_agent_work_count() == 1

    async with TestClient(TestServer(make_app(adapter))) as client:
        replay_response = await client.post(
            "/v1/bounded-runs",
            data=request_body(key="terminal_reconciliation_key"),
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        replay = await replay_response.json()
        reconciled_response = await client.get(
            f"/v1/bounded-runs/{run_id}",
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        reconciled = await reconciled_response.json()

    assert replay_response.status == 202
    assert replay_response.headers["Idempotency-Replayed"] == "true"
    assert replay == {"run_id": run_id, "status": "failed", "replayed": True}
    assert reconciled_response.status == 200
    assert reconciled["status"] == "failed"
    assert reconciled["error"] == "Bounded run failed"
    assert run_id not in adapter._bounded_pending_terminal_statuses
    assert run_id not in adapter._bounded_active_run_tasks
    assert run_id not in adapter._bounded_active_run_agents
    assert adapter.active_agent_work_count() == 0
    durable = adapter._run_idempotency_store.status_for_run(
        adapter._bounded_run_owners[run_id][1], run_id
    )
    assert durable["status"] == reconciled
    assert "synthetic private durable-store detail" not in caplog.text


@pytest.mark.asyncio
async def test_replay_conflict_and_principal_scope_survive_restart(tmp_path):
    module = bounded_module()
    db_path = tmp_path / "runs.db"
    first = make_adapter(tmp_path)
    app = make_app(first)
    agent = completed_agent()
    with patch.object(module, "create_bound_agent", return_value=agent):
        async with TestClient(TestServer(app)) as client:
            created = await client.post(
                "/v1/bounded-runs", data=request_body(),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            created_body = await created.json()
            await wait_terminal(client, created_body["run_id"])
    first._run_idempotency_store.close()

    restarted = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"bounded_admission_registry": make_registry()})
    )
    restarted._run_idempotency_store.close()
    restarted._run_idempotency_store = RunIdempotencyStore(str(db_path))
    restarted._session_db = SessionDB(tmp_path / "state.db")
    app = make_app(restarted)
    with patch.object(module, "create_bound_agent", return_value=agent) as factory:
        async with TestClient(TestServer(app)) as client:
            replay = await client.post(
                "/v1/bounded-runs", data=request_body(),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            replay_body = await replay.json()
            conflict = await client.post(
                "/v1/bounded-runs", data=request_body(message="changed"),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            isolated = await client.post(
                "/v1/bounded-runs", data=request_body(),
                headers={"Authorization": f"Bearer {OTHER_KEY}"},
            )
            isolated_body = await isolated.json()
            await wait_terminal(client, isolated_body["run_id"], key=OTHER_KEY)
    assert replay.status == 202
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert replay_body["run_id"] == created_body["run_id"]
    assert conflict.status == 409
    assert isolated.status == 202
    assert isolated_body["run_id"] != created_body["run_id"]
    factory.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy_change",
    [
        {
            "context_bytes": APPROVED_PROMPT + b"reviewed revision\n",
            "context_digest": sha256(APPROVED_PROMPT + b"reviewed revision\n").hexdigest(),
        },
        {"model": MODEL + "-reviewed-revision"},
    ],
    ids=("approved-context", "model-lane"),
)
async def test_server_owned_policy_change_conflicts_same_body_and_key_after_restart(
    tmp_path, policy_change
):
    module = bounded_module()
    first = make_adapter(tmp_path)
    with patch.object(module, "create_bound_agent", return_value=completed_agent()):
        async with TestClient(TestServer(make_app(first))) as client:
            created = await client.post(
                "/v1/bounded-runs",
                data=request_body(),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            created_body = await created.json()
            await wait_terminal(client, created_body["run_id"])
    first._run_idempotency_store.close()
    first._session_db.close()

    restarted = make_adapter(tmp_path, registry=make_registry(**policy_change))
    with patch.object(module, "create_bound_agent") as factory:
        async with TestClient(TestServer(make_app(restarted))) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=request_body(),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            payload = await response.json()

    assert response.status == 409
    assert payload["error"]["code"] == "bounded_idempotency_conflict"
    factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy_name", "policy_value"),
    (("BOUNDED_MAX_TOKENS", 2048), ("BOUNDED_API_MAX_RETRIES", 2)),
)
async def test_execution_policy_change_conflicts_same_body_and_key_after_restart(
    tmp_path, monkeypatch, policy_name, policy_value
):
    module = bounded_module()
    first = make_adapter(tmp_path)
    with patch.object(module, "create_bound_agent", return_value=completed_agent()):
        async with TestClient(TestServer(make_app(first))) as client:
            created = await client.post(
                "/v1/bounded-runs",
                data=request_body(key="execution_policy_key"),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            created_body = await created.json()
            await wait_terminal(client, created_body["run_id"])
    first._run_idempotency_store.close()
    first._session_db.close()

    monkeypatch.setattr(module, policy_name, policy_value)
    restarted = make_adapter(tmp_path)
    with patch.object(module, "create_bound_agent") as factory:
        async with TestClient(TestServer(make_app(restarted))) as client:
            response = await client.post(
                "/v1/bounded-runs",
                data=request_body(key="execution_policy_key"),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            payload = await response.json()

    assert response.status == 409
    assert payload["error"]["code"] == "bounded_idempotency_conflict"
    factory.assert_not_called()


def test_idempotency_scope_includes_server_resolved_profile():
    module = bounded_module()
    admission_module = importlib.import_module("gateway.platforms.api_server_bound_admission")
    prepared = admission_module.prepare_bound_run(
        raw_body=request_body(),
        authorization=f"Bearer {GATEWAY_KEY}",
        idempotency_key=None,
        registry=make_registry(),
    )

    assert module._idempotency_scope(prepared, profile="default") != module._idempotency_scope(
        prepared, profile="separate-profile"
    )


@pytest.mark.asyncio
async def test_live_bounded_owner_fast_path_is_profile_scoped(tmp_path):
    module = bounded_module()
    adapter = make_adapter(tmp_path)

    @web.middleware
    async def stamp_profile(request, handler):
        token = _api_request_profile.set(request.headers.get("X-Test-Profile"))
        try:
            return await handler(request)
        finally:
            _api_request_profile.reset(token)

    app = web.Application(middlewares=[stamp_profile])
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/bounded-runs"):
            app.router.add_route(method, path, handler)

    profile_a_headers = {
        "Authorization": f"Bearer {GATEWAY_KEY}",
        "X-Test-Profile": "profile-a",
    }
    with patch.object(module, "create_bound_agent", return_value=completed_agent()):
        async with TestClient(TestServer(app)) as client:
            created = await client.post(
                "/v1/bounded-runs",
                data=request_body(key="profile_live_key"),
                headers=profile_a_headers,
            )
            run_id = (await created.json())["run_id"]
            for _ in range(100):
                owner_response = await client.get(
                    f"/v1/bounded-runs/{run_id}", headers=profile_a_headers
                )
                owner_payload = await owner_response.json()
                if owner_payload.get("status") == "completed":
                    break
                await asyncio.sleep(0.01)
            foreign_response = await client.get(
                f"/v1/bounded-runs/{run_id}",
                headers={
                    "Authorization": f"Bearer {GATEWAY_KEY}",
                    "X-Test-Profile": "profile-b",
                },
            )

    assert created.status == 202
    assert owner_response.status == 200
    assert owner_payload["status"] == "completed"
    assert foreign_response.status == 404


@pytest.mark.asyncio
async def test_capacity_rejects_new_run_before_reservation_but_allows_exact_replay(tmp_path):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    with patch.object(module, "create_bound_agent", return_value=completed_agent()):
        async with TestClient(TestServer(make_app(adapter))) as client:
            created = await client.post(
                "/v1/bounded-runs",
                data=request_body(),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            created_body = await created.json()
            await wait_terminal(client, created_body["run_id"])

            blocker = asyncio.create_task(asyncio.Event().wait())
            adapter._active_run_tasks["occupied-generic-run"] = blocker
            adapter._max_concurrent_runs = 1
            real_reserve = adapter._run_idempotency_store.reserve
            adapter._run_idempotency_store.reserve = MagicMock(wraps=real_reserve)
            try:
                replay = await client.post(
                    "/v1/bounded-runs",
                    data=request_body(),
                    headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
                )
                new_run = await client.post(
                    "/v1/bounded-runs",
                    data=request_body(message="new admitted request", key="idem_at_capacity"),
                    headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
                )
                new_payload = await new_run.json()
            finally:
                blocker.cancel()
                with suppress(asyncio.CancelledError):
                    await blocker
                adapter._active_run_tasks.pop("occupied-generic-run", None)

    assert replay.status == 202
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert new_run.status == 429
    assert new_payload["error"]["code"] == "rate_limit_exceeded"
    adapter._run_idempotency_store.reserve.assert_not_called()


@pytest.mark.asyncio
async def test_bounded_status_and_stop_are_owner_scoped_and_generic_controls_cannot_see_run(tmp_path):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    app = make_app(adapter)
    agent = completed_agent()
    release = asyncio.Event()

    def run(**_kwargs):
        while not release.is_set():
            import time
            time.sleep(0.01)
        return {"interrupted": True, "final_response": ""}

    agent.run_conversation.side_effect = run
    agent.interrupt.side_effect = lambda *_a, **_k: release.set()
    with patch.object(module, "create_bound_agent", return_value=agent):
        async with TestClient(TestServer(app)) as client:
            created = await client.post(
                "/v1/bounded-runs", data=request_body(),
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            run_id = (await created.json())["run_id"]
            foreign = await client.get(
                f"/v1/bounded-runs/{run_id}",
                headers={"Authorization": f"Bearer {OTHER_KEY}"},
            )
            generic = await client.get(f"/v1/runs/{run_id}")
            stopped = await client.post(
                f"/v1/bounded-runs/{run_id}/stop",
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            _, terminal = await wait_terminal(client, run_id)
    assert foreign.status == 404
    assert generic.status == 404
    assert stopped.status == 200
    assert terminal["status"] == "cancelled"
    agent.interrupt.assert_called()


@pytest.mark.asyncio
async def test_postconstruction_runtime_drift_never_executes_and_retry_reuses_binding(
    tmp_path
):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    raw_body = request_body(key="runtime_drift_key")
    drifted = completed_agent(model="unauthorized-model")

    with patch.object(module, "create_bound_agent", return_value=drifted):
        async with TestClient(TestServer(make_app(adapter))) as client:
            rejected = await client.post(
                "/v1/bounded-runs",
                data=raw_body,
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            rejected_payload = await rejected.json()

    prepared = importlib.import_module(
        "gateway.platforms.api_server_bound_admission"
    ).prepare_bound_run(
        raw_body=raw_body,
        authorization=f"Bearer {GATEWAY_KEY}",
        idempotency_key=None,
        registry=make_registry(),
    )
    scope = module._idempotency_scope(prepared, profile="default")
    fingerprint = module._idempotency_fingerprint(prepared, raw_body)
    assert rejected.status == 503
    assert rejected_payload["error"]["code"] == "bounded_runtime_unavailable"
    assert adapter._run_idempotency_store.lookup(
        scope, "runtime_drift_key", fingerprint
    ) == ("missing", None)
    drifted.run_conversation.assert_not_called()

    corrected = completed_agent()
    with patch.object(module, "create_bound_agent", return_value=corrected):
        async with TestClient(TestServer(make_app(adapter))) as client:
            accepted = await client.post(
                "/v1/bounded-runs",
                data=raw_body,
                headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
            )
            accepted_payload = await accepted.json()
            _, terminal = await wait_terminal(client, accepted_payload["run_id"])
    assert accepted.status == 202
    assert "Idempotency-Replayed" not in accepted.headers
    assert terminal["status"] == "completed"
    corrected.run_conversation.assert_called_once()


def test_exact_bound_runtime_and_prompt_factory_configuration(tmp_path, monkeypatch):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    case = make_registry().cases[(PRINCIPAL, CASE_SELECTOR)]
    admission_module = importlib.import_module("gateway.platforms.api_server_bound_admission")
    admission = admission_module.admit_bound_run(
        raw_body=request_body(), authorization=f"Bearer {GATEWAY_KEY}",
        idempotency_key=None, registry=make_registry(), session_db=adapter._session_db,
    )
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs["model"]
            self.provider = kwargs["provider"]
            self.requested_provider = kwargs["requested_provider"]
            self.api_mode = kwargs["api_mode"]
            self.base_url = kwargs["base_url"]
            self._credential_pool = kwargs["credential_pool"]
            self.fallback_model = kwargs["fallback_model"]
            self._fallback_model = kwargs["fallback_model"]
            self.request_overrides = kwargs["request_overrides"]
            self._exact_system_prompt = kwargs["exact_system_prompt_bytes"].decode("utf-8")
            self._exact_context_length = kwargs["exact_context_length"]
            self.deny_all_tools = kwargs["deny_all_tools"]
            self.require_durable_history = kwargs["require_durable_history"]
            self.tools = []
            self.max_iterations = kwargs["max_iterations"]
            self.iteration_budget = SimpleNamespace(max_total=kwargs["max_iterations"])
            self.run_budget_seconds = kwargs["run_budget_seconds"]
            self.max_tokens = kwargs["max_tokens"]
            self.reasoning_config = kwargs["reasoning_config"]
            self.save_trajectories = kwargs["save_trajectories"]
            self.verbose_logging = kwargs["verbose_logging"]
            self.quiet_mode = kwargs["quiet_mode"]

    monkeypatch.setattr(module, "AIAgent", FakeAgent)
    agent = module.create_bound_agent(admission, session_db=adapter._session_db)
    assert sha256(captured["exact_system_prompt_bytes"]).hexdigest() == case.context_digest
    assert captured["exact_system_prompt_bytes"] == APPROVED_PROMPT
    assert captured["enabled_toolsets"] == []
    assert captured["deny_all_tools"] is True
    assert captured["require_durable_history"] is True
    assert captured["credential_pool"] is None
    assert captured["fallback_model"] is None
    assert captured["request_overrides"] is None
    assert captured["quiet_mode"] is True
    assert captured["max_iterations"] == 1
    assert captured["run_budget_seconds"] == 120.0
    assert captured["max_tokens"] == 4096
    assert captured["reasoning_config"] == {"enabled": False}
    assert captured["save_trajectories"] is False
    assert captured["verbose_logging"] is False
    assert captured["exact_context_length"] == CONTEXT_LENGTH
    assert (agent.provider, agent.model, agent.api_mode, agent.base_url) == (
        "nous", MODEL, "chat_completions", BASE_URL
    )
    module._validate_bound_agent(admission, agent)


def test_real_bound_factory_sends_exact_approved_prompt_as_first_wire_message(tmp_path, monkeypatch):
    module = bounded_module()
    adapter = make_adapter(tmp_path)
    case = make_registry().cases[(PRINCIPAL, CASE_SELECTOR)]
    admission_module = importlib.import_module("gateway.platforms.api_server_bound_admission")
    admission = admission_module.admit_bound_run(
        raw_body=request_body(), authorization=f"Bearer {GATEWAY_KEY}",
        idempotency_key=None, registry=make_registry(), session_db=adapter._session_db,
    )

    import hermes_cli.config as hermes_config
    from agent.conversation_loop import _restore_or_build_system_prompt
    from agent.turn_request_assembly import assemble_api_request
    from agent.turn_context import (
        _collect_pre_llm_call_context,
        _maybe_title_session_at_turn_start,
        _merge_gateway_notes,
    )

    ambient_config_calls = []

    def forbid_ambient_config(*_args, **_kwargs):
        import inspect
        caller = inspect.stack()[1]
        ambient_config_calls.append(f"{caller.filename}:{caller.lineno}:{caller.function}")
        raise AssertionError("bounded construction must not load ambient config")

    monkeypatch.setattr(hermes_config, "load_config_readonly", forbid_ambient_config)
    with patch("hermes_cli.plugins.get_plugin_context_engine") as plugin_context_engine:
        agent = module.create_bound_agent(admission, session_db=adapter._session_db)
        plugin_context_engine.assert_not_called()
    assert ambient_config_calls == []
    _restore_or_build_system_prompt(agent, None, [])
    assembled = assemble_api_request(
        agent,
        messages=[{"role": "user", "content": "hello"}],
        current_turn_user_idx=0,
        _ext_prefetch_cache=None,
        _plugin_user_context=None,
        moa_config=None,
        active_system_prompt=agent._cached_system_prompt,
        original_user_message="hello",
        pending_moa_prepared_request=None,
        request_logger=module.logger,
    )

    assert assembled.api_messages[0] == {
        "role": "system", "content": APPROVED_PROMPT.decode("utf-8")
    }
    assert sha256(assembled.api_messages[0]["content"].encode("utf-8")).hexdigest() == case.context_digest
    assert agent.tools == []
    provider_profile_calls = []

    def forbid_provider_profile(*args, **kwargs):
        provider_profile_calls.append((args, kwargs))
        raise AssertionError("bounded request construction must not load provider profiles")

    with patch("providers.get_provider_profile", side_effect=forbid_provider_profile):
        api_kwargs = agent._build_api_kwargs(assembled.api_messages, tools_for_api=[])

    assert ambient_config_calls == []
    assert provider_profile_calls == []
    assert api_kwargs["messages"][0] == assembled.api_messages[0]
    assert api_kwargs["model"] == MODEL
    assert not ({"tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"} & api_kwargs.keys())
    with patch.object(agent, "_resolve_env_credentials") as env_resolver:
        assert agent._try_refresh_env_client_credentials() is False
        env_resolver.assert_not_called()
    with patch(
        "hermes_cli.auth.resolve_nous_runtime_credentials"
    ) as nous_resolver:
        assert agent._try_refresh_nous_client_credentials(force=True) is False
        nous_resolver.assert_not_called()
    with patch("agent.title_generator.maybe_auto_title") as title_generator:
        _maybe_title_session_at_turn_start(
            agent, [{"role": "user", "content": "Must not title bounded sessions"}]
        )
        title_generator.assert_not_called()
    messages = [{"role": "user", "content": "Must not load ambient context"}]
    with patch("hermes_cli.lifecycle.invoke_hook") as invoke_hook:
        assert _collect_pre_llm_call_context(
            agent,
            effective_task_id="bounded-task",
            turn_id="bounded-turn",
            original_user_message="hello",
            messages=messages,
            conversation_history=None,
        ) == ""
        invoke_hook.assert_not_called()
    with patch("agent.turn_context.consume_gateway_turn_context_notes") as consume_notes:
        assert _merge_gateway_notes(agent, messages, 0, "") == ""
        consume_notes.assert_not_called()
    with patch("hermes_cli.lifecycle.has_hook") as has_hook, patch(
        "hermes_cli.lifecycle.invoke_hook"
    ) as invoke_hook:
        agent._invoke_api_request_error_hook(
            task_id="bounded-task",
            turn_id="bounded-turn",
            api_request_id="bounded-request",
            api_call_count=1,
            api_start_time=1.0,
            api_kwargs={},
            error_type="SyntheticError",
            error_message="synthetic",
        )
        has_hook.assert_not_called()
        invoke_hook.assert_not_called()


@pytest.mark.asyncio
async def test_dead_owner_nonterminal_status_becomes_durable_interrupted_after_restart(tmp_path):
    adapter = make_adapter(tmp_path)
    registry = make_registry()
    case = registry.cases[(PRINCIPAL, CASE_SELECTOR)]
    scope = bounded_module()._case_scope(PRINCIPAL, case, profile="default")
    run_id = "brun_dead_owner"
    queued = {
        "object": "hermes.bounded_run",
        "run_id": run_id,
        "status": "running",
        "created_at": 1.0,
        "updated_at": 1.0,
        "owner_id": case.owner_id,
        "agent_id": case.agent_id,
        "case_id": case.case_id,
    }
    replay_body = request_body(key="dead_owner_key")
    prepared = importlib.import_module(
        "gateway.platforms.api_server_bound_admission"
    ).prepare_bound_run(
        raw_body=replay_body,
        authorization=f"Bearer {GATEWAY_KEY}",
        idempotency_key=None,
        registry=registry,
    )
    assert adapter._run_idempotency_store.reserve(
        scope,
        "dead_owner_key",
        bounded_module()._idempotency_fingerprint(prepared, replay_body),
        run_id,
        queued,
        owner_pid=999_999_999, owner_started=1,
    )[0] == "created"
    app = make_app(adapter)
    async with TestClient(TestServer(app)) as client:
        replay = await client.post(
            "/v1/bounded-runs",
            data=replay_body,
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        replay_payload = await replay.json()
        response = await client.get(
            f"/v1/bounded-runs/{run_id}",
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        payload = await response.json()

    assert response.status == 200
    assert payload["status"] == "interrupted"
    assert replay.status == 202
    assert replay_payload["status"] == "interrupted"
    stored = adapter._run_idempotency_store.status_for_run(scope, run_id)
    assert stored["status"]["status"] == "interrupted"


def test_conditional_reservation_rollback_is_exact(tmp_path):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    status = {
        "run_id": "run-a",
        "status": "queued",
        "admission_state": "reserved",
    }
    assert store.reserve("scope", "key", "a" * 64, "run-a", status)[0] == "created"
    assert store.rollback_reservation("scope", "key", "b" * 64, "run-a") is False
    assert store.rollback_reservation("scope", "key", "a" * 64, "run-b") is False
    assert store.owns_run("scope", "run-a") is True
    assert store.rollback_reservation("scope", "key", "a" * 64, "run-a") is True
    assert store.owns_run("scope", "run-a") is False

    assert store.reserve("scope", "key", "a" * 64, "run-a", status)[0] == "created"
    published = dict(status, admission_state="published")
    assert store.publish_reservation(
        "scope", "key", "a" * 64, "run-a", published
    ) is True
    assert store.publish_reservation(
        "scope", "key", "a" * 64, "run-a", published
    ) is False
    assert store.rollback_reservation("scope", "key", "a" * 64, "run-a") is False
    assert store.owns_run("scope", "run-a") is True


def test_exact_prompt_constructor_rejects_above_admission_cap():
    from run_agent import AIAgent

    with pytest.raises(ValueError, match="exact_system_prompt_bytes is too large"):
        AIAgent(
            base_url=BASE_URL,
            api_key=MODEL_KEY,
            provider="nous",
            requested_provider="nous",
            api_mode="chat_completions",
            model=MODEL,
            enabled_toolsets=[],
            deny_all_tools=True,
            require_durable_history=True,
            credential_pool=None,
            fallback_model=None,
            request_overrides=None,
            quiet_mode=True,
            exact_system_prompt_bytes=b"x" * 131_073,
            skip_context_files=True,
            load_soul_identity=False,
            skip_memory=True,
            skip_background_review=True,
        )
