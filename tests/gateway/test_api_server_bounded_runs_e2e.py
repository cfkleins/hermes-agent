"""Synthetic provider I/O through the real authenticated bounded HTTP path."""

from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import socket
import traceback
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
import agent.chat_completion_helpers as chat_completion_helpers
import agent.nous_rate_guard as nous_rate_guard
import agent.relay_llm as relay_llm
import agent.turn_empty_response as turn_empty_response
import gateway.platforms.api_server_bounded_runs as bounded_runs
import hermes_cli.lifecycle as hermes_lifecycle
import hermes_cli.middleware as hermes_middleware
import hermes_cli.plugins as hermes_plugins
import hermes_cli.config as hermes_config
import providers
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion

from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.platforms.base import PlatformConfig
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.gateway.test_api_server_bounded_runs import (
    APPROVED_PROMPT,
    CASE_SELECTOR,
    GATEWAY_KEY,
    IDENTITY,
    make_app,
    make_registry,
    request_body,
    wait_terminal,
)


MODEL = "openai/gpt-6-astra"
REPLIES = ("DEMO first bounded answer", "DEMO resumed bounded answer")
CANARIES = (
    "DEMO_GLOBAL_MEMORY_CANARY",
    "DEMO_GLOBAL_USER_CANARY",
    "DEMO_GLOBAL_SOUL_CANARY",
    "DEMO_PROJECT_CONTEXT_CANARY",
)


def completion(content):
    return ChatCompletion.model_validate({
        "id": "chatcmpl-demo-bounded-http",
        "created": 1,
        "model": "openai/gpt-6-astra",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    })


def adapter_for(root):
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"bounded_admission_registry": make_registry()})
    )
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(root / "runs.db"))
    adapter._session_db = SessionDB(root / "state.db")
    return adapter


def message_pairs(messages):
    return [
        (message["role"], message.get("content"))
        for message in messages
        if message.get("role") != "system"
    ]


@pytest.mark.asyncio
async def test_authenticated_http_exchange_resumes_native_history_after_restart(
    tmp_path, monkeypatch, caplog
):
    home = Path(tmp_path / "home")
    home.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        path = tmp_path / name.lower()
        path.mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")
    monkeypatch.setenv("HERMES_DUMP_REQUEST_STDOUT", "1")
    for path, marker in zip(
        (
            home / "memories" / "MEMORY.md",
            home / "memories" / "USER.md",
            home / "SOUL.md",
            tmp_path / "AGENTS.md",
        ),
        CANARIES,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker, encoding="utf-8")

    network_attempts = []

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(sock, address):
        if address[0] in {"127.0.0.1", "::1"}:
            return real_connect(sock, address)
        network_attempts.append(("connect", address))
        raise RuntimeError("unexpected network access in bounded synthetic exchange")

    def guarded_connect_ex(sock, address):
        if address[0] in {"127.0.0.1", "::1"}:
            return real_connect_ex(sock, address)
        network_attempts.append(("connect_ex", address))
        raise RuntimeError("unexpected network access in bounded synthetic exchange")

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host in {"127.0.0.1", "::1"}:
            return real_getaddrinfo(host, *args, **kwargs)
        network_attempts.append(("getaddrinfo", host))
        raise RuntimeError("unexpected DNS access in bounded synthetic exchange")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)

    def forbid_plugin_discovery(*_args, **_kwargs):
        raise AssertionError("bounded synthetic exchange must not discover ambient plugins")

    hermes_plugins._join_background_discovery()
    monkeypatch.setattr(hermes_plugins, "discover_plugins", forbid_plugin_discovery)
    monkeypatch.setattr(hermes_plugins, "_ensure_plugins_discovered", forbid_plugin_discovery)

    lifecycle_calls = []
    lifecycle_checks = []
    middleware_calls = []
    execution_middleware_calls = []
    relay_calls = []
    nous_rate_state_calls = []
    provider_observer_calls = []
    empty_recovery_calls = []
    plugin_classifier_calls = []
    runtime_env_reads = []
    ambient_config_reads = []
    ambient_provider_profile_reads = []

    def forbid_lifecycle_hook(name, *_args, **_kwargs):
        lifecycle_calls.append(name)
        raise AssertionError("bounded synthetic exchange must not invoke ambient lifecycle hooks")

    def forbid_lifecycle_check(name, *_args, **_kwargs):
        lifecycle_checks.append(name)
        return False

    def forbid_middleware(*_args, **_kwargs):
        middleware_calls.append(True)
        raise AssertionError("bounded synthetic exchange must not invoke ambient middleware")

    def record_execution_middleware(request, execute, **kwargs):
        execution_middleware_calls.append((deepcopy(request), kwargs))
        return execute(request)

    real_relay_stream = relay_llm.stream
    real_complete_logical_call = relay_llm.complete_logical_call

    def record_relay_stream(*args, **kwargs):
        relay_calls.append("stream")
        return real_relay_stream(*args, **kwargs)

    def record_complete_logical_call(*args, **kwargs):
        relay_calls.append("complete")
        return real_complete_logical_call(*args, **kwargs)

    def forbid_runtime_env_read(*args, **kwargs):
        runtime_env_reads.append((args, kwargs))
        return args[1]

    real_load_config = hermes_config.load_config

    def record_ambient_config_read(*args, **kwargs):
        caller = traceback.extract_stack(limit=2)[0]
        ambient_config_reads.append((caller.filename, caller.name, caller.lineno))
        return real_load_config(*args, **kwargs)

    real_get_provider_profile = providers.get_provider_profile

    def record_ambient_provider_profile_read(*args, **kwargs):
        caller = traceback.extract_stack(limit=2)[0]
        ambient_provider_profile_reads.append((caller.filename, caller.name, caller.lineno))
        return real_get_provider_profile(*args, **kwargs)

    monkeypatch.setattr(hermes_lifecycle, "invoke_hook", forbid_lifecycle_hook)
    monkeypatch.setattr(hermes_lifecycle, "has_hook", forbid_lifecycle_check)
    monkeypatch.setattr(hermes_middleware, "apply_llm_request_middleware", forbid_middleware)
    monkeypatch.setattr(
        hermes_middleware,
        "run_llm_execution_middleware",
        record_execution_middleware,
    )
    monkeypatch.setattr(relay_llm, "stream", record_relay_stream)
    monkeypatch.setattr(relay_llm, "complete_logical_call", record_complete_logical_call)
    monkeypatch.setattr(
        nous_rate_guard,
        "nous_rate_limit_remaining",
        lambda: nous_rate_state_calls.append("read") or None,
    )
    monkeypatch.setattr(
        nous_rate_guard,
        "clear_nous_rate_limit",
        lambda: nous_rate_state_calls.append("clear"),
    )
    for method_name in (
        "_capture_rate_limits",
        "_capture_credits",
        "_stream_diag_capture_response",
        "_check_openrouter_cache_status",
    ):
        monkeypatch.setattr(
            AIAgent,
            method_name,
            lambda self, *_args, _name=method_name, **_kwargs: provider_observer_calls.append(_name),
        )
    monkeypatch.setattr(chat_completion_helpers, "env_int", forbid_runtime_env_read)
    monkeypatch.setattr(chat_completion_helpers, "env_float", forbid_runtime_env_read)
    monkeypatch.setattr(
        turn_empty_response,
        "interruptible_backoff_sleep",
        lambda *_args, **_kwargs: empty_recovery_calls.append(True) or None,
    )
    monkeypatch.setattr(hermes_config, "load_config", record_ambient_config_read)
    monkeypatch.setattr(providers, "get_provider_profile", record_ambient_provider_profile_read)
    monkeypatch.setattr(
        hermes_plugins,
        "get_plugin_error_classification",
        lambda *args, **kwargs: plugin_classifier_calls.append((args, kwargs)),
    )

    scripted = deque(REPLIES)
    provider_requests = []
    created_agents = []

    def create(_resource, **kwargs):
        request_index = len(provider_requests)
        provider_requests.append(deepcopy(kwargs))
        assert kwargs.get("stream") is True
        if request_index == len(REPLIES):
            raise ConnectionError("synthetic transient provider failure must stay private")
        if request_index == len(REPLIES) + 1:
            def broken_stream():
                yield SimpleNamespace(
                    choices=[SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(
                            content="DEMO partial provider output",
                            tool_calls=None,
                            reasoning_content=None,
                            reasoning=None,
                        ),
                        finish_reason=None,
                    )],
                    model=MODEL,
                    usage=None,
                )
                raise ConnectionError("synthetic partial provider failure must stay private")

            return broken_stream()
        if request_index == len(REPLIES) + 2:
            return iter((SimpleNamespace(
                choices=[SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=None,
                        reasoning_content=None,
                        reasoning=None,
                    ),
                    finish_reason="stop",
                )],
                model=MODEL,
                usage=None,
            ),))
        assert scripted, "unexpected auxiliary or retry provider request"
        reply = scripted.popleft()
        content = SimpleNamespace(
            choices=[SimpleNamespace(
                index=0,
                delta=SimpleNamespace(
                    content=reply,
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                ),
                finish_reason=None,
            )],
            model=MODEL,
            usage=None,
        )
        terminal = SimpleNamespace(
            choices=[SimpleNamespace(
                index=0,
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=None,
                    reasoning_content=None,
                    reasoning=None,
                ),
                finish_reason="stop",
            )],
            model=MODEL,
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
            ),
        )
        return iter((content, terminal))

    monkeypatch.setattr(Completions, "create", create)
    real_create_bound_agent = bounded_runs.create_bound_agent

    def record_bound_agent(*args, **kwargs):
        agent = real_create_bound_agent(*args, **kwargs)
        created_agents.append(agent)
        return agent

    monkeypatch.setattr(bounded_runs, "create_bound_agent", record_bound_agent)

    first = adapter_for(tmp_path)
    async with TestClient(TestServer(make_app(first))) as client:
        accepted = await client.post(
            "/v1/bounded-runs",
            data=request_body(message="DEMO first bounded user", key="e2e_first"),
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        assert accepted.status == 202, (accepted.status, await accepted.text(), network_attempts)
        accepted_body = await accepted.json()
        terminal_response, terminal = await wait_terminal(client, accepted_body["run_id"])
        assert terminal_response.status == 200
        assert terminal["status"] == "completed"
        assert terminal["output"] == REPLIES[0]

    binding = first._session_db.get_session_binding(terminal["session_id"])
    assert binding["session_id"] == terminal["session_id"]
    for key, value in IDENTITY.items():
        assert binding[key] == value
    assert message_pairs(first._session_db.get_messages_as_conversation(terminal["session_id"])) == [
        ("user", "DEMO first bounded user"),
        ("assistant", REPLIES[0]),
    ]
    first._run_idempotency_store.close()
    first._session_db.close()

    restarted = adapter_for(tmp_path)
    async with TestClient(TestServer(make_app(restarted))) as client:
        resumed = await client.post(
            "/v1/bounded-runs",
            data=request_body(message="DEMO second bounded user", key="e2e_second"),
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        resumed_body = await resumed.json()
        _, resumed_terminal = await wait_terminal(client, resumed_body["run_id"])
        assert resumed.status == 202
        assert resumed_terminal["status"] == "completed"
        assert resumed_terminal["output"] == REPLIES[1]

        foreign = await client.get(
            f"/v1/bounded-runs/{resumed_body['run_id']}",
            headers={"Authorization": "Bearer invalid-gateway-token"},
        )
        assert foreign.status == 401

    assert len(provider_requests) == 2
    for request in provider_requests:
        assert request["messages"][0] == {
            "role": "system",
            "content": APPROVED_PROMPT.decode("utf-8"),
        }
        assert "tools" not in request
        assert "tools" not in (request.get("extra_body") or {})
        assert request.get("max_completion_tokens", request.get("max_tokens")) == 4096
        assert "reasoning_effort" not in request
        assert "reasoning" not in request
        serialized = json.dumps(request["messages"])
        assert all(canary not in serialized for canary in CANARIES)

    assert message_pairs(provider_requests[0]["messages"]) == [
        ("user", "DEMO first bounded user"),
    ]
    assert message_pairs(provider_requests[1]["messages"]) == [
        ("user", "DEMO first bounded user"),
        ("assistant", REPLIES[0]),
        ("user", "DEMO second bounded user"),
    ]
    assert message_pairs(restarted._session_db.get_messages_as_conversation(terminal["session_id"])) == [
        ("user", "DEMO first bounded user"),
        ("assistant", REPLIES[0]),
        ("user", "DEMO second bounded user"),
        ("assistant", REPLIES[1]),
    ]
    session = restarted._session_db.get_session(terminal["session_id"])
    assert session["api_call_count"] == 2
    assert session["input_tokens"] == 200
    assert session["output_tokens"] == 20
    # The numeric accumulator stays zero when no price is admitted; status/source
    # are authoritative and prevent this from being represented as zero-cost usage.
    assert session["estimated_cost_usd"] == 0.0
    assert session["cost_status"] == "unknown"
    assert session["cost_source"] == "none"
    assert not scripted

    async with TestClient(TestServer(make_app(restarted))) as client:
        failed_response = await client.post(
            "/v1/bounded-runs",
            data=request_body(message="DEMO bounded failure", key="e2e_failure"),
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        failed_body = await failed_response.json()
        _, failed_terminal = await wait_terminal(client, failed_body["run_id"])
        assert failed_response.status == 202
        assert failed_terminal["status"] == "failed"
        assert failed_terminal["error"] == "Bounded run failed"
        assert "synthetic transient provider failure" not in json.dumps(failed_terminal)
        assert message_pairs(
            restarted._session_db.get_messages_as_conversation(terminal["session_id"])
        ) == [
            ("user", "DEMO first bounded user"),
            ("assistant", REPLIES[0]),
            ("user", "DEMO second bounded user"),
            ("assistant", REPLIES[1]),
        ]

        partial_response = await client.post(
            "/v1/bounded-runs",
            data=request_body(message="DEMO bounded partial failure", key="e2e_partial_failure"),
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        partial_body = await partial_response.json()
        _, partial_terminal = await wait_terminal(client, partial_body["run_id"])
        assert partial_response.status == 202
        assert partial_terminal["status"] == "failed"
        assert partial_terminal["error"] == "Bounded run failed"
        assert "DEMO partial provider output" not in json.dumps(partial_terminal)
        assert "synthetic partial provider failure" not in json.dumps(partial_terminal)
        assert message_pairs(
            restarted._session_db.get_messages_as_conversation(terminal["session_id"])
        ) == [
            ("user", "DEMO first bounded user"),
            ("assistant", REPLIES[0]),
            ("user", "DEMO second bounded user"),
            ("assistant", REPLIES[1]),
        ]

        malformed_response = await client.post(
            "/v1/bounded-runs",
            data=request_body(message="DEMO bounded malformed response", key="e2e_malformed"),
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
        malformed_body = await malformed_response.json()
        _, malformed_terminal = await wait_terminal(client, malformed_body["run_id"])
        assert malformed_response.status == 202
        assert malformed_terminal["status"] == "failed"
        assert malformed_terminal["error"] == "Bounded run failed"
        assert message_pairs(
            restarted._session_db.get_messages_as_conversation(terminal["session_id"])
        ) == [
            ("user", "DEMO first bounded user"),
            ("assistant", REPLIES[0]),
            ("user", "DEMO second bounded user"),
            ("assistant", REPLIES[1]),
        ]

    assert len(provider_requests) == 5
    assert created_agents
    assert {agent._api_max_retries for agent in created_agents} == {1}
    assert provider_requests[2]["messages"][0] == {
        "role": "system",
        "content": APPROVED_PROMPT.decode("utf-8"),
    }
    assert plugin_classifier_calls == []
    assert "synthetic transient provider failure must stay private" not in caplog.text
    assert "synthetic partial provider failure must stay private" not in caplog.text
    assert list((home / "sessions").glob("request_dump_*")) == []
    assert network_attempts == []
    assert lifecycle_calls == []
    assert lifecycle_checks == []
    assert middleware_calls == []
    assert execution_middleware_calls == []
    assert provider_observer_calls == []
    assert relay_calls == []
    assert nous_rate_state_calls == []
    assert empty_recovery_calls == []
    assert runtime_env_reads == []
    agent_config_reads = [
        call for call in ambient_config_reads
        if "\\agent\\" in call[0] or "/agent/" in call[0]
    ]
    if agent_config_reads or ambient_provider_profile_reads:
        pytest.fail(
            "Exact bounded execution used ambient configuration:\n"
            + "\n".join(
                f"config {filename}:{lineno} {name}"
                for filename, name, lineno in sorted(set(agent_config_reads))
            )
            + "\n"
            + "\n".join(
                f"provider_profile {filename}:{lineno} {name}"
                for filename, name, lineno in sorted(set(ambient_provider_profile_reads))
            )
        )
    restarted._run_idempotency_store.close()
    restarted._session_db.close()
