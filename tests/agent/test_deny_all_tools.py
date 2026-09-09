"""Tool authority is independent of a mutable schema snapshot."""
import json
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def make_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: [])
    from run_agent import AIAgent
    def make(**kwargs):
        return AIAgent(model="gpt-4o", provider="openai", api_key="test-only",
                       base_url="https://example.invalid/v1", quiet_mode=True,
                       skip_memory=True, skip_context_files=True,
                       skip_background_review=True, **kwargs)
    return make


@pytest.mark.parametrize("field", ["deny_all_tools", "require_durable_history"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", [], {}])
def test_constructor_rejects_non_bool(field, value):
    from run_agent import AIAgent
    with pytest.raises(TypeError, match=field + " must be a bool"):
        AIAgent(**{field: value})


def test_intent_is_readonly_and_not_schema_derived(make_agent):
    agent = make_agent(deny_all_tools=True, require_durable_history=True)
    agent.tools = [{"type": "function", "function": {"name": "resurrected"}}]
    assert agent.deny_all_tools is True
    assert agent.require_durable_history is True
    for field in ("deny_all_tools", "require_durable_history", "_deny_all_tools", "_require_durable_history"):
        with pytest.raises(AttributeError):
            setattr(agent, field, False)
        with pytest.raises(AttributeError):
            delattr(agent, field)
    ordinary = make_agent()
    assert ordinary.deny_all_tools is False
    assert ordinary.require_durable_history is False


def test_outbound_publication_overrides_resurrected_tools(make_agent):
    agent = make_agent(deny_all_tools=True)
    resurrected = [{"type": "function", "function": {"name": "resurrected", "parameters": {"type": "object", "properties": {}}}}]
    agent.tools = resurrected
    from agent.chat_completion_helpers import build_api_kwargs
    for tools in (None, resurrected):
        result = build_api_kwargs(agent, [{"role": "user", "content": "hello"}], tools)
        assert not result.get("tools")


@pytest.mark.parametrize("path", ["sequential", "concurrent", "invoke", "middleware"])
def test_hallucinated_calls_denied_before_hooks_or_handlers(make_agent, monkeypatch, path):
    agent = make_agent(deny_all_tools=True)
    from agent import tool_executor, agent_runtime_helpers
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    import hermes_cli.middleware as middleware
    recorded = []
    monkeypatch.setitem(INLINE_TOOL_EXECUTORS, "todo", lambda *a, **kw: recorded.append("inline"))
    monkeypatch.setattr(middleware, "apply_tool_request_middleware", lambda *a, **kw: recorded.append("hook"))
    tc = SimpleNamespace(id="denied-call", function=SimpleNamespace(name="todo", arguments="{}"))
    messages = []
    if path == "invoke":
        result = agent_runtime_helpers.invoke_tool(agent, "todo", {}, "test")
    elif path == "middleware":
        result = tool_executor._run_agent_tool_execution_middleware(agent, function_name="todo", function_args={}, effective_task_id="test", tool_call_id="denied-call", execute=lambda args: recorded.append("handler")).result
    else:
        getattr(tool_executor, "execute_tool_calls_" + path)(agent, SimpleNamespace(tool_calls=[tc]), messages, "test")
        result = messages[-1]["content"]
    assert "denied" in json.loads(result)["error"].lower()
    assert recorded == []


@pytest.mark.parametrize("kwargs", [{"api_mode": "codex_app_server"}, {"provider": "copilot-acp"}, {"base_url": "acp://test"}])
def test_independently_tooled_backends_rejected(kwargs):
    from run_agent import AIAgent
    with pytest.raises(ValueError, match="deny_all_tools"):
        AIAgent(deny_all_tools=True, **kwargs)


def test_turn_policy_reaches_raw_dispatch_and_cannot_be_weakened(monkeypatch):
    from agent.tool_execution_policy import with_agent_tool_policy, tools_denied
    from tools.registry import ToolRegistry
    import model_tools
    registry = ToolRegistry()
    calls = []
    registry.register(name="demo_probe", toolset="demo", schema={"name": "demo_probe"},
                      handler=lambda args, **kw: calls.append(args) or "{}")

    @with_agent_tool_policy
    def ordinary_inner(agent):
        assert tools_denied(agent)
        return registry.dispatch("demo_probe", {})

    @with_agent_tool_policy
    def denied_turn(agent):
        assert "denied" in json.loads(ordinary_inner(SimpleNamespace(deny_all_tools=False)))["error"]
        # A raw call must be rejected before even coercing its arguments.
        monkeypatch.setattr(model_tools, "coerce_tool_args", lambda *a: pytest.fail("coercion reached"))
        assert "denied" in json.loads(model_tools.handle_function_call("demo_probe", {}))["error"]
        raise RuntimeError("demo interruption")

    with pytest.raises(RuntimeError, match="demo interruption"):
        denied_turn(SimpleNamespace(deny_all_tools=True))
    assert calls == []
    assert not tools_denied(SimpleNamespace(deny_all_tools=False))
    assert registry.dispatch("demo_probe", {}) == "{}"
    assert calls == [{}]


def test_final_nonstreaming_dispatch_removes_late_injection(make_agent):
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    agent = make_agent(deny_all_tools=True)
    captured = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: captured.append(kwargs) or "demo-reply")))
    result = _dispatch_nonstreaming_api_request(agent,
        {"model": "demo", "messages": [], "tools": [{"type": "web_search"}],
         "extra_body": {"tools": [{"type": "computer_use"}], "demo": True}},
        make_client=lambda *a, **kw: client)
    assert result == "demo-reply"
    assert "tools" not in captured[0]
    assert captured[0]["extra_body"] == {"demo": True}


@pytest.mark.parametrize("api_mode", ["anthropic_messages", "bedrock_converse", "codex_responses", "unknown"])
def test_unverified_tool_free_transports_fail_closed(make_agent, api_mode):
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request
    agent = make_agent(deny_all_tools=True)
    agent.api_mode = api_mode
    with pytest.raises(ValueError, match="deny_all_tools"):
        _dispatch_nonstreaming_api_request(agent, {},
            make_client=lambda *a, **kw: pytest.fail("unverified backend reached"))


def test_bound_session_restores_denial_when_flags_are_omitted(make_agent, tmp_path):
    from hermes_state import SessionDB
    with SessionDB(tmp_path / "bound.db") as db:
        session_id = db.create_bound_session(owner_id="demo-owner", agent_id="demo-agent",
            case_id="demo-case", source="demo-desk", context_digest="c" * 64)
        agent = make_agent(session_db=db, session_id=session_id)
        assert agent.deny_all_tools is True
        assert agent.require_durable_history is True


def test_deny_all_does_not_resolve_default_toolsets(make_agent, monkeypatch):
    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions",
                        lambda **kw: pytest.fail("default tool discovery reached"))
    agent = make_agent(deny_all_tools=True)
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_final_streaming_dispatch_removes_late_injection(make_agent, monkeypatch):
    from agent.chat_completion_helpers import _StreamingCall
    agent = make_agent(deny_all_tools=True)
    captured = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: captured.append(kwargs) or "demo-stream")))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    call = SimpleNamespace(agent=agent, clients=SimpleNamespace(set_client=lambda c: c),
                           last_chunk_time={})
    result = _StreamingCall._open_chat_stream(call,
        {"tools": [{"type": "web_search"}], "extra_body": {"tools": ["demo-tool"]}})
    assert result == "demo-stream"
    assert "tools" not in captured[0]
    assert "tools" not in captured[0]["extra_body"]


def test_exact_request_ignores_ambient_request_dump_flags(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")
    monkeypatch.setenv("HERMES_DUMP_REQUEST_STDOUT", "1")
    from run_agent import AIAgent
    from agent.turn_api_request import build_api_request
    from hermes_state import SessionDB

    prompt = b"DEMO exact prompt"
    db = SessionDB(tmp_path / "state.db")
    session_id = db.create_bound_session(
        owner_id="demo-owner",
        agent_id="demo-agent",
        case_id="demo-case",
        source="demo-source",
        context_digest=sha256(prompt).hexdigest(),
    )
    agent = AIAgent(
        base_url="https://inference-api.nousresearch.com/v1",
        api_key="test-only",
        provider="nous",
        requested_provider="nous",
        api_mode="chat_completions",
        model="openai/gpt-6-astra",
        max_tokens=4096,
        enabled_toolsets=[],
        deny_all_tools=True,
        require_durable_history=True,
        quiet_mode=True,
        exact_system_prompt_bytes=prompt,
        exact_context_length=65_536,
        session_db=db,
        session_id=session_id,
        skip_memory=True,
        skip_context_files=True,
        skip_background_review=True,
    )
    agent._empty_content_retries = 0
    agent._is_user_initiated_turn = False
    agent._dump_api_request_debug = MagicMock()
    result = build_api_request(
        agent,
        api_messages=[
            {"role": "system", "content": prompt.decode("utf-8")},
            {"role": "user", "content": "DEMO private user message"},
        ],
        _moa_prepared_request=None,
        tools_for_api=[],
        system_message="DEMO exact prompt",
        messages=[{"role": "user", "content": "DEMO private user message"}],
        original_user_message="DEMO private user message",
        approx_tokens=4,
        total_chars=25,
        retry_count=0,
        api_call_count=1,
        api_request_id="demo-request",
        api_start_time=1.0,
        effective_task_id="demo-task",
        turn_id="demo-turn",
    )
    assert result.api_kwargs["messages"][1]["content"] == "DEMO private user message"
    agent._dump_api_request_debug.assert_not_called()
    db.close()


def test_non_exact_request_still_honors_ambient_request_dump_flags(
    make_agent, monkeypatch
):
    from agent.turn_api_request import build_api_request

    agent = make_agent()
    agent._empty_content_retries = 0
    agent._is_user_initiated_turn = False
    agent._dump_api_request_debug = MagicMock()
    monkeypatch.setenv("HERMES_DUMP_REQUESTS", "1")

    build_api_request(
        agent,
        api_messages=[{"role": "user", "content": "DEMO generic message"}],
        _moa_prepared_request=None,
        tools_for_api=[],
        system_message="",
        messages=[{"role": "user", "content": "DEMO generic message"}],
        original_user_message="DEMO generic message",
        approx_tokens=4,
        total_chars=20,
        retry_count=0,
        api_call_count=1,
        api_request_id="demo-generic-request",
        api_start_time=1.0,
        effective_task_id=None,
        turn_id="demo-generic-turn",
    )

    agent._dump_api_request_debug.assert_called_once()


def test_exact_provider_call_has_no_ambient_execution_wrappers(monkeypatch):
    from agent.turn_api_call import perform_api_call
    from agent import relay_llm
    import hermes_cli.middleware as middleware

    wrapper_calls = []

    def execution_wrapper(request, execute, **_kwargs):
        wrapper_calls.append("middleware")
        return execute(request)

    def relay_wrapper(request, execute, **_kwargs):
        wrapper_calls.append("relay")
        return execute(request)

    monkeypatch.setattr(middleware, "run_llm_execution_middleware", execution_wrapper)
    monkeypatch.setattr(relay_llm, "execute", relay_wrapper)
    agent = SimpleNamespace(
        _exact_system_prompt="DEMO exact prompt",
        _disable_streaming=True,
        _has_stream_consumers=lambda: False,
        _interruptible_api_call=lambda request: ("provider", request),
        _model_request_active=None,
        _pending_redirect_lock=None,
        _has_pending_redirect=lambda: False,
        api_mode="chat_completions",
        base_url="https://inference-api.nousresearch.com/v1",
        provider="nous",
        client=object(),
        session_id="demo-session",
        platform="api_server",
        model="openai/gpt-6-astra",
        is_subagent=False,
        _fallback_index=0,
    )

    verdict = perform_api_call(
        agent,
        api_kwargs={"messages": [{"role": "system", "content": "DEMO exact prompt"}]},
        _original_api_kwargs={"messages": []},
        _llm_middleware_trace=[],
        _moa_prepared_request=None,
        _retry=SimpleNamespace(),
        thinking_spinner=None,
        retry_count=0,
        api_call_count=1,
        api_request_id="demo-request",
        effective_task_id=None,
        turn_id="demo-turn",
        interrupted=False,
    )

    assert verdict.response[0] == "provider"
    assert wrapper_calls == []


def test_exact_invalid_response_returns_terminal_failure_before_recovery(monkeypatch):
    import agent.turn_recovery as turn_recovery
    import agent.turn_response_check as response_check

    recovery_calls = []
    monkeypatch.setattr(
        turn_recovery,
        "validate_response_shape",
        lambda _agent, _response: (True, "missing choices"),
    )

    def record_recovery(*_args, **_kwargs):
        recovery_calls.append(True)
        return SimpleNamespace(
            action="continue",
            result=None,
            thinking_spinner=None,
            active_system_prompt="DEMO exact prompt",
            retry_count=1,
            compression_attempts=0,
        )

    monkeypatch.setattr(response_check, "retry_invalid_response", record_recovery)
    verdict = response_check.check_api_response(
        SimpleNamespace(
            _exact_system_prompt="DEMO exact prompt",
            quiet_mode=True,
            verbose_logging=False,
            thinking_callback=None,
        ),
        response=SimpleNamespace(choices=[]),
        _retry=SimpleNamespace(),
        thinking_spinner=None,
        messages=[{"role": "user", "content": "DEMO bounded message"}],
        api_messages=[],
        api_kwargs={},
        active_system_prompt="DEMO exact prompt",
        conversation_history=[],
        finish_reason=None,
        retry_count=0,
        max_retries=3,
        compression_attempts=0,
        max_compression_attempts=3,
        length_continue_retries=0,
        truncated_response_parts=[],
        truncated_tool_call_retries=0,
        current_turn_user_idx=0,
        api_call_count=1,
        api_request_id="demo-request",
        api_start_time=0.0,
        effective_task_id="demo-task",
        turn_id="demo-turn",
        _preflight_compression_blocked=False,
        _last_preflight_pressure=None,
    )

    assert verdict.action == "return"
    assert verdict.retry_count == 0
    assert verdict.result == {
        "completed": False,
        "failed": True,
        "error": "Bounded run failed",
    }
    assert recovery_calls == []


@pytest.mark.parametrize("finish_reason", ["content_filter", "length"])
def test_exact_nonterminal_finish_reason_returns_failure_before_recovery(
    monkeypatch, finish_reason
):
    import agent.turn_recovery as turn_recovery
    import agent.turn_response_check as response_check

    recovery_calls = []
    monkeypatch.setattr(
        turn_recovery,
        "validate_response_shape",
        lambda _agent, _response: (False, ""),
    )
    monkeypatch.setattr(
        response_check,
        "_derive_finish_reason",
        lambda _agent, _response, _messages: finish_reason,
    )
    monkeypatch.setattr(
        response_check,
        "handle_content_policy_refusal",
        lambda *_args, **_kwargs: (
            recovery_calls.append("content_filter")
            or SimpleNamespace(
                action="break",
                result=None,
                active_system_prompt="DEMO exact prompt",
            )
        ),
    )
    monkeypatch.setattr(
        response_check,
        "recover_from_truncation",
        lambda *_args, **_kwargs: (
            recovery_calls.append("length")
            or SimpleNamespace(
                action="break",
                result=None,
                messages=[{"role": "user", "content": "DEMO bounded message"}],
                length_continue_retries=0,
                truncated_response_parts=[],
                truncated_tool_call_retries=0,
                retry_count=0,
                compression_attempts=0,
            )
        ),
    )
    verdict = response_check.check_api_response(
        SimpleNamespace(
            _exact_system_prompt="DEMO exact prompt",
            quiet_mode=True,
            verbose_logging=False,
            thinking_callback=None,
        ),
        response=SimpleNamespace(choices=[object()]),
        _retry=SimpleNamespace(),
        thinking_spinner=None,
        messages=[{"role": "user", "content": "DEMO bounded message"}],
        api_messages=[],
        api_kwargs={},
        active_system_prompt="DEMO exact prompt",
        conversation_history=[],
        finish_reason=None,
        retry_count=0,
        max_retries=1,
        compression_attempts=0,
        max_compression_attempts=3,
        length_continue_retries=0,
        truncated_response_parts=[],
        truncated_tool_call_retries=0,
        current_turn_user_idx=0,
        api_call_count=1,
        api_request_id="demo-request",
        api_start_time=0.0,
        effective_task_id="demo-task",
        turn_id="demo-turn",
        _preflight_compression_blocked=False,
        _last_preflight_pressure=None,
    )

    assert verdict.action == "return"
    assert verdict.result == {
        "completed": False,
        "failed": True,
        "error": "Bounded run failed",
    }
    assert recovery_calls == []
