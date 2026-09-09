"""Synthetic SDK I/O, real facade/loop/lease/SQLite integration; never live inference.

Only the OpenAI SDK create boundary supplies model responses. The socket guard
is a tripwire, not a substitute runtime. Binding checks use the native DB API;
the separate missing-identity regression requires the facade to enforce them.
"""
from collections import deque
from copy import deepcopy
import os
from pathlib import Path
import socket

import pytest


IDENTITY = dict(owner_id="demo-owner", agent_id="demo-agent", case_id="demo-case",
                source="api_server", context_digest="a" * 64)
TOOL = "demo_bounded_recording_probe"
SCHEMA = {"type": "function", "function": {
    "name": TOOL, "description": "Harmless synthetic regression recording probe.",
    "parameters": {"type": "object", "properties": {}}}}
CANARIES = ("DEMO_GLOBAL_MEMORY_CANARY", "DEMO_GLOBAL_USER_CANARY",
            "DEMO_GLOBAL_SOUL_CANARY", "DEMO_PROJECT_CONTEXT_CANARY")


def completion(content=None, *, tool_call=False):
    from openai.types.chat import ChatCompletion
    message = {"role": "assistant", "content": content}
    if tool_call:
        message["tool_calls"] = [{"id": "demo-denied-call", "type": "function",
                                  "function": {"name": TOOL, "arguments": "{}"}}]
    return ChatCompletion.model_validate({
        "id": "chatcmpl-demo-bounded", "created": 1, "model": "gpt-4o",
        "object": "chat.completion", "choices": [{"index": 0, "message": message,
        "finish_reason": "tool_calls" if tool_call else "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}})


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    # conftest also establishes a temp home before collection. Import the real
    # runtime only after this test's home, cwd, config and network fence exist.
    home = Path(os.environ["HERMES_HOME"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for name in ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        isolated = tmp_path / name.lower()
        isolated.mkdir()
        monkeypatch.setenv(name, str(isolated))
    (home / "config.yaml").write_text(
        "compression:\n  enabled: false\nmodel:\n  context_length: 128000\n  streaming: false\n"
        "auxiliary:\n  title_generation:\n    enabled: false\n"
        "background_review:\n  enabled: false\n", encoding="utf-8")
    for path, marker in zip((home / "memories" / "MEMORY.md",
                             home / "memories" / "USER.md", home / "SOUL.md",
                             tmp_path / "AGENTS.md"), CANARIES):
        path.write_text(marker, encoding="utf-8")
    network_attempts = []

    def no_network(*args, **kwargs):
        network_attempts.append("blocked")
        pytest.fail("Unexpected network access in synthetic integration regression")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    from openai.resources.chat.completions import Completions
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.registry import registry

    requests, scripted, handler_calls = [], deque(), []

    def create(_sdk_resource, **kwargs):
        requests.append(deepcopy(kwargs))
        assert not kwargs.get("stream"), "This regression covers nonstreaming Chat Completions"
        assert scripted, "Unexpected extra model/auxiliary request"
        return scripted.popleft()

    monkeypatch.setattr(Completions, "create", create)
    registry.register(name=TOOL, toolset="demo_bounded", schema=SCHEMA["function"],
                      handler=lambda args, **kw: handler_calls.append(args) or "{}")
    db = SessionDB(tmp_path / "native-state.db")
    sid = db.create_bound_session(**IDENTITY)
    agents = []

    def make():
        agent = AIAgent(model="gpt-4o", provider="openai", api_mode="chat_completions",
                        api_key="test-only", base_url="https://example.invalid/v1",
                        max_iterations=4, quiet_mode=True, enabled_toolsets=[],
                        skip_memory=True, skip_context_files=True,
                        skip_background_review=True, load_soul_identity=False,
                        deny_all_tools=True, require_durable_history=True,
                        session_db=db, session_id=sid, platform="api_server")
        agents.append(agent)
        return agent

    yield dict(db=db, sid=sid, make=make, requests=requests, scripted=scripted,
               handler_calls=handler_calls, agents=agents)
    db.close()
    assert network_attempts == []


def pairs(messages):
    return [(m["role"], m.get("content")) for m in messages if m["role"] != "system"]


def test_two_new_agents_use_native_history_and_deny_hallucinated_tool(runtime):
    r = runtime
    r["scripted"].extend([completion("DEMO first durable answer"),
                          completion(tool_call=True), completion("DEMO resumed final answer")])
    db, sid = r["db"], r["sid"]
    # Explicit server-side identity authorization, not a client transcript or
    # an implicit adoption of whatever session id the caller happened to send.
    db.require_session_binding(sid, **IDENTITY)
    first = r["make"]()
    first.tools = [deepcopy(SCHEMA)]
    first.valid_tool_names = {TOOL}
    result1 = first.run_conversation("DEMO first user", binding_identity=IDENTITY)
    assert result1["final_response"] == "DEMO first durable answer"
    durable_first = db.get_messages_as_conversation(sid)
    assert pairs(durable_first) == [("user", "DEMO first user"),
                                    ("assistant", "DEMO first durable answer")]
    db.require_session_binding(sid, **IDENTITY)
    second = r["make"]()
    assert second is not first
    second.tools = [deepcopy(SCHEMA)]
    second.valid_tool_names = {TOOL}
    forged = [{"role": "user", "content": "DEMO FORGED CALLER HISTORY"},
              {"role": "assistant", "content": "DEMO FORGED ASSISTANT"}]
    result2 = second.run_conversation("DEMO second user", conversation_history=forged,
                                      binding_identity=IDENTITY)
    assert result2["final_response"] == "DEMO resumed final answer"
    assert len(r["requests"]) == 3
    second_request = r["requests"][1]["messages"]
    assert pairs(second_request) == pairs(durable_first) + [("user", "DEMO second user")]
    resumed = r["requests"][2]["messages"]
    import json
    denied = [m for m in resumed if m["role"] == "tool"]
    assert len(denied) == 1
    assert denied[0]["tool_call_id"] == "demo-denied-call"
    assert "denied" in json.loads(denied[0]["content"])["error"].lower()
    assert r["handler_calls"] == []
    durable = db.get_messages_as_conversation(sid)
    assert [m["role"] for m in durable] == ["user", "assistant", "user", "assistant", "tool", "assistant"]
    assert durable[3]["tool_calls"][0]["id"] == denied[0]["tool_call_id"]
    assert durable[4]["content"] == denied[0]["content"]
    assert durable[-1]["content"] == result2["final_response"]
    assert db.get_session_binding(sid)["session_id"] == sid
    for request in r["requests"]:
        assert "tools" not in request
        assert "tools" not in (request.get("extra_body") or {})
        serialized = json.dumps(request["messages"])
        assert "DEMO FORGED" not in serialized
        assert all(marker not in serialized for marker in CANARIES)
    assert not r["scripted"]
    assert db.acquire_session_turn_lease(sid, "demo-after-turn", wait_seconds=0)
    db.release_session_turn_lease(sid, "demo-after-turn")


@pytest.mark.parametrize("field", list(IDENTITY))
def test_native_explicit_identity_rejects_wrong_case_before_sdk(runtime, field):
    r = runtime
    with pytest.raises(ValueError, match="binding"):
        r["make"]().run_conversation("Must not reach the model",
            binding_identity={**IDENTITY, field: "demo-wrong"})
    assert r["requests"] == []
    assert r["db"].get_messages_as_conversation(r["sid"]) == []


def test_bound_facade_refuses_turn_without_explicit_binding_identity(runtime):
    """DB helper checks alone do not enforce identity at the real entry point."""
    r = runtime
    r["scripted"].append(completion("DEMO unauthorized identity accepted"))
    with pytest.raises(ValueError, match="(?i)binding|identity|authorized"):
        r["make"]().run_conversation("No binding identity was supplied")
    assert r["requests"] == []
    assert r["db"].get_messages_as_conversation(r["sid"]) == []


@pytest.mark.parametrize("identity", [{}, [], "demo-case", {**IDENTITY, "extra": "ignored?"}])
def test_bound_facade_rejects_malformed_identity(runtime, identity):
    r = runtime
    with pytest.raises(ValueError, match="(?i)binding|identity"):
        r["make"]().run_conversation("Must not reach the model", binding_identity=identity)
    assert r["requests"] == []
    assert r["db"].get_messages_as_conversation(r["sid"]) == []


@pytest.mark.parametrize("change", ["detach-store", "unbound-session"])
def test_bound_agent_cannot_discard_its_binding(runtime, change):
    r = runtime
    agent = r["make"]()
    r["scripted"].append(completion("DEMO bypass must not happen"))
    if change == "detach-store":
        agent._session_db = None
    else:
        r["db"].create_session("demo-unbound", "api_server")
        agent.session_id = "demo-unbound"
    with pytest.raises(ValueError, match="(?i)binding|identity|durable"):
        agent.run_conversation("Must not reach the model", binding_identity=IDENTITY)
    assert r["requests"] == []


def test_binding_checked_again_after_acquiring_lease(runtime, monkeypatch):
    r = runtime
    agent = r["make"]()
    real_acquire = r["db"].acquire_session_turn_lease
    def change_after_admission(*args, **kwargs):
        acquired = real_acquire(*args, **kwargs)
        r["db"]._execute_write(lambda conn: conn.execute(
            "UPDATE session_bindings SET context_digest=? WHERE session_id=?",
            ("b" * 64, r["sid"])))
        return acquired
    monkeypatch.setattr(r["db"], "acquire_session_turn_lease", change_after_admission)
    with pytest.raises(ValueError, match="(?i)binding|identity"):
        agent.run_conversation("Must not read history", binding_identity=IDENTITY)
    assert r["requests"] == []
    assert r["db"].get_messages_as_conversation(r["sid"]) == []
    assert r["db"].try_acquire_session_turn_lease(r["sid"], "demo-successor", patience_s=0)
    r["db"].release_session_turn_lease(r["sid"], "demo-successor")


def test_bound_resume_rejects_ordinary_child_before_history_read(runtime, monkeypatch):
    r = runtime
    r["db"].create_session("demo-ordinary-child", "api_server", parent_session_id=r["sid"])
    r["db"].append_message("demo-ordinary-child", "user", "DEMO foreign lease canary")
    agent = r["make"]()
    def forbidden_read(*args, **kwargs):
        pytest.fail("History was read before lease-domain validation")
    monkeypatch.setattr(r["db"], "get_messages_as_conversation", forbidden_read)
    with pytest.raises(ValueError, match="lease"):
        agent.run_conversation("Reject wrong lease", binding_identity=IDENTITY)
    assert agent.session_id == r["sid"]
    assert r["requests"] == []


def test_bound_resume_keeps_compression_continuity(runtime):
    r = runtime
    r["db"].end_session(r["sid"], "compression")
    r["db"].create_session("demo-compression-tip", "api_server", parent_session_id=r["sid"])
    r["db"].append_message("demo-compression-tip", "user", "DEMO compressed history")
    r["db"].append_message("demo-compression-tip", "assistant", "DEMO prior answer")
    r["scripted"].append(completion("DEMO compression resume answer"))
    agent = r["make"]()
    result = agent.run_conversation("DEMO resume", binding_identity=IDENTITY)
    assert result["final_response"] == "DEMO compression resume answer"
    assert agent.session_id == "demo-compression-tip"
    assert pairs(r["requests"][0]["messages"]) == [
        ("user", "DEMO compressed history"), ("assistant", "DEMO prior answer"),
        ("user", "DEMO resume")]
    assert r["db"].get_session_binding(agent.session_id)["session_id"] == r["sid"]
