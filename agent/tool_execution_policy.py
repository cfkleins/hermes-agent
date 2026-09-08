"""Session tool authority; not a memory/context/lifecycle isolation policy."""
import json
from contextvars import ContextVar
from functools import wraps


DENIED_RESULT = json.dumps({"error": "Tool execution denied by deny_all_tools policy"})
_tool_denial = ContextVar("hermes_tool_denial", default=False)
_TOOL_FIELDS = frozenset(("tools", "tool_choice", "parallel_tool_calls", "functions", "function_call"))


def tools_denied(agent):
    return _tool_denial.get() or getattr(agent, "deny_all_tools", False) is True


def with_agent_tool_policy(fn):
    """Nested execution may tighten but never weaken the current turn's denial."""
    @wraps(fn)
    def guarded(agent, *args, **kwargs):
        token = _tool_denial.set(tools_denied(agent))
        try:
            return fn(agent, *args, **kwargs)
        finally:
            _tool_denial.reset(token)
    return guarded


def validate_tool_backend(*, deny_all_tools, api_mode=None, provider=None, base_url=None,
                          acp_command=None):
    if not deny_all_tools:
        return
    provider = (provider or "").lower()
    if (api_mode not in (None, "chat_completions") or provider == "moa"
            or provider.endswith("-acp") or provider == "acp"
            or (base_url or "").lower().startswith("acp://") or acp_command):
        raise ValueError("deny_all_tools requires a verified chat_completions backend")


def enforce_tool_request_policy(agent, payload):
    """Recheck immediately before publication, after request middleware/fallback."""
    if not tools_denied(agent):
        return payload
    validate_tool_backend(deny_all_tools=True, api_mode=agent.api_mode,
                          provider=agent.provider, base_url=agent.base_url,
                          acp_command=getattr(agent, "acp_command", None))
    clean = {key: value for key, value in payload.items() if key not in _TOOL_FIELDS}
    if isinstance(clean.get("extra_body"), dict):
        clean["extra_body"] = {key: value for key, value in clean["extra_body"].items()
                               if key not in _TOOL_FIELDS}
    return clean


def deny_tool_batch(agent, assistant_message, messages):
    """Pair hallucinated calls without entering tool hooks or dispatch machinery."""
    if not tools_denied(agent):
        return False
    from agent.message_sanitization import coalesce_tool_call_id
    for tc in assistant_message.tool_calls:
        messages.append({"role": "tool", "tool_call_id": coalesce_tool_call_id(tc),
                         "name": tc.function.name, "content": DENIED_RESULT})
    return True
