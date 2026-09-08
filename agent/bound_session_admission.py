"""Validate server-declared case identity before a bound turn touches history.

This is consistency enforcement, not HTTP authentication. The transport must
supply identity from its authenticated principal and approved case registry.
"""
from hermes_state_bound_sessions import _IDENTITY_FIELDS


def validate_bound_turn(agent, binding_identity):
    db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    initialized_binding = getattr(agent, "_bound_session_binding", None)
    can_read_binding = callable(getattr(type(db), "get_session_binding", None))
    binding = db.get_session_binding(session_id) if session_id and can_read_binding else None
    if initialized_binding is not None and binding != initialized_binding:
        raise ValueError("Durable session binding changed since agent initialization")
    if binding is None:
        if binding_identity is not None:
            raise ValueError("Explicit binding identity requires a bound durable session")
        return None
    if (getattr(agent, "deny_all_tools", False) is not True
            or getattr(agent, "require_durable_history", False) is not True):
        raise ValueError("Bound session requires its durable execution policy")
    if type(binding_identity) is not dict or set(binding_identity) != set(_IDENTITY_FIELDS):
        raise ValueError("Explicit complete binding identity is required for each bound turn")
    # Copy before any wait so a caller cannot alter the admitted identity in place.
    authorized = db.require_session_binding(session_id, **dict(binding_identity))
    if authorized != binding:
        raise ValueError("Session binding changed during identity validation")
    return dict(authorized)
