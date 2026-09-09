"""Server-owned case bindings in the native session store.

Creation is atomic with the empty session row. No operation adopts or retags an
existing conversation. The binding is authority metadata, not a tool grant or
proof that a particular API/agent invocation enforces it.
"""
import json
import re
import time
import uuid

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}", re.ASCII)
_IDENTITY_FIELDS = ("owner_id", "agent_id", "case_id", "source", "context_digest")


def _identity(owner_id, agent_id, case_id, source, context_digest):
    values = dict(zip(_IDENTITY_FIELDS, (owner_id, agent_id, case_id, source, context_digest)))
    for key, value in values.items():
        pattern = _DIGEST if key == "context_digest" else _ID
        if type(value) is not str or pattern.fullmatch(value) is None:
            raise ValueError(f"Invalid session binding {key}")
    return values


def _validated_binding(row):
    if row is None:
        return None
    binding = dict(row)
    _identity(**{key: binding[key] for key in _IDENTITY_FIELDS})
    if binding["version"] != 1 or binding["tool_policy"] != "deny_all":
        raise ValueError("Unsupported session execution policy")
    return binding


def _binding_on_conn(conn, session_id):
    """Resolve binding ancestry, never using the permissive lease-key fallback.

    Callers hold one SQLite snapshot (or the create write transaction). A binding
    row denotes an independent root, not a grant that may be shadowed by a parent.
    Ordinary parents and explicit forks terminate inheritance.
    """
    seen = set()
    current_id = session_id
    while True:
        if type(current_id) is not str or _ID.fullmatch(current_id) is None:
            raise ValueError("Invalid bound session ancestry ID")
        if current_id in seen:
            raise ValueError("Cyclic session binding ancestry")
        seen.add(current_id)
        session = conn.execute(
            "SELECT parent_session_id, source, model_config FROM sessions WHERE id=?",
            (current_id,),
        ).fetchone()
        binding = _validated_binding(conn.execute(
            "SELECT * FROM session_bindings WHERE session_id=?", (current_id,),
        ).fetchone())
        if session is None:
            if binding is not None or current_id != session_id:
                raise ValueError("Orphaned session binding ancestry")
            return None  # Ordinary agents may not have persisted their first turn yet.
        raw_config = session["model_config"]
        try:
            config = json.loads(raw_config) if raw_config is not None else {}
        except (TypeError, ValueError) as exc:
            raise ValueError("Malformed session binding ancestry config") from exc
        if not isinstance(config, dict):
            raise ValueError("Malformed session binding ancestry config")
        markers = (config.get("_branched_from"), config.get("_delegate_from"))
        for marker in markers:
            if marker is not None and (type(marker) is not str or _ID.fullmatch(marker) is None):
                raise ValueError("Malformed session binding fork marker")
        parent_id = session["parent_session_id"]
        if parent_id is None:
            return binding
        if type(parent_id) is not str or _ID.fullmatch(parent_id) is None:
            raise ValueError("Invalid bound session ancestry parent")
        if parent_id in seen:
            raise ValueError("Cyclic session binding ancestry")
        parent = conn.execute("SELECT end_reason FROM sessions WHERE id=?", (parent_id,)).fetchone()
        if parent is None:
            raise ValueError("Dangling session binding ancestry parent")
        if session["source"] == "tool" or parent_id in markers or parent["end_reason"] != "compression":
            return binding
        if binding is not None:
            raise ValueError("Independent session binding cannot inherit compression ancestry")
        current_id = parent_id


class SessionBoundMixin:
    """Narrow server API; callers must authorize the owner before using it."""

    def create_bound_session(self, *, owner_id, agent_id, case_id, source, context_digest):
        identity = _identity(owner_id, agent_id, case_id, source, context_digest)

        def create(conn):
            row = conn.execute(
                "SELECT * FROM session_bindings WHERE owner_id=? AND agent_id=? AND case_id=?",
                (owner_id, agent_id, case_id),
            ).fetchone()
            if row is not None:
                existing = _binding_on_conn(conn, row["session_id"])
                if any(existing[key] != value for key, value in identity.items()):
                    raise ValueError("Session binding cannot be changed")
                return existing["session_id"]
            session_id = f"bound_{uuid.uuid4().hex}"
            conn.execute(
                "INSERT INTO sessions(id, source, user_id, started_at, pinned) VALUES (?, ?, ?, ?, 1)",
                (session_id, source, owner_id, time.time()),
            )
            conn.execute(
                "INSERT INTO session_bindings(session_id, owner_id, agent_id, case_id, source, "
                "context_digest, version, tool_policy) VALUES (?, ?, ?, ?, ?, ?, 1, 'deny_all')",
                (session_id, owner_id, agent_id, case_id, source, context_digest),
            )
            return session_id

        return self._execute_write(create, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def get_session_binding(self, session_id):
        if type(session_id) is not str or _ID.fullmatch(session_id) is None:
            raise ValueError("Invalid bound session ID")
        with self._read_ctx() as conn:
            # _read_ctx alone is autocommit: pin all ancestry reads to one snapshot.
            # A savepoint also composes with the writer-connection fallback.
            conn.execute("SAVEPOINT bound_binding_read")
            try:
                return _binding_on_conn(conn, session_id)
            finally:
                conn.execute("RELEASE SAVEPOINT bound_binding_read")

    def require_session_binding(self, session_id, *, owner_id, agent_id, case_id, source, context_digest):
        expected = _identity(owner_id, agent_id, case_id, source, context_digest)
        binding = self.get_session_binding(session_id)
        if binding is None or any(binding[key] != value for key, value in expected.items()):
            raise ValueError("Session binding does not match the authorized case")
        return binding
