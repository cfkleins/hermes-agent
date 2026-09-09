"""Native-session admission must load durable history under its actual lease."""
from types import SimpleNamespace

import pytest

from agent.turn_facade_lease import DurableTurnLease, admit_durable_turn_lease
from hermes_state import SessionDB


def agent_for(db, session_id="demo-native"):
    return SimpleNamespace(
        _session_db=db, session_id=session_id, require_durable_history=True,
        _persist_disabled=False, _interrupt_requested=False,
        _emit_status=lambda *_: None,
    )


def admit(agent):
    return admit_durable_turn_lease(
        agent, session_id=agent.session_id, relay_turn_id="demo-turn",
        task_context={"platform": "api_server", "session_id": agent.session_id},
        conversation_history=[{"role": "user", "content": "DEMO stale caller seed"}],
    )


def test_immediate_admission_reloads_after_other_writer_commits(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    reader, writer = SessionDB(path), SessionDB(path)
    try:
        reader.create_session("demo-native", "api_server")
        real_acquire = reader.acquire_session_turn_lease
        monkeypatch.setattr(DurableTurnLease, "build_threads", lambda _: None)

        def acquire_after_commit(*args, **kwargs):
            # The caller already loaded its stale seed. A writer completes
            # before acquisition, so on_wait is never invoked.
            writer.append_message("demo-native", "user", "DEMO durable newer turn")
            return real_acquire(*args, **kwargs)

        monkeypatch.setattr(reader, "acquire_session_turn_lease", acquire_after_commit)
        result = admit(agent_for(reader))
        try:
            assert [m["content"] for m in result.conversation_history] == ["DEMO durable newer turn"]
            assert result.lease is not None
        finally:
            if result.lease:
                result.lease.release()
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize("failure", ["missing-db", "missing-session", "disabled-persistence", "lineage-read", "history-read", "vanished-tip"])
def test_native_admission_fails_closed_and_releases_lease(tmp_path, monkeypatch, failure):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("demo-native", "api_server")
        agent = agent_for(db)
        if failure == "missing-db":
            agent._session_db = None
        elif failure == "missing-session":
            agent.session_id = "demo-absent"
        elif failure == "disabled-persistence":
            agent._persist_disabled = True
        elif failure in ("lineage-read", "history-read"):
            def broken(*_, **__):
                raise RuntimeError("DEMO read failure")
            monkeypatch.setattr(db, "get_compression_tip" if failure == "lineage-read" else "get_messages_as_conversation", broken)
        else:
            monkeypatch.setattr(db, "resolve_resume_session_id", lambda *_, **__: "demo-absent")
        monkeypatch.setattr(DurableTurnLease, "build_threads", lambda _: None)
        with pytest.raises((ValueError, RuntimeError)):
            admit(agent)
        # A refused turn must not strand the row lease.
        assert db.acquire_session_turn_lease("demo-native", "demo-next-owner", wait_seconds=0)
        db.release_session_turn_lease("demo-native", "demo-next-owner")
    finally:
        db.close()


def test_native_resume_rejects_history_outside_held_lease(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("demo-native", "api_server")
        db.create_session("demo-child", "api_server", parent_session_id="demo-native")
        db.append_message("demo-child", "user", "DEMO foreign lease history")
        agent = agent_for(db)
        reads = []
        real_read = db.get_messages_as_conversation
        def read(*args, **kwargs):
            reads.append(args[0])
            return real_read(*args, **kwargs)
        monkeypatch.setattr(db, "get_messages_as_conversation", read)
        monkeypatch.setattr(DurableTurnLease, "build_threads", lambda _: None)
        with pytest.raises(ValueError, match="lease"):
            admit(agent)
        assert reads == []
        assert agent.session_id == "demo-native"
        assert db.try_acquire_session_turn_lease("demo-native", "demo-next", patience_s=0)
        db.release_session_turn_lease("demo-native", "demo-next")


def test_native_compression_tip_is_covered_by_same_lease(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("demo-native", "api_server")
        db.end_session("demo-native", "compression")
        db.create_session("demo-tip", "api_server", parent_session_id="demo-native")
        db.append_message("demo-tip", "user", "DEMO compression history")
        monkeypatch.setattr(DurableTurnLease, "build_threads", lambda _: None)
        agent = agent_for(db)
        result = admit(agent)
        try:
            assert agent.session_id == "demo-tip"
            assert result.conversation_history[0]["content"] == "DEMO compression history"
            assert not db.try_acquire_session_turn_lease("demo-tip", "demo-competitor", patience_s=0)
        finally:
            result.lease.release()


def test_native_rejects_moved_acquired_root_and_releases_original_claim(tmp_path, monkeypatch):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("demo-parent", "api_server")
        db.create_session("demo-child", "api_server", parent_session_id="demo-parent")
        db.append_message("demo-child", "user", "DEMO moved-domain history")
        acquire = db.acquire_session_turn_lease
        def move_after_acquisition(*args, **kwargs):
            result = acquire(*args, **kwargs)
            db.end_session("demo-parent", "compression")
            return result
        monkeypatch.setattr(db, "acquire_session_turn_lease", move_after_acquisition)
        monkeypatch.setattr(db, "get_messages_as_conversation", lambda *a, **kw: pytest.fail("history read after root moved"))
        monkeypatch.setattr(DurableTurnLease, "build_threads", lambda _: None)
        with pytest.raises(ValueError, match="lease"):
            admit(agent_for(db, "demo-child"))
        # end_session preserves the first end_reason; inspect the original claim
        # directly instead of accidentally checking only the newly resolved parent.
        assert db._read_one(
            "SELECT holder FROM session_turn_leases WHERE conversation_id = ?", ("demo-child",)
        ) is None
        assert db.try_acquire_session_turn_lease("demo-child", "demo-successor", patience_s=0)
        db.release_session_turn_lease("demo-child", "demo-successor")


@pytest.mark.parametrize("pool_fallback", [False, True])
def test_native_ownership_check_and_history_read_serialize_metadata_writes(tmp_path, monkeypatch, pool_fallback):
    import sqlite3
    path = tmp_path / "state.db"
    with SessionDB(path) as db, SessionDB(path) as peer:
        db.create_session("demo-parent", "api_server")
        db.create_session("demo-child", "api_server", parent_session_id="demo-parent")
        db.append_message("demo-child", "user", "DEMO serialized history")
        if pool_fallback:
            monkeypatch.setattr(db, "_checkout_read_conn", lambda: None)
        read = db.get_messages_as_conversation
        attempts = []
        def interleaved_read(*args, **kwargs):
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                peer._execute_write(lambda conn: conn.execute(
                    "UPDATE sessions SET end_reason='compression' WHERE id='demo-parent'"
                ), patience_s=0)
            attempts.append("blocked")
            return read(*args, **kwargs)
        monkeypatch.setattr(db, "get_messages_as_conversation", interleaved_read)
        monkeypatch.setattr(DurableTurnLease, "build_threads", lambda _: None)
        result = admit(agent_for(db, "demo-child"))
        try:
            assert attempts == ["blocked"]
            assert result.conversation_history[0]["content"] == "DEMO serialized history"
            assert not peer.try_acquire_session_turn_lease("demo-child", "demo-peer", patience_s=0)
        finally:
            result.lease.release()
