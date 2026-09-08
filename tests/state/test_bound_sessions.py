"""A governed case gets one immutable native session binding, never a guessed ID."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from hermes_state import SessionDB


def identity(**changes):
    return {"owner_id": "demo-owner", "agent_id": "demo-advisor", "case_id": "demo-case",
            "source": "demo-desk", "context_digest": "a" * 64, **changes}


def test_bound_session_creation_is_atomic_and_survives_restart(tmp_path):
    path = tmp_path / "state.db"
    first, second = SessionDB(path), SessionDB(path)
    barrier = Barrier(2)
    def create(db):
        barrier.wait(timeout=10)
        return db.create_bound_session(**identity())
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = list(pool.map(create, (first, second)))
        assert a == b
        first.append_message(a, "user", "DEMO persistent case message")
    finally:
        first.close()
        second.close()
    restarted = SessionDB(path)
    try:
        assert restarted.create_bound_session(**identity()) == a
        binding = restarted.require_session_binding(a, **identity())
        assert binding["session_id"] == a
        assert binding["tool_policy"] == "deny_all"
        assert restarted.get_messages_as_conversation(a)[0]["content"] == "DEMO persistent case message"
        other = restarted.create_bound_session(**identity(case_id="demo-other-case"))
        assert other != a
        assert restarted.get_messages_as_conversation(other) == []
    finally:
        restarted.close()


@pytest.mark.parametrize("change", [
    {"owner_id": "demo-other-owner"}, {"agent_id": "demo-other-advisor"},
    {"case_id": "demo-other-case"}, {"source": "demo-other-source"},
    {"context_digest": "b" * 64}, {"case_id": " demo-case"}, {"owner_id": 7},
])
def test_binding_rejects_identity_context_mismatch_without_retagging(tmp_path, change):
    db = SessionDB(tmp_path / "state.db")
    try:
        session_id = db.create_bound_session(**identity())
        with pytest.raises(ValueError):
            db.require_session_binding(session_id, **identity(**change))
        # No lookup or mismatch may change the bound context/role.
        assert db.require_session_binding(session_id, **identity())["context_digest"] == "a" * 64
        if set(change) <= {"source", "context_digest"}:
            with pytest.raises(ValueError):
                db.create_bound_session(**identity(**change))
        db.create_session("demo-unbound", "api_server")
        with pytest.raises(ValueError):
            db.require_session_binding("demo-unbound", **identity())
    finally:
        db.close()


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    yield store
    store.close()


@pytest.mark.parametrize("operation", ["read", "require", "reuse"])
@pytest.mark.parametrize("damage", [
    "foreign-root", "intermediate-root", "orphan", "dangling", "bad-parent",
    "empty-parent", "self-cycle", "cycle", "bad-json", "array-config", "bad-marker",
])
def test_corrupt_binding_lineage_is_rejected_without_mutation(db, damage, operation):
    root = db.create_bound_session(**identity())
    target = root
    if damage in {"foreign-root", "intermediate-root"}:
        foreign = db.create_bound_session(**identity(owner_id="foreign"))
        db.end_session(foreign, "compression")
        db._execute_write(lambda c: c.execute(
            "UPDATE sessions SET parent_session_id=? WHERE id=?", (foreign, root)))
        if damage == "intermediate-root":
            db.end_session(root, "compression")
            db.create_session("tip", "demo-desk", parent_session_id=root)
            target = "tip"
    elif damage == "orphan":
        db._conn.execute("PRAGMA foreign_keys=OFF")
        try:
            db._execute_write(lambda c: c.execute("DELETE FROM sessions WHERE id=?", (root,)))
        finally:
            db._conn.execute("PRAGMA foreign_keys=ON")
    elif damage in {"dangling", "bad-parent", "empty-parent", "self-cycle", "cycle"}:
        parent = {"dangling": "missing", "bad-parent": " invalid", "empty-parent": "",
                  "self-cycle": root, "cycle": "middle"}[damage]
        if damage == "cycle":
            db.create_session("middle", "demo-desk", parent_session_id=root)
            db.end_session("middle", "compression")
        db.end_session(root, "compression")
        db._conn.execute("PRAGMA foreign_keys=OFF")
        try:
            db._execute_write(lambda c: c.execute(
                "UPDATE sessions SET parent_session_id=? WHERE id=?", (parent, root)))
        finally:
            db._conn.execute("PRAGMA foreign_keys=ON")
    else:
        config = {"bad-json": "{", "array-config": "[]", "bad-marker": '{"_branched_from":7}'}[damage]
        db._execute_write(lambda c: c.execute(
            "UPDATE sessions SET model_config=? WHERE id=?", (config, root)))
    with db._read_ctx() as c:
        before = [tuple(r) for r in c.execute("SELECT * FROM session_bindings ORDER BY session_id")]
    with pytest.raises(ValueError):
        if operation == "reuse":
            db.create_bound_session(**identity())
        elif operation == "require":
            db.require_session_binding(target, **identity())
        else:
            db.get_session_binding(target)
    with db._read_ctx() as c:
        assert [tuple(r) for r in c.execute("SELECT * FROM session_bindings ORDER BY session_id")] == before
    if damage == "orphan":
        assert db.get_session(root) is None


def test_compression_inherits_but_explicit_forks_and_unbound_sessions_do_not(db):
    root = db.create_bound_session(**identity())
    assert db.get_session(root)["pinned"] == 1
    assert db.get_session_binding("not-yet-created") is None
    db.create_session("ordinary", "demo-desk", parent_session_id=root)
    assert db.get_session_binding("ordinary") is None
    assert db.get_session("ordinary")["pinned"] == 0
    db.end_session(root, "compression")
    db.create_session("middle", "demo-desk", parent_session_id=root)
    assert db.get_session("middle")["pinned"] == 1
    db.end_session("middle", "compression")
    db.create_session("tip", "demo-desk", parent_session_id="middle")
    assert db.get_session("tip")["pinned"] == 1
    assert db.require_session_binding("tip", **identity())["session_id"] == root
    assert db.create_bound_session(**identity()) == root
    for marker in ("_branched_from", "_delegate_from"):
        child = "fork" + marker
        db.create_session(child, "demo-desk", parent_session_id=root, model_config={marker: root})
        assert db.get_session_binding(child) is None
        assert db.get_session(child)["pinned"] == 0
        db.end_session(child, "compression")
        # Compression copies the old marker: it does not make this new edge a fork.
        db.create_session(child + "-tip", "demo-desk", parent_session_id=child, model_config={marker: root})
        assert db.get_session_binding(child + "-tip") is None
    db.create_session("tool-child", "tool", parent_session_id=root)
    assert db.get_session_binding("tool-child") is None
    assert db.get_session("tool-child")["pinned"] == 0


def test_default_prune_preserves_complete_bound_compression_lineage(db):
    root = db.create_bound_session(**identity())
    assert db.try_acquire_compression_lock(root, "root-rotator")
    db.publish_compression_child(
        parent_session_id=root,
        child_session_id="middle",
        source="demo-desk",
        messages=[{"role": "system", "content": "root summary"}],
        compression_lock_holder="root-rotator",
    )
    assert db.try_acquire_compression_lock("middle", "middle-rotator")
    db.publish_compression_child(
        parent_session_id="middle",
        child_session_id="tip",
        source="demo-desk",
        messages=[{"role": "system", "content": "middle summary"}],
        compression_lock_holder="middle-rotator",
    )
    db.end_session("tip", "complete")
    old = time.time() - (91 * 86400)
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET ended_at = ? WHERE id IN (?, ?, ?)",
        (old, root, "middle", "tip"),
    ))

    assert db.get_session("middle")["pinned"] == 1
    assert db.get_session("tip")["pinned"] == 1
    assert db.prune_sessions() == 0
    assert all(db.get_session(session_id) is not None for session_id in (root, "middle", "tip"))
    assert db.require_session_binding("tip", **identity())["session_id"] == root
    assert db.create_bound_session(**identity()) == root


def test_binding_insert_failure_rolls_back_session_and_allows_retry(db):
    db._execute_write(lambda c: c.execute(
        "CREATE TRIGGER fail_binding BEFORE INSERT ON session_bindings "
        "BEGIN SELECT RAISE(ABORT, 'binding fault'); END"))
    with pytest.raises(sqlite3.IntegrityError, match="binding fault"):
        db.create_bound_session(**identity())
    with db._read_ctx() as c:
        assert c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM session_bindings").fetchone()[0] == 0
    db._execute_write(lambda c: c.execute("DROP TRIGGER fail_binding"))
    sid = db.create_bound_session(**identity())
    assert db.require_session_binding(sid, **identity())["session_id"] == sid


@pytest.mark.parametrize("force_writer_fallback", [False, True])
def test_binding_read_uses_one_snapshot_during_competing_update(db, monkeypatch, force_writer_fallback):
    import hermes_state_bound_sessions as bindings

    root = db.create_bound_session(**identity())
    db.end_session(root, "compression")
    db.create_session("snapshot-tip", "demo-desk", parent_session_id=root)
    if force_writer_fallback:
        monkeypatch.setattr(db, "_checkout_read_conn", lambda: None)
    original = bindings._validated_binding
    changed = False
    started, finished = Event(), Event()
    errors = []

    def update():
        started.set()
        try:
            db._execute_write(lambda c: c.execute(
                "UPDATE session_bindings SET context_digest=? WHERE session_id=?", ("b" * 64, root)))
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    writer = Thread(target=update, daemon=True)

    def validate(row):
        nonlocal changed
        if not changed:
            changed = True
            writer.start()
            assert started.wait(10), "competing writer did not start"
            if db._wal_active and not force_writer_fallback:
                # WAL must retain the old snapshot even after a concurrent commit.
                assert finished.wait(10), "WAL writer did not commit during the read"
                assert not errors
            else:
                # DELETE/pool fallback serializes reads with the non-reentrant
                # writer lock. Never wait for that writer while holding its lock.
                assert db._lock.locked()
        return original(row)

    monkeypatch.setattr(bindings, "_validated_binding", validate)
    try:
        assert db.get_session_binding("snapshot-tip")["context_digest"] == "a" * 64
    finally:
        if writer.ident is not None:
            writer.join(timeout=10)
    assert not writer.is_alive(), "competing writer did not finish after the read"
    assert finished.is_set() and not errors
    assert db.get_session_binding("snapshot-tip")["context_digest"] == "b" * 64


@pytest.mark.parametrize("different", [False, True])
def test_process_contenders_observe_one_immutable_binding(tmp_path, different):
    path = tmp_path / "state.db"
    SessionDB(path).close()
    gate = tmp_path / "go"
    worker = '''import json,sys,time
from pathlib import Path
from hermes_state import SessionDB
path,gate,ready,digest=sys.argv[1:]
db=SessionDB(Path(path))
Path(ready).touch()
deadline=time.monotonic()+15
while not Path(gate).exists():
    if time.monotonic()>deadline: raise RuntimeError('gate timeout')
    time.sleep(.01)
try:
    sid=db.create_bound_session(owner_id='demo-owner',agent_id='demo-advisor',case_id='demo-case',source='demo-desk',context_digest=digest)
    print(json.dumps({'ok':sid,'digest':digest}))
except ValueError as e:
    print(json.dumps({'error':str(e),'digest':digest}))
finally: db.close()
'''
    procs = []
    try:
        for i, digest in enumerate(["a" * 64, ("b" if different else "a") * 64]):
            procs.append(subprocess.Popen(
                [sys.executable, "-c", worker, str(path), str(gate), str(tmp_path / f"ready{i}"), digest],
                cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        deadline = time.monotonic() + 20
        while not all((tmp_path / f"ready{i}").exists() for i in range(2)):
            for p in procs:
                if p.poll() is not None:
                    out, err = p.communicate(timeout=2)
                    pytest.fail("worker exited before gate: " + out + err)
            assert time.monotonic() < deadline, "worker readiness timeout"
            time.sleep(.01)
        gate.touch()
        results = []
        for p in procs:
            out, err = p.communicate(timeout=25)
            assert p.returncode == 0, err
            results.append(json.loads(out))
        winners = [r for r in results if "ok" in r]
        losers = [r for r in results if "error" in r]
        assert len(winners) == (1 if different else 2)
        assert len(losers) == (1 if different else 0)
        assert all("cannot be changed" in r["error"] for r in losers)
        assert len({r["ok"] for r in winners}) == 1
        restarted = SessionDB(path)
        try:
            sid = winners[0]["ok"]
            assert restarted.get_session(sid) is not None
            assert restarted.require_session_binding(
                sid, **identity(context_digest=winners[0]["digest"]))["session_id"] == sid
            with restarted._read_ctx() as c:
                assert c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
                assert c.execute("SELECT COUNT(*) FROM session_bindings").fetchone()[0] == 1
        finally:
            restarted.close()
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.communicate(timeout=5)
