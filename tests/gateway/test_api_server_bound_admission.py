"""Authenticated, server-owned admission into one native bound session."""
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import socket
from threading import Barrier
from types import MappingProxyType, SimpleNamespace

import pytest
import yaml

from hermes_state import SessionDB


GATEWAY_KEY = "synthetic-gateway-key-never-a-real-credential"
MODEL_KEY = "synthetic-model-key-never-a-real-credential"
PRINCIPAL = "fixture-principal"
SELECTOR = "case_opaque_7d9"
CONTEXT = b"Approved fixture context.\nInterview without tools."
DIGEST = sha256(CONTEXT).hexdigest()
MODEL = "openai/gpt-6-astra"
BASE_URL = "https://bounded-fixture.invalid/v1"
CONTEXT_LENGTH = 1_050_000
IDENTITY = {
    "owner_id": "fixture-owner",
    "agent_id": "fixture-advisor",
    "case_id": "fixture-case",
    "source": "fixture-vcc",
    "context_digest": DIGEST,
}
REQUIRED_SKILLS = ("core-interview", "project-issue-interview")


def admission_module():
    name = "gateway.platforms.api_server_bound_admission"
    assert importlib.util.find_spec(name) is not None, "bounded admission helper is missing"
    return importlib.import_module(name)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        folder = tmp_path / name.lower()
        folder.mkdir()
        monkeypatch.setenv(name, str(folder))
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": []}}), encoding="utf-8")
    attempts = []

    def forbidden(*_args, **_kwargs):
        attempts.append("network")
        pytest.fail("Unexpected network access in bounded admission")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    yield SimpleNamespace(home=home, attempts=attempts)
    assert attempts == []


def make_case(module, **changes):
    values = {
        "principal_id": PRINCIPAL,
        **IDENTITY,
        "context_bytes": CONTEXT,
        "skills": REQUIRED_SKILLS,
        "tool_policy": "deny_all",
        "provider": "nous",
        "model": MODEL,
        "api_mode": "chat_completions",
        "base_url": BASE_URL,
        "api_key": MODEL_KEY,
        "context_length": CONTEXT_LENGTH,
    }
    values.update(changes)
    return module.BoundCase(**values)


def make_registry(module, *, gateway_principals=None, case=None, cases=None):
    return module.BoundAdmissionRegistry(
        gateway_principals=gateway_principals or {GATEWAY_KEY: PRINCIPAL},
        cases=cases or {(PRINCIPAL, SELECTOR): case or make_case(module)},
    )


@pytest.mark.parametrize("field", ["gateway_principals", "cases"])
def test_registry_rejects_duplicate_iterable_authority(field):
    module = admission_module()
    values = {
        "gateway_principals": {GATEWAY_KEY: PRINCIPAL},
        "cases": {(PRINCIPAL, SELECTOR): make_case(module)},
    }
    if field == "gateway_principals":
        values[field] = [(GATEWAY_KEY, PRINCIPAL), (GATEWAY_KEY, "other-principal")]
    else:
        values[field] = [
            ((PRINCIPAL, SELECTOR), make_case(module)),
            ((PRINCIPAL, SELECTOR), make_case(module, case_id="other-case")),
        ]
    with pytest.raises(ValueError, match="Invalid bounded admission registry"):
        module.BoundAdmissionRegistry(**values)


def test_registry_rejects_hostile_mapping_with_duplicate_items():
    module = admission_module()

    class DuplicateItemsMapping(Mapping):
        def __getitem__(self, key):
            if key != GATEWAY_KEY:
                raise KeyError(key)
            return PRINCIPAL

        def __iter__(self):
            return iter((GATEWAY_KEY,))

        def __len__(self):
            return 2

        def items(self):
            return ((GATEWAY_KEY, PRINCIPAL), (GATEWAY_KEY, "other-principal"))

    with pytest.raises(ValueError, match="Invalid bounded admission registry"):
        module.BoundAdmissionRegistry(
            gateway_principals=DuplicateItemsMapping(),
            cases={(PRINCIPAL, SELECTOR): make_case(module)},
        )


def test_registry_sanitizes_mapping_failures_without_exception_chain():
    module = admission_module()

    class ExplodingMapping(Mapping):
        def __getitem__(self, _key):
            raise KeyError

        def __iter__(self):
            return iter(())

        def __len__(self):
            return 1

        def items(self):
            raise RuntimeError(GATEWAY_KEY)

    with pytest.raises(ValueError, match="Invalid bounded admission registry") as caught:
        module.BoundAdmissionRegistry(
            gateway_principals=ExplodingMapping(),
            cases={(PRINCIPAL, SELECTOR): make_case(module)},
        )
    assert caught.value.__cause__ is None
    assert GATEWAY_KEY not in str(caught.value)
    assert GATEWAY_KEY not in repr(caught.value)


def body(*, text="Please begin the approved interview.", selector=SELECTOR):
    return json.dumps({"input": text, "case": selector}, separators=(",", ":")).encode()


def admit(module, db, registry, **changes):
    values = {
        "raw_body": body(),
        "authorization": f"Bearer {GATEWAY_KEY}",
        "idempotency_key": "run_fixture_001",
        "registry": registry,
        "session_db": db,
    }
    values.update(changes)
    return module.admit_bound_run(**values)


class NoTouchDB:
    def __getattr__(self, _name):
        pytest.fail("Invalid admission reached native state")


@pytest.mark.parametrize("authorization", [
    None, "", 7, "bearer " + GATEWAY_KEY, "Bearer", "Bearer  " + GATEWAY_KEY,
    "Bearer " + GATEWAY_KEY + " ", "Bearer " + GATEWAY_KEY + ",Bearer other",
    "Bearer wrong-key", ["Bearer " + GATEWAY_KEY, "Bearer other"],
])
def test_authenticates_raw_bytes_before_parse_or_any_state(isolated, authorization):
    module = admission_module()
    registry = make_registry(module)
    with pytest.raises(module.BoundAdmissionError) as caught:
        admit(module, NoTouchDB(), registry, raw_body=b"{not-json", authorization=authorization)
    assert caught.value.code == "unauthorized"
    assert "not-json" not in str(caught.value)
    assert GATEWAY_KEY not in repr(caught.value)


@pytest.mark.parametrize("raw", [
    "not-bytes", bytearray(b"{}"), b"\xff", b"", b"[]", b"null", b"{",
    b'{"input":"x","case":"case_opaque_7d9","input":"y"}',
    b'{"input":NaN,"case":"case_opaque_7d9"}',
    b'{"input":Infinity,"case":"case_opaque_7d9"}',
    b'{"input":"x","case":"case_opaque_7d9","model":"other"}',
    b'{"input":"x"}',
])
def test_exact_json_contract_fails_before_runtime_or_state(isolated, monkeypatch, raw):
    module = admission_module()
    monkeypatch.setattr(module, "resolve_bound_runtime", lambda **_kw: pytest.fail("runtime touched"))
    with pytest.raises(module.BoundAdmissionError) as caught:
        admit(module, NoTouchDB(), make_registry(module), raw_body=raw)
    assert caught.value.code == "invalid_request"


@pytest.mark.parametrize("payload", [
    {"input": "", "case": SELECTOR},
    {"input": 7, "case": SELECTOR},
    {"input": True, "case": SELECTOR},
    {"input": "bad\x00text", "case": SELECTOR},
    {"input": "bad\u200btext", "case": SELECTOR},
    {"input": "x" * 16001, "case": SELECTOR},
    {"input": "ok", "case": ""},
    {"input": "ok", "case": 7},
    {"input": "ok", "case": "bad selector"},
    {"input": "ok", "case": "x" * 129},
])
def test_body_types_caps_and_controls_fail_before_runtime_or_state(isolated, monkeypatch, payload):
    module = admission_module()
    monkeypatch.setattr(module, "resolve_bound_runtime", lambda **_kw: pytest.fail("runtime touched"))
    with pytest.raises(module.BoundAdmissionError) as caught:
        admit(module, NoTouchDB(), make_registry(module), raw_body=json.dumps(payload).encode())
    assert caught.value.code == "invalid_request"


def test_raw_body_cap_fails_before_runtime_or_state(isolated, monkeypatch):
    module = admission_module()
    monkeypatch.setattr(module, "resolve_bound_runtime", lambda **_kw: pytest.fail("runtime touched"))
    raw = b'{"input":"' + b"x" * 65536 + b'","case":"' + SELECTOR.encode() + b'"}'
    with pytest.raises(module.BoundAdmissionError) as caught:
        admit(module, NoTouchDB(), make_registry(module), raw_body=raw)
    assert caught.value.code == "invalid_request"


@pytest.mark.parametrize("key", [None, "", 7, " bad", "bad key", "bad\nkey", "x" * 129])
def test_idempotency_key_is_required_before_runtime_or_state(isolated, monkeypatch, key):
    module = admission_module()
    monkeypatch.setattr(module, "resolve_bound_runtime", lambda **_kw: pytest.fail("runtime touched"))
    with pytest.raises(module.BoundAdmissionError) as caught:
        admit(module, NoTouchDB(), make_registry(module), idempotency_key=key)
    assert caught.value.code == "invalid_idempotency_key"


def test_foreign_principal_and_case_fail_before_runtime_or_state(isolated, monkeypatch):
    module = admission_module()
    monkeypatch.setattr(module, "resolve_bound_runtime", lambda **_kw: pytest.fail("runtime touched"))
    registry = make_registry(module, gateway_principals={GATEWAY_KEY: "foreign-principal"})
    with pytest.raises(module.BoundAdmissionError) as caught:
        admit(module, NoTouchDB(), registry)
    assert caught.value.code == "not_found"


def test_context_bytes_are_digest_verified_before_runtime_or_state(isolated, monkeypatch):
    module = admission_module()
    monkeypatch.setattr(module, "resolve_bound_runtime", lambda **_kw: pytest.fail("runtime touched"))
    changed = make_case(module, context_bytes=CONTEXT + b" modified")
    with pytest.raises(module.BoundAdmissionError) as caught:
        admit(module, NoTouchDB(), make_registry(module, case=changed))
    assert caught.value.code == "invalid_server_policy"


def test_runtime_error_is_sanitized_and_precedes_state(isolated, monkeypatch):
    module = admission_module()

    def fail(**_kwargs):
        raise RuntimeError(MODEL_KEY + CONTEXT.decode())

    monkeypatch.setattr(module, "resolve_bound_runtime", fail)
    with pytest.raises(module.BoundAdmissionError) as caught:
        admit(module, NoTouchDB(), make_registry(module))
    rendered = repr(caught.value)
    assert caught.value.code == "runtime_unavailable"
    assert MODEL_KEY not in rendered and CONTEXT.decode() not in rendered


def test_real_lane_creates_requires_and_reuses_native_binding(isolated):
    module = admission_module()
    registry = make_registry(module)
    path = isolated.home / "native-state.db"
    with SessionDB(path) as db:
        first = admit(module, db, registry)
        second = admit(module, db, registry)
        assert first.session_id == second.session_id
        binding = db.require_session_binding(first.session_id, **IDENTITY)
        assert dict(first.binding_identity) == {name: binding[name] for name in IDENTITY}
        assert first.principal_id == PRINCIPAL
        assert first.input_text == "Please begin the approved interview."
        assert first.idempotency_key == "run_fixture_001"
        assert first.context_bytes == CONTEXT
        assert first.skills == REQUIRED_SKILLS
        assert first.tool_policy == "deny_all"
        assert (first.runtime.provider, first.runtime.model, first.runtime.api_mode, first.runtime.base_url) == (
            "nous", MODEL, "chat_completions", BASE_URL)
        assert first.runtime.api_key == MODEL_KEY
        assert db.get_messages_as_conversation(first.session_id) == []
        with pytest.raises(FrozenInstanceError):
            first.session_id = "other"
        with pytest.raises(TypeError):
            first.binding_identity["owner_id"] = "other"
        assert isinstance(first.binding_identity, MappingProxyType)
        rendered = repr((registry, first))
        assert GATEWAY_KEY not in rendered and MODEL_KEY not in rendered and CONTEXT.decode() not in rendered


@pytest.mark.parametrize("change", [
    {"context_digest": "f" * 64}, {"source": "fixture-other-source"},
])
def test_existing_native_binding_conflict_is_sanitized(isolated, change):
    module = admission_module()
    registry = make_registry(module)
    with SessionDB(isolated.home / "native-state.db") as db:
        db.create_bound_session(**{**IDENTITY, **change})
        with pytest.raises(module.BoundAdmissionError) as caught:
            admit(module, db, registry)
        assert caught.value.code == "binding_conflict"
        assert DIGEST not in repr(caught.value)
        with db._read_ctx() as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM session_bindings").fetchone()[0] == 1


def test_registry_owns_immutable_exact_policy_and_rejects_client_authority(isolated):
    module = admission_module()
    auth = {GATEWAY_KEY: PRINCIPAL}
    cases = {(PRINCIPAL, SELECTOR): make_case(module)}
    registry = make_registry(module, gateway_principals=auth, cases=cases)
    auth[GATEWAY_KEY] = "foreign"
    cases.clear()
    assert registry.gateway_principals[GATEWAY_KEY] == PRINCIPAL
    assert registry.cases[(PRINCIPAL, SELECTOR)].skills == REQUIRED_SKILLS
    with pytest.raises(TypeError):
        registry.cases[(PRINCIPAL, SELECTOR)] = make_case(module)
    for forbidden in ("session_id", "history", "context", "provider", "model", "instructions", "tools", "skills"):
        raw = json.dumps({"input": "ok", "case": SELECTOR, forbidden: "override"}).encode()
        with pytest.raises(module.BoundAdmissionError) as caught:
            admit(module, NoTouchDB(), registry, raw_body=raw)
        assert caught.value.code == "invalid_request"


def test_two_native_db_threads_converge_on_one_bound_session(isolated):
    module = admission_module()
    registry = make_registry(module)
    path = isolated.home / "native-state.db"
    first, second = SessionDB(path), SessionDB(path)
    barrier = Barrier(2)

    def run(db):
        barrier.wait(timeout=10)
        return admit(module, db, registry).session_id

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            session_ids = list(pool.map(run, (first, second)))
        assert session_ids[0] == session_ids[1]
        assert first.require_session_binding(session_ids[0], **IDENTITY)["session_id"] == session_ids[0]
        with first._read_ctx() as conn:
            assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
            assert conn.execute("SELECT COUNT(*) FROM session_bindings").fetchone()[0] == 1
    finally:
        first.close()
        second.close()
