"""Offline synthetic grants; real native DPAPI and NTFS permission checks."""
import base64
import importlib
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

pytestmark = pytest.mark.windows_only


def jwt(subject="synthetic-owner", ttl=600, padding="x" * 600):
    enc = lambda x: base64.urlsafe_b64encode(json.dumps(x).encode()).decode().rstrip("=")
    return enc({"alg": "RS256"}) + "." + enc(dict(sub=subject, exp=time.time()+ttl,
        scope="inference:invoke", padding=padding)) + ".synthetic-signature"


def payload(**kw):
    return dict(access_token=jwt(), refresh_token="synthetic-refresh", token_type="Bearer",
                scope="inference:invoke", **kw)


@pytest.fixture
def fx(tmp_path, monkeypatch):
    import win32api
    import win32con
    import win32security
    def forbidden(*a, **k):
        raise AssertionError("network forbidden")
    for obj, name in [(socket.socket, "connect"), (socket, "create_connection"), (socket, "getaddrinfo")]:
        monkeypatch.setattr(obj, name, forbidden)
    m = importlib.import_module("hermes_cli.bounded_nous_credentials")
    directory = tmp_path / "private"
    directory.mkdir()
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    finally:
        token.Close()
    acl = win32security.ACL()
    acl.AddAccessAllowedAceEx(win32security.ACL_REVISION,
        win32con.OBJECT_INHERIT_ACE | win32con.CONTAINER_INHERIT_ACE, win32con.GENERIC_ALL, user)
    win32security.SetNamedSecurityInfo(str(directory), win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        None, None, acl, None)
    source = directory / "independent.dpapi"
    return m, dict(source_path=source, marker_path=source.with_name(source.name + ".uncertain"))


def enroll(m, kw, ttl=600):
    data = payload()
    data["access_token"] = jwt(ttl=ttl)
    m.enroll_nous_credentials(source_path=kw["source_path"], token_payload=data)
    return data


def test_lock_marker_checked_before_decrypt(fx, monkeypatch):
    from contextlib import contextmanager
    m, kw = fx
    enroll(m, kw)
    original = m._lease
    @contextmanager
    def lease(source):
        with original(source):
            kw["marker_path"].write_bytes(b"uncertain")
            yield
    monkeypatch.setattr(m, "_lease", lease)
    monkeypatch.setattr(m, "_read", lambda p: pytest.fail("decrypted blocked grant"))
    with pytest.raises(m.CredentialBlocked, match="uncertain"):
        m.isolated_nous_credentials(**kw)


def test_partial_committed_pair_is_not_restored(fx, monkeypatch):
    m, kw = fx
    enroll(m, kw, ttl=100)
    rotated = payload(); rotated["refresh_token"] = "synthetic-rotated"
    transport(m, monkeypatch, lambda request: httpx.Response(200, json=rotated))
    original = m._persist
    def fail_after_commit(path, value):
        original(path, value)
        raise OSError("synthetic-secret-error")
    monkeypatch.setattr(m, "_persist", fail_after_commit)
    with pytest.raises(m.CredentialBlocked):
        m.isolated_nous_credentials(**kw)
    assert m._read(kw["source_path"])["state"]["refresh_token"] == "synthetic-rotated"
    assert kw["marker_path"].exists()


def test_no_other_file_reads_and_shorter_server_cutoff(fx, monkeypatch):
    from pathlib import Path
    m, kw = fx
    data = payload(expires_in=200)
    m.enroll_nous_credentials(source_path=kw["source_path"], token_payload=data)
    original = Path.open
    def local_only(path, *args, **kwargs):
        assert path.parent == kw["source_path"].parent, "foreign store read"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", local_only)
    current = m.isolated_nous_credentials(**kw)
    assert current.expires_at < time.time()+201
    clock = time.time()
    monkeypatch.setattr(m.time, "time", lambda: clock+30)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload(expires_in=400))
    transport(m, monkeypatch, handler)
    assert m.isolated_nous_credentials(**kw).expires_at > current.expires_at
    assert len(calls) == 1


def test_interface_is_isolated(fx):
    m, kw = fx
    assert callable(getattr(m, "isolated_nous_credentials", None))
    assert callable(getattr(m, "enroll_nous_credentials", None))
    assert not hasattr(m, "coordinated_nous_credentials")


@pytest.mark.parametrize("value", ["", "x"*8193, "a b", "a,b", "a${b}", "a\n", "a\t", "é", " x"])
def test_provider_key_rejected(fx, value):
    m, _ = fx
    with pytest.raises(m.CredentialBlocked):
        m.validate_provider_key(value)


def test_native_enrollment_cached_ciphertext_and_exclusive(fx):
    m, kw = fx
    data = enroll(m, kw)
    source = kw["source_path"]
    before = source.read_bytes()
    assert data["access_token"].encode() not in before
    assert b"synthetic-refresh" not in before
    assert m.validate_provider_key("x"*8192) == "x"*8192
    assert len(data["access_token"]) > 512
    result = m.isolated_nous_credentials(**kw)
    assert result.access_token == data["access_token"]
    assert result.access_token not in repr(result)
    assert not hasattr(result, "refresh_token")
    assert source.read_bytes() == before
    with pytest.raises(m.CredentialBlocked):
        enroll(m, kw)
    assert source.read_bytes() == before


@pytest.mark.parametrize("change", [dict(token_type="Basic"), dict(scope="profile"),
    dict(client_id="other"), dict(portal_base_url="https://invalid.test"),
    dict(expires_in=float("nan")), dict(expires_at=1), dict(refresh_token=""), dict(agent_key="legacy")])
def test_invalid_enrollment_never_creates_store(fx, change):
    m, kw = fx
    data = payload(); data.update(change)
    with pytest.raises(m.CredentialBlocked):
        m.enroll_nous_credentials(source_path=kw["source_path"], token_payload=data)
    assert not kw["source_path"].exists()


def transport(m, monkeypatch, handler):
    def factory(**kwargs):
        assert kwargs["verify"] is True and kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        tr = kwargs.pop("transport")
        assert tr._pool._retries == 0
        tr.close()
        return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(m, "_client_factory", factory)


@pytest.mark.parametrize("bad", [None, "subject", "scope", "route", "ttl", "missing", "network", "redirect", "persist"])
def test_rotation_single_post_persist_before_validation(fx, monkeypatch, bad):
    m, kw = fx
    enroll(m, kw, ttl=100)
    calls = []
    rotated = payload(); rotated.update(access_token=jwt(subject="other" if bad == "subject" else "synthetic-owner"),
        refresh_token="synthetic-rotated", expires_in=400)
    if bad == "scope": rotated["scope"] = "profile"
    if bad == "route": rotated["client_id"] = "other"
    if bad == "ttl": rotated["expires_in"] = 10
    if bad == "missing": rotated.pop("refresh_token")
    def handler(request):
        calls.append(request)
        assert kw["marker_path"].exists()
        assert str(request.url) == "https://portal.nousresearch.com/api/oauth/token"
        assert request.method == "POST"
        assert request.headers["x-nous-refresh-token"] == "synthetic-refresh"
        assert request.content == b"grant_type=refresh_token&client_id=hermes-cli"
        assert "authorization" not in request.headers
        if bad == "network": raise RuntimeError("synthetic-secret-error")
        if bad == "redirect": return httpx.Response(307, headers={"location": "https://invalid.test"})
        return httpx.Response(200, json=rotated)
    transport(m, monkeypatch, handler)
    if bad == "persist":
        monkeypatch.setattr(m, "_persist", lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic-secret-error")))
    if bad is None:
        result = m.isolated_nous_credentials(**kw)
        assert result.access_token == rotated["access_token"]
        assert result.expires_at < time.time()+401
        assert m.isolated_nous_credentials(**kw).expires_at == result.expires_at
        assert not kw["marker_path"].exists()
    else:
        with pytest.raises(m.CredentialBlocked) as err:
            m.isolated_nous_credentials(**kw)
        assert "synthetic-secret-error" not in str(err.value)
        assert kw["marker_path"].exists()
        with pytest.raises(m.CredentialBlocked, match="uncertain"):
            m.isolated_nous_credentials(**kw)
        if bad not in ("missing", "network", "redirect", "persist"):
            obj = m._read(kw["source_path"])
            assert obj["subject"] == "synthetic-owner"
            assert obj["state"]["refresh_token"] == "synthetic-rotated"
    assert len(calls) == 1


def test_native_replacement_temp_acl_and_peers(fx, monkeypatch):
    m, kw = fx
    enroll(m, kw, ttl=100)
    checked = []
    original = m._require_private_acl
    def check(path):
        original(path)
        if path.name.startswith(".nous-"):
            checked.append(path)
            assert path.stat().st_size == 0
    monkeypatch.setattr(m, "_require_private_acl", check)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload(expires_in=400))
    transport(m, monkeypatch, handler)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: m.isolated_nous_credentials(**kw), range(2)))
    assert len(calls) == 1 and len(checked) >= 2
    assert results[0].access_token == results[1].access_token
    original(kw["source_path"])


@pytest.mark.parametrize("bad", ["hardlink", "ciphertext", "marker", "expiry", "subject", "schema", "alias", "minimum"])
def test_cached_hardening_no_network(fx, bad):
    m, kw = fx
    enroll(m, kw)
    source = kw["source_path"]
    if bad == "hardlink": source.with_name("alias").hardlink_to(source)
    elif bad == "ciphertext": source.write_bytes(b"invalid-ciphertext")
    elif bad == "marker": kw["marker_path"].write_bytes(b"uncertain")
    elif bad == "minimum": kw["min_ttl_seconds"] = 179
    else:
        obj = m._read(source)
        if bad == "expiry": obj["state"]["expires_at"] = time.time()+10000
        if bad == "subject": obj["subject"] = "other"
        if bad == "schema": obj["version"] = 2
        if bad == "alias": obj["state"]["agent_key"] = "legacy"
        m._persist(source, obj)
    before = source.read_bytes()
    with pytest.raises(m.CredentialBlocked): m.isolated_nous_credentials(**kw)
    assert source.read_bytes() == before
