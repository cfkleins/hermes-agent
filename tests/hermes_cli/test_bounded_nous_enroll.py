"""Offline device authorization: synthetic HTTP only, never an owner browser."""
import base64
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

pytestmark = pytest.mark.windows_only


def test_cli_is_implemented():
    assert importlib.util.find_spec("hermes_cli.bounded_nous_enroll") is not None


@pytest.fixture
def cli():
    try:
        return importlib.import_module("hermes_cli.bounded_nous_enroll")
    except ModuleNotFoundError:
        pytest.fail("isolated enrollment CLI has not been implemented")


@pytest.fixture
def source(tmp_path, monkeypatch, cli):
    # HTTP controller tests isolate ACL policy; native ACL tests below do not.
    monkeypatch.setattr(cli, "_require_private_acl", lambda path: None)
    private = tmp_path / "private"
    private.mkdir()
    return private / "nous-login.dpapi"


class Clock:
    def __init__(self):
        self.now = 0
        self.waits = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def device(**changes):
    result = dict(device_code="synthetic-device-secret", user_code="SYNTHETIC-USER",
                  verification_uri="https://portal.nousresearch.com/device",
                  verification_uri_complete="https://portal.nousresearch.com/device?user_code=SYNTHETIC-USER",
                  expires_in=100, interval=3)
    result.update(changes)
    return result


def tokens():
    return dict(access_token="synthetic-access", refresh_token="synthetic-refresh",
                scope="inference:invoke", token_type="Bearer")


def run(cli, source, replies, **changes):
    requests, opened, saved = [], [], []
    clock = Clock()
    def respond(request):
        requests.append((request, clock.now))
        reply = replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        status, payload = reply
        return httpx.Response(status, content=json.dumps(payload).encode("utf-8"),
                              headers={"Content-Type": "application/json"})
    kwargs = dict(transport=httpx.MockTransport(respond), opener=opened.append,
                  persist=lambda **kw: saved.append(kw), clock=clock.monotonic,
                  sleep=clock.sleep)
    kwargs.update(changes)
    return lambda: cli.run_enrollment(source_path=source, **kwargs), requests, opened, saved, clock


def test_pending_slow_down_then_success_is_pinned_and_secret_safe(cli, source, capsys, caplog):
    action, requests, opened, saved, clock = run(cli, source, [
        (200, device()), (400, {"error": "authorization_pending"}),
        (400, {"error": "slow_down"}), (200, tokens())])
    with caplog.at_level(logging.DEBUG):
        action()
    assert [t for _, t in requests] == [0, 3, 6, 14]
    assert clock.waits == [3, 3, 8]
    assert [str(req.url) for req, _ in requests] == [
        "https://portal.nousresearch.com/api/oauth/device/code",
        *["https://portal.nousresearch.com/api/oauth/token"] * 3]
    for req, _ in requests:
        assert req.method == "POST"
        assert req.extensions["timeout"] == dict(connect=15, read=15, write=15, pool=15)
    assert parse_qs(requests[0][0].content.decode()) == {
        "client_id": ["hermes-cli"], "scope": ["inference:invoke"]}
    assert parse_qs(requests[1][0].content.decode()) == {
        "grant_type": ["urn:ietf:params:oauth:grant-type:device_code"],
        "client_id": ["hermes-cli"], "device_code": [device()["device_code"]]}
    assert opened == [device()["verification_uri_complete"]]
    assert saved == [{"source_path": source, "token_payload": tokens()}]
    captured = capsys.readouterr()
    assert [json.loads(line) for line in captured.out.splitlines()] == [
        {"status": "awaiting_owner_login"}, {"status": "enrolled"}]
    assert captured.err == ""
    assert not caplog.records
    assert not source.exists()  # The fake persistence callable stores nothing.
    assert all(p.stat().st_size == 0 for p in source.parent.iterdir())


@pytest.mark.parametrize("status,payload", [
    (400, {"error": "invalid_grant", "error_description": "synthetic-secret"}),
    (400, {"error": "access_denied"}), (400, {"error": "expired_token"}),
    (401, {"error": "authorization_pending"}), (429, {"error": "slow_down"}),
    (500, {"error": "authorization_pending"}), (302, {"error": "authorization_pending"}),
    (200, {"access_token": "synthetic-secret"}), (400, ["authorization_pending"]),
    (200, dict(tokens(), token_type="Other")),
    (200, dict(tokens(), scope="billing:manage")),
])
def test_non_authorized_poll_results_abort_without_retry(cli, source, status, payload, capsys):
    action, requests, opened, saved, _ = run(cli, source, [(200, device()), (status, payload)])
    with pytest.raises(cli.EnrollmentError):
        action()
    assert len(requests) == 2
    assert not saved
    assert "enrolled" not in capsys.readouterr().out


@pytest.mark.parametrize("failure", [httpx.ConnectError("synthetic-secret"),
                                     httpx.ReadTimeout("synthetic-secret")])
@pytest.mark.parametrize("during_token", [False, True])
def test_network_errors_never_restart(cli, source, failure, during_token):
    replies = ([(200, device())] if during_token else []) + [failure]
    action, requests, _, saved, _ = run(cli, source, replies)
    with pytest.raises(cli.EnrollmentError, match="^network_error$"):
        action()
    assert len(requests) == (2 if during_token else 1)
    assert not saved


@pytest.mark.parametrize("field", list(device()))
def test_missing_device_fields_prevent_browser_and_token_poll(cli, source, field):
    payload = device()
    del payload[field]
    action, requests, opened, saved, _ = run(cli, source, [(200, payload)])
    with pytest.raises(cli.EnrollmentError):
        action()
    assert len(requests) == 1
    assert not opened and not saved


@pytest.mark.parametrize("change", [
    {"interval": 0}, {"interval": -1}, {"interval": True}, {"interval": 1.5},
    {"interval": "3"}, {"interval": float("inf")}, {"interval": 901},
    {"expires_in": 901}, {"expires_in": 0}, {"expires_in": True},
    {"device_code": ""}, {"user_code": None},
])
def test_invalid_device_values_abort(cli, source, change):
    action, requests, opened, _, _ = run(cli, source, [(200, device(**change))])
    with pytest.raises(cli.EnrollmentError):
        action()
    assert len(requests) == 1 and not opened


@pytest.mark.parametrize("url", [
    "http://portal.nousresearch.com/device?x=secret", "https://evil.example/device",
    "https://portal.nousresearch.com.evil.example/", "https://user@portal.nousresearch.com/",
    "https://portal.nousresearch.com:444/", "https://portal.nousresearch.com/#secret",
    "https://portal.nousresearch.com\\@evil.example/", "file:///C:/synthetic-secret",
    "https://portal.nousresearch.com/\nsecret", "https://portal.nousresearch.com/#",
])
def test_invalid_verification_uri_never_opens(cli, source, url):
    action, requests, opened, _, _ = run(cli, source, [(200, device(verification_uri_complete=url))])
    with pytest.raises(cli.EnrollmentError):
        action()
    assert len(requests) == 1 and not opened


def test_existing_source_prevents_all_http_and_browser(cli, source):
    source.write_bytes(b"synthetic-existing-ciphertext")
    action, requests, opened, _, _ = run(cli, source, [])
    with pytest.raises(cli.EnrollmentError, match="^source_exists$"):
        action()
    assert not requests and not opened
    assert source.read_bytes() == b"synthetic-existing-ciphertext"


@pytest.mark.parametrize("kind", ["filename", "parent", "uncertainty"])
def test_unusable_service_target_prevents_authorization(cli, source, kind):
    if kind == "filename":
        source = source.with_name("wrong.dpapi")
    elif kind == "parent":
        source = source.parent.parent / source.name
    else:
        source.with_name(source.name + ".uncertain").write_text("DEMO uncertainty")
    action, requests, opened, saved, _ = run(cli, source, [(200, device()), (200, tokens())])
    with pytest.raises(cli.EnrollmentError):
        action()
    assert not requests and not opened and not saved


def test_acl_failure_prevents_all_http(cli, source, monkeypatch):
    def refuse(path):
        raise cli.EnrollmentError("unsafe_acl")
    monkeypatch.setattr(cli, "_require_private_acl", refuse)
    action, requests, _, _, _ = run(cli, source, [])
    with pytest.raises(cli.EnrollmentError):
        action()
    assert requests == []


def test_deadline_never_polls_at_or_after_expiry(cli, source):
    action, requests, _, saved, clock = run(cli, source, [
        (200, device(expires_in=6)), (400, {"error": "authorization_pending"})])
    with pytest.raises(cli.EnrollmentError, match="^authorization_timeout$"):
        action()
    assert [t for _, t in requests] == [0, 3]
    assert not saved and clock.now <= 6


def test_browser_failure_never_polls(cli, source):
    def fail(url):
        raise OSError(url)
    action, requests, _, saved, _ = run(cli, source, [(200, device())], opener=fail)
    with pytest.raises(cli.EnrollmentError, match="^browser_failed$"):
        action()
    assert len(requests) == 1 and not saved


def test_persist_failure_never_reports_enrolled(cli, source, capsys):
    def fail(**kw):
        raise ValueError(repr(kw))
    action, _, _, _, _ = run(cli, source, [(200, device()), (200, tokens())], persist=fail)
    with pytest.raises(cli.EnrollmentError, match="^persistence_failed$"):
        action()
    assert capsys.readouterr().out == '{"status":"awaiting_owner_login"}\n'


@pytest.mark.windows_only
def test_native_enrollment_lock_blocks_duplicate_before_http(cli, source):
    action, requests, _, _, _ = run(cli, source, [])
    with cli._enrollment_lock(source):
        with pytest.raises(cli.EnrollmentError, match="^enrollment_busy$"):
            action()
    assert not requests
    # Stale empty lock filename is reusable after handle close.
    with cli._enrollment_lock(source):
        pass


def test_source_absolute_required(cli, source):
    action, requests, _, _, _ = run(cli, Path("relative.dpapi"), [])
    with pytest.raises(cli.EnrollmentError):
        action()
    assert not requests


@pytest.mark.parametrize("failure", [KeyboardInterrupt("synthetic-secret"), ValueError("synthetic-secret")])
def test_main_sanitizes_interrupt_and_unexpected_exceptions(cli, source, monkeypatch, capsys, failure):
    def fail(**kwargs):
        raise failure
    monkeypatch.setattr(cli, "run_enrollment", fail)
    assert cli.main(["--source", str(source)]) != 0
    out = capsys.readouterr()
    assert out.out == ""
    assert json.loads(out.err) == {"status": "error", "code": (
        "interrupted" if isinstance(failure, KeyboardInterrupt) else "enrollment_failed")}
    assert "synthetic-secret" not in out.err


def test_main_invalid_args_never_echo_values(cli, capsys):
    assert cli.main(["--unexpected-synthetic-secret"]) != 0
    out = capsys.readouterr()
    assert out.out == ""
    assert json.loads(out.err) == {"status": "error", "code": "invalid_arguments"}


def protect_directory(path):
    import win32api
    import ntsecuritycon
    import win32con
    import win32security
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    finally:
        token.Close()
    acl = win32security.ACL()
    acl.AddAccessAllowedAceEx(win32security.ACL_REVISION,
        win32con.CONTAINER_INHERIT_ACE | win32con.OBJECT_INHERIT_ACE, ntsecuritycon.FILE_ALL_ACCESS, user)
    win32security.SetNamedSecurityInfo(str(path), win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        None, None, acl, None)


def test_real_acl_and_dpapi_persistence_without_default_auth(cli, tmp_path, monkeypatch, capsys):
    # No fake ACL, native lock, real fresh DPAPI file + readback API. Only wire
    # traffic/browser are synthetic. Blocking imports prevents accidental resolver use.
    for name in ("hermes_cli.auth", "hermes_cli.auth_store", "hermes_cli.auth_device_flow"):
        monkeypatch.setitem(sys.modules, name, None)
    from hermes_cli import bounded_nous_credentials as store
    def forbidden(*args, **kwargs):
        pytest.fail("enrollment must not refresh, infer, or consult a shared store")
    monkeypatch.setattr(store, "_refresh_access_token", forbidden)
    private = tmp_path / "private"
    private.mkdir()
    protect_directory(private)
    source = private / "nous-login.dpapi"
    claims = dict(sub="synthetic-owner-subject", exp=time.time() + 3600, scope="inference:invoke")
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    payload = dict(tokens(), access_token="synthetic." + encoded + ".signature")
    action, requests, opened, _, _ = run(cli, source, [(200, device()), (200, payload)], persist=None)
    action()
    raw = source.read_bytes()
    assert not raw.startswith(b"{")
    for secret in (payload["access_token"], payload["refresh_token"], claims["sub"],
                   device()["device_code"], device()["user_code"]):
        assert secret.encode() not in raw
    credential = store.isolated_nous_credentials(source_path=source,
        marker_path=source.with_name(source.name + ".uncertain"))
    assert credential.access_token == payload["access_token"]
    assert credential.expires_at == claims["exp"]
    assert len(requests) == 2 and len(opened) == 1
    assert capsys.readouterr().out.endswith('{"status":"enrolled"}\n')


def test_native_broad_acl_refused_before_http(cli, tmp_path):
    import ntsecuritycon
    import win32con
    import win32security
    private = tmp_path / "private"
    private.mkdir()
    protect_directory(private)
    sd = win32security.GetFileSecurity(str(private), win32security.DACL_SECURITY_INFORMATION)
    acl = sd.GetSecurityDescriptorDacl()
    acl.AddAccessAllowedAceEx(win32security.ACL_REVISION, 0, ntsecuritycon.FILE_GENERIC_READ,
                             win32security.ConvertStringSidToSid("S-1-1-0"))
    win32security.SetNamedSecurityInfo(str(private), win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        None, None, acl, None)
    action, requests, opened, _, _ = run(cli, private / "nous-login.dpapi", [])
    with pytest.raises(cli.EnrollmentError, match="^unsafe_acl$"):
        action()
    assert not requests and not opened


def test_reparse_ancestor_rejected_before_http(cli, source, monkeypatch):
    original = Path.lstat
    def lstat(path):
        info = original(path)
        if path == source.parent:
            from types import SimpleNamespace
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
        return info
    monkeypatch.setattr(Path, "lstat", lstat)
    action, requests, _, _, _ = run(cli, source, [])
    with pytest.raises(cli.EnrollmentError, match="^linked_path$"):
        action()
    assert not requests


def test_default_client_and_transport_are_pinned(cli, source, monkeypatch):
    clients, transports = [], []
    real_client = httpx.Client
    mock = httpx.MockTransport(lambda request: httpx.Response(400, json={"error": "stop"}))
    def make_transport(**kwargs):
        transports.append(kwargs)
        return mock
    def make_client(**kwargs):
        clients.append(kwargs)
        return real_client(**kwargs)
    monkeypatch.setattr(httpx, "HTTPTransport", make_transport)
    monkeypatch.setattr(httpx, "Client", make_client)
    with pytest.raises(cli.EnrollmentError, match="^device_http_error$"):
        cli.run_enrollment(source_path=source, persist=lambda **kw: None, opener=lambda url: None)
    assert transports == [{"verify": True, "trust_env": False, "retries": 0}]
    assert clients == [{"verify": True, "trust_env": False, "follow_redirects": False,
                        "timeout": 15, "transport": mock}]


@pytest.mark.parametrize("token_stage", [False, True])
@pytest.mark.parametrize("redirect", [False, True])
def test_non_json_or_redirect_never_follows_or_retries(cli, source, token_stage, redirect):
    requests, opened, saved = [], [], []
    def respond(request):
        requests.append(request)
        if token_stage and len(requests) == 1:
            return httpx.Response(200, json=device())
        return httpx.Response(302 if redirect else 400, content=b"synthetic-secret-not-json",
                              headers={"Location": "https://evil.example/synthetic-secret"})
    clock = Clock()
    with pytest.raises(cli.EnrollmentError):
        cli.run_enrollment(source_path=source, transport=httpx.MockTransport(respond),
            opener=opened.append, persist=lambda **kw: saved.append(kw), clock=clock.monotonic,
            sleep=clock.sleep)
    assert len(requests) == (2 if token_stage else 1) and not saved
    assert all(request.url.host == "portal.nousresearch.com" for request in requests)


def test_owner_browser_uses_only_startfile(cli, monkeypatch):
    opened = []
    monkeypatch.setattr(os, "startfile", opened.append)
    cli._open_owner_browser(device()["verification_uri_complete"])
    assert opened == [device()["verification_uri_complete"]]


def test_status_flushes_before_browser_and_enrolled_follows_persistence(cli, source, monkeypatch):
    events = []
    class Output:
        def write(self, text):
            events.append(("write", text))
        def flush(self):
            events.append(("flush", None))
    monkeypatch.setattr(sys, "stdout", Output())
    def opened(url):
        assert events[-1] == ("flush", None)
        events.append(("browser", None))
    def saved(**kwargs):
        assert not any("enrolled" in (text or "") for _, text in events)
        events.append(("persisted_readback", None))
    action, _, _, _, _ = run(cli, source, [(200, device()), (200, tokens())], opener=opened, persist=saved)
    action()
    assert events.index(("persisted_readback", None)) < events.index(("write", '{"status":"enrolled"}'))
    assert events[-1] == ("flush", None)

