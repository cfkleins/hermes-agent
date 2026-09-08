"""Owner-only isolated device enrollment; invoke with ``python -m ... --source ABS``.

No normal login/resolver/store participates. Browser URI and OAuth payloads stay
in memory; the isolated persistence API exclusively writes DPAPI and reads back.
This is a dedicated, single-threaded CLI, not a service endpoint. Injected seams
are for offline tests only and cannot be selected by command-line arguments.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import stat
import sys
import time
from urllib.parse import urlsplit

import httpx

from hermes_cli.auth_constants import (
    DEFAULT_NOUS_CLIENT_ID, DEFAULT_NOUS_PORTAL_URL, DEFAULT_NOUS_SCOPE,
    DEVICE_CODE_GRANT_TYPE,
)

MAX_AUTHORIZATION_SECONDS = 900
_PERSISTENCE_FAILURES = {
    reason: "persistence_" + reason for reason in (
        "invalid_provider_key", "unsafe_permissions", "unsafe_acl_owner", "unsafe_acl",
        "unsupported_acl", "absolute_path_required", "ambiguous_path", "missing_path",
        "linked_path", "unsafe_file", "unsafe_file_handle", "duplicate_json_key",
        "invalid_json_number", "oversize_store", "persistence_readback_failed",
        "lease_timeout", "invalid_expiry", "invalid_jwt", "invalid_claims",
        "invalid_subject", "missing_invoke_scope", "unsupported_payload_schema",
        "routing_mismatch", "token_type_required", "subject_mismatch", "scope_mismatch",
        "expired_enrollment", "fresh_store_required", "enrollment_failed_owner_relogin",
    )
}


class EnrollmentError(RuntimeError):
    """Fixed internal code only; never construct from provider/OS error text."""


def _require_private_acl(path: Path) -> None:
    if os.name != "nt":
        raise EnrollmentError("windows_required")
    import win32api
    import win32con
    import win32security
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    finally:
        token.Close()
    allowed = {win32security.ConvertSidToStringSid(user), "S-1-5-18", "S-1-5-32-544"}
    sd = win32security.GetFileSecurity(str(path), win32security.OWNER_SECURITY_INFORMATION |
                                      win32security.DACL_SECURITY_INFORMATION)
    if win32security.ConvertSidToStringSid(sd.GetSecurityDescriptorOwner()) not in allowed:
        raise EnrollmentError("unsafe_acl")
    acl = sd.GetSecurityDescriptorDacl()
    if acl is None or (path.is_dir() and not sd.GetSecurityDescriptorControl()[0] &
                       win32security.SE_DACL_PROTECTED):
        raise EnrollmentError("unsafe_acl")
    for index in range(acl.GetAceCount()):
        ace = acl.GetAce(index)
        if ace[0][0] not in (win32security.ACCESS_ALLOWED_ACE_TYPE,
                             win32security.ACCESS_DENIED_ACE_TYPE):
            raise EnrollmentError("unsafe_acl")
        # Include inherit-only grants: this parent will create sensitive children.
        if ace[0][0] == win32security.ACCESS_ALLOWED_ACE_TYPE and ace[1] and (
                win32security.ConvertSidToStringSid(ace[2]) not in allowed):
            raise EnrollmentError("unsafe_acl")


def _check_path(path: Path, *, absent: bool = False) -> None:
    if (not path.is_absolute() or ".." in path.parts or
            (os.name == "nt" and (len(path.drive) != 2 or path.drive[1] != ":")) or
            any(part.endswith((" ", ".")) or ":" in part for part in path.parts[1:])):
        raise EnrollmentError("invalid_source_path")
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            if part == path:
                return
            raise EnrollmentError("missing_parent") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise EnrollmentError("linked_path")
        if part != path and not stat.S_ISDIR(info.st_mode):
            raise EnrollmentError("invalid_parent")
        if part == path:
            if absent:
                raise EnrollmentError("source_exists")
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise EnrollmentError("unsafe_lock")


@contextmanager
def _enrollment_lock(source: Path):
    """Nonblocking Windows share-deny handle, held until persistence completes.

    Keep the empty sidecar: unlink-after-unlock permits split-lock races. A
    crashed process releases its handle, so the same empty filename is reusable.
    """
    if os.name != "nt":
        raise EnrollmentError("windows_required")
    import win32con
    import win32file
    lock = source.with_name(source.name + ".enrollment.lock")
    _check_path(lock)
    if lock.exists():
        _require_private_acl(lock)
    try:
        handle = win32file.CreateFile(str(lock), win32con.GENERIC_READ | win32con.GENERIC_WRITE,
                                      0, None, win32con.OPEN_ALWAYS,
                                      win32file.FILE_FLAG_OPEN_REPARSE_POINT, None)
    except win32file.error:
        raise EnrollmentError("enrollment_busy") from None
    try:
        info = win32file.GetFileInformationByHandle(handle)
        if info[0] & (0x400 | win32con.FILE_ATTRIBUTE_DIRECTORY) or info[7] != 1:
            raise EnrollmentError("unsafe_lock")
        _require_private_acl(lock)
        yield
    finally:
        handle.Close()


def _verification_uri(value) -> str:
    if (not isinstance(value, str) or not value or len(value) > 8192 or
            any(ord(c) <= 32 or ord(c) >= 127 for c in value) or "\\" in value or "#" in value):
        raise EnrollmentError("invalid_device_response")
    try:
        url = urlsplit(value)
        if (url.scheme != "https" or url.hostname != urlsplit(DEFAULT_NOUS_PORTAL_URL).hostname
                or url.username is not None or url.password is not None or url.port not in (None, 443)
                or url.netloc not in ("portal.nousresearch.com", "portal.nousresearch.com:443")):
            raise EnrollmentError("invalid_device_response")
    except ValueError:
        raise EnrollmentError("invalid_device_response") from None
    return value


def _device_payload(payload):
    if not isinstance(payload, dict):
        raise EnrollmentError("invalid_device_response")
    for field in ("device_code", "user_code"):
        value = payload.get(field)
        if (not isinstance(value, str) or not value or len(value) > 8192 or
                any(ord(c) <= 32 or ord(c) >= 127 for c in value)):
            raise EnrollmentError("invalid_device_response")
    for field in ("interval", "expires_in"):
        value = payload.get(field)
        if type(value) is not int or not 0 < value <= MAX_AUTHORIZATION_SECONDS:
            raise EnrollmentError("invalid_device_response")
    _verification_uri(payload.get("verification_uri"))
    _verification_uri(payload.get("verification_uri_complete"))
    return payload


def _json_response(response):
    try:
        payload = response.json()
    except (ValueError, UnicodeError):
        raise EnrollmentError("invalid_response") from None
    if not isinstance(payload, dict):
        raise EnrollmentError("invalid_response")
    return payload


def _token_payload(payload):
    for field in ("access_token", "refresh_token"):
        value = payload.get(field)
        if (not isinstance(value, str) or not value or len(value) > 8192 or
                any(ord(c) <= 32 or ord(c) >= 127 for c in value)):
            raise EnrollmentError("invalid_token_response")
    scope = payload.get("scope")
    if (payload.get("token_type") != "Bearer" or not isinstance(scope, str) or
            DEFAULT_NOUS_SCOPE not in scope.split() or "error" in payload):
        raise EnrollmentError("invalid_token_response")
    # JWT subject, expiry and optional route validation belong to persistence API.
    return payload


def _poll(client, device, deadline, clock, sleep):
    interval = device["interval"]
    while True:
        # Wait the server interval even before the first poll; never clamp it down.
        if clock() + interval >= deadline:
            raise EnrollmentError("authorization_timeout")
        sleep(interval)
        if clock() >= deadline:
            raise EnrollmentError("authorization_timeout")
        response = client.post(DEFAULT_NOUS_PORTAL_URL + "/api/oauth/token", data={
            "grant_type": DEVICE_CODE_GRANT_TYPE, "client_id": DEFAULT_NOUS_CLIENT_ID,
            "device_code": device["device_code"]})
        if clock() >= deadline:
            raise EnrollmentError("authorization_timeout")
        if response.status_code not in (200, 400):
            raise EnrollmentError("token_http_error")
        payload = _json_response(response)
        if response.status_code == 200:
            return _token_payload(payload)
        error = payload.get("error")
        if error == "slow_down":
            interval += 5  # RFC 8628: applies to this and every subsequent poll.
        elif error != "authorization_pending":
            raise EnrollmentError("authorization_rejected")


def _open_owner_browser(url):
    # ShellExecute URI dispatch, not a command line, subprocess, or webbrowser fallback.
    os.startfile(url)


def run_enrollment(*, source_path: Path, transport=None, opener=None, persist=None,
                   clock=time.monotonic, sleep=time.sleep) -> None:
    """Perform exactly one owner flow; only pending/slow_down permit more polls."""
    previous_logging_disable = logging.root.manager.disable
    logging.disable(sys.maxsize)  # Dedicated CLI: never emit HTTP headers/body debug logs.
    try:
        source = Path(source_path)
        if source.name != "nous-login.dpapi" or source.parent.name != "private":
            raise EnrollmentError("invalid_source_path")
        _check_path(source, absent=True)
        marker = source.with_name(source.name + ".uncertain")
        _check_path(marker, absent=True)
        _require_private_acl(source.parent)
        with _enrollment_lock(source):
            _check_path(source, absent=True)
            _check_path(marker, absent=True)
            _require_private_acl(source.parent)
            if persist is None:
                # This API exclusively creates the dedicated DPAPI store and verifies
                # readback itself. Do not call a resolver (it could refresh a grant).
                from hermes_cli.bounded_nous_credentials import enroll_nous_credentials
                persist = enroll_nous_credentials
            with httpx.Client(verify=True, trust_env=False, follow_redirects=False,
                              timeout=15, transport=transport if transport is not None else
                              httpx.HTTPTransport(verify=True, trust_env=False, retries=0)) as client:
                started = clock()
                response = client.post(DEFAULT_NOUS_PORTAL_URL + "/api/oauth/device/code",
                                       data={"client_id": DEFAULT_NOUS_CLIENT_ID,
                                             "scope": DEFAULT_NOUS_SCOPE})
                if response.status_code != 200:
                    raise EnrollmentError("device_http_error")
                device = _device_payload(_json_response(response))
                deadline = started + device["expires_in"]
                if clock() >= deadline:
                    raise EnrollmentError("authorization_timeout")
                print('{"status":"awaiting_owner_login"}', flush=True)
                try:
                    (opener if opener is not None else _open_owner_browser)(device["verification_uri_complete"])
                except Exception:
                    raise EnrollmentError("browser_failed") from None
                payload = _poll(client, device, deadline, clock, sleep)
            try:
                persist(source_path=source, token_payload=payload)
            except Exception as error:
                # Emit only a fixed known code; never exception text or payload values.
                code = _PERSISTENCE_FAILURES.get(str(error), "persistence_failed")
                raise EnrollmentError(code) from None
            print('{"status":"enrolled"}', flush=True)
    except httpx.HTTPError:
        raise EnrollmentError("network_error") from None
    except EnrollmentError:
        raise
    except Exception:
        raise EnrollmentError("enrollment_failed") from None
    finally:
        logging.disable(previous_logging_disable)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise EnrollmentError("invalid_arguments")


def main(argv=None) -> int:
    try:
        parser = _Parser(prog="bounded-nous-enroll", allow_abbrev=False)
        parser.add_argument("--source", required=True, type=Path)
        args = parser.parse_args(argv)
        run_enrollment(source_path=args.source)
        return 0
    except KeyboardInterrupt:
        code = "interrupted"
    except EnrollmentError as error:
        # Only locally defined codes can escape; never arbitrary exception text.
        allowed = {"invalid_arguments", "windows_required", "unsafe_acl", "invalid_source_path",
                   "missing_parent", "linked_path", "invalid_parent", "source_exists", "unsafe_lock",
                   "enrollment_busy", "invalid_device_response", "invalid_response", "invalid_token_response",
                   "authorization_timeout", "token_http_error", "authorization_rejected", "device_http_error",
                   "browser_failed", "persistence_failed", "network_error", "enrollment_failed"}
        allowed.update(_PERSISTENCE_FAILURES.values())
        code = str(error) if str(error) in allowed else "enrollment_failed"
    except Exception:
        code = "enrollment_failed"
    print(json.dumps({"status": "error", "code": code}, separators=(",", ":")), file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
