"""Private independently enrolled Nous grant. No shared/profile store resolution.

DPAPI CurrentUser protects the entire envelope. JWT claims are local consistency
checks, not signature verification; enrollment must receive a fresh TLS device
flow response. Incomplete rotation requires owner relogin, never spent-pair retry.
"""
from __future__ import annotations

import base64
import json
import math
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from hermes_cli.auth_nous import _refresh_access_token

_client_factory = httpx.Client
_ROUTE = {"client_id": "hermes-cli", "portal_base_url": "https://portal.nousresearch.com",
          "inference_base_url": "https://inference-api.nousresearch.com/v1", "token_type": "Bearer"}
_FIELDS = set(_ROUTE) | {"access_token", "refresh_token", "scope", "expires_at"}


class CredentialBlocked(RuntimeError):
    """Sanitized non-retry failure; callers must not log locals or fall back."""


@dataclass(frozen=True, repr=False)
class NousAccessCredential:
    access_token: str
    expires_at: float


def validate_provider_key(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 8192 or any(
            ord(c) <= 32 or ord(c) >= 127 or c in ",$ {}" for c in value):
        raise CredentialBlocked("invalid_provider_key")
    return value


def _require_private_acl(path: Path) -> None:
    """Read-only conservative permission check; never repair existing ACLs."""
    if os.name != "nt":
        info = path.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise CredentialBlocked("unsafe_permissions")
        return
    import win32api
    import win32con
    import win32security
    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    finally:
        token.Close()
    sd = win32security.GetFileSecurity(str(path), win32security.OWNER_SECURITY_INFORMATION |
                                      win32security.DACL_SECURITY_INFORMATION)
    allowed = {win32security.ConvertSidToStringSid(user), "S-1-5-18", "S-1-5-32-544"}
    if win32security.ConvertSidToStringSid(sd.GetSecurityDescriptorOwner()) not in allowed:
        raise CredentialBlocked("unsafe_acl_owner")
    acl = sd.GetSecurityDescriptorDacl()
    if acl is None:
        raise CredentialBlocked("unsafe_acl")
    for index in range(acl.GetAceCount()):
        ace = acl.GetAce(index)
        kind, flags = ace[0]
        if flags & win32con.INHERIT_ONLY_ACE:
            continue
        if kind not in (win32security.ACCESS_ALLOWED_ACE_TYPE, win32security.ACCESS_DENIED_ACE_TYPE):
            raise CredentialBlocked("unsupported_acl")
        if kind == win32security.ACCESS_ALLOWED_ACE_TYPE and ace[1] and (
                win32security.ConvertSidToStringSid(ace[2]) not in allowed):
            raise CredentialBlocked("unsafe_acl")


def _path_check(path: Path, *, optional: bool = False) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise CredentialBlocked("absolute_path_required")
    # Reject Windows alternate streams and ambiguous trailing-dot/space aliases.
    if any(":" in p or p.endswith((".", " ")) for p in path.parts[1:]):
        raise CredentialBlocked("ambiguous_path")
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            if part == path and optional:
                return
            raise CredentialBlocked("missing_path") from None
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CredentialBlocked("linked_path")
        if part == path and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise CredentialBlocked("unsafe_file")


def _checked_fd(path, fd):
    _path_check(path)
    info, named = os.fstat(fd), path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
            (info.st_dev, info.st_ino) != (named.st_dev, named.st_ino)):
        raise CredentialBlocked("unsafe_file_handle")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise CredentialBlocked("duplicate_json_key")
        result[key] = value
    return result


def _bad_constant(value):
    raise CredentialBlocked("invalid_json_number")


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_bad_constant)


def _protect(value):
    import win32crypt
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False).encode("utf-8")
    # UI forbidden, CurrentUser (LOCAL_MACHINE deliberately absent).
    return win32crypt.CryptProtectData(raw, "isolated-nous-v1", None, None, None, 1)


def _read_bytes(path):
    _path_check(path)
    _require_private_acl(path)
    _require_private_acl(path.parent)
    with path.open("rb") as stream:
        _checked_fd(path, stream.fileno())
        if os.fstat(stream.fileno()).st_size > 1024 * 1024:
            raise CredentialBlocked("oversize_store")
        raw = stream.read(1024 * 1024 + 1)
        _checked_fd(path, stream.fileno())
    if len(raw) > 1024 * 1024:
        raise CredentialBlocked("oversize_store")
    return raw


def _read(path):
    import win32crypt
    raw = _read_bytes(path)
    return _json(win32crypt.CryptUnprotectData(raw, None, None, None, 1)[1])


def _exclusive(path, raw):
    _path_check(path, optional=True)
    _require_private_acl(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        _checked_fd(path, stream.fileno())
        _require_private_acl(path)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if _read_bytes(path) != raw:
        raise CredentialBlocked("persistence_readback_failed")


def _persist(path, value):
    """Encrypt before any disk write; prove temporary ACL before ciphertext."""
    _path_check(path)
    _require_private_acl(path.parent)
    raw = _protect(value)
    fd, name = tempfile.mkstemp(prefix=".nous-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            _checked_fd(temporary, stream.fileno())
            _require_private_acl(temporary)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _path_check(path)
        _require_private_acl(path)
        os.replace(temporary, path)
        if _read_bytes(path) != raw:
            raise CredentialBlocked("persistence_readback_failed")
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _lease(source):
    """Exact sidecar kernel lease; never unlink it (avoids split lock inodes)."""
    path = source.with_name(source.name + ".lease")
    _path_check(source, optional=True)
    _path_check(path, optional=True)
    _require_private_acl(source.parent)
    if path.exists():
        _require_private_acl(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
    acquired = False
    try:
        _checked_fd(path, fd)
        _require_private_acl(path)
        deadline = time.monotonic() + 15
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise CredentialBlocked("lease_timeout") from None
                time.sleep(0.02)
        _checked_fd(path, fd)
        yield
    finally:
        if acquired:
            os.lseek(fd, 0, os.SEEK_SET)
            if os.name == "nt":
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise CredentialBlocked("invalid_expiry")
    return float(value)


def _claims(token):
    validate_provider_key(token)
    parts = token.split(".")
    if len(parts) != 3 or not all(parts):
        raise CredentialBlocked("invalid_jwt")
    claims = _json(base64.b64decode(parts[1] + "=" * (-len(parts[1]) % 4), altchars=b"-_", validate=True))
    if not isinstance(claims, dict):
        raise CredentialBlocked("invalid_claims")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject or subject.strip() != subject:
        raise CredentialBlocked("invalid_subject")
    _number(claims.get("exp"))
    _scope(claims.get("scope"))
    return claims


def _scope(value):
    if not isinstance(value, str) or "inference:invoke" not in value.split():
        raise CredentialBlocked("missing_invoke_scope")
    return set(value.split())


def _payload(data, subject, now):
    if not isinstance(data, dict) or set(data) - (_FIELDS | {"expires_in"}):
        raise CredentialBlocked("unsupported_payload_schema")
    for key, value in _ROUTE.items():
        if data.get(key, value) != value:
            raise CredentialBlocked("routing_mismatch")
    if data.get("token_type") != "Bearer":
        raise CredentialBlocked("token_type_required")
    claims = _claims(data.get("access_token"))
    if claims["sub"] != subject:
        raise CredentialBlocked("subject_mismatch")
    validate_provider_key(data.get("refresh_token"))
    # _claims already requires invoke in the JWT. Require it independently in
    # the response; unrelated scope names need not match or become authority.
    _scope(data.get("scope"))
    expiry = _number(claims["exp"])
    if "expires_in" in data:
        ttl = _number(data["expires_in"])
        if ttl <= 0:
            raise CredentialBlocked("invalid_expiry")
        expiry = min(expiry, now + ttl)
    if "expires_at" in data:
        declared = _number(data["expires_at"])
        if declared > claims["exp"]:
            raise CredentialBlocked("expiry_mismatch")
        expiry = min(expiry, declared)
    if expiry <= now:
        raise CredentialBlocked("expired_enrollment")
    return {**_ROUTE, "access_token": data["access_token"], "refresh_token": data["refresh_token"],
            "scope": data["scope"], "expires_at": expiry}


def _cached(root):
    if not isinstance(root, dict) or set(root) != {"version", "subject", "state"} or type(root["version"]) is not int or root["version"] != 1:
        raise CredentialBlocked("unsupported_store_schema")
    state = root["state"]
    if not isinstance(state, dict) or set(state) != _FIELDS:
        raise CredentialBlocked("unsupported_state_schema")
    claims = _claims(state["access_token"])
    if claims["sub"] != root["subject"]:
        raise CredentialBlocked("subject_mismatch")
    for key, value in _ROUTE.items():
        if state[key] != value:
            raise CredentialBlocked("routing_mismatch")
    validate_provider_key(state["refresh_token"])
    # Preserve the same required-permission check on persisted readback.
    _scope(state["scope"])
    expiry = _number(state["expires_at"])
    if expiry > claims["exp"]:
        raise CredentialBlocked("expiry_mismatch")
    return expiry


def enroll_nous_credentials(*, source_path: Path, token_payload: dict) -> None:
    """Persist only a freshly obtained independent grant; never overwrite."""
    try:
        source = Path(source_path)
        with _lease(source):
            _path_check(source, optional=True)
            if source.exists() or source.with_name(source.name + ".uncertain").exists():
                raise CredentialBlocked("fresh_store_required")
            subject = _claims(token_payload.get("access_token"))["sub"]
            state = _payload(token_payload, subject, time.time())
            _exclusive(source, _protect(dict(version=1, subject=subject, state=state)))
    except CredentialBlocked:
        raise
    except Exception:
        raise CredentialBlocked("enrollment_failed_owner_relogin") from None


def isolated_nous_credentials(*, source_path: Path, marker_path: Path,
                              min_ttl_seconds: int = 180) -> NousAccessCredential:
    """Return cached grant or execute one pinned, fail-closed rotation."""
    try:
        source, marker = Path(source_path), Path(marker_path)
        if marker != source.with_name(source.name + ".uncertain"):
            raise CredentialBlocked("exact_marker_required")
        if type(min_ttl_seconds) is not int or min_ttl_seconds < 180:
            raise CredentialBlocked("minimum_ttl_required")
        with _lease(source):
            _path_check(marker, optional=True)
            if marker.exists():
                raise CredentialBlocked("uncertain_rotation_blocked_owner_relogin")
            root = _read(source)
            expiry = _cached(root)
            if expiry > time.time() + min_ttl_seconds:
                return NousAccessCredential(root["state"]["access_token"], expiry)
            _exclusive(marker, b'{"status":"uncertain_rotation"}\n')
            with _client_factory(verify=True, trust_env=False, follow_redirects=False,
                    timeout=15, transport=httpx.HTTPTransport(verify=True, trust_env=False, retries=0)) as client:
                started = time.time()
                refreshed = _refresh_access_token(client=client, portal_base_url=_ROUTE["portal_base_url"],
                    client_id=_ROUTE["client_id"], refresh_token=root["state"]["refresh_token"])
            if not isinstance(refreshed, dict) or any(not isinstance(refreshed.get(k), str) or not refreshed[k]
                    for k in ("access_token", "refresh_token")):
                raise CredentialBlocked("rotation_missing_tokens_owner_relogin")
            # Commit rotated pair BEFORE any postvalidation. Preserve the pinned
            # subject and leave marker present until validated metadata is durable.
            root["state"].update({k: refreshed[k] for k in ("access_token", "refresh_token")})
            _persist(source, root)
            state = _payload(refreshed, root["subject"], started)
            if state["expires_at"] <= time.time() + min_ttl_seconds:
                raise CredentialBlocked("insufficient_rotated_ttl_owner_relogin")
            root["state"] = state
            _persist(source, root)
            _path_check(marker)
            marker.unlink()
            return NousAccessCredential(state["access_token"], state["expires_at"])
    except CredentialBlocked:
        raise
    except Exception:
        raise CredentialBlocked("credential_failed_owner_relogin") from None
