"""Isolated authenticated admission for server-owned bounded runs.

This helper is intentionally not wired to an HTTP route or run executor. It
accepts only raw request bytes and two headers, resolves all authority from an
immutable server registry, and mutates only the native SessionDB after every
request, policy, context, and runtime check succeeds.
"""
from dataclasses import dataclass, field
from hashlib import sha256
import hmac
import json
import re
from collections.abc import Mapping
from types import MappingProxyType
import unicodedata

from gateway.platforms.api_server_bound_runtime import BoundRuntime, resolve_bound_runtime


MAX_RAW_BODY_BYTES = 65_536
MAX_INPUT_CHARS = 16_000
MAX_CASE_SELECTOR_CHARS = 128
MAX_IDEMPOTENCY_KEY_CHARS = 128
MAX_CONTEXT_BYTES = 131_072
_REQUIRED_BODY_KEYS = frozenset(("input", "case"))
_REQUIRED_HTTP_BODY_KEYS = frozenset(("message", "case_selector", "idempotency_key"))
_REQUIRED_SKILLS = ("core-interview", "project-issue-interview")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", re.ASCII)
_DIGEST = re.compile(r"[0-9a-f]{64}", re.ASCII)
_IDENTITY_FIELDS = ("owner_id", "agent_id", "case_id", "source", "context_digest")


class BoundAdmissionError(ValueError):
    """Sanitized public failure with a stable machine-readable code."""

    _MESSAGES = {
        "unauthorized": "Bounded admission unauthorized",
        "invalid_request": "Invalid bounded request",
        "invalid_idempotency_key": "Invalid idempotency key",
        "not_found": "Authorized case not found",
        "invalid_server_policy": "Invalid bounded server policy",
        "runtime_unavailable": "Bounded runtime unavailable",
        "binding_conflict": "Bounded session binding conflict",
        "state_unavailable": "Bounded session state unavailable",
    }

    def __init__(self, code: str):
        self.code = code
        super().__init__(self._MESSAGES[code])


@dataclass(frozen=True, slots=True)
class BoundCase:
    """One server-approved principal/case policy; never client constructed."""

    principal_id: str
    owner_id: str
    agent_id: str
    case_id: str
    source: str
    context_digest: str
    context_bytes: bytes = field(repr=False, compare=False)
    skills: tuple[str, ...] = _REQUIRED_SKILLS
    tool_policy: str = "deny_all"
    provider: str = ""
    model: str = ""
    api_mode: str = ""
    base_url: str = ""
    context_length: int = 0
    api_key: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True, slots=True, init=False)
class BoundAdmissionRegistry:
    """Defensive immutable copies of authentication and case authority maps."""

    gateway_principals: Mapping[str, str] = field(repr=False)
    cases: Mapping[tuple[str, str], BoundCase] = field(repr=False)

    def __init__(self, *, gateway_principals: Mapping[str, str],
                 cases: Mapping[tuple[str, str], BoundCase]):
        try:
            principals = _strict_mapping_copy(gateway_principals)
            case_map = _strict_mapping_copy(cases)
        except Exception:
            raise ValueError("Invalid bounded admission registry") from None
        if not principals or not case_map:
            raise ValueError("Invalid bounded admission registry")
        for secret, principal in principals.items():
            if (type(secret) is not str or not secret or len(secret) > 512
                    or _has_control_or_space(secret) or not _valid_token(principal)):
                raise ValueError("Invalid bounded admission registry")
        for key, case in case_map.items():
            if (type(key) is not tuple or len(key) != 2 or not _valid_token(key[0])
                    or not _valid_token(key[1]) or not isinstance(case, BoundCase)
                    or case.principal_id != key[0]):
                raise ValueError("Invalid bounded admission registry")
        object.__setattr__(self, "gateway_principals", MappingProxyType(principals))
        object.__setattr__(self, "cases", MappingProxyType(case_map))


@dataclass(frozen=True, slots=True)
class BoundAdmission:
    """Validated launch material; context and credentials are repr-hidden."""

    principal_id: str
    input_text: str = field(repr=False)
    idempotency_key: str
    session_id: str
    binding_identity: Mapping[str, str]
    context_bytes: bytes = field(repr=False, compare=False)
    skills: tuple[str, ...]
    tool_policy: str
    runtime: BoundRuntime = field(repr=False)

    @property
    def exact_system_prompt_bytes(self) -> bytes:
        return self.context_bytes


@dataclass(frozen=True, slots=True)
class PreparedBoundAdmission:
    """Validated launch material that has not touched native session state."""

    principal_id: str
    input_text: str = field(repr=False)
    idempotency_key: str
    binding_identity: Mapping[str, str]
    context_bytes: bytes = field(repr=False, compare=False)
    skills: tuple[str, ...]
    tool_policy: str
    runtime: BoundRuntime = field(repr=False)

    @property
    def exact_system_prompt_bytes(self) -> bytes:
        return self.context_bytes


class _InvalidJSON(ValueError):
    pass


def _strict_mapping_copy(value: object) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError
    pairs = list(value.items())
    if len(pairs) != len(value):
        raise ValueError
    result = {}
    for key, item in pairs:
        if key in result:
            raise ValueError
        result[key] = item
    if len(result) != len(value):
        raise ValueError
    return result


def _has_control_or_space(value: str) -> bool:
    return any(ch.isspace() or unicodedata.category(ch).startswith("C") for ch in value)


def _has_control(value: str) -> bool:
    return any(unicodedata.category(ch).startswith("C") for ch in value)


def _valid_token(value: object) -> bool:
    return type(value) is str and _TOKEN.fullmatch(value) is not None


def _authenticate(authorization: object, registry: BoundAdmissionRegistry) -> str:
    if type(authorization) is not str or not authorization.startswith("Bearer "):
        raise BoundAdmissionError("unauthorized")
    secret = authorization[7:]
    if not secret or len(secret) > 512 or _has_control_or_space(secret) or "," in secret:
        raise BoundAdmissionError("unauthorized")
    matches = [principal for candidate, principal in registry.gateway_principals.items()
               if hmac.compare_digest(secret, candidate)]
    if len(matches) != 1:
        raise BoundAdmissionError("unauthorized")
    return matches[0]


def authenticate_bound_authorization(
    authorization: str, registry: BoundAdmissionRegistry
) -> str:
    """Authenticate one exact bounded Bearer value without touching request bytes."""
    if not isinstance(registry, BoundAdmissionRegistry):
        raise BoundAdmissionError("invalid_server_policy")
    return _authenticate(authorization, registry)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidJSON
        result[key] = value
    return result


def _reject_constant(_value):
    raise _InvalidJSON


def _parse_request(raw_body: object) -> tuple[str, str, str | None]:
    if type(raw_body) is not bytes or not raw_body or len(raw_body) > MAX_RAW_BODY_BYTES:
        raise BoundAdmissionError("invalid_request")
    try:
        text = raw_body.decode("utf-8", errors="strict")
        payload = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, _InvalidJSON, RecursionError):
        raise BoundAdmissionError("invalid_request") from None
    if type(payload) is not dict or set(payload) not in (
        _REQUIRED_BODY_KEYS, _REQUIRED_HTTP_BODY_KEYS
    ):
        raise BoundAdmissionError("invalid_request")
    if set(payload) == _REQUIRED_HTTP_BODY_KEYS:
        input_text, selector = payload["message"], payload["case_selector"]
        embedded_key = _validate_idempotency_key(payload["idempotency_key"])
    else:
        input_text, selector, embedded_key = payload["input"], payload["case"], None
    if (type(input_text) is not str or not input_text or len(input_text) > MAX_INPUT_CHARS
            or _has_control(input_text)):
        raise BoundAdmissionError("invalid_request")
    if not _valid_token(selector) or len(selector) > MAX_CASE_SELECTOR_CHARS:
        raise BoundAdmissionError("invalid_request")
    return input_text, selector, embedded_key


def _validate_idempotency_key(value: object) -> str:
    if not _valid_token(value) or len(value) > MAX_IDEMPOTENCY_KEY_CHARS:
        raise BoundAdmissionError("invalid_idempotency_key")
    return value


def _validated_case(case: BoundCase, principal: str) -> dict[str, str]:
    identity = {name: getattr(case, name) for name in _IDENTITY_FIELDS}
    if (case.principal_id != principal or not _valid_token(case.principal_id)
            or any(not _valid_token(identity[name]) for name in _IDENTITY_FIELDS[:-1])
            or type(case.context_digest) is not str
            or _DIGEST.fullmatch(case.context_digest) is None
            or type(case.context_bytes) is not bytes or not case.context_bytes
            or len(case.context_bytes) > MAX_CONTEXT_BYTES
            or type(case.skills) is not tuple or case.skills != _REQUIRED_SKILLS
            or case.tool_policy != "deny_all"
            or type(case.context_length) is not int
            or case.context_length < 65_536
            or case.context_length > 16_777_216):
        raise BoundAdmissionError("invalid_server_policy")
    if not hmac.compare_digest(sha256(case.context_bytes).hexdigest(), case.context_digest):
        raise BoundAdmissionError("invalid_server_policy")
    return identity


def prepare_bound_run(*, raw_body: bytes, authorization: str, idempotency_key: str | None,
                      registry: BoundAdmissionRegistry) -> PreparedBoundAdmission:
    """Authenticate, validate, and resolve without mutating native session state."""
    if not isinstance(registry, BoundAdmissionRegistry):
        raise BoundAdmissionError("invalid_server_policy")

    # Authentication intentionally precedes UTF-8 decoding and JSON parsing.
    principal = _authenticate(authorization, registry)
    input_text, selector, embedded_key = _parse_request(raw_body)
    idempotency_key = _validate_idempotency_key(
        embedded_key if embedded_key is not None else idempotency_key
    )
    case = registry.cases.get((principal, selector))
    if case is None:
        raise BoundAdmissionError("not_found")
    identity = _validated_case(case, principal)

    try:
        runtime = resolve_bound_runtime(
            provider=case.provider,
            model=case.model,
            api_mode=case.api_mode,
            base_url=case.base_url,
            context_length=case.context_length,
            api_key=case.api_key,
        )
    except Exception:
        raise BoundAdmissionError("runtime_unavailable") from None

    return PreparedBoundAdmission(
        principal_id=principal,
        input_text=input_text,
        idempotency_key=idempotency_key,
        binding_identity=MappingProxyType(identity),
        context_bytes=case.context_bytes,
        skills=case.skills,
        tool_policy=case.tool_policy,
        runtime=runtime,
    )


def complete_bound_run(prepared: PreparedBoundAdmission, session_db) -> BoundAdmission:
    """Create or require the native binding after external admission is durable."""
    if not isinstance(prepared, PreparedBoundAdmission):
        raise BoundAdmissionError("invalid_server_policy")

    try:
        identity = dict(prepared.binding_identity)
        session_id = session_db.create_bound_session(**identity)
        binding = session_db.require_session_binding(session_id, **identity)
    except ValueError:
        raise BoundAdmissionError("binding_conflict") from None
    except Exception:
        raise BoundAdmissionError("state_unavailable") from None
    frozen_identity = MappingProxyType({name: binding[name] for name in _IDENTITY_FIELDS})
    return BoundAdmission(
        principal_id=prepared.principal_id,
        input_text=prepared.input_text,
        idempotency_key=prepared.idempotency_key,
        session_id=session_id,
        binding_identity=frozen_identity,
        context_bytes=prepared.context_bytes,
        skills=prepared.skills,
        tool_policy=prepared.tool_policy,
        runtime=prepared.runtime,
    )


def admit_bound_run(*, raw_body: bytes, authorization: str, idempotency_key: str | None,
                    registry: BoundAdmissionRegistry, session_db) -> BoundAdmission:
    """Preserved one-call API for authenticated native bound-session admission."""
    prepared = prepare_bound_run(
        raw_body=raw_body,
        authorization=authorization,
        idempotency_key=idempotency_key,
        registry=registry,
    )
    return complete_bound_run(prepared, session_db)
