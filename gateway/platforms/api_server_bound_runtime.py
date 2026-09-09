"""Exact lane preflight for a future bounded ``_RunLaunch`` agent factory.

Not connected to the API adapter. Inputs must come from approved server config,
never browser fields. This increment supports ONLY canonical Nous with an
explicit static key and HTTPS endpoint: that real resolver path returns before
pool selection/refresh. Other paths need their own no-recovery proof.

The caller owns isolated home/config/secret scope and must construct from this
result without generic runtime resolution, defaults, model recovery or fallback.
This is not authentication, configuration provenance or a lifecycle sandbox.
"""
from dataclasses import dataclass, field
import unicodedata
from urllib.parse import urlsplit


class BoundRuntimeError(ValueError):
    """A lane cannot be resolved exactly; messages never contain input values."""


@dataclass(frozen=True, slots=True)
class BoundRuntime:
    """In-memory construction values, not a persistence/HTTP serialization DTO."""

    provider: str
    model: str
    api_mode: str
    base_url: str
    context_length: int
    api_key: str = field(repr=False, compare=False)


def _require_literal(value: str, name: str) -> None:
    if (not isinstance(value, str) or not value or value.lower() == "auto"
            or "${" in value
            or any(ch.isspace() or unicodedata.category(ch).startswith("C") for ch in value)):
        raise BoundRuntimeError(f"Invalid bounded {name}")


def resolve_bound_runtime(*, provider: str, model: str, api_mode: str,
                          base_url: str, context_length: int,
                          api_key: str) -> BoundRuntime:
    """Resolve once, verifying the public result BEFORE any agent construction.

    No normalization: even a changed trailing slash is a mismatch. API mode is
    an expectation, not an override sent to the resolver. Nous does not return
    a model field; the explicit target remains authoritative in that case. If a
    resolver does return a model, require exact equality (including nonempty).
    No credential pool or request/header overrides may survive this preflight.
    """
    for name, value in (("provider", provider), ("model", model),
                        ("api_mode", api_mode), ("base_url", base_url), ("api_key", api_key)):
        _require_literal(value, name)
    if provider != "nous":
        raise BoundRuntimeError("Unsupported bounded provider")
    if api_mode != "chat_completions":
        raise BoundRuntimeError("Unsupported bounded API mode")
    if (type(context_length) is not int or context_length < 65_536
            or context_length > 16_777_216):
        raise BoundRuntimeError("Invalid bounded context_length")
    try:
        url = urlsplit(base_url)
        valid_url = (url.scheme == "https" and bool(url.hostname) and url.port != 0
                     and url.username is None and url.password is None
                     and "?" not in base_url and "#" not in base_url and "\\" not in base_url)
    except ValueError:
        valid_url = False
    if not valid_url:
        raise BoundRuntimeError("Invalid bounded base_url")

    from hermes_cli.auth import has_usable_secret
    from hermes_cli import runtime_provider as runtime_provider_module

    if not has_usable_secret(api_key):
        raise BoundRuntimeError("Invalid bounded api_key")
    try:
        # Do not enter resolve_runtime_provider's generic ladder. Its shared
        # model-config rung may auto-discover a configured local endpoint before
        # reaching explicit credentials. This bounded lane is already canonical
        # and validated, so invoke only Hermes's disabled-provider guard and
        # explicit-key builder with an approved minimal model config.
        runtime_provider_module._raise_if_provider_disabled(provider)
        runtime = runtime_provider_module._resolve_explicit_runtime(
            provider=provider,
            requested_provider=provider,
            model_cfg={"default": model},
            explicit_base_url=base_url,
            explicit_api_key=api_key,
            target_model=model,
        )
    except Exception:
        # Resolver/auth errors can embed credentials; do not log or chain them.
        raise BoundRuntimeError("Bounded runtime resolution failed") from None

    if not isinstance(runtime, dict):
        raise BoundRuntimeError("Invalid bounded runtime result")
    expected = {"requested_provider": provider, "provider": provider,
                "api_mode": api_mode, "base_url": base_url, "api_key": api_key}
    if any(runtime.get(name) != value for name, value in expected.items()):
        raise BoundRuntimeError("Bounded runtime lane mismatch")
    if "model" in runtime and runtime["model"] != model:
        raise BoundRuntimeError("Bounded runtime model mismatch")
    if (runtime.get("credential_pool") is not None or runtime.get("extra_headers")
            or runtime.get("request_overrides")):
        raise BoundRuntimeError("Unsupported bounded runtime overrides")
    return BoundRuntime(provider=runtime["provider"], model=model,
                        api_mode=runtime["api_mode"], base_url=runtime["base_url"],
                        context_length=context_length,
                        api_key=runtime["api_key"])
