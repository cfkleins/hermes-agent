"""Offline configured Nous lane resolution; no factory or API transport is wired.

Real resolver/config imports happen after isolation. Only the resolver's returned
metadata/error is injected for adversarial contract cases; success and protocol
mismatch exercise actual resolution. The local launch witness is NOT a factory.
"""
import importlib
import json
import os
from pathlib import Path
import socket
import traceback
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import yaml


MODEL = "openai/gpt-6-astra"
BASE_URL = "https://bounded-fixture.invalid/v1"
CONTEXT_LENGTH = 1_050_000
SYNTHETIC_KEY = "synthetic-bounded-key-not-a-credential"


@pytest.fixture
def offline(tmp_path, monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
                 "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        folder = tmp_path / name.lower()
        folder.mkdir()
        monkeypatch.setenv(name, str(folder))
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("network")
        pytest.fail("Unexpected network access in offline lane regression")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setenv("BOUNDED_FIXTURE_KEY", SYNTHETIC_KEY)
    config = {
        # An unrelated stored/default model must NOT replace the approved target.
        "model": {"provider": "nous", "default": "anthropic/fixture-default",
                  "base_url": BASE_URL},
        "nous": {"anthropic_wire": "native"},
        "bounded_fixture": {"provider": "nous", "model": MODEL,
                            "api_mode": "chat_completions", "base_url": BASE_URL,
                            "api_key": "${BOUNDED_FIXTURE_KEY}",
                            "context_length": CONTEXT_LENGTH},
        "fallback_model": {"provider": "openrouter", "model": "fixture-fallback"},
        "model_aliases": {MODEL: {"provider": "custom", "model": "fixture-alias",
                                  "base_url": "https://other.invalid/v1"}},
        "plugins": {"enabled": []},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    # The explicit-key path may READ local Nous state, but must never select or
    # refresh this deliberately unusable pool or borrow another provider's key.
    (home / "auth.json").write_text(json.dumps({
        "providers": {"nous": {}},
        "credential_pool": {"nous": [{"id": "fixture-pool", "provider": "nous",
            "api_key": "synthetic-conflicting-pool-key",
            "base_url": "https://pool-conflict.invalid/v1"}]},
    }), encoding="utf-8")
    from hermes_cli import config as config_module
    from hermes_cli import runtime_provider as rp

    def forbidden_pool(*args, **kwargs):
        pytest.fail("Explicit bounded lane must not load/select a credential pool")

    monkeypatch.setattr(rp, "load_pool", forbidden_pool)
    loaded = config_module.load_config()
    assert loaded["bounded_fixture"]["api_key"] == SYNTHETIC_KEY
    assert loaded["model"]["default"] != MODEL
    assert Path(rp.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
    yield SimpleNamespace(home=home, lane=loaded["bounded_fixture"], rp=rp,
                          config=config, calls=[], attempts=attempts)
    assert attempts == []


def bound_module():
    # An assertion failure (not a collection error) is the initial RED witness.
    name = "gateway.platforms.api_server_bound_runtime"
    assert importlib.util.find_spec(name) is not None, "bounded runtime resolver is missing"
    return importlib.import_module(name)


def record_resolver(offline, monkeypatch, *, patch_result=None, error=None):
    real = offline.rp._resolve_explicit_runtime

    def record(**kwargs):
        # Record routing only; never serialize credentials or approved model config.
        offline.calls.append({k: v for k, v in kwargs.items()
                              if k not in {"explicit_api_key", "model_cfg"}})
        if error is not None:
            raise error
        result = real(**kwargs)
        if patch_result is not None:
            result.update(patch_result)
        return result

    monkeypatch.setattr(offline.rp, "_resolve_explicit_runtime", record)


def launch_witness(module, lane, constructed):
    resolved = module.resolve_bound_runtime(**lane)
    # Deliberately stop at the construction boundary: no real lifecycle, SDK,
    # history or inference is needed to witness rejection BEFORE this point.
    constructed.append(resolved)
    return resolved


def test_real_configured_lane_is_exact_and_repr_safe(offline, monkeypatch, caplog):
    module = bound_module()
    record_resolver(offline, monkeypatch)
    constructed = []
    resolved = launch_witness(module, offline.lane, constructed)
    assert len(constructed) == 1
    assert (resolved.provider, resolved.model, resolved.api_mode, resolved.base_url) == (
        offline.lane["provider"], MODEL, "chat_completions", BASE_URL)
    assert resolved.api_key == SYNTHETIC_KEY
    assert resolved.context_length == CONTEXT_LENGTH
    assert SYNTHETIC_KEY not in repr(resolved)
    assert SYNTHETIC_KEY not in caplog.text
    assert offline.calls == [{"provider": "nous", "requested_provider": "nous",
                              "explicit_base_url": BASE_URL,
                              "target_model": MODEL}]
    print("OFFLINE_IMPORTS", offline.rp.__file__, module.__file__)
    print("OFFLINE_LANE", resolved.provider, resolved.model, resolved.api_mode,
          resolved.base_url, "resolver_calls", len(offline.calls),
          "construction_boundary_reached", len(constructed),
          "network_attempts", len(offline.attempts))
    with pytest.raises(FrozenInstanceError):
        resolved.model = "fixture-replacement"
    with pytest.raises(TypeError):
        json.dumps(resolved)


@pytest.mark.parametrize("field,value", [
    ("provider", ""), ("provider", None), ("provider", "auto"),
    ("provider", "nous\n"), ("provider", "Nous"), ("provider", "nous-portal"),
    ("provider", "openai-codex"), ("provider", "custom:fixture"),
    ("model", ""), ("model", None), ("model", 123), ("model", "auto"),
    ("model", " model"), ("model", "model\x00"), ("model", "model\x7f"),
    ("model", "model\u0085"), ("model", "model\u200b"),
    ("api_mode", ""), ("api_mode", "auto"), ("api_mode", "responses"),
    ("api_mode", "codex_responses"), ("api_mode", "anthropic_messages"),
    ("api_mode", "chat_completions\r"),
    ("base_url", ""), ("base_url", None), ("base_url", "auto"),
    ("base_url", "http://bounded-fixture.invalid/v1"),
    ("base_url", "https://user:secret@bounded-fixture.invalid/v1"),
    ("base_url", "https://bounded-fixture.invalid/v1?key=secret"),
    ("base_url", "https://bounded-fixture.invalid/v1#secret"),
    ("base_url", "https://bounded-fixture.invalid:bad/v1"),
    ("base_url", "https:///v1"), ("base_url", BASE_URL + "\n"),
    ("api_key", ""), ("api_key", None), ("api_key", "placeholder"),
    ("api_key", "${MISSING_FIXTURE_KEY}"), ("api_key", "bad\nkey"),
])
def test_invalid_lane_fails_before_resolution_or_construction(offline, monkeypatch, field, value):
    module = bound_module()
    record_resolver(offline, monkeypatch)
    constructed = []
    with pytest.raises(module.BoundRuntimeError):
        launch_witness(module, {**offline.lane, field: value}, constructed)
    assert offline.calls == []
    assert constructed == []


@pytest.mark.parametrize("patch_result", [
    {"provider": "openrouter"}, {"provider": "nous "},
    {"requested_provider": "auto"}, {"requested_provider": None},
    {"model": "fixture-wrong-model"}, {"model": ""}, {"model": None},
    {"api_mode": "codex_responses"}, {"api_mode": "anthropic_messages"},
    {"base_url": "https://other.invalid/v1"}, {"base_url": BASE_URL + "/"},
    {"api_key": "synthetic-other-key"}, {"api_key": ""},
    {"credential_pool": object()},
    {"extra_headers": {"Authorization": "synthetic-extra-secret"}},
    {"request_overrides": {"extra_body": {"model": "fixture-other-model"}}},
])
def test_resolver_drift_fails_before_construction_without_retry(offline, monkeypatch, patch_result):
    module = bound_module()
    record_resolver(offline, monkeypatch, patch_result=patch_result)
    constructed = []
    with pytest.raises(module.BoundRuntimeError):
        launch_witness(module, offline.lane, constructed)
    assert len(offline.calls) == 1
    assert offline.calls[0]["target_model"] == MODEL
    assert constructed == []


def test_real_model_protocol_conflict_is_not_forced_to_chat(offline, monkeypatch):
    module = bound_module()
    record_resolver(offline, monkeypatch)
    constructed = []
    lane = {**offline.lane, "model": "anthropic/fixture-target"}
    with pytest.raises(module.BoundRuntimeError):
        launch_witness(module, lane, constructed)
    assert len(offline.calls) == 1
    assert offline.calls[0]["target_model"] == lane["model"]
    assert constructed == []


@pytest.mark.parametrize("kind", ["auth", "resolution"])
def test_resolution_error_has_no_secret_or_recovery(offline, monkeypatch, kind, caplog):
    module = bound_module()
    error_type = offline.rp.AuthError if kind == "auth" else RuntimeError
    record_resolver(offline, monkeypatch, error=error_type(SYNTHETIC_KEY))
    constructed = []
    with pytest.raises(module.BoundRuntimeError) as caught:
        launch_witness(module, offline.lane, constructed)
    assert len(offline.calls) == 1
    assert constructed == []
    assert SYNTHETIC_KEY not in "".join(traceback.format_exception(caught.value))
    assert SYNTHETIC_KEY not in caplog.text


def test_real_disabled_provider_does_not_try_configured_fallback(offline, monkeypatch):
    module = bound_module()
    offline.config["providers"] = {"nous": {"enabled": False}}
    (offline.home / "config.yaml").write_text(yaml.safe_dump(offline.config), encoding="utf-8")
    record_resolver(offline, monkeypatch)
    constructed = []
    with pytest.raises(module.BoundRuntimeError):
        launch_witness(module, offline.lane, constructed)
    assert offline.calls == []
    assert constructed == []


def test_explicit_lane_never_enters_ambient_local_model_discovery(offline, monkeypatch):
    module = bound_module()
    offline.config["model"] = {
        "provider": "custom",
        "default": "",
        "base_url": "http://127.0.0.1:9/v1",
    }
    (offline.home / "config.yaml").write_text(yaml.safe_dump(offline.config), encoding="utf-8")

    import requests
    attempted = []

    def forbidden_send(_session, request, **_kwargs):
        attempted.append((request.method, request.url))
        raise AssertionError("ambient model discovery attempted HTTP")

    monkeypatch.setattr(requests.sessions.Session, "send", forbidden_send)
    resolved = module.resolve_bound_runtime(**offline.lane)

    assert resolved.model == MODEL
    assert attempted == []
