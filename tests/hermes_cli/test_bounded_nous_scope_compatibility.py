"""DEMO-only scope interoperability without expanding the required permission."""
import base64
import json

import httpx
import pytest

from tests.hermes_cli.test_bounded_nous_credentials import fx, payload, enroll, transport

pytestmark = pytest.mark.windows_only

SCOPES = [
    ("inference:invoke profile", "inference:invoke"),
    ("inference:invoke", "openid inference:invoke"),
    ("inference:invoke profile", "openid inference:invoke"),
]


def scoped_payload(jwt_scope, response_scope):
    data = payload()
    header, body, signature = data["access_token"].split(".")
    claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    claims["scope"] = jwt_scope
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    data["access_token"] = ".".join((header, body, signature))
    data["scope"] = response_scope
    return data


@pytest.mark.parametrize("jwt_scope,response_scope", SCOPES)
def test_enrollment_and_cached_read_require_invoke_not_identical_scope_lists(fx, monkeypatch, jwt_scope, response_scope):
    m, kw = fx
    data = scoped_payload(jwt_scope, response_scope)
    monkeypatch.setattr(m, "_client_factory", lambda **k: pytest.fail("cached read must not authenticate"))
    m.enroll_nous_credentials(source_path=kw["source_path"], token_payload=data)
    raw = kw["source_path"].read_bytes()
    result = m.isolated_nous_credentials(**kw)
    assert result.access_token == data["access_token"]
    assert kw["source_path"].read_bytes() == raw
    saved = m._read(kw["source_path"])
    assert saved["state"]["scope"] == response_scope
    assert saved["state"]["access_token"] == data["access_token"]
    assert not kw["marker_path"].exists()


@pytest.mark.parametrize("jwt_scope,response_scope", SCOPES)
def test_rotation_and_reopen_preserve_different_scope_lists(fx, monkeypatch, jwt_scope, response_scope):
    m, kw = fx
    enroll(m, kw, ttl=100)
    data = scoped_payload(jwt_scope, response_scope)
    data["refresh_token"] = "DEMO-rotated-refresh"
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=data)
    transport(m, monkeypatch, respond)
    result = m.isolated_nous_credentials(**kw)
    assert result.access_token == data["access_token"]
    assert m.isolated_nous_credentials(**kw).access_token == data["access_token"]
    assert len(requests) == 1
    assert m._read(kw["source_path"])["state"]["scope"] == response_scope
    assert not kw["marker_path"].exists()


@pytest.mark.parametrize("stage", ["enrollment", "cached", "rotation"])
@pytest.mark.parametrize("jwt_scope,response_scope", [("profile", "inference:invoke"), ("inference:invoke", "profile")])
def test_missing_invoke_in_either_source_remains_blocked(fx, monkeypatch, stage, jwt_scope, response_scope):
    m, kw = fx
    data = scoped_payload(jwt_scope, response_scope)
    requests = []
    if stage == "enrollment":
        with pytest.raises(m.CredentialBlocked, match="missing_invoke_scope"):
            m.enroll_nous_credentials(source_path=kw["source_path"], token_payload=data)
        assert not kw["source_path"].exists()
        return
    enroll(m, kw, ttl=100 if stage == "rotation" else 600)
    if stage == "cached":
        root = m._read(kw["source_path"])
        root["state"].update(access_token=data["access_token"], scope=response_scope)
        m._persist(kw["source_path"], root)
        monkeypatch.setattr(m, "_client_factory", lambda **k: pytest.fail("invalid cache must not authenticate"))
    else:
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=data)
        transport(m, monkeypatch, respond)
    with pytest.raises(m.CredentialBlocked, match="missing_invoke_scope"):
        m.isolated_nous_credentials(**kw)
    if stage == "rotation":
        assert len(requests) == 1
        assert kw["marker_path"].exists()
        with pytest.raises(m.CredentialBlocked, match="uncertain"):
            m.isolated_nous_credentials(**kw)
        assert len(requests) == 1
