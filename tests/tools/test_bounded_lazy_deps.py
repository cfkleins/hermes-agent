"""Bounded bootstrap may never install optional dependencies at runtime."""
import pytest

import hermes_bounded_bootstrap as bootstrap
from tools import lazy_deps


def test_bounded_ensure_stops_before_discovery_or_configuration(monkeypatch):
    monkeypatch.setattr(bootstrap, '_active', True)
    calls = []
    monkeypatch.setattr(lazy_deps, 'feature_missing', lambda feature: calls.append(feature) or ())
    with pytest.raises(lazy_deps.FeatureUnavailable, match='bounded'):
        lazy_deps.ensure('provider.bedrock', prompt=False)
    assert calls == []


def test_bounded_install_specs_is_denied_without_installer(monkeypatch):
    monkeypatch.setattr(bootstrap, '_active', True)
    calls = []
    monkeypatch.setattr(lazy_deps, '_allow_lazy_installs', lambda: True)
    monkeypatch.setattr(lazy_deps, '_venv_pip_install', lambda *a, **kw: calls.append(a) or lazy_deps._InstallResult(True, '', ''))
    result = lazy_deps.install_specs(['boto3==1.34.59'])
    assert result.blocked and not result.ok
    assert calls == []


def test_ordinary_ensure_retains_already_installed_behavior(monkeypatch):
    monkeypatch.setattr(bootstrap, '_active', False)
    calls = []
    monkeypatch.setattr(lazy_deps, 'feature_missing', lambda feature: calls.append(feature) or ())
    lazy_deps.ensure('provider.bedrock', prompt=False)
    assert calls == ['provider.bedrock']
