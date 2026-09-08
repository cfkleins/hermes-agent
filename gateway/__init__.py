"""Hermes Gateway public types, loaded only when requested.

Keep package import inert: ``python -m gateway.bounded_service`` must set its
explicit home before any runtime/config module takes an import-time snapshot.
"""
from importlib import import_module

_EXPORTS = {
    'GatewayConfig': '.config', 'PlatformConfig': '.config',
    'HomeChannel': '.config', 'load_gateway_config': '.config',
    'SessionContext': '.session', 'SessionStore': '.session',
    'build_session_context_prompt': '.session',
    'DeliveryRouter': '.delivery', 'DeliveryTarget': '.delivery',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
