"""Import-safe, one-way process isolation for the dedicated bounded service.

Not an environment feature flag: only the validated bootstrap can opt in,
before importing any runtime module. This is not a filesystem sandbox.
"""
import sys

_active = False


def activate():
    global _active
    if any(name in sys.modules for name in ('run_agent', 'model_tools', 'hermes_cli.config')):
        raise RuntimeError('Bounded service requires a fresh interpreter')
    _active = True


def active():
    return _active
