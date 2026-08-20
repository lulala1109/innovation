"""Archived cross-model adapters outside the default experiment factory.

These adapters are retained for future validation only. They have separate
runtime constraints and have not been validated against the current
safety-state forward-attack contract.
"""

from importlib import import_module


_EXPORT_MODULES = {
    "PhiModel": "models.optional.phi",
    "VoxtralModel": "models.optional.voxtral",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
