"""Waveform-bounded attack implementations.

Imports are lazy so command-line help does not load model stacks.
"""

from importlib import import_module


_EXPORT_MODULES = {
    "AttackResult": "attacks.base",
    "BaseWavAttacker": "attacks.base",
    "PGDAttacker": "attacks.pgd",
    "LayerAdaptivePGDAttacker": "attacks.layer_adaptive_pgd",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
