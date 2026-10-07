"""Stage-2 causal refusal-bottleneck experiments.

The package is intentionally separate from the frozen Stage-1/RQ1 pipeline.
It may read validated RQ1 artifacts, but it never writes into their namespace.
"""

from importlib import import_module


_EXPORTS = {
    "BehaviorLabel": ("rq2.artifacts", "BehaviorLabel"),
    "FrozenRQ1Bundle": ("rq2.rq1_bundle", "FrozenRQ1Bundle"),
    "InterventionAudit": ("rq2.interventions", "InterventionAudit"),
    "InterventionSpec": ("rq2.interventions", "InterventionSpec"),
    "RQ2Config": ("rq2.config", "RQ2Config"),
    "RQ2ConfigError": ("rq2.config", "RQ2ConfigError"),
    "TrialKey": ("rq2.artifacts", "TrialKey"),
    "load_rq2_config": ("rq2.config", "load_rq2_config"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(module_name), attribute)


__all__ = sorted(_EXPORTS)
