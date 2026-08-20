"""Data preparation APIs for safety-state experiments."""

from importlib import import_module



_EXPORT_MODULES = {
    "CONFIDENCE_Z": "data.sampling",
    "balanced_sample": "data.sampling",
    "exclude_rows": "data.sampling",
    "fpc_sample_size": "data.sampling",
    "proportional_allocation": "data.sampling",
    "stratified_sample": "data.sampling",
    "MANIFEST_COLUMNS": "data.build_safety_pairs",
    "STATE_AUDIO_COLUMNS": "data.build_safety_pairs",
    "ManifestValidationError": "data.build_safety_pairs",
    "build_manifest": "data.build_safety_pairs",
    "pair_prompt_tables": "data.build_safety_pairs",
    "select_state_sets": "data.build_safety_pairs",
    "stable_pair_id": "data.build_safety_pairs",
    "validate_manifest": "data.build_safety_pairs",
    "DatasetSchemaError": "data.datasets",
    "assign_stratum": "data.datasets",
    "load_advbench": "data.datasets",
    "load_jbb": "data.datasets",
    "normalize_prompt_table": "data.datasets",
    "read_table": "data.datasets",
    "write_table": "data.datasets",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "CONFIDENCE_Z",
    "MANIFEST_COLUMNS",
    "STATE_AUDIO_COLUMNS",
    "DatasetSchemaError",
    "ManifestValidationError",
    "assign_stratum",
    "balanced_sample",
    "build_manifest",
    "exclude_rows",
    "fpc_sample_size",
    "load_advbench",
    "load_jbb",
    "normalize_prompt_table",
    "pair_prompt_tables",
    "proportional_allocation",
    "read_table",
    "select_state_sets",
    "stable_pair_id",
    "stratified_sample",
    "validate_manifest",
    "write_table",
]
