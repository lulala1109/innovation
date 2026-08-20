"""Behavioral and perceptual evaluation for generic safety runs.

Every public symbol is imported lazily so command-line help and lightweight
metrics remain usable without initializing optional judges.
"""

from importlib import import_module


_EXPORT_MODULES = {
    "JailbreakEvalEvaluator": "evaluation.behavior",
    "LlamaGuardEvaluator": "evaluation.behavior",
    "StrongRejectEvaluator": "evaluation.behavior",
    "StrongRejectScore": "evaluation.behavior",
    "evaluate_response": "evaluation.behavior",
    "SafetyEvaluationCase": "evaluation.evaluate_safety_runs",
    "calculate_summary": "evaluation.evaluate_safety_runs",
    "evaluate_safety_runs": "evaluation.evaluate_safety_runs",
    "load_safety_cases": "evaluation.evaluate_safety_runs",
    "OptionalMetricDependencyError": "evaluation.perceptual",
    "perturbation_linf": "evaluation.perceptual",
    "perturbation_l2": "evaluation.perceptual",
    "perturbation_rms": "evaluation.perceptual",
    "signal_to_noise_ratio_db": "evaluation.perceptual",
    "signal_to_perturbation_ratio_db": "evaluation.perceptual",
    "snr_db": "evaluation.perceptual",
    "spr_db": "evaluation.perceptual",
    "compute_perturbation_metrics": "evaluation.perceptual",
    "compute_pesq": "evaluation.perceptual",
    "compute_stoi": "evaluation.perceptual",
    "compute_perceptual_metrics": "evaluation.perceptual",
    "AudioBudgetViolationError": "evaluation.evaluate_audio_quality",
    "SPR_DEFINITION": "evaluation.evaluate_audio_quality",
    "discover_run_files": "evaluation.evaluate_audio_quality",
    "evaluate_audio_quality": "evaluation.evaluate_audio_quality",
    "evaluate_run_audio_quality": "evaluation.evaluate_audio_quality",
    "evaluate_audio_quality_run": "evaluation.evaluate_audio_quality",
    "verify_linf_budget": "evaluation.evaluate_audio_quality",
    "edit_distance": "evaluation.task_utility",
    "word_error_rate": "evaluation.task_utility",
    "character_error_rate": "evaluation.task_utility",
    "wer": "evaluation.task_utility",
    "cer": "evaluation.task_utility",
    "compute_wer": "evaluation.task_utility",
    "compute_cer": "evaluation.task_utility",
    "compute_transcript_metrics": "evaluation.task_utility",
    "evaluate_task_utility": "evaluation.task_utility",
    "evaluate_run_task_utility": "evaluation.task_utility",
    "evaluate_task_utility_batch": "evaluation.task_utility",
    "batch_evaluate_task_utility": "evaluation.task_utility",
    "compare_paired_methods": "evaluation.paired_statistics",
    "paired_method_statistics": "evaluation.paired_statistics",
    "paired_bootstrap_ci": "evaluation.paired_statistics",
    "paired_cohens_dz": "evaluation.paired_statistics",
    "paired_permutation_test": "evaluation.paired_statistics",
    "paired_statistics": "evaluation.paired_statistics",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
