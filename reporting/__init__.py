"""Lazy public exports for safety-state tables and visualizations."""

from importlib import import_module


_EXPORTS = (
    "OptionalReportingDependencyError",
    "build_bottleneck_path_table",
    "build_heatmap_table",
    "build_method_comparison_table",
    "build_parser",
    "build_patching_table",
    "build_quality_tradeoff_table",
    "build_report_tables",
    "create_report",
    "generate_report",
    "main",
    "plot_activation_patching",
    "plot_bottleneck_path",
    "plot_method_comparison",
    "plot_quality_tradeoff",
    "plot_safety_heatmap",
    "write_csv_table",
)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("reporting.generate_report"), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
