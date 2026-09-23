#!/usr/bin/env python3
"""Verify the generated artifact contract without requiring input data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXPECTED_FILES = (
    "00_provenance.txt",
    "figure_roc_curves.png",
    "roc_auc_analysis.csv",
    "mortality_by_quartile_test.txt",
    "mortality_by_quartile.csv",
    "psili_threshold_counts.json",
    "breathing_patterns.csv",
    "etiology_breakdown.csv",
    "comorbidity_breakdown.csv",
    "table1_full_categories.csv",
    "posthoc_power.txt",
    "figure_p01_vs_rass.png",
    "p01_vs_rass_stats.txt",
    "p01_by_rass.csv",
    "figure_qq_plots.png",
    "figure_residual_diagnostics.png",
    "residual_diagnostics.csv",
    "figure_nonlinear_fits.png",
    "spline_model_comparison.csv",
    "de_tfdi_correlation.txt",
    "vif_diagnostics.csv",
    "specification_robustness_summary.txt",
    "specification_robustness.csv",
    "table4_rebuilt_sqrt.md",
    "figure_table4_backtransformed.png",
    "table4_backtransformed_predictions.csv",
    "table4_diagnostics.txt",
    "de_tfdi_nested_models.csv",
    "table4_rebuilt_sqrt.csv",
    "competing_models_fit.csv",
    "competing_models_comparison.csv",
    "oxygenation_integrity_notes.txt",
    "oxygenation_integrity.csv",
    "sample_size_reconciliation.json",
    "missingness_report.csv",
)

CSV_COLUMNS = {
    "table4_rebuilt_sqrt.csv": {"panel", "predictor", "n", "beta", "ci_lower", "ci_upper", "p_value", "significant"},
    "competing_models_comparison.csv": {"model", "n", "predictor", "estimate", "ci_lower", "ci_upper", "p_value", "significant_at_0.05"},
    "vif_diagnostics.csv": {"model", "predictor", "VIF", "tolerance", "interpretation"},
    "roc_auc_analysis.csv": {"predictor", "n", "orientation", "AUC", "optimal_cutoff_Youden", "sensitivity_at_cutoff", "specificity_at_cutoff"},
}
JSON_KEYS = {
    "sample_size_reconciliation.json": {
        "n_enrolled_rows_in_datafile", "n_with_P0_1", "n_complete_model_A_with_lung_injury",
        "n_complete_model_B_without_lung_injury", "n_missing_DE", "n_missing_lung_injury",
    },
    "psili_threshold_counts.json": {
        "n_above_1.5", "pct_above_1.5", "n_above_2.0", "pct_above_2.0",
        "n_above_3.5", "pct_above_3.5", "actual_above_3.5",
    },
}


def verify(output_dir: Path) -> int:
    failures: list[str] = []
    if not output_dir.is_dir():
        print(f"error: output directory not found: {output_dir}", file=sys.stderr)
        return 1

    paths: dict[str, Path] = {}
    for name in EXPECTED_FILES:
        path = output_dir / name
        paths[name] = path
        if not path.is_file():
            failures.append(f"{name}: missing")
        elif path.stat().st_size == 0:
            failures.append(f"{name}: empty")

    for name, columns in CSV_COLUMNS.items():
        path = paths[name]
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            frame = pd.read_csv(path)
            missing = sorted(columns.difference(frame.columns))
            if missing:
                failures.append(f"{name}: missing columns {', '.join(missing)}")
        except Exception as exc:
            failures.append(f"{name}: invalid CSV ({exc})")

    for name, keys in JSON_KEYS.items():
        path = paths[name]
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                failures.append(f"{name}: expected an object")
            else:
                missing = sorted(keys.difference(value))
                if missing:
                    failures.append(f"{name}: missing keys {', '.join(missing)}")
        except Exception as exc:
            failures.append(f"{name}: invalid JSON ({exc})")

    for path in output_dir.iterdir():
        if path.suffix.lower() == ".csv" and path.stat().st_size:
            try:
                pd.read_csv(path)
            except Exception as exc:
                failures.append(f"{path.name}: invalid CSV ({exc})")
        elif path.suffix.lower() == ".json" and path.stat().st_size:
            try:
                json.loads(path.read_text())
            except Exception as exc:
                failures.append(f"{path.name}: invalid JSON ({exc})")

    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return 1
    print(f"Verified {len(EXPECTED_FILES)} artifacts in {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return verify(args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
