#!/usr/bin/env python3
"""Summarize EXP-CORR-001 CSV files using only the Python standard library."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


EXPECTED_SOLVERS = {
    "sequential-cpu-bta",
    "cpu-pascal-btdma",
    "gpu-pascal-btdma",
}


def read_accuracy_rows(raw_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(raw_dir.glob("accuracy_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append({key: value.strip() for key, value in row.items()})
    return rows


def finite_float(value: str) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else math.nan


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: summarize_results.py RAW_DIR OUTPUT_DIR", file=sys.stderr)
        return 2

    raw_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_accuracy_rows(raw_dir)
    if not rows:
        print(f"No accuracy_*.csv files found under {raw_dir}", file=sys.stderr)
        return 1

    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    coverage: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = (row["m"], row["family"], row["target_K"], row["solver"])
        grouped[key].append(row)
        coverage[row["input_id"]].add(row["solver"])

    summary_path = output_dir / "accuracy_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "m", "family", "target_K", "solver", "system_count",
            "residual_median", "residual_max", "error_median", "error_max",
            "failure_count", "nonfinite_count", "status_available",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(grouped, key=lambda item: (int(item[0]), item[1], float(item[2]), item[3])):
            group = grouped[key]
            residuals = [finite_float(row["residual_norm"]) for row in group]
            errors = [finite_float(row["relative_solution_error"]) for row in group]
            residuals_finite = [value for value in residuals if math.isfinite(value)]
            errors_finite = [value for value in errors if math.isfinite(value)]
            writer.writerow({
                "m": key[0],
                "family": key[1],
                "target_K": key[2],
                "solver": key[3],
                "system_count": len(group),
                "residual_median": statistics.median(residuals_finite) if residuals_finite else "nan",
                "residual_max": max(residuals_finite) if residuals_finite else "nan",
                "error_median": statistics.median(errors_finite) if errors_finite else "nan",
                "error_max": max(errors_finite) if errors_finite else "nan",
                "failure_count": sum(row["failed"] == "yes" for row in group),
                "nonfinite_count": sum(row["nonfinite"] == "yes" for row in group),
                "status_available": "yes" if all(row["status_available"] == "yes" for row in group) else "no",
            })

    missing = {
        input_id: sorted(EXPECTED_SOLVERS - solvers)
        for input_id, solvers in coverage.items()
        if solvers != EXPECTED_SOLVERS
    }
    coverage_path = output_dir / "paired_input_coverage.txt"
    with coverage_path.open("w", encoding="utf-8") as handle:
        handle.write(f"unique_inputs={len(coverage)}\n")
        handle.write(f"fully_paired_inputs={len(coverage) - len(missing)}\n")
        handle.write(f"incomplete_inputs={len(missing)}\n")
        for input_id, solvers in sorted(missing.items()):
            handle.write(f"{input_id}: missing={','.join(solvers)}\n")

    print(summary_path)
    print(coverage_path)
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
