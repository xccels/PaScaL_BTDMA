#!/usr/bin/env python3
"""Collect rank-max five-phase timing records from Mango stdout logs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


INTEGER_FIELDS = [
    "repeat",
    "p_total",
    "p_dir",
    "reduced_rows",
    "m",
    "N",
    "nsys",
    "nlocal_min",
    "nlocal_max",
]
FLOAT_FIELDS = [
    "step1_s",
    "step2_s",
    "step3_s",
    "step4_s",
    "step5_s",
    "total_s",
    "step3_fraction",
    "phase_sum_s",
    "phase_sum_over_total",
]
RAW_FIELDS = [
    "experiment_id",
    "cell_id",
    "source_log",
    "cartesian_topology",
    *INTEGER_FIELDS,
    *FLOAT_FIELDS,
    "timer_scope",
    "synchronization",
    "rank_aggregation",
]


def parse_log(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("PHASE_RAW,"):
            continue
        values = [item.strip() for item in line.split(",")[1:]]
        expected = len(INTEGER_FIELDS) + len(FLOAT_FIELDS)
        if len(values) != expected:
            raise ValueError(
                f"{path}:{line_number}: expected {expected} PHASE_RAW values, "
                f"found {len(values)}"
            )
        row: dict[str, object] = {
            "experiment_id": "EXP-PHASE-001",
            "source_log": str(path),
            "timer_scope": "solver_call_only",
            "synchronization": "cuda_sync_at_phase_boundaries;mpi_barrier_before_total",
            "rank_aggregation": "MPI_MAX_per_phase_and_total_per_repeat",
        }
        for key, value in zip(INTEGER_FIELDS, values[: len(INTEGER_FIELDS)]):
            row[key] = int(value)
        for key, value in zip(FLOAT_FIELDS, values[len(INTEGER_FIELDS) :]):
            row[key] = float(value)
        row["cell_id"] = f"PHASE-GPU-{int(row['p_total']):02d}"
        row["cartesian_topology"] = f"1x1x{int(row['p_dir'])}"
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no PHASE_RAW records found")
    return rows


def validate(rows: list[dict[str, object]]) -> None:
    seen: set[tuple[int, int]] = set()
    for row in rows:
        p_total = int(row["p_total"])
        repeat = int(row["repeat"])
        key = (p_total, repeat)
        if key in seen:
            raise ValueError(f"duplicate record for P={p_total}, repeat={repeat}")
        seen.add(key)
        if p_total not in {2, 4, 8}:
            raise ValueError(f"unexpected P={p_total}; fixed cases are 2, 4, and 8")
        if int(row["m"]) != 8 or int(row["N"]) != 2048 or int(row["nsys"]) != 16384:
            raise ValueError("a record does not match fixed m=8, N=2048, nsys=16384")
        total = float(row["total_s"])
        if total <= 0.0:
            raise ValueError(f"non-positive total time for P={p_total}, repeat={repeat}")
        expected_fraction = float(row["step3_s"]) / total
        if abs(expected_fraction - float(row["step3_fraction"])) > 1.0e-10:
            raise ValueError(f"inconsistent Step 3 fraction for P={p_total}, repeat={repeat}")
    observed_process_counts = {int(row["p_total"]) for row in rows}
    if observed_process_counts != {2, 4, 8}:
        raise ValueError(
            "the complete fixed experiment requires records for P=2, 4, and 8; "
            f"found {sorted(observed_process_counts)}"
        )


def write_raw(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    summary_fields = [
        "experiment_id",
        "cell_id",
        "cartesian_topology",
        "p_total",
        "p_dir",
        "reduced_rows",
        "m",
        "N",
        "nsys",
        "n_repeats",
        "representative_repeat",
        "aggregation_rule",
        *FLOAT_FIELDS,
        "total_median_s",
        "total_mean_s",
        "total_pstdev_s",
    ]
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["p_total"]), []).append(row)

    summaries: list[dict[str, object]] = []
    for p_total in sorted(grouped):
        group = grouped[p_total]
        representative = min(group, key=lambda item: float(item["total_s"]))
        totals = [float(item["total_s"]) for item in group]
        summary: dict[str, object] = {
            "experiment_id": representative["experiment_id"],
            "cell_id": representative["cell_id"],
            "cartesian_topology": representative["cartesian_topology"],
            "p_total": p_total,
            "p_dir": representative["p_dir"],
            "reduced_rows": representative["reduced_rows"],
            "m": representative["m"],
            "N": representative["N"],
            "nsys": representative["nsys"],
            "n_repeats": len(group),
            "representative_repeat": representative["repeat"],
            "aggregation_rule": "minimum rank-max total; same-repeat phase vector",
            "total_median_s": statistics.median(totals),
            "total_mean_s": statistics.fmean(totals),
            "total_pstdev_s": statistics.pstdev(totals),
        }
        for field in FLOAT_FIELDS:
            summary[field] = representative[field]
        summaries.append(summary)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for log in args.logs:
        rows.extend(parse_log(log))
    rows.sort(key=lambda item: (int(item["p_total"]), int(item["repeat"])))
    validate(rows)
    write_raw(args.raw, rows)
    write_summary(args.summary, rows)
    print(f"Collected {len(rows)} repeats into {args.raw}")
    print(f"Wrote representative summaries to {args.summary}")


if __name__ == "__main__":
    main()
