#!/usr/bin/env python3
"""Collect structured Heat3D logs into raw and workflow summary CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


INTEGER_KEYS = {
    "nx",
    "ny",
    "nz",
    "np1",
    "np2",
    "np3",
    "mpi_ranks",
    "nsteps",
    "nref",
}


def read_cases(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def parse_result(path: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    inside = False
    block_count = 0
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if line == "HEAT3D_RESULT_BEGIN":
            if inside:
                raise ValueError(f"nested result block in {path}")
            inside = True
            block_count += 1
            continue
        if line == "HEAT3D_RESULT_END":
            inside = False
            continue
        if inside and "=" in line:
            key, value = line.split("=", 1)
            value = value.strip().replace("D", "E").replace("d", "e")
            values[key.strip()] = int(value) if key.strip() in INTEGER_KEYS else float(value)

    if inside or block_count != 1:
        raise ValueError(f"expected one complete result block in {path}, found {block_count}")
    required = INTEGER_KEYS | {"dt", "final_time", "l2_error", "total_loop_s", "btdma_s"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"missing {missing} in {path}")
    return values


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(workflow: str, log_dir: Path, cases: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    pattern = re.compile(r"^(?P<case>.+)_r(?P<repeat>\d+)\.log$")
    rows: list[dict[str, object]] = []
    for path in sorted(log_dir.glob("*.log")):
        match = pattern.match(path.name)
        if not match:
            continue
        case_id = match.group("case")
        if case_id not in cases:
            raise ValueError(f"unknown case id {case_id} in {path.name}")
        case = cases[case_id]
        if case["workflow"] != workflow:
            continue
        result = parse_result(path)
        expected = {
            "nx": int(case["n"]),
            "ny": int(case["n"]),
            "nz": int(case["n"]),
            "np1": int(case["np1"]),
            "np2": int(case["np2"]),
            "np3": int(case["np3"]),
            "mpi_ranks": int(case["gpus"]),
            "nsteps": int(case["nsteps"]),
            "nref": 1024,
        }
        for key, expected_value in expected.items():
            if result[key] != expected_value:
                raise ValueError(
                    f"{path.name}: {key}={result[key]} does not match cases.tsv ({expected_value})"
                )
        rows.append(
            {
                "case_id": case_id,
                "workflow": workflow,
                "repeat": int(match.group("repeat")),
                "mango_nodes": int(case["mango_nodes"]),
                **result,
                "log_file": path.name,
            }
        )

    expected_cases = {name for name, row in cases.items() if row["workflow"] == workflow}
    observed_cases = {str(row["case_id"]) for row in rows}
    missing_cases = sorted(expected_cases - observed_cases)
    if missing_cases:
        raise ValueError(f"missing logs for cases: {', '.join(missing_cases)}")
    return rows


def convergence_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["nx"])].append(row)

    summary: list[dict[str, object]] = []
    previous_h: float | None = None
    previous_error: float | None = None
    for n in sorted(grouped):
        group = grouped[n]
        errors = [float(row["l2_error"]) for row in group]
        h = 1.0 / (n + 1)
        error = statistics.median(errors)
        order = ""
        if previous_h is not None and previous_error is not None and error > 0.0:
            order = math.log(previous_error / error) / math.log(previous_h / h)
        summary.append(
            {
                "N": n,
                "h": h,
                "dt": float(group[0]["dt"]),
                "nsteps": int(group[0]["nsteps"]),
                "final_time": float(group[0]["final_time"]),
                "l2_error_median": error,
                "l2_error_min": min(errors),
                "l2_error_max": max(errors),
                "observed_order": order,
                "repeats": len(group),
            }
        )
        previous_h = h
        previous_error = error
    return summary


def scaling_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["mpi_ranks"])].append(row)
    if 8 not in grouped:
        raise ValueError("P=8 baseline is required")

    baseline = min(float(row["total_loop_s"]) for row in grouped[8])
    summary: list[dict[str, object]] = []
    for p in sorted(grouped):
        group = grouped[p]
        times = [float(row["total_loop_s"]) for row in group]
        errors = [float(row["l2_error"]) for row in group]
        best = min(times)
        speedup = baseline / best
        summary.append(
            {
                "P": p,
                "mango_nodes": int(group[0]["mango_nodes"]),
                "topology": f"{group[0]['np1']}x{group[0]['np2']}x{group[0]['np3']}",
                "N": int(group[0]["nx"]),
                "nsteps": int(group[0]["nsteps"]),
                "total_loop_s_min": best,
                "total_loop_s_median": statistics.median(times),
                "speedup_from_P8": speedup,
                "efficiency_percent": 100.0 * speedup / (p / 8.0),
                "l2_error_max": max(errors),
                "repeats": len(group),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", choices=("convergence", "scaling"))
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.tsv"))
    args = parser.parse_args()

    cases = read_cases(args.cases)
    raw_rows = collect(args.workflow, args.log_dir, cases)
    write_csv(args.raw, raw_rows)
    summary_rows = convergence_summary(raw_rows) if args.workflow == "convergence" else scaling_summary(raw_rows)
    write_csv(args.summary, summary_rows)
    print(f"Wrote {args.raw} and {args.summary}")


if __name__ == "__main__":
    main()
