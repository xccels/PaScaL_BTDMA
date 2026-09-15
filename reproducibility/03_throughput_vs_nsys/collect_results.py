#!/usr/bin/env python3
"""Extract throughput benchmark records and make manuscript-panel CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


FIELDS = [
    "method",
    "backend",
    "precision",
    "logical_P",
    "mpi_ranks",
    "N_global",
    "Nlocal_min",
    "Nlocal_max",
    "nsys",
    "m",
    "warmups",
    "repeats",
    "repeat",
    "total_s",
    "compute_s",
    "communication_s",
    "phase1_s",
    "phase2_s",
    "phase3_s",
    "phase4_s",
    "phase5_s",
]

INTEGER_FIELDS = {
    "logical_P",
    "mpi_ranks",
    "N_global",
    "Nlocal_min",
    "Nlocal_max",
    "nsys",
    "m",
    "warmups",
    "repeats",
    "repeat",
}
TIMING_FIELDS = [
    "total_s",
    "compute_s",
    "communication_s",
    "phase1_s",
    "phase2_s",
    "phase3_s",
    "phase4_s",
    "phase5_s",
]
GROUP_FIELDS = [field for field in FIELDS if field not in {"repeat", *TIMING_FIELDS}]
SUMMARY_FIELDS = [
    *GROUP_FIELDS,
    "samples",
    "selected_repeat",
    "total_s",
    "compute_s",
    "communication_s",
    "throughput_work_per_s",
    "throughput_Gwork_per_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="benchmark log files")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_records(paths: Sequence[Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    expected_columns = len(FIELDS) + 1
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.startswith("THROUGHPUT_RESULT,"):
                    continue
                values = next(csv.reader([line]))
                if len(values) != expected_columns:
                    raise ValueError(
                        f"{path}:{line_number}: expected {expected_columns} columns, "
                        f"found {len(values)}"
                    )
                record: Dict[str, object] = dict(zip(FIELDS, values[1:]))
                for field in INTEGER_FIELDS:
                    record[field] = int(record[field])
                for field in TIMING_FIELDS:
                    record[field] = float(record[field])
                records.append(record)
    if not records:
        raise ValueError("no THROUGHPUT_RESULT records were found")
    return records


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: DefaultDict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        key = tuple(record[field] for field in GROUP_FIELDS)
        grouped[key].append(record)

    summary: List[Dict[str, object]] = []
    for key, group in grouped.items():
        selected = min(group, key=lambda row: (float(row["total_s"]), int(row["repeat"])))
        work = int(selected["nsys"]) * int(selected["N_global"]) * int(selected["m"]) ** 3
        compute_time = float(selected["compute_s"])
        if compute_time <= 0.0:
            raise ValueError(f"nonpositive computation time for configuration {key}")
        row: Dict[str, object] = dict(zip(GROUP_FIELDS, key))
        row.update(
            {
                "samples": len(group),
                "selected_repeat": selected["repeat"],
                "total_s": selected["total_s"],
                "compute_s": compute_time,
                "communication_s": selected["communication_s"],
                "throughput_work_per_s": work / compute_time,
                "throughput_Gwork_per_s": work / compute_time / 1.0e9,
            }
        )
        summary.append(row)

    return sorted(
        summary,
        key=lambda row: (
            str(row["backend"]),
            int(row["logical_P"]),
            int(row["m"]),
            int(row["nsys"]),
        ),
    )


def main() -> None:
    args = parse_args()
    records = load_records(args.inputs)
    records.sort(
        key=lambda row: (
            str(row["backend"]),
            int(row["logical_P"]),
            int(row["m"]),
            int(row["nsys"]),
            int(row["repeat"]),
        )
    )
    summary = summarize(records)
    panel_a = [row for row in summary if int(row["logical_P"]) == 8]
    panel_b = [
        row
        for row in summary
        if row["backend"] == "gpu" and int(row["m"]) == 8
    ]

    write_csv(args.output_dir / "results_raw.csv", FIELDS, records)
    write_csv(args.output_dir / "results_summary.csv", SUMMARY_FIELDS, summary)
    write_csv(args.output_dir / "panel_a_P8_cpu_gpu.csv", SUMMARY_FIELDS, panel_a)
    write_csv(args.output_dir / "panel_b_gpu_m8.csv", SUMMARY_FIELDS, panel_b)
    print(
        f"collected {len(records)} repeats for {len(summary)} configurations; "
        f"wrote CSV files under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
