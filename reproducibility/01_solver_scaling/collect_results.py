#!/usr/bin/env python3
"""Extract BTDMA_RESULT records and create raw and aggregate CSV files."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple


FIELDS = [
    "case",
    "variant",
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
NON_GROUP_FIELDS = set(["repeat"] + TIMING_FIELDS)
GROUP_FIELDS = [field for field in FIELDS if field not in NON_GROUP_FIELDS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="benchmark log files")
    parser.add_argument("--raw", required=True, type=Path, help="raw CSV output")
    parser.add_argument("--summary", required=True, type=Path, help="summary CSV output")
    return parser.parse_args()


def load_records(paths: Sequence[Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    expected_columns = len(FIELDS) + 1
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.startswith("BTDMA_RESULT,"):
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
        raise ValueError("no BTDMA_RESULT records were found")
    return records


def write_raw(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def write_summary(path: Path, records: Sequence[Dict[str, object]]) -> None:
    grouped: DefaultDict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in GROUP_FIELDS)].append(record)

    statistic_fields = []
    for timing in TIMING_FIELDS:
        stem = timing[:-2] if timing.endswith("_s") else timing
        statistic_fields.extend(
            [f"{stem}_min_s", f"{stem}_median_s", f"{stem}_mean_s", f"{stem}_stdev_s"]
        )
    output_fields = [*GROUP_FIELDS, "samples", *statistic_fields]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        for key in sorted(grouped, key=lambda item: tuple(map(str, item))):
            group = grouped[key]
            row: Dict[str, object] = dict(zip(GROUP_FIELDS, key))
            row["samples"] = len(group)
            for timing in TIMING_FIELDS:
                values = [float(record[timing]) for record in group]
                stem = timing[:-2] if timing.endswith("_s") else timing
                row[f"{stem}_min_s"] = min(values)
                row[f"{stem}_median_s"] = statistics.median(values)
                row[f"{stem}_mean_s"] = statistics.fmean(values)
                row[f"{stem}_stdev_s"] = statistics.stdev(values) if len(values) > 1 else 0.0
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    records = load_records(args.inputs)
    write_raw(args.raw, records)
    write_summary(args.summary, records)
    print(f"collected {len(records)} measurements from {len(args.inputs)} log files")


if __name__ == "__main__":
    main()
