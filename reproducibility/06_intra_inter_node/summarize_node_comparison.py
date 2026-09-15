#!/usr/bin/env python3
"""Create raw and four-row table CSVs from the fixed node comparison logs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple


FIELDS = [
    "method",
    "regime",
    "repeat",
    "nodes",
    "gpus_per_node",
    "mpi_ranks",
    "local_ranks",
    "m",
    "N",
    "nsys",
    "Nlocal",
    "phase1_s",
    "phase2_s",
    "phase3_s",
    "phase4_s",
    "phase5_s",
    "computation_s",
    "communication_s",
    "total_s",
    "wall_s",
    "timing_gap_s",
]

INTEGER_FIELDS = {
    "repeat",
    "nodes",
    "gpus_per_node",
    "mpi_ranks",
    "local_ranks",
    "m",
    "N",
    "nsys",
    "Nlocal",
}
FLOAT_FIELDS = set(FIELDS) - INTEGER_FIELDS - {"method", "regime"}
EXPECTED_GROUPS = {
    ("conventional_alltoall", "intra-node"),
    ("conventional_alltoall", "inter-node"),
    ("pascal_btdma", "intra-node"),
    ("pascal_btdma", "inter-node"),
}
GROUP_ORDER = [
    ("conventional_alltoall", "intra-node"),
    ("conventional_alltoall", "inter-node"),
    ("pascal_btdma", "intra-node"),
    ("pascal_btdma", "inter-node"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="four benchmark log files")
    parser.add_argument("--raw", required=True, type=Path, help="per-repeat CSV")
    parser.add_argument("--table", required=True, type=Path, help="four-row table CSV")
    return parser.parse_args()


def load_records(paths: Sequence[Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    expected_columns = len(FIELDS) + 1
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.startswith("NODE_RAW,"):
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
                for field in FLOAT_FIELDS:
                    record[field] = float(record[field])
                record["source_log"] = str(path)
                record["source_line"] = line_number
                validate_record(record)
                records.append(record)
    if not records:
        raise ValueError("no NODE_RAW records were found")
    return records


def validate_record(record: Dict[str, object]) -> None:
    key = (str(record["method"]), str(record["regime"]))
    if key not in EXPECTED_GROUPS:
        raise ValueError(f"unexpected method/regime pair: {key}")
    fixed = {
        "mpi_ranks": 4,
        "m": 8,
        "N": 2048,
        "nsys": 16384,
        "Nlocal": 512,
    }
    for field, expected in fixed.items():
        if record[field] != expected:
            raise ValueError(f"{field} must be {expected}; found {record[field]}")
    if record["regime"] == "intra-node":
        expected_placement = (1, 4, 4)
    else:
        expected_placement = (4, 1, 1)
    actual_placement = (
        record["nodes"],
        record["gpus_per_node"],
        record["local_ranks"],
    )
    if actual_placement != expected_placement:
        raise ValueError(
            f"placement mismatch for {record['regime']}: {actual_placement}"
        )

    phases = [float(record[f"phase{index}_s"]) for index in range(1, 6)]
    if record["method"] == "conventional_alltoall":
        expected_computation = phases[2]
    else:
        expected_computation = phases[0] + phases[2] + phases[4]
    expected_communication = phases[1] + phases[3]
    expected_total = expected_computation + expected_communication
    assert_close(float(record["computation_s"]), expected_computation, "computation_s")
    assert_close(float(record["communication_s"]), expected_communication, "communication_s")
    assert_close(float(record["total_s"]), expected_total, "total_s")


def assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=5.0e-12, abs_tol=1.0e-15):
        raise ValueError(f"{name} is inconsistent with the phase timers")


def write_raw(path: Path, records: Iterable[Dict[str, object]]) -> None:
    output_fields = [*FIELDS, "source_log", "source_line"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(records)


def representative_rows(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: DefaultDict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["method"]), str(record["regime"]))].append(record)
    missing = EXPECTED_GROUPS - set(grouped)
    extra = set(grouped) - EXPECTED_GROUPS
    if missing or extra:
        raise ValueError(f"incomplete comparison: missing={sorted(missing)}, extra={sorted(extra)}")

    selected = {
        key: min(grouped[key], key=lambda row: float(row["total_s"]))
        for key in EXPECTED_GROUPS
    }
    speedups = {
        regime: float(selected[("conventional_alltoall", regime)]["total_s"])
        / float(selected[("pascal_btdma", regime)]["total_s"])
        for regime in ("intra-node", "inter-node")
    }

    rows: List[Dict[str, object]] = []
    for key in GROUP_ORDER:
        record = selected[key]
        total = float(record["total_s"])
        communication = float(record["communication_s"])
        row = {
            "method": record["method"],
            "regime": record["regime"],
            "nodes": record["nodes"],
            "gpus_per_node": record["gpus_per_node"],
            "mpi_ranks": record["mpi_ranks"],
            "m": record["m"],
            "N": record["N"],
            "nsys": record["nsys"],
            "selected_repeat": record["repeat"],
            "samples": len(grouped[key]),
            "representative_rule": "minimum phase-sum total",
            "t_total_s": total,
            "t_computation_s": record["computation_s"],
            "t_communication_s": communication,
            "communication_fraction_percent": 100.0 * communication / total,
            "speedup_vs_alltoall": (
                1.0 if record["method"] == "conventional_alltoall"
                else speedups[str(record["regime"])]
            ),
            "wall_s_diagnostic": record["wall_s"],
            "timing_gap_s_diagnostic": record["timing_gap_s"],
            "source_log": record["source_log"],
            "source_line": record["source_line"],
        }
        rows.append(row)
    return rows


def write_table(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    records = load_records(args.inputs)
    records.sort(
        key=lambda row: (
            GROUP_ORDER.index((str(row["method"]), str(row["regime"]))),
            int(row["repeat"]),
        )
    )
    write_raw(args.raw, records)
    table_rows = representative_rows(records)
    write_table(args.table, table_rows)
    print(f"wrote {len(records)} raw records and {len(table_rows)} table rows")


if __name__ == "__main__":
    main()
