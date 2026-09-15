#!/usr/bin/env python3
"""Extract one selected communication timing per configuration from job logs."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


EXPECTED_PLATFORMS = ("cpu", "gpu")
EXPECTED_METHODS = ("alltoall", "pascal")
EXPECTED_M = (2, 5, 8)
EXPECTED_N = (512, 1024, 2048, 4096)
EXPECTED_NPROCS = {"cpu": 192, "gpu": 8}
EXPECTED_P = 8
EXPECTED_NSYS = 128 * 128


@dataclass(frozen=True)
class Result:
    platform: str
    method: str
    p_label: int
    nprocs: int
    nsys: int
    m: int
    n_global: int
    repeat: int
    warmups: int
    repeats: int
    forward: float
    backward: float
    total: float
    source_file: str
    source_line: int

    @property
    def key(self) -> tuple[str, str, int, int, int, int, int]:
        return (
            self.platform,
            self.method,
            self.p_label,
            self.nprocs,
            self.nsys,
            self.m,
            self.n_global,
        )


def parse_result_line(line: str, source: Path, line_number: int) -> Result | None:
    if not line.startswith("RESULT,"):
        return None
    fields = next(csv.reader([line]))
    if len(fields) != 14:
        raise ValueError(f"{source}:{line_number}: expected 14 RESULT fields, got {len(fields)}")
    try:
        return Result(
            platform=fields[1],
            method=fields[2],
            p_label=int(fields[3]),
            nprocs=int(fields[4]),
            nsys=int(fields[5]),
            m=int(fields[6]),
            n_global=int(fields[7]),
            repeat=int(fields[8]),
            warmups=int(fields[9]),
            repeats=int(fields[10]),
            forward=float(fields[11]),
            backward=float(fields[12]),
            total=float(fields[13]),
            source_file=str(source),
            source_line=line_number,
        )
    except ValueError as error:
        raise ValueError(f"{source}:{line_number}: invalid RESULT value: {error}") from error


def read_results(paths: Iterable[Path]) -> list[Result]:
    results: list[Result] = []
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                result = parse_result_line(raw_line.strip(), path, line_number)
                if result is not None:
                    results.append(result)
    return results


def validate_result(result: Result) -> None:
    if result.platform not in EXPECTED_PLATFORMS:
        raise ValueError(f"unexpected platform {result.platform!r} in {result.source_file}:{result.source_line}")
    if result.method not in EXPECTED_METHODS:
        raise ValueError(f"unexpected method {result.method!r} in {result.source_file}:{result.source_line}")
    if result.p_label != EXPECTED_P or result.nsys != EXPECTED_NSYS:
        raise ValueError(f"unexpected fixed parameters in {result.source_file}:{result.source_line}")
    if result.nprocs != EXPECTED_NPROCS[result.platform]:
        raise ValueError(f"unexpected process count in {result.source_file}:{result.source_line}")
    if result.m not in EXPECTED_M or result.n_global not in EXPECTED_N:
        raise ValueError(f"unexpected m or N in {result.source_file}:{result.source_line}")
    if result.repeat < 1 or result.repeats < 1 or result.repeat > result.repeats:
        raise ValueError(f"invalid repetition metadata in {result.source_file}:{result.source_line}")
    if min(result.forward, result.backward, result.total) < 0.0:
        raise ValueError(f"negative timing in {result.source_file}:{result.source_line}")
    tolerance = 1.0e-10 * max(1.0, abs(result.total))
    if result.total + tolerance < max(result.forward, result.backward):
        raise ValueError(f"rank-maximum total is smaller than a phase in {result.source_file}:{result.source_line}")
    if result.total > result.forward + result.backward + tolerance:
        raise ValueError(f"rank-maximum total exceeds phase-max sum in {result.source_file}:{result.source_line}")


def expected_keys() -> set[tuple[str, str, int, int, int, int, int]]:
    return {
        (platform, method, EXPECTED_P, EXPECTED_NPROCS[platform], EXPECTED_NSYS, m, n_global)
        for platform in EXPECTED_PLATFORMS
        for method in EXPECTED_METHODS
        for m in EXPECTED_M
        for n_global in EXPECTED_N
    }


def write_selected(results: list[Result], output: TextIO, require_complete: bool) -> None:
    groups: dict[tuple[str, str, int, int, int, int, int], list[Result]] = defaultdict(list)
    for result in results:
        validate_result(result)
        groups[result.key].append(result)

    missing = sorted(expected_keys() - set(groups))
    if missing:
        message = f"missing {len(missing)} of {len(expected_keys())} expected configurations"
        if require_complete:
            raise ValueError(message)
        print(f"warning: {message}", file=sys.stderr)

    fieldnames = [
        "platform", "method", "P_label", "nprocs", "nsys", "m", "N",
        "repeat", "warmups", "repeats", "comm_forward_s", "comm_backward_s",
        "comm_total_s", "samples", "selection", "source_file", "source_line",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for key in sorted(groups):
        samples = groups[key]
        selected = min(samples, key=lambda item: item.total)
        writer.writerow(
            {
                "platform": selected.platform,
                "method": selected.method,
                "P_label": selected.p_label,
                "nprocs": selected.nprocs,
                "nsys": selected.nsys,
                "m": selected.m,
                "N": selected.n_global,
                "repeat": selected.repeat,
                "warmups": selected.warmups,
                "repeats": selected.repeats,
                "comm_forward_s": f"{selected.forward:.16e}",
                "comm_backward_s": f"{selected.backward:.16e}",
                "comm_total_s": f"{selected.total:.16e}",
                "samples": len(samples),
                "selection": "min_rank_max_total",
                "source_file": selected.source_file,
                "source_line": selected.source_line,
            }
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the minimum rank-maximum communication total for each benchmark configuration."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Slurm stdout logs containing RESULT rows")
    parser.add_argument("-o", "--output", type=Path, help="write CSV here instead of standard output")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless all 48 CPU/GPU, method, m, and N configurations are present",
    )
    args = parser.parse_args()

    try:
        results = read_results(args.logs)
        if not results:
            raise ValueError("no RESULT rows were found")
        if args.output is None:
            write_selected(results, sys.stdout, args.require_complete)
        else:
            with args.output.open("w", newline="", encoding="utf-8") as stream:
                write_selected(results, stream, args.require_complete)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
