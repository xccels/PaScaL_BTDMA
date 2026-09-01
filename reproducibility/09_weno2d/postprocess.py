#!/usr/bin/env python3
"""Combine WENO2D metrics and export plotting-ready contour CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


EXPECTED = {
    "cfl1p0": (1.0, 1621),
    "cfl1p5": (1.5, 1080),
    "cfl2p0": (2.0, 810),
}


def read_one_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one data row, found {len(rows)}")
    row = rows[0]
    case_id = row["case_id"]
    if case_id not in EXPECTED:
        raise ValueError(f"{path}: unknown case_id {case_id!r}")
    expected_cfl, expected_steps = EXPECTED[case_id]
    if not math.isclose(float(row["cfl_nominal"]), expected_cfl):
        raise ValueError(f"{path}: CFL does not match {case_id}")
    if int(row["nstep"]) != expected_steps:
        raise ValueError(f"{path}: step count does not match {case_id}")
    if (int(row["nx"]), int(row["ny"])) != (516, 516):
        raise ValueError(f"{path}: expected a 516 x 516 grid")
    if (int(row["npx"]), int(row["npy"]), int(row["gpus"])) != (2, 2, 4):
        raise ValueError(f"{path}: expected a 2 x 2, four-GPU run")
    return row


def parse_legacy_vtk(path: Path) -> tuple[dict[str, object], dict[str, list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    dimensions: tuple[int, int, int] | None = None
    origin: tuple[float, float, float] | None = None
    spacing: tuple[float, float, float] | None = None
    point_count: int | None = None
    fields: dict[str, list[float]] = {}
    index = 0

    while index < len(lines):
        parts = lines[index].split()
        if not parts:
            index += 1
            continue
        keyword = parts[0].upper()
        if keyword == "DIMENSIONS":
            dimensions = tuple(map(int, parts[1:4]))
        elif keyword == "ORIGIN":
            origin = tuple(map(float, parts[1:4]))
        elif keyword == "SPACING":
            spacing = tuple(map(float, parts[1:4]))
        elif keyword == "POINT_DATA":
            point_count = int(parts[1])
        elif keyword == "SCALARS":
            if point_count is None:
                raise ValueError(f"{path}: SCALARS appears before POINT_DATA")
            name = parts[1]
            index += 1
            if index >= len(lines) or not lines[index].startswith("LOOKUP_TABLE"):
                raise ValueError(f"{path}: missing LOOKUP_TABLE for {name}")
            values: list[float] = []
            while len(values) < point_count:
                index += 1
                if index >= len(lines):
                    raise ValueError(f"{path}: incomplete field {name}")
                values.extend(float(value) for value in lines[index].split())
            if len(values) != point_count:
                raise ValueError(f"{path}: too many values in field {name}")
            fields[name] = values
        index += 1

    if dimensions is None or origin is None or spacing is None or point_count is None:
        raise ValueError(f"{path}: incomplete structured-points header")
    if dimensions[0] * dimensions[1] * dimensions[2] != point_count:
        raise ValueError(f"{path}: DIMENSIONS and POINT_DATA disagree")
    required = {"u_final", "u_initial", "u_error"}
    if set(fields) != required:
        raise ValueError(f"{path}: expected fields {sorted(required)}, found {sorted(fields)}")
    metadata: dict[str, object] = {
        "dimensions": dimensions,
        "origin": origin,
        "spacing": spacing,
        "point_count": point_count,
    }
    return metadata, fields


def write_contour_csv(
    path: Path,
    metadata: dict[str, object],
    fields: dict[str, list[float]],
) -> float:
    nx, ny, nz = metadata["dimensions"]  # type: ignore[misc]
    x0, y0, _ = metadata["origin"]  # type: ignore[misc]
    dx, dy, _ = metadata["spacing"]  # type: ignore[misc]
    if nz != 1:
        raise ValueError("the WENO2D output must have one plane")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "u_initial", "u_final", "u_error"])
        for j in range(ny):
            for i in range(nx):
                offset = j * nx + i
                writer.writerow(
                    [
                        f"{x0 + i * dx:.16e}",
                        f"{y0 + j * dy:.16e}",
                        f"{fields['u_initial'][offset]:.16e}",
                        f"{fields['u_final'][offset]:.16e}",
                        f"{fields['u_error'][offset]:.16e}",
                    ]
                )
    return math.fsum(abs(value) for value in fields["u_error"]) * dx * dy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--contour-dir", required=True, type=Path)
    args = parser.parse_args()

    combined: list[dict[str, str]] = []
    seen: set[str] = set()
    for metrics_path in args.metrics:
        row = read_one_row(metrics_path)
        case_id = row["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate case_id {case_id!r}")
        seen.add(case_id)

        vtk_path = Path(row["vtk_file"])
        if not vtk_path.is_absolute():
            vtk_path = metrics_path.parent / vtk_path
        metadata, fields = parse_legacy_vtk(vtk_path)
        expected_dimensions = (int(row["nx"]), int(row["ny"]), 1)
        if metadata["dimensions"] != expected_dimensions:
            raise ValueError(
                f"{vtk_path}: expected DIMENSIONS {expected_dimensions}, "
                f"found {metadata['dimensions']}"
            )
        contour_path = args.contour_dir / f"{case_id}_contour.csv"
        l1_from_vtk = write_contour_csv(contour_path, metadata, fields)
        row["l1_from_vtk"] = f"{l1_from_vtk:.16e}"
        row["l1_consistency_abs"] = f"{abs(l1_from_vtk - float(row['l1_error'])):.16e}"
        row["contour_csv"] = str(contour_path)
        combined.append(row)

    order = {case_id: index for index, case_id in enumerate(EXPECTED)}
    combined.sort(key=lambda row: order[row["case_id"]])
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(combined[0])
    with args.summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)

    print(f"Wrote {args.summary}")
    for row in combined:
        print(f"Wrote {row['contour_csv']}")


if __name__ == "__main__":
    main()
