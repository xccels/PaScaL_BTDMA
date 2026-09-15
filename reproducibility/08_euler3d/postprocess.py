#!/usr/bin/env python3
"""Collect the Euler run log and per-rank centreline files into CSV summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FIELDS = ("rho", "pressure", "vorticity_z")
REPORTED_L2 = {
    "rho": 1.94e-3,
    "pressure": 2.00e-3,
    "vorticity_z": 4.43e-4,
}


def parse_log(path: Path):
    config = None
    validation = {}
    timing = {}
    solver_phase = {}

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line.startswith("RUN_CONFIG_CSV,"):
            parts = line.split(",")
            if len(parts) == 13 and parts[1] != "n1":
                config = {
                    "n1": int(parts[1]),
                    "n2": int(parts[2]),
                    "n3": int(parts[3]),
                    "np1": int(parts[4]),
                    "np2": int(parts[5]),
                    "np3": int(parts[6]),
                    "m": int(parts[7]),
                    "nstep": int(parts[8]),
                    "dt": float(parts[9]),
                    "t_final": float(parts[10]),
                    "cfl": float(parts[11]),
                    "epsilon": float(parts[12]),
                }
        elif line.startswith("VALIDATION_CSV,"):
            parts = line.split(",")
            if len(parts) == 3 and parts[1] != "field":
                validation[parts[1]] = float(parts[2])
        elif line.startswith("TIMING_CSV,"):
            parts = line.split(",")
            if len(parts) == 4 and parts[1] != "scope":
                timing[parts[1]] = (float(parts[2]), int(parts[3]))
        elif line.startswith("SOLVER_PHASE_CSV,"):
            parts = line.split(",")
            if len(parts) == 4 and parts[1] != "phase":
                solver_phase[parts[1]] = (float(parts[2]), int(parts[3]))

    if config is None:
        raise ValueError(f"RUN_CONFIG_CSV data row not found in {path}")
    if set(validation) != set(FIELDS):
        raise ValueError(f"missing validation fields in {path}: {sorted(set(FIELDS)-set(validation))}")
    return config, validation, timing, solver_phase


def load_centreline(run_dir: Path, stage: str):
    rows = {}
    for path in sorted(run_dir.glob(f"centerline_{stage}_rank*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                index = int(row["global_i"])
                if index in rows:
                    raise ValueError(f"duplicate global_i={index} in {stage} centreline")
                rows[index] = {key: float(value) for key, value in row.items() if key != "global_i"}
    if not rows:
        raise ValueError(f"no centerline_{stage}_rank*.csv files in {run_dir}")
    return rows


def relative_l2(initial, final, field):
    numerator = sum((final[i][field] - initial[i][field]) ** 2 for i in initial)
    denominator = sum(initial[i][field] ** 2 for i in initial)
    return math.sqrt(numerator / denominator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    config, validation, timing, solver_phase = parse_log(run_dir / "run.log")
    initial = load_centreline(run_dir, "initial")
    final = load_centreline(run_dir, "final")
    if set(initial) != set(final):
        raise ValueError("initial and final centreline indices differ")
    if len(initial) != config["n1"]:
        raise ValueError(f"centreline has {len(initial)} points; expected {config['n1']}")

    with (run_dir / "centerline_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "global_i", "x", "rho_initial", "rho_final",
            "pressure_initial", "pressure_final",
            "vorticity_z_initial", "vorticity_z_final",
        ])
        for index in sorted(initial):
            writer.writerow([
                index,
                initial[index]["x"],
                initial[index]["rho"],
                final[index]["rho"],
                initial[index]["pressure"],
                final[index]["pressure"],
                initial[index]["vorticity_z"],
                final[index]["vorticity_z"],
            ])

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "name", "value", "reference"])
        for name, value in config.items():
            writer.writerow(["configuration", name, value, ""])
        is_manuscript_case = (
            (config["n1"], config["n2"], config["n3"]) == (256, 256, 256)
            and (config["np1"], config["np2"], config["np3"]) == (2, 2, 2)
            and config["m"] == 5
            and config["nstep"] == 256
            and math.isclose(config["t_final"], 32.0)
        )
        for field in FIELDS:
            writer.writerow(["full_grid_relative_l2", field, validation[field], ""])
            writer.writerow(["centreline_relative_l2", field, relative_l2(initial, final, field), ""])
            if is_manuscript_case:
                writer.writerow(["manuscript_reported_relative_l2", field, REPORTED_L2[field], "scope not recorded in donor"])
        for name, (seconds, timed_steps) in timing.items():
            writer.writerow(["rank_max_timing_seconds", name, seconds, f"timed_steps={timed_steps}"])
        for name, (seconds, timed_steps) in solver_phase.items():
            writer.writerow(["rank_max_solver_phase_seconds", name, seconds, f"timed_steps={timed_steps}"])

    print(run_dir / "summary.csv")
    print(run_dir / "centerline_profiles.csv")


if __name__ == "__main__":
    main()
