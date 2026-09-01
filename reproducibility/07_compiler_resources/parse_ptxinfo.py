#!/usr/bin/env python3
"""Parse NVHPC -gpu=ptxinfo output and compute bounded theoretical occupancy."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


ENTRY_RE = re.compile(r"Compiling entry function '([^']+)'")
PROPERTIES_RE = re.compile(r"Function properties for\s+(\S+)")
STACK_RE = re.compile(
    r"(\d+) bytes stack frame,\s*(\d+) bytes spill stores,\s*"
    r"(\d+) bytes spill loads"
)
REGISTERS_RE = re.compile(r"Used\s+(\d+)\s+registers")
DECLARATION_RE = re.compile(
    r"attributes\((global|device)\)\s+subroutine\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
MODULE_RE = re.compile(r"^\s*module\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE | re.MULTILINE)


@dataclass
class ResourceRecord:
    ptx_symbol: str
    registers: int | None = None
    stack_bytes: int | None = None
    spill_store_bytes: int | None = None
    spill_load_bytes: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="compile_manifest.tsv")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--threads-per-block", type=int, default=64)
    parser.add_argument("--warp-size", type=int, default=32)
    parser.add_argument("--registers-per-sm", type=int, default=65536)
    parser.add_argument("--max-warps-per-sm", type=int, default=64)
    parser.add_argument("--max-threads-per-sm", type=int, default=2048)
    parser.add_argument("--max-blocks-per-sm", type=int, default=32)
    parser.add_argument("--register-allocation-unit-per-warp", type=int, default=256)
    return parser.parse_args()


def declared_symbols(source: Path) -> tuple[str, dict[str, str]]:
    source_text = source.read_text(encoding="utf-8")
    module_match = MODULE_RE.search(source_text)
    if module_match is None:
        raise SystemExit(f"No Fortran module declaration found in {source}")
    module_name = module_match.group(1).lower()
    symbols: dict[str, str] = {}
    for kind, name in DECLARATION_RE.findall(source_text):
        symbols[name.lower()] = "entry_kernel" if kind.lower() == "global" else "device_helper"
    if not symbols:
        raise SystemExit(f"No CUDA Fortran routines found in {source}")
    return module_name, symbols


def human_symbol(
    ptx_symbol: str, module_name: str, declarations: dict[str, str]
) -> tuple[str, str]:
    lowered = ptx_symbol.lower()
    for name in sorted(declarations, key=len, reverse=True):
        if lowered.endswith(f"{module_name}_{name}_"):
            return name, declarations[name]
    return ptx_symbol, "unmapped"


def parse_log(log_path: Path) -> tuple[list[ResourceRecord], set[str]]:
    records: list[ResourceRecord] = []
    by_symbol: dict[str, ResourceRecord] = {}
    entry_symbols: set[str] = set()
    current: ResourceRecord | None = None

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ENTRY_RE.search(line)
        if match:
            entry_symbols.add(match.group(1))
            continue

        match = PROPERTIES_RE.search(line)
        if match:
            symbol = match.group(1)
            current = by_symbol.get(symbol)
            if current is None:
                current = ResourceRecord(ptx_symbol=symbol)
                by_symbol[symbol] = current
                records.append(current)
            continue

        if current is None:
            continue

        match = STACK_RE.search(line)
        if match:
            current.stack_bytes = int(match.group(1))
            current.spill_store_bytes = int(match.group(2))
            current.spill_load_bytes = int(match.group(3))
            continue

        match = REGISTERS_RE.search(line)
        if match:
            current.registers = int(match.group(1))

    return records, entry_symbols


def theoretical_occupancy(registers: int, args: argparse.Namespace) -> dict[str, int | float]:
    warps_per_block = math.ceil(args.threads_per_block / args.warp_size)
    registers_per_warp_raw = registers * args.warp_size
    allocation_unit = args.register_allocation_unit_per_warp
    registers_per_warp = math.ceil(registers_per_warp_raw / allocation_unit) * allocation_unit
    registers_per_block = registers_per_warp * warps_per_block

    blocks_by_registers = args.registers_per_sm // registers_per_block
    blocks_by_threads = args.max_threads_per_sm // args.threads_per_block
    blocks_by_warps = args.max_warps_per_sm // warps_per_block
    resident_blocks = min(
        blocks_by_registers,
        blocks_by_threads,
        blocks_by_warps,
        args.max_blocks_per_sm,
    )
    resident_warps = resident_blocks * warps_per_block
    occupancy = 100.0 * resident_warps / args.max_warps_per_sm
    return {
        "registers_allocated_per_block": registers_per_block,
        "resident_blocks_per_sm": resident_blocks,
        "resident_warps_per_sm": resident_warps,
        "theoretical_occupancy_percent": occupancy,
    }


def optional(value: int | float | None, *, decimals: int | None = None) -> str:
    if value is None:
        return ""
    if decimals is not None:
        return f"{value:.{decimals}f}"
    return str(value)


def markdown_table(rows: list[dict[str, str]], kind: str) -> list[str]:
    selected = [row for row in rows if row["symbol_kind"] == kind]
    title = "Entry kernels" if kind == "entry_kernel" else "Device helpers"
    output = [f"## {title}", ""]
    if not selected:
        output.extend(["No matching compiler records were parsed.", ""])
        return output

    output.extend(
        [
            "| Build | m | Symbol | Registers/thread | Stack B/thread | Spill store B/thread | Spill load B/thread | Theoretical occupancy |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected:
        occupancy = row["theoretical_occupancy_percent"]
        output.append(
            "| {build_id} | {m} | `{symbol_name}` | {registers_per_thread} | "
            "{stack_frame_bytes_per_thread} | {spill_store_bytes_per_thread} | "
            "{spill_load_bytes_per_thread} | {occupancy} |".format(
                **row, occupancy=(occupancy + "%" if occupancy else "n/a")
            )
        )
    output.append("")
    return output


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    base_dir = manifest_path.parent
    module_name, declarations = declared_symbols(args.source.resolve())

    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        builds = list(csv.DictReader(stream, delimiter="\t"))
    if not builds:
        raise SystemExit(f"No compile records in {manifest_path}")

    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    fatal_errors: list[str] = []
    for build in builds:
        log_path = Path(build["log_path"])
        if not log_path.is_absolute():
            log_path = base_dir / log_path
        if not log_path.is_file():
            warnings.append(f"{build['build_id']}: missing log {log_path}")
            continue

        records, ptx_entries = parse_log(log_path)
        if not records:
            warnings.append(f"{build['build_id']}: no ptxinfo function records parsed")
        seen_source_symbols: set[str] = set()

        for record in records:
            symbol_name, kind = human_symbol(record.ptx_symbol, module_name, declarations)
            if kind != "unmapped":
                seen_source_symbols.add(symbol_name)
            source_says_entry = kind == "entry_kernel"
            ptx_says_entry = record.ptx_symbol in ptx_entries
            if source_says_entry != ptx_says_entry and kind != "unmapped":
                warnings.append(
                    f"{build['build_id']}: source/PTX entry classification differs for "
                    f"{record.ptx_symbol}"
                )

            occupancy_fields: dict[str, int | float] = {}
            if kind == "entry_kernel" and record.registers is not None:
                occupancy_fields = theoretical_occupancy(record.registers, args)

            role = ""
            if symbol_name == "btdma_many_modi_gpu_v2":
                role = "optimized_modified_thomas_entry"
            elif kind == "entry_kernel":
                role = "other_entry_kernel"
            elif kind == "device_helper":
                role = "device_helper"

            rows.append(
                {
                    "build_id": build["build_id"],
                    "m": build["m"],
                    "compile_status": build["compile_status"],
                    "analysis_role": role,
                    "symbol_kind": kind,
                    "symbol_name": symbol_name,
                    "ptx_symbol": record.ptx_symbol,
                    "registers_per_thread": optional(record.registers),
                    "stack_frame_bytes_per_thread": optional(record.stack_bytes),
                    "spill_store_bytes_per_thread": optional(record.spill_store_bytes),
                    "spill_load_bytes_per_thread": optional(record.spill_load_bytes),
                    "threads_per_block": (
                        str(args.threads_per_block) if kind == "entry_kernel" else ""
                    ),
                    "registers_allocated_per_block": optional(
                        occupancy_fields.get("registers_allocated_per_block")
                    ),
                    "resident_blocks_per_sm": optional(
                        occupancy_fields.get("resident_blocks_per_sm")
                    ),
                    "resident_warps_per_sm": optional(
                        occupancy_fields.get("resident_warps_per_sm")
                    ),
                    "theoretical_occupancy_percent": optional(
                        occupancy_fields.get("theoretical_occupancy_percent"), decimals=3
                    ),
                    "occupancy_basis": (
                        "compiler registers plus cc90 architectural limits; not achieved occupancy"
                        if occupancy_fields
                        else "not applicable to a separately reported device helper"
                        if kind == "device_helper"
                        else "unavailable"
                    ),
                }
            )

        missing = sorted(set(declarations) - seen_source_symbols)
        if missing:
            warnings.append(
                f"{build['build_id']}: declared routines absent from ptxinfo: "
                + ", ".join(missing)
            )

        if build["compile_status"] == "0":
            target_records = [
                record
                for record in rows
                if record["build_id"] == build["build_id"]
                and record["symbol_name"] == "btdma_many_modi_gpu_v2"
            ]
            if len(target_records) != 1:
                fatal_errors.append(
                    f"{build['build_id']}: expected exactly one "
                    "btdma_many_modi_gpu_v2 resource record"
                )
            elif any(
                not target_records[0][field]
                for field in (
                    "registers_per_thread",
                    "stack_frame_bytes_per_thread",
                    "spill_store_bytes_per_thread",
                    "spill_load_bytes_per_thread",
                    "theoretical_occupancy_percent",
                )
            ):
                fatal_errors.append(
                    f"{build['build_id']}: incomplete resource fields for "
                    "btdma_many_modi_gpu_v2"
                )

    fieldnames = [
        "build_id",
        "m",
        "compile_status",
        "analysis_role",
        "symbol_kind",
        "symbol_name",
        "ptx_symbol",
        "registers_per_thread",
        "stack_frame_bytes_per_thread",
        "spill_store_bytes_per_thread",
        "spill_load_bytes_per_thread",
        "threads_per_block",
        "registers_allocated_per_block",
        "resident_blocks_per_sm",
        "resident_warps_per_sm",
        "theoretical_occupancy_percent",
        "occupancy_basis",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    markdown: list[str] = [
        "# Compiler-resource audit results",
        "",
        "This report contains compile-only evidence. It does not establish runtime ",
        "performance, correctness for m > 8, achieved occupancy, or production support.",
        "",
        "## Build status",
        "",
        "| Build | m | Compile status | Raw log | Full command |",
        "|---|---:|---:|---|---|",
    ]
    for build in builds:
        markdown.append(
            f"| {build['build_id']} | {build['m']} | {build['compile_status']} | "
            f"`{build['log_path']}` | `{build['command_path']}` |"
        )
    markdown.append("")
    markdown.extend(markdown_table(rows, "entry_kernel"))
    markdown.extend(markdown_table(rows, "device_helper"))
    markdown.extend(
        [
            "## Occupancy calculation boundary",
            "",
            f"The table uses {args.threads_per_block} threads per block, "
            f"{args.warp_size} threads per warp, {args.registers_per_sm} 32-bit "
            f"registers per SM, {args.max_warps_per_sm} warps per SM, "
            f"{args.max_threads_per_sm} threads per SM, and "
            f"{args.register_allocation_unit_per_warp}-register allocation units "
            "per warp. Stack-frame bytes are reported separately and are not "
            "converted into measured local-memory traffic.",
            "",
        ]
    )
    if warnings:
        markdown.extend(["## Parser warnings", ""])
        markdown.extend(f"- {warning}" for warning in warnings)
        markdown.append("")
    if fatal_errors:
        markdown.extend(["## Parser errors", ""])
        markdown.extend(f"- {error}" for error in fatal_errors)
        markdown.append("")

    args.markdown.write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote {len(rows)} resource rows to {args.csv}")
    print(f"Wrote Markdown report to {args.markdown}")
    if warnings:
        print(f"Parser warnings: {len(warnings)} (see Markdown report)")
    if fatal_errors:
        print(f"Parser errors: {len(fatal_errors)} (see Markdown report)")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
