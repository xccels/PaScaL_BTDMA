#!/usr/bin/env python3
"""Generate isolated fixed-workspace variants for a compile-only audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


AUDIT_SIZES = (8, 9, 10, 12)
FIXED_TOKEN = "1:8"
EXPECTED_REPLACEMENTS = 98
EXPECTED_SOURCE_SHA256 = "811642b0cdbea34c5b4b1189474af2441071275c253a8529db3a6ebb37b54322"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Copy the production GPU source and enlarge every audited 1:8 "
            "extent consistently. The production source is never edited."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=script_dir.parent.parent / "lib_btdma" / "mod_btdma_gpu_v2.f90",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "build" / "variants",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="TSV manifest path (default: OUTPUT_DIR/variant_manifest.tsv)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(AUDIT_SIZES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else output_dir / "variant_manifest.tsv"
    )

    sizes = tuple(args.sizes)
    if sizes != AUDIT_SIZES:
        raise SystemExit(
            f"This audit is fixed to sizes {AUDIT_SIZES}; received {sizes}."
        )
    if not source.is_file():
        raise SystemExit(f"Production source not found: {source}")

    source_hash = sha256(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "Refusing to generate variants from an unaudited source revision: "
            f"expected SHA-256 {EXPECTED_SOURCE_SHA256}, found {source_hash}. "
            "Re-audit every fixed workspace before updating the expected hash."
        )

    original = source.read_text(encoding="utf-8")
    replacement_count = original.count(FIXED_TOKEN)
    if replacement_count != EXPECTED_REPLACEMENTS:
        raise SystemExit(
            "Refusing to generate variants: expected "
            f"{EXPECTED_REPLACEMENTS} occurrences of {FIXED_TOKEN!r}, found "
            f"{replacement_count}. Re-audit all fixed workspaces first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    for size in sizes:
        build_id = f"W{size}"
        variant_dir = output_dir / build_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        destination = variant_dir / "mod_btdma_gpu_v2.f90"

        replacement = f"1:{size}"
        generated_body = original.replace(FIXED_TOKEN, replacement)
        if size != 8 and FIXED_TOKEN in generated_body:
            raise SystemExit(f"Unreplaced {FIXED_TOKEN!r} token in {build_id}")
        if generated_body.count(replacement) != EXPECTED_REPLACEMENTS:
            raise SystemExit(
                f"Unexpected replacement count in {build_id}: "
                f"{generated_body.count(replacement)}"
            )

        header = (
            "! AUTO-GENERATED FOR A COMPILE-ONLY RESOURCE AUDIT.\n"
            "! The production library is unchanged; this file is not a "
            "supported runtime implementation.\n"
            f"! AUDIT_WORKSPACE_EXTENT={size}; "
            f"AUDITED_REPLACEMENTS={EXPECTED_REPLACEMENTS}.\n"
        )
        destination.write_text(header + generated_body, encoding="utf-8")
        rows.append(
            {
                "build_id": build_id,
                "m": size,
                "production_source": str(source),
                "production_source_sha256": source_hash,
                "generated_source": str(destination),
                "generated_source_sha256": sha256(destination),
                "extent_token_from": FIXED_TOKEN,
                "extent_token_to": replacement,
                "replacement_count": EXPECTED_REPLACEMENTS,
            }
        )

    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} isolated variants in {output_dir}")
    print(f"Variant manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
