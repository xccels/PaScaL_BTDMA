# Compile-only GPU resource audit

This directory prepares the compiler-resource evidence requested for block sizes
`m = 8, 9, 10, 12`.  It does not run the solver.  The production library under
`../../lib_btdma/` is read but never modified.

The current optimized GPU library remains supported and recommended only for
`m <= 8`.  The `m = 9, 10, 12` files generated here are isolated compile probes:
they show how compiler-reported resources change when every fixed `1:8`
workspace extent in the current kernel and its device helpers is enlarged
consistently.  They do not establish correctness, runtime performance, achieved
occupancy, or production support for those block sizes.

## Run on Mango

From this directory:

```bash
source ../../00_enviroment.sh
./build_mango.sh
```

Alternatively, submit the compile-only job:

```bash
sbatch run_compile_mango.slurm
```

The Mango defaults are NVIDIA HPC SDK/HPC-X 25.3, CUDA 12.8, `cc90`, `-O3`,
and `-gpu=ptxinfo`.  Override `MPIFC`, `GPU_ARCH`, or `OPT` only when the change
is intentionally part of a new audit.

## What is generated

Each invocation writes a timestamped directory under `output/` containing:

- `environment.txt`: compiler version, target flags, repository commit, and
  source/script hashes;
- `variant_manifest.tsv`: source and generated-variant hashes plus the number
  of audited extent replacements;
- `raw/W*/command.txt`: the full shell-escaped compile command;
- `raw/W*/ptxinfo.log`: unmodified combined stdout/stderr from the variant
  compilation;
- `compile_manifest.tsv`: build status and paths for all four cases;
- `resource_table.csv` and `resource_table.md`: entry kernels and device
  helpers parsed separately.

Generated source copies and object/module files are placed under `build/`.
Both `build/` and `output/` are ignored by the repository.

## Safety guard

The audited source has SHA-256
`811642b0cdbea34c5b4b1189474af2441071275c253a8529db3a6ebb37b54322` and
contains exactly 98 occurrences of the fixed extent token `1:8`.
`generate_variants.py` refuses to continue if either check changes.  These
guards prevent a source update from silently leaving one local array or
initialization slice at extent eight.  If the production source is changed,
audit all fixed workspaces again before updating the expected hash and count.

## Interpreting occupancy

`parse_ptxinfo.py` reports a clearly labelled **theoretical occupancy** for
entry kernels only.  The default `cc90` calculation uses 64 threads per block,
64 warps and 65,536 32-bit registers per SM, and a 256-register allocation
unit per warp.  The calculation is based on compiler-reported registers and
architectural block/thread/warp limits.  It is not achieved occupancy and does
not replace Nsight Compute or another runtime profiler.  Device helpers have no
standalone launch occupancy; their occupancy fields are therefore left blank.

The primary comparison row is
`btdma_many_modi_gpu_v2`.  The other entry kernels and the device helpers are
retained because their stack and spill behavior must not be conflated with the
modified-Thomas entry kernel.
