# Solver accuracy and conditioning

This directory implements the solver-only experiment used for the accuracy and
robustness study.  It is independent of the PDE convergence cases.

No full Mango result is included here.  The scripts, drivers, fixed generator,
raw-data contract, and summarizer are provided so that the result can be
generated and audited.

## Full experiment at a glance

- FP64, non-cyclic block tridiagonal systems
- `m = 2, 5, 8`
- `N = 2048` global block rows
- `n_sys = 128^2 = 16384` for each `(m,K)` case
- `K = 10^2, 10^6, 10^10, 10^14`
- 8192 global-conditioning systems and 8192 diagonal-block-conditioning
  systems in each case
- fixed seed `271828183`
- prescribed `x*`, followed by `b = A x*`
- identical logical inputs for the sequential CPU Block Thomas reference, CPU
  PaScaL_BTDMA, and GPU PaScaL_BTDMA
- per-system normalized residual and relative solution error
- median, maximum, non-finite count, and failure count in the summary

The all-to-all baseline is deliberately absent because this experiment tests
solver correctness, not communication performance.

## Controlled matrix families

The generator implements the two constructions documented in the manuscript
plan.  For the global family it starts from `T_K (x) I_m`.  For the
diagonal-block family it starts from `T_2 (x) S_K`, where the singular values of
`S_K` range geometrically from one to `1/K`.  Independent deterministic Givens
products are applied on the left and right, block by block.  These transforms
preserve the block-tridiagonal sparsity and, in exact arithmetic, the singular
values.

The raw columns `theoretical_global_condition` and
`theoretical_max_diagonal_condition` therefore describe the construction, not
a dense SVD of every `N*m` matrix.  A manuscript claim about *realized* FP64
condition-number ranges still requires a separately documented numerical
estimator or verification step, especially at `K=10^14`.

## Memory strategy

The 16384 systems are never stored as one full coefficient tensor.  A run uses
deterministic chunks (64 systems by default), solves them, writes per-system
records, and regenerates the next chunk.  The same `input_id` encodes generator
version, seed, `m`, `N`, system ID, family, and `K`; the summarizer checks that
each input is present for all three solver paths.

## Build on Mango

From this directory:

```bash
source ../../00_enviroment.sh
./build_mango.sh
```

The build uses the packaged CPU and optimized GPU libraries under
`../../lib_btdma`.  The default GPU target is `cc90,cuda12.8`.

## Run

The default is a deliberately small smoke run:

```bash
mkdir -p output
sbatch run_mango.slurm
```

Run the complete matrix explicitly:

```bash
mkdir -p output
sbatch --export=ALL,CASE=full run_mango.slurm
```

Run only the exact-singular probes:

```bash
mkdir -p output
sbatch --export=ALL,CASE=singular run_mango.slurm
```

The full configuration uses eight MPI ranks and eight GPUs on one Mango node.
`N=2048` is divided evenly across the ranks.  `CHUNK_SIZE` may be changed to a
divisor of 8192, but changing `N_SYSTEMS_PER_FAMILY` in a full evidence run
changes the experiment and must be reported.

## Outputs

Each submission writes to `output/<job-id>-<case>/`, preventing a smoke, full,
or singular run from overwriting an earlier result. The `raw/accuracy_*.csv`
files inside that directory contain one row per system and solver. Important
fields are:

- canonical `input_id` and generator metadata;
- target and construction-controlled condition numbers;
- solver path and resource count;
- normalized residual and relative solution error;
- API-status availability, CUDA runtime status, non-finite flag, and failure
  flag.

The predeclared `failed` rule is a nonzero sequential-reference `dgesv` status,
a non-finite result, a CUDA runtime failure, or normalized residual above
`1e-8`.  Forward error is reported but is not itself a fixed pass/fail test,
because its expected magnitude grows with conditioning.

`summarize_results.py` creates the following files under the same job-specific
directory:

- `summary/accuracy_summary.csv` with median, maximum, and counts;
- `summary/paired_input_coverage.txt` with missing solver paths.

The singular probes use two distinct inputs:

1. a singular first diagonal block embedded in a globally nonsingular system;
2. a singular graph-Laplacian system whose individual diagonal blocks are
   invertible.

Their CSV files and process exit codes are kept separately from the finite
condition-number sweep.  See `SOURCE_AUDIT.md` before interpreting them: the
current library APIs do not propagate a factorization status or warning.

## Evidence boundary

- The CPU driver and generator can be compiled and smoke-tested without a GPU.
- The GPU drivers require Mango's NVHPC/CUDA-aware MPI environment.
- File presence and a successful smoke run do not establish the full accuracy
  curves.
- Results must remain pending until the `CASE=full` raw CSVs, environment log,
  paired-input coverage, and singular-probe logs have been reviewed.
