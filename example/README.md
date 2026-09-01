# Minimal CPU and GPU examples

The two programs solve the same deterministic, diagonally dominant,
non-cyclic block tridiagonal problem.  The CPU and GPU sources are kept
separate so that each backend's array ordering and API calls remain visible.

Both programs use:

- block size `m = 2`;
- 16 independent systems;
- eight local block rows per MPI rank;
- a prescribed analytical vector `x_ref`;
- a right-hand side constructed as `b = A*x_ref`; and
- global normalized-residual and relative-solution-error checks.

## Build

```bash
cd example
source ../00_enviroment.sh
./build_mango.sh
```

Normally this example is built from the repository root with `./02_build.sh`.

## Run through Slurm

```bash
cd example
mkdir -p output
sbatch run_mango.slurm
```

The job uses two MPI ranks and two GPUs.  A successful run ends with:

```text
RESULT: PASS
```

The GPU example selects devices from the MPI node-local rank rather than the
global rank, so the same program can be used on a single node or across nodes.
