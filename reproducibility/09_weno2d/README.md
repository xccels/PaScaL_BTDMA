# 09 — WENO5 Zalesak slotted-disk application

This directory reproduces the manuscript's two-dimensional scalar-advection
application. It is intentionally fixed to the reported configuration instead
of providing a general application framework.

## Fixed cases

| Case | Nominal CFL | Uniform steps to one rotation | Grid | MPI/GPU layout | Block size |
|---|---:|---:|---:|---:|---:|
| `cfl1p0` | 1.0 | 1621 | `516 x 516` | `2 x 2` (4 GPUs) | 3 |
| `cfl1p5` | 1.5 | 1080 | `516 x 516` | `2 x 2` (4 GPUs) | 3 |
| `cfl2p0` | 2.0 | 810 | `516 x 516` | `2 x 2` (4 GPUs) | 3 |

The machine-readable copy of this table is [`cases.tsv`](cases.tsv). The
program accepts only these three case IDs and checks the grid, decomposition,
rank count, and block size before the simulation starts.

Each step uses frozen signed-upwind WENO5 weights, the five-stage L-stable
SDIRK4 integrator, two directional approximate-factorisation sweeps, and ten
batched PaScaL_BTDMA calls (five stages times two directions). The exact
solution after one rotation is the initial slotted disk.

## Build on Mango

From this directory:

```bash
source ../../00_enviroment.sh
./build_mango.sh
```

The build script first creates the shared GPU library from `../../lib_btdma`
and then links `bin/weno2d_gpu` against that library. The application constants
are fixed to `N=516`, `m=3`, and `NP=2 x 2`. The Mango defaults are NVHPC/HPC-X
25.3, CUDA 12.8, and `cc90`; compiler and GPU-target overrides use the same
environment variables documented in `../../00_enviroment.sh`.

## Run the three manuscript cases

Submit one case per allocation so each result has an independent environment
manifest and log:

```bash
sbatch run_mango.slurm cfl1p0
sbatch run_mango.slurm cfl1p5
sbatch run_mango.slurm cfl2p0
```

Each job uses one Mango node, four MPI ranks, and one GPU per rank. Results are
written below `output/<job-id>/<case-id>/`.

## Outputs

For each case, the driver writes:

- `metrics_<case-id>.csv`: nominal/effective CFL, step count, time step, L1 and
  Linf errors, extrema, rank-0 diagnostic timing (excluding the first step),
  and VTK path;
- `output_weno5_<case-id>_516x516_final.vtk`: `u_final`, `u_initial`, and
  `u_error` on the complete grid;
- `run.log`: complete program output;
- `manifest.txt`: module, compiler, GPU, Git, and source-hash provenance.

The job then runs [`postprocess.py`](postprocess.py). It combines the metrics
into `summary.csv`, recomputes the L1 error from the VTK error field, and writes
`contours/<case-id>_contour.csv` with this plotting-ready schema:

```text
x,y,u_initial,u_final,u_error
```

To combine completed cases later:

```bash
python3 postprocess.py \
  output/*/cfl1p0/metrics_cfl1p0.csv \
  output/*/cfl1p5/metrics_cfl1p5.csv \
  output/*/cfl2p0/metrics_cfl2p0.csv \
  --summary output/weno2d_summary.csv \
  --contour-dir output/contours
```

The manuscript reports L1 errors of `5.68e-3`, `5.91e-3`, and `6.29e-3` for
the three cases. These are comparison references, not values generated or
hard-coded by this package. A Mango run is required before claiming that the
public driver reproduces them.

See [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md) for the donor-source audit
and the discrepancies corrected in this public version.
