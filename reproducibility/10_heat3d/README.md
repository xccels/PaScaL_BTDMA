# 3D fourth-order heat-equation application

This directory provides two fixed Mango recipes for the manuscript's three-dimensional heat-equation application. The application uses Crank--Nicolson time integration, Douglas--Gunn directional splitting, and a fourth-order five-point central difference. Pairing adjacent stencil rows produces the `m=2` block tridiagonal systems solved by the shared GPU library in `../../lib_btdma`.

## Fixed workflows

### Spatial convergence

The manufactured solution is

`u(x,y,z,t) = sin(pi*x) sin(pi*y) sin(pi*z) exp(-3*pi^2*t)`

on the unit cube with `alpha=1` and homogeneous Dirichlet boundary conditions. The six grids are `N={32,64,128,256,512,1024}` internal points per direction. Every grid uses 100 time steps and the same

`dt = 1/[2*alpha*(N_ref+1)^2]`, with `N_ref=1024`.

The executable reports the volume-normalized pointwise `L2` error. `postprocess.py` calculates the observed order between adjacent grids without imposing a pass threshold.

### Strong scaling

The global grid and step count remain fixed at `1024^3` and 100. The Mango recipes use eight GPUs per node:

| GPUs | Mango nodes | Cartesian decomposition | Local internal grid |
| ---: | ---: | --- | --- |
| 8 | 1 | `2x2x2` | `512x512x512` |
| 16 | 2 | `2x2x4` | `512x512x256` |
| 32 | 4 | `2x4x4` | `512x256x256` |
| 64 | 8 | `4x4x4` | `256x256x256` |

The primary scaling metric is the rank-maximum wall time of the complete 100-step application loop. It includes field copies, RHS and matrix construction, three PaScaL_BTDMA solves per step, unpacking, boundary treatment, and halo exchange. Plan creation, initialisation, and the final error calculation are outside this timer.

## Build and run on Mango

From this directory:

```bash
source ../../00_enviroment.sh
./build_mango.sh all
sbatch run_convergence_mango.slurm
sbatch run_scaling_mango.slurm
```

Use `./build_mango.sh convergence` or `./build_mango.sh scaling` to build only one workflow. A single row can be built with its `case_id` from `cases.tsv`. The Slurm files intentionally leave project/account and site-specific partition selection to the submission environment.

Each Slurm recipe repeats every complete case three times by default. Set `REPEATS=1` at submission time for a single reproduction run. Results are written under `output/<job-id>/`:

- `manifest.txt`: source hashes, repository revision, compiler/MPI modules, GPU and allocation information.
- `logs/*.log`: complete application output and one structured `HEAT3D_RESULT` block.
- `*_raw.csv`: one row per repetition.
- `convergence_summary.csv`: median/minimum/maximum error and observed spatial order.
- `scaling_summary.csv`: minimum and median loop time, speedup, and efficiency relative to the eight-GPU Mango run.

## Evidence boundary

The published 8--64 GPU times were collected on the GH200-based Hangang pilot with four GPUs per node. These Mango scripts preserve the global grid, step count, GPU counts, and Cartesian decompositions, but use the Mango node layout of eight GPUs per node. They therefore generate a new platform-specific data set and must not be described as reproducing the published absolute times unless the hardware and placement are matched.

No GPU run was performed while assembling this directory. The scripts, parser, case table, and source wiring were checked statically. See [PROVENANCE.md](PROVENANCE.md) for the donor lineage and corrected script inconsistencies.
