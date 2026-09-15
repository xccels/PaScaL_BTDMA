# 3D compressible Euler isentropic vortex

This directory reproduces the manuscript's periodic three-dimensional isentropic-vortex application with the GPU PaScaL_BTDMA library in `../../lib_btdma`. It is a fixed Mango recipe, not a general CFD application framework.

## Manuscript case

| Item | Setting |
| --- | --- |
| Equations | Compressible Euler equations with artificial diffusion |
| Implicit method | Crank--Nicolson with three Beam--Warming directional sweeps |
| Block size | `m = 5` conservative variables |
| Domain | `[0,32]^3`, periodic in all directions |
| Vortex | `beta = 5`, `R_c = 2`, free-stream velocity `(1,0,0)` |
| Grid | `256 x 256 x 256` |
| MPI/GPU layout | `2 x 2 x 2` = 8 ranks and 8 GPUs |
| Time step | `dt = dx = 0.125` |
| End time | `t = 32`, one periodic advection |
| Steps | 256 |
| Reported CFL | approximately 3.25; the executable reports the value computed from its initial state |

After one period, the exact solution is the initial vortex translated once through the periodic domain. The executable reports full-grid relative L2 errors for density, pressure, and spanwise vorticity, while the postprocessor also reports the corresponding centreline changes. The manuscript values `1.94e-3`, `2.00e-3`, and `4.43e-4` are copied into `summary.csv` as separate comparison records. The donor does not record whether all three published values used the full grid or the plotted centreline, so they are not attached to either metric as pass criteria.

## Run on Mango

From this directory:

```bash
sbatch run_mango.slurm
```

The job loads the repository's Mango environment, rebuilds the shared GPU library, compiles one fixed executable, and runs it with eight MPI ranks. Results are written to `output/<job-id>/`.

For an interactive build only:

```bash
source ../../00_enviroment.sh
./build_mango.sh
```

`NSTEPS` can shorten a developer smoke test without changing the default manuscript run. For example, `N1=32 N2=32 N3=32 NSTEPS=2 ./build_mango.sh` builds a two-step executable. Such a run is a code check, not manuscript evidence.

## Outputs

- `environment.txt`: commit, compiler/MPI, and GPU environment.
- `run.log`: configuration, measured CFL, validation values, application timing, and the five PaScaL_BTDMA solver phases.
- `centerline_initial_rank*.csv` and `centerline_final_rank*.csv`: rank-local pieces of the x-centreline.
- `summary.csv`: parsed configuration, full-grid L2 errors, centreline diagnostics, and rank-maximum timings.
- `centerline_profiles.csv`: one ordered table for plotting initial and final density, pressure, and spanwise-vorticity profiles.

The application timer excludes the first time step as a warm-up. Every reported timing is reduced with `MPI_MAX`, so the CSV records the slowest-rank time rather than rank 0 alone.

## Evidence boundary

The scripts and source have been checked statically, but the `256^3` eight-GPU case has not been run during package preparation. Do not treat the manuscript comparison values as regenerated results. See [PROVENANCE.md](PROVENANCE.md) for the donor source, the corrected Jacobian history, and the old build/job mismatch.
