# Source provenance and reconciled discrepancies

The application source was copied from:

`proj14_bench_rewrite/4_app_euler/euler3d_mpi_gpu.f90`

at repository commit `9d9c19d127f0265905373cfae64ac5b6b541c9a7`. Its SHA-256 before packaging was:

`2c0a7f5756112b6f5f5c7088952f9f54dba88a3eeb102392465a34ffe7baffb4`

After the packaging-only instrumentation described below, the packaged source SHA-256 is:

`539935bb01193b081e1e3011dfbe5844ddcc11b57dbad1809611c982882adeac`

The public-package copy keeps the donor's Euler discretisation and directional cyclic BTDMA calls. Packaging changes add an explicit process-count check, consistent build/run names, the translated-vortex validation for density, pressure, and spanwise vorticity, rank-maximum timing records, and compact centreline CSV output. The shared PaScaL_BTDMA source is linked from `../../lib_btdma`; it is not duplicated here.

The donor executable prints a full-grid relative L2 error for density only. It does not preserve the calculation that produced the manuscript's pressure and vorticity error values, nor does it state whether those two values were reduced over the full grid or the plotted centreline. This package therefore emits both a full-grid analytic-reference metric and a centreline initial/final metric. The three manuscript values are retained as separately labelled comparison records, not as automatic pass thresholds.

## Jacobian history

The donor directory contains two versions. `_backup0_euler.f90` evaluates both off-diagonal blocks with the Jacobian at the current cell. The current `euler3d_mpi_gpu.f90` evaluates the lower and upper blocks with the neighbouring states in all three directions. The latter matches the coefficient equations in `Manuscript_v6.tex` and `Manuscript_v7.tex` and is the version packaged here. The note `4_app_euler/tmp.md` describes the older mismatch and is stale with respect to the current donor source.

## Original build/run mismatch

The original build script defaults to a `128 x 128 x 128` grid and produces a name containing all three grid sizes and `m=5`. Its job script requests two nodes and invokes a differently named `1024 x 1024` executable. This package replaces both with the fixed executable `build/euler3d_mpi_gpu.out` and the manuscript configuration `256 x 256 x 256`, `2 x 2 x 2`, and eight MPI ranks/GPUs.

No large Euler run was performed while preparing this package. Numerical values produced by a future Mango run must be retained as run evidence rather than inferred from the manuscript.
