# Source provenance and evidence boundary

## Application source

The packaged source was copied from

`proj14_bench_rewrite/3_app_heateq/v1_3d/heat3d_mpi_gpu.f90`

at repository commit `9d9c19d127f0265905373cfae64ac5b6b541c9a7`. The donor SHA-256 is

`217ddf5027e514dcb569d691504350e4c129c8c2792324e17556c33e04280ef8`.

The packaged source SHA-256 after the instrumentation below is

`3e7324dca40889d431a79fe04820057dd68ec774f340e14a92a70363c2691e85`.

The numerical kernels, Douglas--Gunn sequence, fourth-order stencil, boundary treatment, manufactured solution, and three non-cyclic GPU BTDMA calls are retained. Packaging changes add a compile-time `N_ref`, process/topology validation, node-local GPU selection, a synchronized rank-maximum application timer, and a structured result block. The PaScaL_BTDMA implementation is linked from `../../lib_btdma` and is not duplicated.

## Donor script corrections

The donor directory mixes exploratory builds and runs:

- its current build script defaults to five steps, whereas the manuscript convergence and scaling evidence uses 100;
- its job files invoke several prebuilt names without creating those exact executables in the job;
- convergence and strong-scaling cases are mixed in broad job lists; and
- the application prints rank-zero timings without a rank-maximum reduction.

This package places every fixed case in `cases.tsv`, uses 100 steps for both workflows, builds names directly from that table, separates convergence from scaling, and records rank-maximum timings.

## Published GH200 evidence versus Mango rerun

The manuscript's original fixed-`1024^3`, 100-step measurements used the Hangang pilot and the following process layouts:

| GPUs | Hangang nodes | Cartesian decomposition | Original raw-log SHA-256 |
| ---: | ---: | --- | --- |
| 8 | 2 | `2x2x2` | `04b6f9b7674ad133d52402ec347998d9fa7778421abb99ddd5bdbaf3f9a8f6aa` |
| 16 | 4 | `2x2x4` | `1098344387abe92c95d5ceec2c156033e154f5e8196c097800a22a29643879d0` |
| 32 | 8 | `2x4x4` | `e27d0ec1a8683ddb12b469bf057de252d6fccdfb60f9e61fbbb1790fb0290b66` |
| 64 | 16 | `4x4x4` | `4adb40d5a96130b58d6fda68b7059052bebcc06adcac9e0f06dfac3279a83728` |

The convergence sweep was preserved in a separate 100-step raw log with SHA-256 `463ca128e1ee4d9111ffbe0e2637118613963b6af95037fdb00985d38cdfd1a2`.

Those internal raw logs are provenance records, not generated output bundled in this directory. Mango has eight GPUs per node in the supplied recipes, so its 8/16/32/64-GPU placements use 1/2/4/8 nodes. A future Mango run is new platform evidence. Absolute timing agreement with the GH200/Hangang data is neither assumed nor used as an automatic pass condition.
