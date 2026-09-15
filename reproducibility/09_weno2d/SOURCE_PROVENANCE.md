# Source provenance and known discrepancies

## Sources inspected

- Repository revision: `9d9c19d127f0265905373cfae64ac5b6b541c9a7`
- Development driver: `proj14_bench_rewrite/5_weno/weno_mpi_gpu.f90`
- Development-driver SHA-256: `b873dff4fa5ee6e888c58b15d6470622a39627764ec58b0206a68f23d537f84e`
- Core solver used here: `../../lib_btdma/mod_btdma_gpu_v2.f90`
- Manuscript references: `rejected_munuscript/Manuscript_v6.tex` and
  `revised_munuscript/Manuscript_v7.tex`, Subsection
  `Scalar advection: WENO5 stencil grouping`. Both contain the same case matrix.

The public driver was copied exactly from the development driver and then
restricted to the manuscript matrix. The numerical kernels and the WENO5,
SDIRK4, approximate-factorisation, halo-exchange, and PaScaL_BTDMA call paths
remain in the driver. The PaScaL_BTDMA implementation itself is linked from
`../../lib_btdma`; it is not duplicated in this directory.

## Development-source discrepancies corrected here

1. The development build script defaulted to a `252 x 252` grid. The manuscript
   case is `516 x 516` on a `2 x 2` MPI/GPU decomposition.
2. The development driver fixed `CFL_target` to `2.0`, although the manuscript
   reports CFL values `1.0`, `1.5`, and `2.0`.
3. The development VTK filename was fixed to `cfl2.0`, independently of the
   selected case.
4. The development Slurm comments still referred to `N=256` and `m=2` while
   invoking a `252 x 252`, `m=3` executable.
5. The development README explicitly states that the code had not been built or
   run. Accordingly, source presence is not treated as verification of the
   manuscript values.

The public driver accepts only `cfl1p0`, `cfl1p5`, and `cfl2p0` and maps them to
`1621`, `1080`, and `810` uniform steps, respectively. It aborts if the build or
launch does not use the fixed `516 x 516`, `m=3`, `2 x 2`, four-rank case.

## Verification status

- Manuscript/source condition audit: completed.
- Shell and Python static checks: recorded during package preparation.
- NVHPC compilation on Mango: pending.
- Four-GPU executions and comparison with the manuscript contours and errors:
  pending.
