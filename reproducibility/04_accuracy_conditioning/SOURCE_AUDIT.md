# Singularity status and warning audit

This note records what the packaged solver source exposes **before** any Mango
experiment is run.  It is not a runtime result.

Audited source snapshot: Git commit
`9d9c19d127f0265905373cfae64ac5b6b541c9a7`; hashes are listed in
`../../lib_btdma/SOURCE_PROVENANCE.md`.

## CPU path

- `mod_btdma_cpu.f90` calls LAPACK `dgesv` and receives `info` (for example,
  lines 28 and 38 in the packaged snapshot).
- The implementation does not test `info` after those calls.
- `btdma_many_mpi` has no status output argument (lines 242--271).
- Consequently, the current public CPU call cannot return a singular-factor
  status to its caller and does not issue a library warning.

The independent sequential CPU reference in this experiment also calls
`dgesv`, but unlike the library path it checks and records `info`.  This makes
the reference status observable without changing the library.

## GPU path

- `btdma_many_mpi_gpu_v2` has no solver-status output argument (lines
  110--146).
- The device routine `gesv_mrhs2` divides by the current diagonal entries
  during elimination and back substitution (lines 905--957).  It has no pivot
  test, status buffer, or warning path.
- A CUDA runtime error from `cudaDeviceSynchronize` is recorded separately by
  the experiment.  It must not be interpreted as a factorization-status code.

## What the exact-singular probes can establish

The two small probes record whether the public call returns, whether its output
contains non-finite values, its residual, the CUDA runtime status, and the
process exit status captured by `run_mango.slurm`.  They cannot manufacture a
library status or warning that the current API does not provide.

## Pending library change

If the revised library is required to detect and warn on singular or
near-singular factors, a core change is necessary.  The minimum useful change
is to check CPU `dgesv` information, add a GPU pivot/factor test and per-system
status buffer, propagate an aggregate status through the public CPU/GPU APIs,
and define one documented user-warning policy.  This experiment intentionally
does not implement that API change.
