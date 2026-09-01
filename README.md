# PaScaL_BTDMA

PaScaL_BTDMA is an MPI-parallel library for batches of block tridiagonal
systems. The library source under [`lib_btdma/`](lib_btdma/) is the primary
deliverable. A small CPU/GPU example and fixed manuscript-reproduction recipes
are provided as supporting material.

## Start here

Edit `00_enviroment.sh` when the compiler, MPI, CUDA, or GPU target changes.
The supplied file contains the Mango login-node interactive environment that
was used for the verified build and run.

```bash
source ./00_enviroment.sh
./01_setup.sh
./02_build.sh
```

The default build creates the CPU/GPU libraries and both minimal examples.
To compile all reproducibility programs as well, use:

```bash
./02_build.sh all
```

The example constructs a system with a prescribed solution and reports both a
normalized residual and a relative solution error. A successful CPU or GPU run
ends with `RESULT: PASS`.

## Repository layout

| Path | Role |
| --- | --- |
| [`lib_btdma/`](lib_btdma/) | Public CPU and optimized GPU solver source |
| [`example/`](example/) | Minimal runnable CPU and GPU interfaces |
| [`reproducibility/`](reproducibility/) | Fixed benchmark and application recipes |
| [`00_enviroment.sh`](00_enviroment.sh) | User-editable compiler, MPI, CUDA, and GPU-target environment |
| [`01_setup.sh`](01_setup.sh) | Toolchain check and build-directory setup |
| [`02_build.sh`](02_build.sh) | Library/example build, with optional complete reproducibility build |
| [`build_mango.sh`](build_mango.sh) | Internal CPU/GPU static-library build retained for existing scripts |

## Library contents

| File | Purpose |
| --- | --- |
| `mpiutil.f90` | MPI partitioning and packing utilities used by the CPU solver and applications |
| `mod_btdma_cpu.f90` | MPI-parallel CPU implementation |
| `mod_cudatools.f90` | CUDA Fortran dense-block device routines |
| `mod_btdma_gpu_v2.f90` | Optimized MPI + CUDA Fortran implementation |

The optimized GPU implementation assigns one independent block tridiagonal
system to each CUDA thread, with 64 threads per CUDA block.  Its current
supported and recommended block-size range is `m <= 8`.

The CPU and GPU interfaces use different array orderings:

- CPU matrices: `(m, m, nsys, nrow_sub)`; CPU right-hand side: `(m, nsys, nrow_sub)`
- GPU matrices: `(nsys, nrow_sub, m, m)`; GPU right-hand side: `(nsys, nrow_sub, m)`

## Build environment

`00_enviroment.sh` is the single build-environment file. Its supplied values
target the Mango environment rather than trying to detect arbitrary compilers
and clusters. On another system, edit that file and leave the build scripts
unchanged.

The build creates separate CPU and GPU module directories and static libraries:

```text
include/cpu/
include/gpu/
lib/libpascal_btdma_cpu.a
lib/libpascal_btdma_gpu.a
```

The CPU executable must link LAPACK and BLAS.  The GPU executable must be
compiled and linked with NVHPC CUDA Fortran flags.

## Minimal example

The [`example/`](example/) directory contains matching CPU and GPU programs.
Both construct a deterministic system with a prescribed solution, solve it,
and report the normalized residual and relative solution error with a final
`PASS` or `FAIL` result.

## Reproducibility material

The optional [`reproducibility/`](reproducibility/) directory is organized by
experiment purpose. Every local build script sources the root
`00_enviroment.sh`; the Slurm run scripts retain their experiment-specific
resource placement. Generated measurements are kept separate from the source
package until the corresponding run and provenance checks have completed.

## Verification boundary

The Mango NVHPC toolchain and CUDA-aware MPI path have been used successfully
for the current build and run work. The reorganized `01_setup.sh` and
`02_build.sh` entry points still require one confirmation run on Mango.
`./02_build.sh all` compiles the optional reproducibility programs; it does not
execute every experiment or create manuscript evidence. File presence alone is
not treated as a reproduced numerical or performance result.

## License

PaScaL_BTDMA is distributed under the MIT License.
