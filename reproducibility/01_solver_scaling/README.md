# Solver strong and weak scaling

This directory reproduces the isolated FP64 solver-scaling experiment. Its purpose is to compare the computation/communication trade-off of conventional global all-to-all reassembly and PaScaL_BTDMA, and to separate the original GPU implementation from the optimized transposed-layout implementation. It is a performance experiment, not a correctness test; run the repository examples or the dedicated accuracy experiment independently before using these timings.

No historical timing is shipped here. Every file under `output/` is created by a fresh run and is ignored by Git.

## Test matrix

| Quantity | Strong scaling | Weak scaling |
|---|---:|---:|
| precision | FP64 | FP64 |
| block size `m` | `2, 5, 8` | `2, 5, 8` |
| logical scaling unit `P` | `2, 4, 8` | `2, 4, 8` |
| batch size `nsys` | `128^2 = 16384` | `128^2 = 16384` |
| global block rows `N` | `2048` | `256 P = 512, 1024, 2048` |

The CPU and GPU meanings of `P` are intentionally different because that is how the manuscript benchmark and the legacy run scripts were defined:

- CPU: `P` is the number of sockets. One MPI rank runs on each core, with 24 ranks per socket, so `P={2,4,8}` means `48,96,192` MPI ranks.
- GPU: `P` is the number of GPUs. One MPI rank runs on each GPU, so `P={2,4,8}` also means `2,4,8` MPI ranks.

Consequently, the weak-scaling value `Nsub=256` is per logical scaling unit. It is per GPU/rank on the GPU, but per 24-rank CPU socket on the CPU. The machine-readable output records `logical_P`, `mpi_ranks`, `Nlocal_min`, and `Nlocal_max` so this distinction cannot be lost. This mapping follows the legacy CPU commands (`48/96/192` ranks with `N=512/1024/2048`) rather than silently treating a socket as one MPI rank.

## Compared implementations

| Executable | Method | Solver source |
|---|---|---|
| `bench_cpu_alltoall` | CPU conventional all-to-all reassembly followed by sequential BTDMA | `../../lib_btdma/mod_btdma_cpu.f90` |
| `bench_cpu_pascal` | CPU PaScaL_BTDMA | `../../lib_btdma/mod_btdma_cpu.f90` |
| `bench_gpu_alltoall` | GPU conventional all-to-all reassembly followed by the original sequential GPU BTDMA | `legacy/mod_btdma_gpu.f90` |
| `bench_gpu_pascal_unoptimized` | original multi-kernel GPU PaScaL_BTDMA with `(m,m,nsys,Nlocal)` layout | `legacy/mod_btdma_gpu.f90` |
| `bench_gpu_pascal_optimized` | optimized/fused GPU PaScaL_BTDMA with `(nsys,Nlocal,m,m)` layout | `../../lib_btdma/mod_btdma_gpu_v2.f90` |

The local legacy module is an unchanged copy of `proj14_bench_rewrite/lib_btdma/mod_btdma_gpu.f90`; it is kept here because it is a benchmark comparator, not part of the distributed core API. The CPU and optimized GPU variants always compile from the public core through the relative path `../../lib_btdma`.

The five drivers were rewritten from these legacy entry points:

- `0_alltoall/sample_cpu_a2a.f90`
- `0_alltoall/sample_gpu_a2a.f90`
- `1_btdma/sample_cpu.f90`
- `1_btdma/sample_gpu.f90`
- `2_btmda_opt/sample_gpu.f90`

The rewrite makes sizes runtime arguments, records every repeat, maps GPUs by node-local rank, uses a deterministic diagonally dominant non-cyclic system, and reports the slowest-rank time. It also corrects two plan-selection mistakes in the legacy CPU all-to-all driver: matrix `C` is unpacked with the matrix plan, and the returned solution is unpacked with the center/vector (`A`) side of the vector plan.

## Mango build

The build is pinned to the Mango toolchain used for this package:

```bash
module purge
module load nvidia-hpc-sdk-hpcx/25.3 cuda/12.8
./build_mango.sh
```

`build_mango.sh` uses `mpifort`, `-O3`, CUDA Fortran, and `-gpu=cc90,cuda12.8`. CPU linking names LAPACK and BLAS explicitly. Set `PASCAL_MPIFC`, `PASCAL_EXTRA_CPU_FLAGS`, or `PASCAL_EXTRA_GPU_FLAGS` only when auditing a deliberate toolchain variation; record such a variation with the results.

The build creates `build/` and `bin/`, both ignored by Git. A successful build produces exactly five executables matching the table above.

## Mango runs

Submit CPU and GPU jobs separately because their resource mappings differ:

```bash
sbatch run_strong_cpu_mango.slurm
sbatch run_weak_cpu_mango.slurm
sbatch run_strong_gpu_mango.slurm
sbatch run_weak_gpu_mango.slurm
```

Defaults are two warm-ups and five recorded repeats. They can be changed at submission time, for example:

```bash
WARMUPS=3 REPEATS=10 sbatch run_strong_gpu_mango.slurm
```

The scripts assume the benchmark topology used by their Slurm headers:

- CPU allocation: four nodes, two 24-core sockets per node, 48 MPI ranks per node.
- GPU allocation: one node with eight visible GPUs, one rank per GPU.

Do not run the scripts unchanged on a different topology and label the result as the manuscript configuration. In particular, an eight-node/one-GPU-per-node run is an inter-node experiment, not a reproduction of the single-node eight-GPU scaling curve. If Mango exposes a different Slurm MPI plugin, set `MANGO_SRUN_MPI=pmi2` (or the site-provided value) instead of the default `pmix`.

Each job records the loaded modules, compiler version, host information, Git revision when available, and GPU inventory for GPU runs. The scripts do not build inside the timed job steps.

## Timing and output

Initialization, allocation, plan creation, warm-up iterations, and cleanup are outside the recorded interval. Each recorded value is reduced with `MPI_MAX`, so it represents the slowest rank for that metric.

Every repeat emits one `BTDMA_RESULT` CSV record. The common phase columns mean:

| Column | PaScaL_BTDMA | conventional all-to-all |
|---|---|---|
| `phase1_s` | local modified Thomas procedure | `0` |
| `phase2_s` | forward boundary exchange | forward full-data redistribution |
| `phase3_s` | reduced-system solution | sequential BTDMA after reassembly |
| `phase4_s` | backward boundary exchange | backward solution redistribution |
| `phase5_s` | local update | `0` |

`compute_s` is phases 1+3+5 for PaScaL_BTDMA and phase 3 for all-to-all. `communication_s` is phases 2+4. `total_s` is an independent outside wall-clock interval; small differences from `compute_s + communication_s` can occur because every column is reduced independently and timing calls have overhead.

Logs and CSV files are written under `output/<job-id>/`. To collect an arbitrary set of logs again:

```bash
python3 collect_results.py output/<job-id>/*/*.log \
  --raw output/results_raw.csv \
  --summary output/results_summary.csv
```

The summary reports the minimum, median, mean, and sample standard deviation. Retain the raw CSV and `environment.txt` with any figure or table derived from a run.

## Verification status

The package has been checked locally for file paths, shell syntax, Python syntax, consistent test matrices, and source/API correspondence. It has not been compiled with NVHPC 25.3 or executed on Mango in this preparation step. CUDA-aware HPC-X behavior, GPU binding, the availability of an eight-GPU Mango node, runtime memory capacity, and production timings therefore remain to be verified on the target allocation. The generated results must not be presented as reproducing the paper until those checks and the full runs complete.

