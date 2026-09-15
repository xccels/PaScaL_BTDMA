# Throughput versus batch size

This directory reproduces the solver-throughput workload used for the two
panels of the manuscript's throughput figure. It measures the CPU and optimized
GPU PaScaL_BTDMA implementations from `../../lib_btdma`; it does not run an
application or the conventional all-to-all baseline.

No timing result is distributed in this directory. Files under `output/` are
created only by a fresh run.

## Exact test matrix

All cases use FP64 and `N=2048`. The tested batch sizes are
`nsys={16^2,32^2,64^2,128^2,256^2}`.

| Figure data | Backend | `P` | `m` |
| --- | --- | ---: | --- |
| panel (a) | CPU | 8 sockets | 2, 5, 8 |
| panel (a) | GPU | 8 GPUs | 2, 5, 8 |
| panel (b) | GPU | 2, 4, 8 GPUs | 8 |

The `P=8`, GPU, `m=8` run belongs to both panels and is executed only once.
The complete matrix therefore contains 15 CPU configurations and 25 GPU
configurations.

`P` follows the manuscript convention rather than having the same resource
meaning on both backends:

- CPU `P=8` means eight 24-core sockets, four nodes, and 192 MPI ranks.
- GPU `P={2,4,8}` means the same number of GPUs and MPI ranks, all within one
  eight-H200 node.

The CPU code and workload can be replayed on Mango, but the historical CPU
figure was measured on the AMD EPYC 74F3 platform described in the manuscript.
A Mango CPU replay is not evidence that the historical hardware timings were
independently reproduced. The GPU run likewise requires the intended H200
node before it can be compared directly with the manuscript figure.

## What is timed

Allocation, plan creation, array initialization, warm-up iterations, and
cleanup are outside the recorded interval. Each repeat records the slowest-rank
time using `MPI_MAX`.

The computation time used for throughput is the sum of PaScaL_BTDMA Steps 1,
3, and 5. Steps 2 and 4 form the communication time. The reported throughput is

```text
throughput = nsys * N * m^3 / computation_time
```

This is the manuscript's block-work throughput definition, not a hardware
counter measurement of floating-point instructions. The collector selects the
repeat with the minimum total solver time and uses the computation time from
that same repeat. Every repeat remains available in `results_raw.csv`.

## Build on Mango

From this directory:

```bash
source ../../00_enviroment.sh
./build_mango.sh
```

The build uses NVHPC-HPCX 25.3, CUDA 12.8, and `cc90`. It first builds the core
libraries through `../../build_mango.sh`, then creates:

```text
bin/throughput_cpu
bin/throughput_gpu
```

## Run on Mango

CPU and GPU resources cannot be requested by the same Slurm header, so the one
run script is submitted twice with explicit resource requests.

CPU panel data:

```bash
sbatch --partition=cpu --nodes=4 --ntasks-per-node=48 --mem=0 --exclusive \
  --export=ALL,MODE=cpu run_mango.slurm
```

GPU panel data:

```bash
sbatch --partition=gpu --nodes=1 --ntasks-per-node=8 --gres=gpu:8 --exclusive \
  --export=ALL,MODE=gpu run_mango.slurm
```

The defaults are two warm-ups and five recorded repeats. To change them while
retaining the values in the environment record:

```bash
sbatch --partition=gpu --nodes=1 --ntasks-per-node=8 --gres=gpu:8 --exclusive \
  --export=ALL,MODE=gpu,WARMUPS=3,REPEATS=10 run_mango.slurm
```

Set `MANGO_SRUN_MPI` only if Mango's site configuration requires an `srun`
plugin other than the default `pmix`.

## Output and aggregation

Each submission writes its environment, raw logs, and CSV files under
`output/<job-id>/<cpu|gpu>/`. The collector creates:

- `results_raw.csv`: every recorded repeat;
- `results_summary.csv`: one minimum-total-time row per configuration;
- `panel_a_P8_cpu_gpu.csv`: the available `P=8` CPU/GPU rows;
- `panel_b_gpu_m8.csv`: the available GPU `m=8` rows.

After both jobs finish, combine their logs into one complete set:

```bash
python3 collect_results.py \
  output/<cpu-job-id>/cpu/*.log \
  output/<gpu-job-id>/gpu/*.log \
  --output-dir output/combined
```

The complete combined summary has 40 configurations. Preserve the raw logs,
`environment.txt` files, and unmodified CSV output with any regenerated figure.

## Source provenance and verification boundary

The drivers were rewritten from
`proj14_bench_rewrite/1_btdma/sample_cpu.f90` and
`proj14_bench_rewrite/2_btmda_opt/sample_gpu.f90`. They call the public CPU and
optimized GPU library modules directly through `../../lib_btdma` and use the
current supported block-size range `m<=8`.

This preparation step does not include an NVHPC build or a Mango execution.
Compiler compatibility, Slurm placement, CUDA-aware MPI behavior, available
memory at `nsys=256^2`, and the resulting performance values must still be
verified on Mango. This is a performance workload; numerical correctness is
checked separately by the repository examples and accuracy experiment.
