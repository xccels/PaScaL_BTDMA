# Five-phase GPU solver timing

This directory measures the five phases of the optimized GPU
PaScaL_BTDMA solver.  It is the controlled solver benchmark used to inspect
the sequential reduced-system solve separately from application work.

The fixed cases are:

| Quantity | Value |
|---|---:|
| Backend | optimized GPU PaScaL_BTDMA |
| Precision | FP64 |
| Block size, `m` | 8 |
| Global block rows, `N` | 2048 |
| Independent systems, `n_sys` | 16384 (`128^2`) |
| MPI processes / GPUs, `P` | 2, 4, 8 |
| Sweep-direction processes, `n_proc` | `P` |
| Reduced-system block rows | `2 P` |

This standalone benchmark uses `MPI_COMM_WORLD` as one sweep-direction
communicator.  Its recorded topology is therefore `1x1xP`; it is not a claim
about the Cartesian topology of an application.

## What is timed

The timer boundaries follow the optimized solver source in
`../../lib_btdma/mod_btdma_gpu_v2.f90`:

1. **Step 1, local reduction:** the modified local Block Thomas kernel.
2. **Step 2, forward exchange:** GPU packing, four blocking
   `MPI_Alltoallv` calls for `A`, `B`, `C`, and `D`, and GPU unpacking.
3. **Step 3, reduced solve:** the GPU kernel that solves the reduced systems,
   each containing `2 P` block rows.
4. **Step 4, backward exchange:** GPU packing, one blocking
   `MPI_Alltoallv` call for the boundary solution, and GPU unpacking.
5. **Step 5, local update:** the GPU kernel that recovers the local solution.

The existing phase timer calls `cudaDeviceSynchronize()` at each phase
boundary.  The total timer starts after coefficient initialization, a device
synchronization, and an MPI barrier.  It ends after the solver returns and a
final device synchronization.  Plan construction, allocation, initialization,
warm-up solves, MPI reductions, and file output are outside the reported total.

There is no MPI barrier between individual phases.  For every repeat, each
reported phase and the total are reduced with `MPI_MAX` over all ranks.  This
preserves a conservative wall-time view of each phase.  Because a different
rank can be the maximum for each phase, the sum of the five rank-max phase
times is also reported and need not equal the rank-max total exactly.

## Mango build and run

From this directory:

```bash
source ../../00_enviroment.sh
./build_mango.sh
sbatch run_mango.slurm
```

The job requests one Mango GPU node and runs `P=2,4,8` sequentially.  Defaults
are one warm-up and five measured repeats.  They can be changed at submission:

```bash
sbatch --export=ALL,WARMUPS=2,REPEATS=10 run_mango.slurm
```

Each run requires one MPI rank per GPU and CUDA-aware MPI.

## Output and aggregation

The Slurm job creates `output/<job-id>/` containing:

- `environment.txt`: modules, compiler, MPI, GPU, topology, and source commit;
- `phase_P{2,4,8}.log`: untouched program output for every repeat;
- `phase_raw.csv`: every repeat after rank-max aggregation;
- `phase_summary.csv`: one representative row per process count.

The summary selects the repeat with the smallest rank-max `total_s` for each
`P`, matching the minimum-time convention of the earlier scaling driver.  The
five phase times and `step3_fraction` are copied from that same repeat.  The
parser never combines independently minimized phase values.  It additionally
reports the median and population standard deviation of `total_s` across all
repeats so run stability remains visible.

`step3_fraction` is computed within each repeat as
`step3_s / total_s`.  Results are not included in this directory until the
Mango job has actually been run.

## Scope

The deterministic diagonal-block system keeps the timed arithmetic path
well-defined and has the exact solution `x=1`.  This directory measures solver
phases only.  It is not the condition-number accuracy study, an application
timing breakdown, or evidence for process counts beyond the three measured
cases.
