# Communication time versus global system length

This experiment measures how the communication phase changes with the global
block-row count `N`.  It compares the conventional full-data all-to-all
redistribution with PaScaL_BTDMA on CPU and GPU.  It does not measure solver
correctness or total time, and it does not assume the expected trend has been
observed until the jobs below have actually run.

## Fixed matrix

| Field | Values |
|---|---|
| Resource label | `P=8` |
| Independent systems | `nsys=128^2=16384` |
| Block size | `m={2,5,8}` |
| Global block rows | `N={512,1024,2048,4096}` |
| Precision | FP64 |
| Methods | conventional all-to-all, PaScaL_BTDMA |
| Platforms | CPU, GPU |

`P` follows the manuscript's experimental-setup definition.  On CPU, `P=8`
means eight sockets with 24 MPI ranks per socket (`nprocs=192`, four two-socket
nodes).  On GPU, it means eight GPUs with one MPI rank per GPU (`nprocs=8`, one
eight-GPU node).  The executables reject other process counts so an accidental
small run cannot be labelled as this figure's `P=8` condition.

The CPU and GPU measurements are separate platform experiments.  Combining
their curves on one plot does not make their hardware or network directly
comparable.  In particular, a Mango rerun is new evidence and is equivalent to
the manuscript platform only if the recorded hardware and rank placement match
the original setup.

## What is timed

Every measured repeat emits the maximum phase time across MPI ranks.  The
reported `comm_total_s` is the maximum, over ranks, of each rank's
`forward+backward`; it is not the sum of two independently reduced phase
maxima.  The two phase maxima are retained for diagnosis.

| Method | `comm_forward_s` | `comm_backward_s` |
|---|---|---|
| all-to-all | pack, `MPI_Alltoallv`, and unpack of full `A`, `B`, `C`, and RHS arrays | pack, `MPI_Alltoallv`, and unpack of the full solution array |
| PaScaL CPU | core Step 2 timer: reduced boundary `A`, `B`, `C`, and RHS exchange | current CPU core Step 4 timer: reduced `A`, `B`, `C`, and solution exchange |
| PaScaL GPU | `mod_btdma_gpu_v2` Step 2 timer: reduced boundary `A`, `B`, `C`, and RHS exchange | `mod_btdma_gpu_v2` Step 4 timer: reduced solution exchange |

The current CPU and GPU cores therefore do not have identical Step 4 payloads.
That source-level difference is recorded rather than hidden.  GPU timestamps
include device synchronisation at phase boundaries.  The all-to-all local
block-Thomas solve occurs between the two timed redistribution phases and is
excluded from communication time.

The drivers initialise a deterministic diagonally dominant block-tridiagonal
system (`A=C=-0.25 I`, `B=2 I`, RHS `=1`) and zero the physical end couplings.
This avoids the singular dense blocks produced by filling every block element
with the same scalar.  It is only workload initialisation; this directory does
not replace the independent correctness tests.

## Source provenance

- The experiment conditions come from the communication-versus-`N` figure in
  `Manuscript_v6.tex` and `Manuscript_v7.tex`.
- The conventional redistribution follows `../../../0_alltoall` from the private
  benchmark tree.  `baseline_btdma_gpu.f90` contains only its old-layout local
  GPU block-Thomas phase, renamed to avoid presenting it as public library API.
- CPU PaScaL uses `mod_btdma_cpu`; optimised GPU PaScaL explicitly uses
  `mod_btdma_gpu_v2`.
- Both use the public core at `../../lib_btdma`.  `build_mango.sh` validates
  that relative path and links the libraries built by `../../build_mango.sh`.

No historical timing numbers or generated result files are included here.

## Build on Mango

From this directory:

```bash
source ../../00_enviroment.sh
./build_mango.sh all
```

The scripts default to NVHPC/HPC-X 25.3, CUDA 12.8, and `cc90` through the
repository setup.  Override `MPIFC`, `GPU_ARCH`, or `OPT` only when the change
is also recorded with the resulting data.  The GPU executable requires
CUDA-aware MPI for device-buffer `MPI_Alltoallv` calls.

## Submit the fixed matrix

Allocation-specific Mango account and partition options belong on the `sbatch`
command line.  The intended placements are:

```bash
sbatch --nodes=1 --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 \
  --export=ALL,MODE=gpu run_mango.slurm

sbatch --nodes=4 --ntasks=192 --ntasks-per-node=48 \
  --export=ALL,MODE=cpu run_mango.slurm
```

If the local Mango Slurm version uses a different GPU allocation option, adapt
only that site-specific option while retaining one rank per GPU and record the
final command.  The script checks the node and task counts, runs both methods
for all 12 `(m,N)` pairs, and prints environment metadata before the results.
Defaults are two warm-ups and five measured repeats; override with, for
example, `--export=ALL,MODE=gpu,WARMUPS=3,REPEATS=10`.

The output contract is:

```text
RESULT,platform,method,P_label,nprocs,nsys,m,N,repeat,warmups,repeats,comm_forward_s,comm_backward_s,comm_total_s
```

Each `RESULT` row is already a rank-maximum measurement for one repeat.  To
apply the manuscript's minimum-repeat policy without mixing phases from
different repeats:

```bash
./extract_results.py btdma_comm_N_*.out -o communication_vs_N.csv --require-complete
```

The extractor selects the row with the minimum `comm_total_s` in each of the
48 configurations and retains that same row's forward and backward values.  It
also records the selected source file and line.  Multiple input logs are
allowed, but the minimum is then selected across all supplied runs, so keep the
provenance of those logs together.

## Verification status and limitations

- The four drivers, scripts, and extractor require Mango compilation and job
  execution; this package does not contain a completed benchmark result.
- The historical source printed a minimum from rank 0.  This package instead
  uses a per-repeat MPI rank maximum before minimum selection, which is safer
  for distributed wall time but is not numerically identical to that legacy
  reduction policy.
- CPU and GPU Step 4 payloads differ in the current public source, as documented
  above.  A future source change must update both the timer-scope row and this
  README before comparing new data.
- Hardware identity, CPU socket binding, GPU binding, CUDA-aware MPI operation,
  compiler version, network path, run-to-run variation, and the full 48-case
  result matrix remain unverified until recorded on Mango.
- Plot generation is deliberately outside this small experiment package; the
  extracted CSV is the traceable input to any later figure workflow.
