# Fixed four-GPU intra-node/inter-node comparison

This experiment compares the optimized GPU PaScaL_BTDMA solver with the
conventional GPU all-to-all redistribution baseline while holding the total
resource count fixed.

| Quantity | Fixed value |
| --- | ---: |
| Block size | `m = 8` |
| Global block rows | `N = 2048` |
| Independent systems | `nsys = 128^2 = 16384` |
| MPI processes / GPUs | `nproc = 4` |
| Intra-node placement | 1 node x 4 GPUs, 4 ranks on the node |
| Inter-node placement | 4 nodes x 1 GPU, 1 rank per node |
| Precision | FP64 |

The drivers reject any other rank placement. This prevents an accidental
single-node run from being labelled as inter-node data.

Both methods use the same deterministic, nonsingular block-tridiagonal input:
`A=-0.25 I`, `B=2.50 I`, `C=-0.25 I`, and right-hand side `D=1`.  The strong
block-diagonal dominance prevents singular-block behavior from contaminating
the timing comparison. Solver accuracy is checked separately by the
accuracy-and-conditioning experiment.

## What is timed

Each repeat is initialized outside the timed region. The phase timers include
the device synchronization already used by the solver source.

- PaScaL_BTDMA computation: Steps 1, 3, and 5.
- PaScaL_BTDMA communication: Steps 2 and 4.
- Conventional all-to-all computation: the block-Thomas solve after the
  forward redistribution.
- Conventional all-to-all communication: the forward and backward
  redistribution phases. These phases include packing, `MPI_Alltoallv`,
  unpacking, and required device synchronization; they are not pure network
  latency measurements.

For each repeat, the driver identifies the rank with the largest phase-sum
total and reports that rank's matching phase, computation, and communication
times, so `t_total = t_computation + t_communication` remains a consistent
breakdown. An enclosing wall-clock timer is reduced separately with `MPI_MAX`
and retained as a diagnostic.

## Build on Mango

From this directory:

```bash
./build_mango.sh
```

The build reuses `../../lib_btdma` for the public PaScaL_BTDMA core. The
conventional baseline additionally uses the exact donor block-Thomas kernel
preserved under `legacy/`; see `legacy/SOURCE_PROVENANCE.md`.

The MPI installation must support CUDA-aware `MPI_Alltoallv`, as in the Mango
NVHPC/HPC-X environment loaded by `../../00_enviroment.sh`.

## Run the two placements

Submit both fixed-resource jobs after building:

```bash
sbatch run_intra_mango.slurm
sbatch run_inter_mango.slurm
```

The default protocol is two warmups followed by five measured repeats. It can
be changed at submission time, for example:

```bash
sbatch --export=ALL,WARMUPS=2,REPEATS=10 run_inter_mango.slurm
```

Each job writes its logs, source checksums, allocation record, host/rank map,
and GPU information under `output/<job-id>/<regime>/`.

## Make the four-row table

After both jobs finish, provide the four log paths to the parser:

```bash
mkdir -p output/final
python3 summarize_node_comparison.py \
  output/<intra-job-id>/intra-node/alltoall.log \
  output/<intra-job-id>/intra-node/pascal.log \
  output/<inter-job-id>/inter-node/alltoall.log \
  output/<inter-job-id>/inter-node/pascal.log \
  --raw output/final/node_comparison_raw.csv \
  --table output/final/node_comparison_table.csv
```

The table CSV contains exactly four rows, ordered by method and then placement.
For each row it selects the repeat with the minimum phase-sum total, records the
communication fraction, and computes the PaScaL_BTDMA speedup against the
all-to-all result from the same placement. The raw CSV retains every repeat and
its source log/line.

## Historical-data boundary

Previously recovered intra-node and inter-node values are separate provenance
evidence. They are not copied into this directory as newly generated output and
are not used by the parser. Fresh files under `output/` become evidence only
with their accompanying manifest and checksums. If archived historical raw
files are distributed later, they should be stored separately with their
original source paths, allocation records, and SHA-256 manifest.

This fixed four-GPU comparison does not by itself establish scaling with node
count or unlimited multi-node scalability.
