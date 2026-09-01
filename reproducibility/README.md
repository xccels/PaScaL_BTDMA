# Reproducibility material

This directory contains optional experiment drivers.  The PaScaL_BTDMA library
itself is provided separately in `../lib_btdma/`.

Each numbered directory is self-contained at the experiment level: its README
states the purpose and fixed test matrix, and its scripts show the Mango build
and run commands directly. Generated measurements are not committed as if they
were verified results.

| Directory | Purpose |
| --- | --- |
| [`01_solver_scaling`](01_solver_scaling/) | CPU/GPU strong- and weak-scaling benchmarks |
| [`02_communication_vs_N`](02_communication_vs_N/) | Communication time as the global system length changes |
| [`03_throughput_vs_nsys`](03_throughput_vs_nsys/) | Throughput as the batch size changes |
| [`04_accuracy_conditioning`](04_accuracy_conditioning/) | Solver-only residual, solution error, conditioning, and singular probes |
| [`05_phase_timing`](05_phase_timing/) | Five-phase timing and reduced-system cost |
| [`06_intra_inter_node`](06_intra_inter_node/) | Fixed-resource intra-node and inter-node comparison |
| [`07_compiler_resources`](07_compiler_resources/) | Compile-only register, stack, spill, and theoretical occupancy evidence |
| [`08_euler3d`](08_euler3d/) | Three-dimensional isentropic-vortex application |
| [`09_weno2d`](09_weno2d/) | WENO5 Zalesak slotted-disk application |
| [`10_heat3d`](10_heat3d/) | Heat-equation convergence and strong scaling |

## Suggested order

1. Run the minimal example in [`../example/`](../example/) to verify the CPU
   and GPU interfaces.
2. Run `04_accuracy_conditioning` before interpreting performance data.
3. Run `07_compiler_resources` for the compile-only block-size audit.
4. Run the performance cases `01`, `02`, `03`, `05`, and `06` with their fixed
   allocations.
5. Run the three application cases `08`, `09`, and `10`.

Every result set should retain its environment record, source hashes, raw
repeats, and parser output. The Mango heat-equation scaling recipe is a new
platform run; it is not identical to the original GH200/Hangang placement.
