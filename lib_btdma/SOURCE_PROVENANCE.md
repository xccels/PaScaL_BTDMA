# Source provenance

The four source files in this directory were copied without modification from
the `proj14_bench_rewrite/lib_btdma` development tree at Git commit
`9d9c19d127f0265905373cfae64ac5b6b541c9a7`.

| File | SHA-256 |
| --- | --- |
| `mpiutil.f90` | `7329f702403a1a9a0a91bf98356e313bac2050ea78d9cb5a3f4e82c8c97ab874` |
| `mod_cudatools.f90` | `30c223d7a7696ef8d8406f61e3543dd0dfe4cb1a35ac6a09b0ddbf6609bd0fc1` |
| `mod_btdma_cpu.f90` | `cc16833393e3e081780d506a599e40224e3fbdbf902c4d98bd0513c04cbaecb9` |
| `mod_btdma_gpu_v2.f90` | `811642b0cdbea34c5b4b1189474af2441071275c253a8529db3a6ebb37b54322` |

The optimized GPU implementation uses fixed-size thread-local work arrays of
extent eight.  Its supported and recommended range is therefore `m <= 8`.

