# Conventional all-to-all solver kernel

`mod_btdma_gpu.f90` is an exact copy of the conventional GPU block-Thomas
kernel used by the donor all-to-all benchmark:

```text
../../../../0_alltoall/lib_btdma/mod_btdma_gpu.f90
```

SHA-256 at the time of copying:

```text
06b4b6ed2273931a8667a38fe75fe692332b089ef78bd99f64c559c09047e315
```

The public library core remains `../../lib_btdma`. This local copy is included
only because the conventional redistribution baseline needs its sequential
GPU block-Thomas kernel and that baseline is not part of the public API.
