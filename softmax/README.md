# softmax

Row softmax, `exp(x − max) / sum`, over 16,384 rows of 4,096 float32 values (256 MiB). It does little arithmetic per byte, so the target is a plain copy of the same bytes.

One block of 256 threads per row, in three passes: a block-wide max, a block-wide sum of `exp(x − max)`, then the output. Each reduction is a shuffle tree inside every warp, one value per warp through shared memory, and a second shuffle tree.

| Kernel | Time (3 runs) | Bandwidth |
|---|---|---|
| copy (the floor) | 1.371–1.387 ms | 387–392 GB/s |
| softmax | 1.415–1.417 ms | 379 GB/s |

97% of the copy, 88% of the 432 GB/s peak. Worst error 7.17e-09 against a double-precision CPU reference. RTX 4080 Laptop GPU, performance power mode, 2026.

The kernel reads each row three times and still moves the bytes of one pass: Nsight Compute shows 268.48 MB read from DRAM against the copy's 268.46 MB. The second and third reads hit in L2.

```bash
./run.sh          # the copy control, then the softmax
```
