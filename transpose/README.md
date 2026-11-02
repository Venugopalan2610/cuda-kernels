# transpose

Transpose a 16,384 × 16,384 float32 matrix: 1 GiB in, 1 GiB out, far larger than the 48 MB L2, so every byte comes from DRAM. A transpose does no arithmetic, so the target is a plain copy of the same bytes.

`transpose.cu` holds every version in the order I wrote them:

| Kernel | Block | Time | Bandwidth | vs copy |
|---|---|---|---|---|
| `copy_kernel` (the floor) | 32×8 | 5.630 ms | 381 GB/s | 100% |
| `transpose_naive` | 32×8 | 12.313 ms | 174 GB/s | 46% |
| `transpose_tiled`, shared memory | 32×32 | 6.301 ms | 341 GB/s | 89% |
| `transpose_tiled8`, 4 values per thread | 32×8 | 5.911 ms | 363 GB/s | 95% |
| cuBLAS `cublasSgeam` | chosen by cuBLAS | 5.535 ms | 388 GB/s | 102% |

Bandwidth counts one read and one write of the matrix. All runs back to back from `run.sh`, RTX 4080 Laptop GPU, performance power mode, 2026.

- **naive:** each warp reads 32 neighbouring floats from a row and writes them 64 KB apart, down a column of the output.
- **tiled:** the tile moves the column access from DRAM into shared memory, so both the DRAM read and the DRAM write are coalesced. The `+1` in `tile[TILE][TILE+1]` shifts each row by one bank; without it, a column of the tile sits in one bank and the read is a 32-way bank conflict.
- **tiled8:** a 32×8 block in which each thread moves 4 values. My explanation, not yet confirmed with Nsight Compute: an SM holds 1,536 threads, so six 256-thread blocks fit where only one 1,024-thread block did, and each thread has 4 loads in flight instead of 1.

```bash
./run.sh          # builds with -lcublas and runs every mode
```
