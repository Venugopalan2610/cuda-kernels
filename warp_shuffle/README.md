# warp_shuffle

One warp of 32 threads reduces its 32 values with a butterfly of `__shfl_xor_sync` at offsets 16, 8, 4, 2, 1, with no shared memory. After the five steps every lane holds the result, which the harness checks lane by lane.

The reduction is a max, and the harness checks a max. The kernel's name, `warp_sum`, and its header comment still say sum.

```bash
./run.sh          # prints each lane and PASS or FAIL
```
