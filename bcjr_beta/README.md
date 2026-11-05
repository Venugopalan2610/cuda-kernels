# bcjr_beta

One step of the BCJR backward recursion from [BCJR-QAT](https://arxiv.org/abs/2605.10655), my paper on trainable 2-bit trellis quantization. It is the mirror of [bcjr_alpha](../bcjr_alpha/). For each of 16 batch rows and 65,536 trellis states:

```
log_beta_t[p] = logsumexp over c in 0..15 of ( log_local_next[s] + log_beta_next[s] ),   s = ((p << 4) | c) & 0xFFFF
```

The reference is my Triton code from the paper, [quant-2bit](https://github.com/Venugopalan2610/quant-2bit), as the paper calls it: an add kernel (local + beta), a fill, then the step kernel with a successor table.

| Kernel | Nsight Compute, one launch | Loop of 200 launches | vs floor |
|---|---|---|---|
| floor (`torch.add`, same 3 arrays) | | 30.0 µs | 1.00× |
| Triton (my paper), 3 kernels | 126.78 µs | 118.3 µs | 3.95× |
| CUDA (my rewrite) | 29.54 µs | 32.9 µs | 1.10× |

The floor and the loop are the same as in bcjr_alpha. Nsight Compute times each launch alone, and the loop times launches back to back, so the two columns do not compare with each other.

## The layout

The 16 predecessors that share a set of successors are the ones with the same bottom 12 bits: the shift pushes the top 4 bits out, and the mask drops them. In a plain layout (one thread per state, consecutive states in a block), those 16 predecessors are 4,096 apart, so they sit in 16 different blocks.

My first tiled version gave each block 256 consecutive predecessors and copied their 4,096 successors into shared memory. Every successor came into the SMs 16 times, once per block that needed it:

| Nsight Compute | Tiled (first try) | Regrouped (this kernel) |
|---|---|---|
| Time | 67.23 µs | 29.54 µs |
| DRAM → L2 | 8.40 MB | 8.40 MB |
| L2 → L1 | 131.9 MB | 8.39 MB |
| L2 busy | 77.35% | 21.77% |

The kernel here regroups the states. One block owns 16 bottoms and all 16 tops of each, `p = (top << 12) | bottom`. Those 256 predecessors share only 256 successors, `s = blockIdx.x * 256 + t`, so each thread copies one value into a 16 × 17 shared tile, and each successor comes in once. The L2 → L1 traffic drops to the size of the data, the same 8.39 MB as alpha.

```bash
./run.sh                                  # check, then time both
python3 test_beta.py --check              # check only
ncu --set full --profile-from-start off -o beta_both python3 profile_beta.py
```

RTX 4080 Laptop GPU, performance power mode, 2026. The test passes against Triton on inputs between −30 and 0 and on inputs near −1000, with a tolerance of 1e-4 + 1e-6 × |value|.
