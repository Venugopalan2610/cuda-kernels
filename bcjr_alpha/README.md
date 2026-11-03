# bcjr_alpha

One step of the BCJR forward recursion from [BCJR-QAT](https://arxiv.org/abs/2605.10655), my paper on trainable 2-bit trellis quantization. For each of 16 batch rows and 65,536 trellis states:

```
log_alpha_t[s] = log_local_t[s] + logsumexp over j in 0..15 of log_alpha_prev[(s >> 4) | (j << 12)]
```

The reference is my fused Triton kernel from the paper's code, [quant-2bit](https://github.com/Venugopalan2610/quant-2bit). The CUDA kernel uses one thread per state, and computes the 16 predecessors with a shift and an OR.

| Kernel | Nsight Compute, one launch | Loop of 200 launches | vs floor |
|---|---|---|---|
| floor (`torch.add`, same 3 arrays) | | 30.0 µs | 1.00× |
| Triton (my paper) | 82.88 µs | 80.4 µs | 2.68× |
| CUDA (my rewrite) | 32.51 µs | 33.6 µs | 1.12× |

The floor reads 2 arrays and writes 1, with no math: the bytes of one alpha step. The loop rotates through 8 input sets (96 MB, past the 48 MB L2).

Worst error against Triton: 1.91e-06 on inputs between −30 and 0, and 1.22e-04 on inputs near −1000 (2 float32 steps at that size). RTX 4080 Laptop GPU, performance power mode, 2026.

The Triton kernel reads its predecessors from an 8 MiB int64 table, and every batch row reads the table again from L2. Nsight Compute shows 133.50 MB from L2 to L1 for Triton against 8.39 MB for the CUDA kernel. At 8 batch rows, the Triton number drops to 63.94 MB.

```bash
./run.sh                                  # check, then time both kernels
python3 test_alpha.py --check             # check only
ncu --set full --profile-from-start off -o alpha_both python3 profile_alpha.py      # B = 16
ncu --set full --profile-from-start off -o alpha_b8   python3 profile_alpha.py 8    # B = 8
```

Here the harness is Python, because the reference is a Triton kernel: `test_alpha.py` compiles `alpha.cu` with nvcc into a shared library and calls it through ctypes.
