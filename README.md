# cuda-kernels

Small CUDA exercises, each measured against the limit of the hardware it runs on.

| Exercise | What it asks | Result |
|---|---|---|
| [transpose](transpose/) | Transpose a 16,384 × 16,384 float matrix (1 GiB) | 12.313 ms naive → 5.911 ms tiled; a plain copy takes 5.630 ms, cuBLAS 5.535 ms |
| [softmax](softmax/) | Row softmax over 16,384 rows of 4,096 floats | 1.416 ms, 97% of a plain copy of the same bytes |
| [pointer_chase](pointer_chase/) | One thread follows a chain of dependent loads through arrays from 16 KB to 512 MB | the latency of each level of the memory hierarchy, in clock ticks |
| [warp_shuffle](warp_shuffle/) | One warp reduces its 32 values with shuffles | every lane ends with the result |

## Who wrote what

I wrote the kernels: the `.cu` file named after each exercise (`transpose.cu`, `softmax.cu`, `chase.cu`, `warp_shuffle.cu`). An AI tutor wrote the harnesses (`harness.cu`) and the `run.sh` scripts, which set up the data, check every output against the CPU, and time the runs. The comments at the top of each kernel file are the exercise as it was set; the code under them is mine, as I wrote it.

## Running them

Each folder builds and runs with one script:

```bash
cd transpose && ./run.sh
```

The scripts call `/usr/local/cuda/bin/nvcc` with `-arch=sm_89`. Change the architecture for a card other than an Ada one (an RTX 40-series).

All results here are from an RTX 4080 Laptop GPU (432 GB/s peak memory bandwidth, 48 MB L2) in 2026, with the laptop in its performance power mode. Your numbers will differ. The comparisons with the plain copy are the part worth checking.

Write-ups: [venugopalaniyengar.si](https://venugopalaniyengar.si/projects/)
