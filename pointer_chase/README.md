# pointer_chase

One thread follows a chain of loads in which every address is the value of the load before it, so no two loads can overlap. The time per load is then the latency of whichever level of the memory hierarchy holds the array. The harness walks arrays from 16 KB to 512 MB and prints the clock ticks per load; the steps in that curve are the L1, the L2 and DRAM.

The default stride is 128 bytes, so every load lands on a new cache line. Pass another stride in bytes as the first argument.

```bash
./run.sh          # 128-byte stride
./run.sh 4        # every int in turn
```
