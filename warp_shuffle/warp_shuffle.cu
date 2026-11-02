// Your kernel: one warp sums its 32 values with shuffles.
// Launched with 1 block of 32 threads. Lane 0 writes the sum to out[0].

// The input of each lane. The harness uses the same function for the CPU sum.
__host__ __device__ int input(int lane) { return (lane * 7) % 32; }

__global__ void warp_sum(int* out)
{
    int lane = threadIdx.x % 32;
    int x = input(lane);

    // TODO: the tree with the offsets 16, 8, 4, 2, 1;
    for(int i = 16; i > 0; i=i/2) {
        int val = __shfl_xor_sync(0xffffffff, x, i);
        x = x > val ? x: val;
    }

    out[lane] = x;
}
