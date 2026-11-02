// Your kernel: one warp finds the max of its 32 values with shuffles.
// Launched with 1 block of 32 threads. Every lane writes the max to out[lane].

// The input of each lane. The harness uses the same function for the CPU max.
__host__ __device__ int input(int lane) { return (lane * 7) % 32; }

__global__ void warp_max(int* out)
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
