// Your kernel. Write it yourself.
//
// in, out: rows x cols, row major. out[r][c] = exp(in[r][c] - m) / d, per row r.
//
// The harness launches one block per row:
//   gridDim  = rows
//   blockDim = BLOCK from the command line (a multiple of 32, at most 1024)
// so blockIdx.x is the row, and threadIdx.x walks along the columns.
//
// Your plan, from 2026-10-27:
//   1. each thread:  walk its columns, keep a running max (and later a running sum)
//   2. each warp:    xor tree over its 32 values                  (registers)
//   3. lane 0:       slots[warp] = warp result                    (shared memory)
//   4. barrier:      __syncthreads()
//   5. every warp:   load the slots, identity in the empty lanes, xor tree
//   6. a second slot array for the second reduction, so no warp overwrites a slot that another warp still reads
//   7. each thread:  write exp(x - m) / d for its columns

#define ARRAY 32

#include <cmath>
__global__ void softmax_kernel(const float* in, float* out, int rows, int cols)
{
    size_t row_start = (size_t)blockIdx.x * cols;
    float max = -INFINITY;
    // block stride loop
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float x = in[row_start + c];
        if(max < x) {
            max = x;
        }
    }

    //warp reduce
    for(int i = 16; i > 0; i=i/2) {
        float val = __shfl_xor_sync(0xffffffff, max, i);
        max = max > val ? max: val;
    }

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    __shared__ float slots[ARRAY];

    if(lane == 0) slots[warp] = max;

    __syncthreads();

    //block reduce
    int numOfWarps = blockDim.x / 32;
    float v = lane < numOfWarps ? slots[lane] : -INFINITY;
    for(int i = 16; i > 0; i=i/2) {
        float val = __shfl_xor_sync(0xffffffff, v, i);
        v = v > val ? v: val;
    }
    max = v;
    
    __shared__ float slots2[ARRAY];

    float sum = 0;
    for (int c = threadIdx.x; c < cols; c+=blockDim.x) {
        float x = in[row_start + c];
        sum += exp(x - max);
    }

    float d = 0;
    for(int i = 16; i> 0; i=i/2) {
        float val = __shfl_down_sync(0xffffffff, sum, i);
        sum += val;
    }
    d = sum;

    if(lane == 0) slots2[warp] = d;

    __syncthreads();

    float v1 = lane < numOfWarps ? slots2[lane] : 0;
    for(int i = 16; i > 0; i=i/2) {
        float val = __shfl_xor_sync(0xffffffff, v1, i);
        v1 = v1 + val;
    }
    d = v1;

    for (int c = threadIdx.x; c < cols; c+=blockDim.x) {
        out[row_start + c] = exp(in[row_start+c] - max)/d;
    }
}
