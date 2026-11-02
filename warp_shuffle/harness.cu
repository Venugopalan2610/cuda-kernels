// Harness: launches your warp_max with one warp, then checks out[0..31] against the CPU max.
// Every lane must write its x to out[lane]. A lane that does not write keeps -1.
// Usage: ./shuffle
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "warp_shuffle.cu"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main()
{
    const int lanes = 32;
    int* d_out;
    CHECK(cudaMalloc(&d_out, lanes * sizeof(int)));
    CHECK(cudaMemset(d_out, 0xff, lanes * sizeof(int)));   // -1, so a missing write is visible

    warp_max<<<1, lanes>>>(d_out);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    int got[lanes];
    CHECK(cudaMemcpy(got, d_out, lanes * sizeof(int), cudaMemcpyDeviceToHost));

    int want = input(0);
    for (int lane = 1; lane < lanes; lane++) want = max(want, input(lane));

    printf("CPU max = %d\n", want);
    int wrong = 0;
    for (int lane = 0; lane < lanes; lane++) {
        printf("lane %2d: %6d%s\n", lane, got[lane], got[lane] == want ? "" : "  <- wrong");
        if (got[lane] != want) wrong++;
    }
    printf("check: %s (%d wrong of %d lanes)\n", wrong ? "FAIL" : "PASS", wrong, lanes);
    cudaFree(d_out);
    return wrong ? 1 : 0;
}
