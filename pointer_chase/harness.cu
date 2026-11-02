// Harness: builds the chain for each array size, runs your chase kernel, prints ticks per load.
// Usage: ./chase [stride_bytes]   (default 128 bytes: every load lands on a new cache line)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "chase.cu"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main(int argc, char** argv)
{
    int stride_bytes = argc > 1 ? atoi(argv[1]) : 128;
    int stride = stride_bytes / (int)sizeof(int);

    int *d_next, *d_sink;
    long long* d_ticks;
    CHECK(cudaMalloc(&d_sink, sizeof(int)));
    CHECK(cudaMalloc(&d_ticks, sizeof(long long)));

    printf("stride %d bytes\n%12s %16s\n", stride_bytes, "array", "ticks/load");
    for (size_t bytes = 16 << 10; bytes <= (size_t)512 << 20; bytes *= 2) {
        size_t n = bytes / sizeof(int);
        std::vector<int> next(n);
        for (size_t k = 0; k < n; k++) next[k] = (int)((k + stride) % n);

        CHECK(cudaMalloc(&d_next, bytes));
        CHECK(cudaMemcpy(d_next, next.data(), bytes, cudaMemcpyHostToDevice));

        int lap = (int)(n / stride);                 // loads for one full walk of the array
        int steps = lap > (1 << 16) ? lap : (1 << 16);
        chase<<<1, 1>>>(d_next, lap, d_sink, d_ticks);   // warm-up: one walk fills the caches
        chase<<<1, 1>>>(d_next, steps, d_sink, d_ticks); // the timed run
        CHECK(cudaGetLastError());

        long long ticks;
        CHECK(cudaMemcpy(&ticks, d_ticks, sizeof(ticks), cudaMemcpyDeviceToHost));
        if (bytes < (1 << 20)) printf("%9zu KB %16lld\n", bytes >> 10, ticks);
        else                   printf("%9zu MB %16lld\n", bytes >> 20, ticks);
        CHECK(cudaFree(d_next));
    }
    return 0;
}
