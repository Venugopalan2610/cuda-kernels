// Harness for the softmax exercise.
//
// Usage: ./softmax ROWS COLS BLOCK [softmax|copy]
//   softmax: your kernel in softmax.cu
//   copy:    the control. It reads each row once and writes it once,
//            so it moves the bytes of an ideal softmax.
//
// Steps in main():
//   1. read the arguments
//   2. fill the input with scores from -60 to +60
//   3. run the kernel once and check every output against the CPU
//   4. time 20 runs and print ms and GB/s
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "softmax.cu"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

typedef void (*RowKernel)(const float*, float*, int, int);

// ---------------------------------------------------------------------------
// The control kernel: one block per row, each thread copies its columns.
// ---------------------------------------------------------------------------
__global__ void copy_rows(const float* in, float* out, int rows, int cols)
{
    size_t row_start = (size_t)blockIdx.x * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        out[row_start + c] = in[row_start + c];
}

// ---------------------------------------------------------------------------
// 1. Arguments
// ---------------------------------------------------------------------------
struct Args {
    int rows, cols, block;
    const char* mode;   // "softmax" or "copy"
    bool copy;
};

static Args parse_args(int argc, char** argv)
{
    if (argc != 4 && argc != 5) {
        printf("usage: %s ROWS COLS BLOCK [softmax|copy]\n", argv[0]);
        exit(1);
    }
    Args a;
    a.rows  = atoi(argv[1]);
    a.cols  = atoi(argv[2]);
    a.block = atoi(argv[3]);
    a.mode  = argc == 5 ? argv[4] : "softmax";
    a.copy  = strcmp(a.mode, "copy") == 0;

    if (!a.copy && strcmp(a.mode, "softmax") != 0) {
        printf("unknown mode %s\n", a.mode);
        exit(1);
    }
    if (a.block % 32 != 0 || a.block < 32 || a.block > 1024) {
        printf("BLOCK must be a multiple of 32, from 32 to 1024\n");
        exit(1);
    }
    return a;
}

// ---------------------------------------------------------------------------
// 2. Input
// Scores from -60 to +60. exp(60) is about 1e26, and a sum of many of them
// overflows a float. So a softmax without the max fails on this input.
// ---------------------------------------------------------------------------
static void fill_scores(std::vector<float>& x)
{
    srand(1);
    for (size_t k = 0; k < x.size(); k++)
        x[k] = 120.0f * rand() / RAND_MAX - 60.0f;
}

// ---------------------------------------------------------------------------
// 3. Check
// ---------------------------------------------------------------------------

// The softmax of one row, in double precision, as the reference.
static void cpu_softmax_row(const float* x, double* y, int cols)
{
    double m = x[0];
    for (int c = 1; c < cols; c++) m = fmax(m, (double)x[c]);

    double d = 0;
    for (int c = 0; c < cols; c++) d += exp(x[c] - m);

    for (int c = 0; c < cols; c++) y[c] = exp(x[c] - m) / d;
}

// Compares every output with the reference. Prints one line, returns the number wrong.
static size_t check_output(const Args& a, const std::vector<float>& in, const std::vector<float>& out)
{
    const double abs_tol = 1e-6, rel_tol = 1e-4;
    std::vector<double> want(a.cols);
    size_t wrong = 0;
    double worst = 0;

    for (int r = 0; r < a.rows; r++) {
        const float* x = &in[(size_t)r * a.cols];
        const float* y = &out[(size_t)r * a.cols];

        if (a.copy) for (int c = 0; c < a.cols; c++) want[c] = x[c];
        else        cpu_softmax_row(x, want.data(), a.cols);

        for (int c = 0; c < a.cols; c++) {
            double err = fabs(y[c] - want[c]);
            // Written with ! so that a NaN output also counts as wrong.
            if (!(err <= abs_tol + rel_tol * fabs(want[c]))) wrong++;
            if (!(err <= worst)) worst = err;
        }
    }
    size_t count = (size_t)a.rows * a.cols;
    printf("check: %s (%zu wrong of %zu, worst error %.3g)\n", wrong ? "FAIL" : "PASS", wrong, count, worst);
    return wrong;
}

// ---------------------------------------------------------------------------
// 4. Timing
// ---------------------------------------------------------------------------

// Average time of one launch, in ms, over `reps` launches.
static float time_kernel(RowKernel kernel, const Args& a, const float* d_in, float* d_out, int reps)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int k = 0; k < reps; k++)
        kernel<<<a.rows, a.block>>>(d_in, d_out, a.rows, a.cols);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / reps;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // 1. Arguments
    Args a = parse_args(argc, argv);
    RowKernel kernel = a.copy ? copy_rows : softmax_kernel;
    size_t count = (size_t)a.rows * a.cols;
    size_t bytes = count * sizeof(float);

    // 2. Input, on the CPU and on the GPU
    std::vector<float> h_in(count), h_out(count);
    fill_scores(h_in);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_out, 0xff, bytes));   // NaN, so a missing write is visible

    // 3. One run, then check every output. This run also warms up the GPU.
    kernel<<<a.rows, a.block>>>(d_in, d_out, a.rows, a.cols);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    size_t wrong = check_output(a, h_in, h_out);

    // 4. Timing. GB/s counts one read and one write of each row.
    float ms = time_kernel(kernel, a, d_in, d_out, 20);
    printf("%s rows %d, cols %d, block %d, grid %d: %.3f ms, %.1f GB/s (one read + one write of each row)\n",
           a.mode, a.rows, a.cols, a.block, a.rows, ms, 2.0 * bytes / ms / 1e6);

    cudaFree(d_in);
    cudaFree(d_out);
    return wrong ? 1 : 0;
}
