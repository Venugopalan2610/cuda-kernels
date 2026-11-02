// Harness: fills the input, launches your kernel, checks the result, prints the time.
// Usage: ./transpose ROWS COLS BX BY [naive|copy|tiled|tiled8|cublas]   (block BX x BY)
//   copy:  the control (same bytes, all coalesced)
//   tiled:  your shared-memory transpose, one 32 x 32 tile per block (use BX = BY = 32)
//   tiled8: the same tile with 32 x 8 blocks, 4 rows per thread (use BX = 32, BY = 8)
//   cublas: cublasSgeam with op(A) = transpose (BX, BY are ignored). Build with -lcublas
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "transpose.cu"

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

int main(int argc, char** argv)
{
    if (argc != 5 && argc != 6) { printf("usage: %s ROWS COLS BX BY [naive|copy|tiled|tiled8|cublas]\n", argv[0]); return 1; }
    const char* mode = argc == 6 ? argv[5] : "naive";
    bool copy = strcmp(mode, "copy") == 0, tiled = strcmp(mode, "tiled") == 0, tiled8 = strcmp(mode, "tiled8") == 0, cublas = strcmp(mode, "cublas") == 0;
    if (!copy && !tiled && !tiled8 && !cublas && strcmp(mode, "naive") != 0) { printf("unknown mode %s\n", mode); return 1; }
    auto kernel = copy ? copy_kernel : tiled ? transpose_tiled : tiled8 ? transpose_tiled8 : transpose_naive;
    int bx = atoi(argv[3]), by = atoi(argv[4]);
    int rows = atoi(argv[1]), cols = atoi(argv[2]);
    size_t count = (size_t)rows * cols, bytes = count * sizeof(float);

    std::vector<float> h_in(count), h_out(count);
    for (size_t k = 0; k < count; k++) h_in[k] = (float)(k % 1000003);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    CHECK(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(bx, by), grid((cols + bx - 1) / bx, (rows + by - 1) / by);
    if (tiled || tiled8) grid = dim3((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);   // one tile per block
    // cuBLAS is column major: it sees the row-major rows x cols input as a cols x rows matrix.
    // So C (rows x cols in its view, ldc = rows) = transpose of A (cols x rows, lda = cols).
    cublasHandle_t handle;
    if (cublas && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { printf("cublasCreate failed\n"); return 1; }
    const float alpha = 1.0f, beta = 0.0f;
    auto run = [&]() {
        if (cublas)
            cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols,
                        &alpha, d_in, cols, &beta, d_out, rows, d_out, rows);
        else
            kernel<<<grid, block>>>(d_in, d_out, rows, cols);
    };
    run();   // warm-up, also the checked run
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    size_t wrong = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (h_out[copy ? (size_t)r * cols + c : (size_t)c * rows + r] != h_in[(size_t)r * cols + c]) wrong++;
    printf("check: %s (%zu wrong of %zu)\n", wrong ? "FAIL" : "PASS", wrong, count);

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    const int reps = 20;
    CHECK(cudaEventRecord(t0));
    for (int k = 0; k < reps; k++) run();
    CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1));
    float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1)); ms /= reps;
    // Print only the launch values that this mode really used.
    char launch[96];
    if (cublas)
        snprintf(launch, sizeof launch, "launch chosen by cuBLAS (see ncu)");
    else if (tiled || tiled8)
        snprintf(launch, sizeof launch, "block %dx%d, grid %dx%d, tile %dx%d", bx, by, grid.x, grid.y, TILE, TILE);
    else
        snprintf(launch, sizeof launch, "block %dx%d, grid %dx%d, no tile", bx, by, grid.x, grid.y);
    printf("%s rows %d, cols %d, %s: %.3f ms, %.1f GB/s (read + write of the needed bytes)\n",
           mode, rows, cols, launch, ms, 2.0 * bytes / ms / 1e6);

    if (cublas) cublasDestroy(handle);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
