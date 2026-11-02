// Your kernel. Write it yourself.
//
// in:  rows x cols, row major
// out: cols x rows, row major
//
// The harness launches a 2D grid:
//   blockDim = (BX, BY) from the command line
//   gridDim  = (ceil(cols / BX), ceil(rows / BY))
// so blockIdx.x / threadIdx.x walk along the columns of the input,
// and blockIdx.y / threadIdx.y walk down its rows.

__global__ void transpose_naive(const float* in, float* out, int rows, int cols)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < cols && j < rows) {
        out[i*rows + j] = in[j*cols + i];
    }
}

// The control: the same bytes in and out, every access coalesced. Write it yourself.
__global__ void copy_kernel(const float* in, float* out, int rows, int cols)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < cols && j < rows) {
        out[j*cols + i] = in[j*cols + i];
    }
}

#define TILE 32
__global__ void transpose_tiled(const float* in, float* out, int rows, int cols)
{
    __shared__ float tile[TILE][TILE+1];
    int x0 = blockIdx.x * TILE;
    int y0 = blockIdx.y * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = x0 + tx;
    int j = y0 + ty;

    if (i < cols && j < rows) {
        tile[ty][tx] = in[(j)*cols + i];
    }
    __syncthreads();
    if ((x0 + ty) < cols && (y0 + tx) < rows) {
        out[(x0 + ty) * rows + (y0 + tx)] = tile[tx][ty];
    }
}

__global__ void transpose_tiled8(const float* in, float* out, int rows, int cols)
{
    __shared__ float tile[TILE][TILE+1];
    int x0 = blockIdx.x * TILE;
    int y0 = blockIdx.y * TILE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = x0 + tx;
    int j = y0 + ty;

    for (int k = 0; k < TILE; k+=blockDim.y) {
        if (i < cols && (j + k) < rows) {
            tile[ty + k][tx] = in[(j + k)*cols + i];
        }
    }
    __syncthreads();
    for (int k = 0; k < TILE; k+=blockDim.y) {
        if ((x0 + ty + k) < cols && (y0 + tx) < rows) {
            out[(x0 + ty + k) * rows + (y0 + tx)] = tile[tx][ty+k];
        }
    }
}
