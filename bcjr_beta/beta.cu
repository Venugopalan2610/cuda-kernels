// Your CUDA beta step. The test (test_beta.py) calls beta_step() below.
//
// log_beta_t[b, p] = logsumexp over c in 0..15 of ( log_local_next[b, s] + log_beta_next[b, s] ),
//                    s = ((p << 4) | c) & 0xFFFF

__global__ void beta_step_kernel(const float* log_beta_next,
                                 const float* log_local_next,
                                 float* log_beta_t,
                                 int S)
{
    int t = threadIdx.x;
    int b = blockDim.y * blockIdx.y + threadIdx.y;

    int top    = t / 16;
    int bottom = blockIdx.x * 16 + t % 16;
    int p      = (top << 12) | bottom; //all bottom bits same in a state share all same successors

    __shared__ float TILE[16][17];
    int s = blockIdx.x * 256 + t;
    TILE[t / 16][t % 16] = log_local_next[b*S + s] + log_beta_next[b*S + s];

    __syncthreads();

    float sum = 0.0f;
    float max = -INFINITY;
    float v[16];
    for (int j = 0; j < 16; j++) {
        v[j] = TILE[t % 16][j];
        if (max < v[j]) {
            max = v[j];
        }
    }
    for (int j = 0; j < 16; j++) {
        sum += expf(v[j] - max);
    }

    log_beta_t[b*S + p] = max + logf(sum); // write part
}

extern "C" void beta_step(const float* log_beta_next,
                          const float* log_local_next,
                          float* log_beta_t,
                          int B, int S)
{
    dim3 block(256, 1);
    dim3 grid((S + block.x -1)/block.x, (B + block.y -1) / block.y);
    beta_step_kernel<<<grid, block>>>(log_beta_next, log_local_next, log_beta_t, S);
}

