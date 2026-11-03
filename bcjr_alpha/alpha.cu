// Your CUDA alpha step. The test (test_alpha.py) calls alpha_step() below.
//
// log_alpha_t[b, s] = log_local_t[b, s] + logsumexp over j in 0..15 of log_alpha_prev[b, (s >> 4) | (j << 12)]

#include <cmath>
__global__ void alpha_step_kernel(const float* log_alpha_prev,
                                  const float* log_local_t,
                                  float* log_alpha_t,
                                  int S)
{
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    int b = blockDim.y * blockIdx.y + threadIdx.y;

    float sum = 0.0f;
    float max = -INFINITY;
    float v[16];
    for (int j = 0; j<16; j++) {
        int p = s >> 4 | j << 12;
        v[j] = log_alpha_prev[b*S + p];
        if (max < v[j]) {
            max = v[j];
        }
    }
    for (int j = 0; j< 16; j++) {
        sum += expf(v[j] - max);
    }
    log_alpha_t[b*S + s] = max + log_local_t[b*S + s] + logf(sum);
}

extern "C" void alpha_step(const float* log_alpha_prev,
                           const float* log_local_t,
                           float* log_alpha_t,
                           int B, int S)
{
    dim3 block(256, 1);
    dim3 grid((S + block.x -1)/block.x, (B + block.y -1) / block.y);
    alpha_step_kernel<<<grid, block>>>(log_alpha_prev, log_local_t, log_alpha_t, S);
}
