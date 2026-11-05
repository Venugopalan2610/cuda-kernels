"""Run the Triton beta step and the CUDA beta kernel once each, for ncu.

The warm-up runs (Triton compiles its kernels on the first call) happen with the
profiler off. Only the measured launches run with the profiler on. The Triton call
launches 3 kernels: an add (local + beta), a fill, then the step.

Run (the optional last argument is B, the number of batch rows; the default is 16):
    ncu --set full --profile-from-start off -o beta_both \
        python profile_beta.py
    ncu-ui beta_both.ncu-rep
"""

import sys

import torch

import test_beta
from test_beta import beta_step_triton, build_pred_succ_tables, \
    compile_if_changed, load_kernel, make_inputs

if len(sys.argv) > 1:
    test_beta.B = int(sys.argv[1])
print("B =", test_beta.B)

compile_if_changed()
cuda_beta = load_kernel()
_, succs = build_pred_succ_tables(device="cuda")
log_beta_next, log_local_next = make_inputs(seed=0)
out = torch.empty_like(log_beta_next)

# Warm-up, not profiled.
beta_step_triton(log_beta_next, log_local_next, succs)
cuda_beta(log_beta_next, log_local_next, out)
torch.cuda.synchronize()

torch.cuda.profiler.start()
beta_step_triton(log_beta_next, log_local_next, succs)   # launches 1-3: Triton (add, fill, step)
cuda_beta(log_beta_next, log_local_next, out)            # last launch: CUDA
torch.cuda.synchronize()
torch.cuda.profiler.stop()
