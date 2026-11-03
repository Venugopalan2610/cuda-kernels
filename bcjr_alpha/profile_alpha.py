"""Run the Triton alpha kernel and your CUDA alpha kernel once each, for ncu.

The warm-up runs (Triton compiles its kernel on the first call) happen with the
profiler off. Only the 2 measured launches run with the profiler on.

Run (the optional last argument is B, the number of batch rows; the default is 16):
    ncu --set full --profile-from-start off -o alpha_both \
        python profile_alpha.py
    ncu --set full --profile-from-start off -o alpha_b8 \
        python profile_alpha.py 8
    ncu-ui alpha_both.ncu-rep
"""

import sys

import torch

import test_alpha
from test_alpha import alpha_step_triton, build_pred_succ_tables, \
    compile_if_changed, load_kernel, make_inputs

if len(sys.argv) > 1:
    test_alpha.B = int(sys.argv[1])
print("B =", test_alpha.B)

compile_if_changed()
cuda_alpha = load_kernel()
preds, _ = build_pred_succ_tables(device="cuda")
log_alpha_prev, log_local_t = make_inputs(seed=0)
out = torch.empty_like(log_alpha_prev)

# Warm-up, not profiled.
alpha_step_triton(log_alpha_prev, log_local_t, preds)
cuda_alpha(log_alpha_prev, log_local_t, out)
torch.cuda.synchronize()

torch.cuda.profiler.start()
alpha_step_triton(log_alpha_prev, log_local_t, preds)   # launch 1: Triton
cuda_alpha(log_alpha_prev, log_local_t, out)            # launch 2: yours
torch.cuda.synchronize()
torch.cuda.profiler.stop()
