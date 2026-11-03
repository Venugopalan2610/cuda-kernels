"""Test and timer for your CUDA alpha step.

What it does:
  1. Compiles alpha.cu with nvcc into alpha.so (only if alpha.cu changed).
  2. Makes random inputs of the real size (B = 16, S = 65536).
  3. Runs the Triton kernel from the paper as the reference.
  4. Runs your kernel and compares the outputs.
  5. Times both, and a floor: torch.add over the same 3 arrays (2 reads, 1 write).

The contract your alpha.cu must meet (one C function, called with device pointers):

    extern "C" void alpha_step(const float* log_alpha_prev,  // (B, S), row-major
                               const float* log_local_t,     // (B, S), row-major
                               float* log_alpha_t,           // (B, S), output
                               int B, int S);

    log_alpha_t[b, s] = log_local_t[b, s]
                        + logsumexp over j in 0..15 of log_alpha_prev[b, (s >> 4) | (j << 12)]

Launch on the default stream (the <<<grid, block>>> form with no stream argument).

Run:
    python test_alpha.py            # check + time
    python test_alpha.py --check    # check only

Needs PyTorch, Triton and nvcc, and a clone of github.com/Venugopalan2610/quant-2bit
at ~/quant-2bit (or set QUANT_2BIT_SRC to its src folder).
"""

import ctypes
import os
import subprocess
import sys

import torch

# The Triton reference lives in the BCJR-QAT repo (github.com/Venugopalan2610/quant-2bit).
sys.path.insert(0, os.path.expanduser(os.environ.get("QUANT_2BIT_SRC", "~/quant-2bit/src")))
from bcjr.bcjr_triton import alpha_step_triton            # noqa: E402
from bcjr.forward_backward import build_pred_succ_tables  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "alpha.cu")
LIBRARY = os.path.join(HERE, "alpha.so")

B = 16
S = 1 << 16
TOLERANCE = 1e-4         # fp32 noise: the sum order differs between kernels
RELATIVE = 1e-6          # fp32 has about 7 digits: near -1000, one step is 6e-5
ROTATING_SETS = 8        # 8 sets x 12 MB > L2, so every run reads from DRAM


def compile_if_changed():
    if os.path.exists(LIBRARY) and os.path.getmtime(LIBRARY) > os.path.getmtime(SOURCE):
        return
    command = ["nvcc", "-O3", "-arch=sm_89", "-shared", "-Xcompiler", "-fPIC",
               SOURCE, "-o", LIBRARY]
    print("compile:", " ".join(command))
    subprocess.run(command, check=True)


def load_kernel():
    library = ctypes.CDLL(LIBRARY)
    library.alpha_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_int]
    library.alpha_step.restype = None

    def run(log_alpha_prev, log_local_t, out):
        library.alpha_step(log_alpha_prev.data_ptr(), log_local_t.data_ptr(),
                           out.data_ptr(), B, S)
    return run


def make_inputs(seed, deep=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # Values like a real step: log probabilities spread over a wide range.
    log_alpha_prev = -30 * torch.rand(B, S, device="cuda", generator=generator)
    if deep:
        # Late in the sequence, alphas are far below zero: exp() of them is 0 in fp32.
        log_alpha_prev = log_alpha_prev - 1000
    log_local_t = -10 * torch.rand(B, S, device="cuda", generator=generator)
    return log_alpha_prev, log_local_t


def check(cuda_alpha, preds, deep=False):
    print("deep test (alphas near -1000):" if deep else "normal test (alphas between -30 and 0):")
    log_alpha_prev, log_local_t = make_inputs(seed=0, deep=deep)
    expected = alpha_step_triton(log_alpha_prev, log_local_t, preds)
    got = torch.full_like(expected, float("nan"))
    cuda_alpha(log_alpha_prev, log_local_t, got)
    torch.cuda.synchronize()

    allowed = TOLERANCE + RELATIVE * expected.abs()
    error = (got - expected).abs()
    worst = error.max().item()
    print(f"max abs error: {worst:.2e}   (allowed {TOLERANCE:.0e} + {RELATIVE:.0e} x |value|)")
    if torch.isnan(got).any() or (error > allowed).any():
        b, s = divmod(int(torch.nan_to_num(error / allowed, nan=float("inf")).argmax()), S)
        print(f"FAIL. Worst place: b = {b}, s = {s}")
        print(f"  expected {expected[b, s].item():.6f}, got {got[b, s].item():.6f}")
        return False
    print("PASS")
    return True


def time_us(step, repeats=200):
    """Mean time of one step, in microseconds, with inputs that rotate through DRAM."""
    sets = [make_inputs(seed=i) + (torch.empty(B, S, device="cuda"),)
            for i in range(ROTATING_SETS)]
    for i in range(20):
        step(*sets[i % ROTATING_SETS])
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(repeats):
        step(*sets[i % ROTATING_SETS])
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000 / repeats


def main():
    compile_if_changed()
    cuda_alpha = load_kernel()
    preds, _ = build_pred_succ_tables(device="cuda")

    passed = check(cuda_alpha, preds) and check(cuda_alpha, preds, deep=True)
    if not passed:
        sys.exit(1)
    if "--check" in sys.argv:
        return

    def triton_alone(prev, local, out):
        alpha_step_triton(prev, local, preds)

    def floor(prev, local, out):
        torch.add(prev, local, out=out)   # the same bytes: read 2 arrays, write 1, no math

    rows = [("floor", time_us(floor)),
            ("triton", time_us(triton_alone)),
            ("yours", time_us(cuda_alpha))]
    floor_us = rows[0][1]
    print()
    for name, us in rows:
        print(f"{name:>8}: {us:7.1f} us   {us / floor_us:4.2f}x floor")


if __name__ == "__main__":
    main()
