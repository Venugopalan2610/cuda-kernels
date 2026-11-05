"""Test and timer for the CUDA beta step.

The same test as test_alpha.py, for the mirror recursion.

The contract beta.cu meets (one C function, called with device pointers):

    extern "C" void beta_step(const float* log_beta_next,   // (B, S), row-major
                              const float* log_local_next,  // (B, S), row-major
                              float* log_beta_t,            // (B, S), output
                              int B, int S);

    log_beta_t[b, p] = logsumexp over c in 0..15 of
                       ( log_local_next[b, s] + log_beta_next[b, s] ),   s = ((p << 4) | c) & 0xFFFF

Launch on the default stream (the <<<grid, block>>> form with no stream argument).

Run:
    python test_beta.py            # check + time
    python test_beta.py --check    # check only

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
from bcjr.bcjr_triton import beta_step_triton             # noqa: E402
from bcjr.forward_backward import build_pred_succ_tables  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "beta.cu")
LIBRARY = os.path.join(HERE, "beta.so")

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
    library.beta_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_int]
    library.beta_step.restype = None

    def run(log_beta_next, log_local_next, out):
        library.beta_step(log_beta_next.data_ptr(), log_local_next.data_ptr(),
                           out.data_ptr(), B, S)
    return run


def make_inputs(seed, deep=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # Values like a real step: log probabilities spread over a wide range.
    log_beta_next = -30 * torch.rand(B, S, device="cuda", generator=generator)
    if deep:
        # Early in the sequence, betas are far below zero: exp() of them is 0 in fp32.
        log_beta_next = log_beta_next - 1000
    log_local_next = -10 * torch.rand(B, S, device="cuda", generator=generator)
    return log_beta_next, log_local_next


def check(cuda_beta, succs, deep=False):
    print("deep test (betas near -1000):" if deep else "normal test (betas between -30 and 0):")
    log_beta_next, log_local_next = make_inputs(seed=0, deep=deep)
    expected = beta_step_triton(log_beta_next, log_local_next, succs)
    got = torch.full_like(expected, float("nan"))
    cuda_beta(log_beta_next, log_local_next, got)
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
    cuda_beta = load_kernel()
    _, succs = build_pred_succ_tables(device="cuda")

    passed = check(cuda_beta, succs) and check(cuda_beta, succs, deep=True)
    if not passed:
        sys.exit(1)
    if "--check" in sys.argv:
        return

    def triton_as_called(nxt, local, out):
        beta_step_triton(nxt, local, succs)   # the full call: an add kernel, then the step

    def floor(nxt, local, out):
        torch.add(nxt, local, out=out)   # the same bytes: read 2 arrays, write 1, no math

    rows = [("floor", time_us(floor)),
            ("triton", time_us(triton_as_called)),
            ("yours", time_us(cuda_beta))]
    floor_us = rows[0][1]
    print()
    for name, us in rows:
        print(f"{name:>8}: {us:7.1f} us   {us / floor_us:4.2f}x floor")


if __name__ == "__main__":
    main()
