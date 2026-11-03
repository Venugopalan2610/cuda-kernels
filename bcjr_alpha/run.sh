#!/bin/bash
# Build alpha.cu, check it against the Triton kernel from BCJR-QAT (normal and deep inputs), then time both.
# Needs PyTorch, Triton, nvcc, and github.com/Venugopalan2610/quant-2bit at ~/quant-2bit
# (or QUANT_2BIT_SRC=/path/to/quant-2bit/src). PYTHON picks the interpreter.
set -e
cd "$(dirname "$0")"
"${PYTHON:-python3}" test_alpha.py
