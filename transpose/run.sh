#!/bin/bash
# Build, then run every mode on the 16,384 x 16,384 matrix (past the 48 MB L2).
# Usage: ./transpose ROWS COLS BX BY MODE
set -e
cd "$(dirname "$0")"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -o transpose harness.cu -lcublas
n=16384
./transpose $n $n 32 8  naive
./transpose $n $n 32 8  copy
./transpose $n $n 32 32 tiled
./transpose $n $n 32 8  tiled8
./transpose $n $n 32 2  tiled8
./transpose $n $n 32 8  cublas
