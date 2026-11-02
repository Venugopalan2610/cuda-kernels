#!/bin/bash
# Build, then run the control and your softmax on 16,384 rows of 4,096 (256 MB, past the 48 MB L2).
# Usage: ./softmax ROWS COLS BLOCK MODE
set -e
cd "$(dirname "$0")"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -o softmax harness.cu
./softmax 16384 4096 256 copy
./softmax 16384 4096 256 softmax
