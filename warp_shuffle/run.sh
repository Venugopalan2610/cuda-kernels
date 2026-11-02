#!/bin/bash
# Build, then run one warp through your shuffle tree.
set -e
cd "$(dirname "$0")"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -o shuffle harness.cu
./shuffle
