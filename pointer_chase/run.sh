#!/bin/bash
# Build, then chase arrays from 16 KB to 512 MB.
set -e
cd "$(dirname "$0")"
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -o chase harness.cu
./chase "$@"
