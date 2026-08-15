#!/usr/bin/env bash
# Build qwen3-gpu-infer (CUDA 12.x, sm_80).
set -e
cd "$(dirname "$0")/.."
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
echo "build done: build/tinyinfer"
