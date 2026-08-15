#!/usr/bin/env bash
# Batch demo: run the 5 prompts from the plan (M4).
set -e
cd "$(dirname "$0")/.."
MODEL="${1:-/root/llm-resource/Models/Qwen3_0.6B.BF16.gguf}"
./build/tinyinfer -m "$MODEL" -n 128 -f demo_prompts.txt
