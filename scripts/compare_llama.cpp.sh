#!/usr/bin/env bash
# Compare generation with the reference llama-server (M3/M5 helper).
# Usage: compare_llama.cpp.sh [prompt]
set -e
cd "$(dirname "$0")/.."
PROMPT="${1:-介绍下深圳}"
echo "===== qwen3-gpu-infer ====="
./build/tinyinfer -m /root/llm-resource/Models/Qwen3_0.6B.BF16.gguf -n 64 -f <(printf '%s\n' "$PROMPT") 2>&1 | sed -n '/\[output\]/p'
echo "===== llama-server (raw completion, temp=0) ====="
curl -s --max-time 120 http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"n_predict\":64,\"temperature\":0,\"cache_prompt\":false}" \
  | python3 -c "import json,sys; print('[output]', json.load(sys.stdin).get('content',''))"
