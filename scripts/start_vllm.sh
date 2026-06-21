#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

set -a
source .env
set +a

#MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL="Qwen/Qwen3-0.6B"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 1024 \
    --max-num-seqs 4
    