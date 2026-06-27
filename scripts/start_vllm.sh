#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

set -a
source .env
set +a

# Self-hosted target model on the H100. (Override to Qwen3-0.6B only for CPU dev.)
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Sane starting config for 1x H100 80GB; fine-tuning belongs to Phase 1/6.
#   --max-model-len 8192        : workload is ~1.5-3K-token prompts + short SQL out; 8K leaves headroom.
#   --gpu-memory-utilization .90: max out KV-cache so we have concurrency headroom toward the >=10 RPS SLO.
#   --max-num-seqs 64           : initial in-flight batch size; raise/lower while chasing the SLO.
#   --tensor-parallel-size 1    : single GPU. (Qwen3-30B-A3B is MoE with ~3B active -> fits one H100.)
#   vLLM exposes /metrics by default -> Prometheus scrapes it, no extra flag needed.
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --tensor-parallel-size 1
