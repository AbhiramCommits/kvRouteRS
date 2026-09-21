#!/usr/bin/env bash
# Launch vLLM with prefix caching for the kvRouteRS router.
#
# Tested against: vLLM v0.6.4.post1 (flags verified against the upstream
# documentation for that release; this machine has no GPU, so the command was
# not executed here — run tests/vllm_e2e.rs on a GPU host to validate).
#
# Key flag:
#   --enable-prefix-caching   enables the block-level prefix cache, which is
#                             what makes cache-aware routing measurable.
# The cache hit rate is reported on /metrics as:
#   vllm:gpu_prefix_cache_hit_rate
# (older builds may report a different name; set `engine_cache_hit_metric`
# in the router config to match your build).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"

python -m pip install "vllm==0.6.4.post1"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --enable-prefix-caching \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
