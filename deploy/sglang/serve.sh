#!/usr/bin/env bash
# Launch SGLang with RadixAttention for the kvRouteRS router.
#
# Tested against: SGLang v0.4.4.post2 (flags verified against the upstream
# documentation for that release; this machine has no GPU, so the command was
# not executed here).
#
# RadixAttention is SGLang's default scheduler (a radix-tree prefix cache), so
# no flag enables it; `--disable-radix-cache` would turn it OFF. `--enable-metrics`
# exposes Prometheus metrics on /metrics.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"

python -m pip install "sglang[all]==0.4.4.post2"

exec python -m sglang.launch_server \
  --model "$MODEL" \
  --port "$PORT" \
  --enable-metrics \
  --mem-fraction-static 0.85
