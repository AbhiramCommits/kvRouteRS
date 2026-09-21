# Benchmarks

Full methodology, reproduction commands, raw numbers, and the caveats you
should carry into any decision made on these numbers.

## What is measured

The benchmark replays **the same seeded trace** against three router policies
and compares:

- **hit rate (router)** — `Δkvrouter_cache_hit_blocks /
  Δkvrouter_prompt_blocks_total`, i.e. the router's belief about cache
  residency, over the run.
- **hit rate (engine)** — per-request `x-kv-prefix-hit` header from the mock
  (matched blocks / prompt blocks), i.e. what the *engine's own LRU* actually
  served. This is the number that matters for TTFT.
- **TTFT p50/p95/p99** — client-side time from scheduled send to the first
  streamed chunk.
- **e2e p50/p95** — client-side time to stream completion.
- **req/s, tok/s** — throughput over the whole run.
- **imbalance** — max/mean of per-worker in-flight samples polled during the
  run (1.0 = perfectly balanced).

## Hardware and software configuration

| Item | Value |
|---|---|
| Host | Apple Silicon MacBook Pro (M-series), macOS |
| Container runtime | Docker Desktop |
| Backends | `mocks/mock_vllm.py` — **no GPU**. TTFT is *simulated*: `50ms +
  5ms/token × (1 − 0.8 × hit_fraction)` prefill, 30ms inter-token decode |
| Engine cache | In-process LRU of 512-char block hashes, capacity **80 blocks** per mock |
| Replicas | 4 mocks per policy router |
| Router | `router-server` in Docker (round_robin :18080, cache_aware :18081,
  disaggregated :18082), `prompt_block_chars: 512`, `cache_weight: 1.0`,
  `load_weight: 0.5` |
| Traces | `shared_prefix`: 59 distinct 1536-char system prompts × 8 repeats
  (472 requests) · `multi_turn`: 31 conversations × 8 turns (248) ·
  `random`: 256 fully random prompts |
| Arrival process | Poisson, 8 req/s, seed 42 (deterministic) |
| Response length | 32 tokens per request, streaming |
| Cache state | Cold start per run: mocks and routers restarted before every
  (policy, shape) run |

The mock is deliberately *not* a GPU: the point is the routing effect, which
is measurable with zero GPUs because the mock models prefill cost and prefix
hits faithfully. Absolute latencies are synthetic; the *ratios between
policies* are the finding.

## Reproduction

```bash
docker compose up -d --build                      # 4 mocks + 3 routers
python3 -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt
bench/.venv/bin/python bench/ab.py --reset --outdir bench/results
```

`--reset` restarts mocks and routers before the run for cold caches. The
replica-count sweep (chart #2) needs a local router binary
(`--router-binary target/release/router-server`); skip it with
`--skip-sweep`. See `python bench/ab.py --help` for the trace knobs.

CI runs a reduced version on every PR and **fails if the cache-aware
shared_prefix hit rate drops below 0.55** (`.github/workflows/bench.yml`) —
the routing win is an enforced invariant, not a one-time claim.

## Raw numbers

From the run committed with this repository (seed 42):

### shared_prefix

| policy | hit rate (router) | hit rate (engine) | ttft p50 | ttft p95 | ttft p99 | e2e p50 | e2e p95 | req/s | tok/s | imbalance |
|---|---|---|---|---|---|---|---|---|---|---|
| round_robin | 0.375 | 0.003 | 1.282 | 1.326 | 1.347 | 2.317 | 2.368 | 7.8 | 250.8 | 1.01 |
| cache_aware | 0.656 | 0.656 | 0.548 | 1.292 | 1.322 | 1.585 | 2.340 | 7.9 | 253.6 | 1.08 |
| disaggregated | 0.656 | 0.000 | 1.307 | 1.352 | 1.371 | 2.349 | 2.409 | 7.8 | 250.6 | 1.15 |

### multi_turn

| policy | hit rate (router) | hit rate (engine) | ttft p50 | ttft p95 | ttft p99 | e2e p50 | e2e p95 | req/s | tok/s | imbalance |
|---|---|---|---|---|---|---|---|---|---|---|
| round_robin | 0.070 | 0.046 | 0.534 | 0.936 | 0.941 | 1.586 | 1.973 | 6.9 | 221.4 | 1.00 |
| cache_aware | 0.299 | 0.218 | 0.499 | 0.712 | 0.934 | 1.524 | 1.758 | 7.0 | 223.1 | 1.04 |
| disaggregated | 0.325 | 0.241 | 0.454 | 0.728 | 0.734 | 1.507 | 1.757 | 6.9 | 222.1 | 1.47 |

### random (control)

| policy | hit rate (router) | hit rate (engine) | ttft p50 | ttft p95 | ttft p99 | e2e p50 | e2e p95 | req/s | tok/s | imbalance |
|---|---|---|---|---|---|---|---|---|---|---|
| round_robin | 0.000 | 0.000 | 0.198 | 0.217 | 0.373 | 1.249 | 1.326 | 9.1 | 291.2 | 1.00 |
| cache_aware | 0.000 | 0.000 | 0.199 | 0.206 | 0.294 | 1.258 | 1.305 | 9.1 | 291.6 | 1.11 |
| disaggregated | 0.000 | 0.000 | 0.226 | 0.235 | 0.254 | 1.291 | 1.338 | 9.1 | 291.1 | 1.74 |

Full per-request samples are in `bench/results/ab_results.json`.

## Confidence caveats

- **No GPU.** The mock simulates prefill/decode latency and cache hits; it
  does not model GPU contention, batching, or real memory pressure. Direction
  and ratios should transfer to vLLM/SGLang; absolute numbers will not.
- **Single host.** Everything ran in one Docker daemon on one machine.
  Network jitter is loopback-scale; multi-node deployments will add transfer
  and health-probe variance.
- **Fixed workload shape.** 59 prefixes / 31 conversations / seed 42. The hit
  rates move with the ratio of working-set size to per-replica cache
  capacity (the replica sweep charts this explicitly). Re-tune for your mix.
- **The disaggregated numbers are a floor, not a ceiling.** The simulated
  KV transfer is a flat 15–20ms sleep, and the prefill pool (2 replicas)
  cannot hold the 177-block working set, so its engine hit rate is ~0 by
  construction. Real NIXL-style transfer would flip this row.
- **Cold-start per run.** First-visit prefixes are always misses; hit rates
  include the warm-up ramp. Long-running production numbers would be higher.
