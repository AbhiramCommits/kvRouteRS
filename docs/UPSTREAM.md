# Upstream observations

Three concrete gaps worth filing as issues or discussions against vLLM or
SGLang, each with the evidence this repository gathered. Feel free to take
these upstream.

## 1. vLLM: prefix-cache hit rate is a process-wide gauge — routers need per-request feedback

**What we found.** `vllm:gpu_prefix_cache_hit_rate` is one cumulative,
process-wide gauge with no labels beyond `model_name`. A router that wants to
*learn* which prefixes a specific worker holds must infer them from its own
dispatch history, because the engine reports nothing per request and nothing
per prefix.

**Evidence.**
- `crates/router-server/tests/vllm_e2e.rs` documents a 0.30 tolerance between
  the router's predicted hit rate (512-character blocks) and the gauge's
  token-block rate — the tolerance *is* the granularity gap.
- Our mock had to invent a per-request header (`x-kv-prefix-hit`) so the
  router's belief index can be scored against engine truth; real vLLM offers
  no equivalent, which is why `kvrouter_hit_rate_prediction_error` can only
  be computed from the aggregate gauge in production.

**Suggestion.** Expose per-request prefix-hit metadata (matched block count /
block ids — `usage.prompt_tokens_details.cached_tokens` is close but token-
not block-granular), and label the `/metrics` gauge by at least model and
engine instance so a router can attribute hits to workers.

## 2. SGLang: radix-cache hit-rate metrics are not stable or documented

**What we found.** While vLLM has one documented series, SGLang's `/metrics`
radix-cache series names vary by build, and we could not pin a stable name.

**Evidence.**
- `deploy/sglang/README.md` says, verbatim: "the exact series names vary by
  build. Check `/metrics` on your build and point the router at the hit-rate
  metric" — which is why `engine_cache_hit_metric` is configurable
  (`crates/router-core/src/config.rs`), an indirection that exists only
  because of this gap.

**Suggestion.** Stabilize and document a radix-cache hit-rate metric (e.g.
`sglang:radix_cache_hit_rate`) with a type/help string, plus per-request
radix-hit metadata mirroring item 1.

## 3. Neither engine exposes cache *content* — routers must keep shadow belief indexes that go stale

**What we found.** There is no API to ask "which KV blocks are currently
resident in engine E?" — prefix caching is entirely passive. Routers
therefore maintain a shadow belief index that can only be corrected by
timeout (TTL), by observation (the aggregate gauge), or by proxy (purging on
pod death). Every stale belief costs a redundant prefill, and the router
cannot distinguish engine-side eviction from a cold cache.

**Evidence.**
- The disaggregated benchmark row: the router predicted 0.656 hit rate
  against the prefill pool while the engines actually served 0.000 — the
  2-replica prefill pool could not hold the 177-block working set, and the
  router had no way to know until the engine metric said so
  (`docs/BENCHMARKS.md`, disaggregated row).
- The router's entire stale-entry machinery — TTL (300s default), LRU cap,
  purge-on-worker-death, and `kvrouter_hit_rate_prediction_error` — exists as
  compensation for this missing API (`docs/ARCHITECTURE.md`, "Failure modes").

**Suggestion.** Expose lightweight KV-cache residency metadata: per-engine
block occupancy (or an eviction event stream / monotonic eviction counter),
so routers can invalidate beliefs instead of guessing. This is exactly the
information a NIXL-style transfer protocol already needs, and it would let
Dynamo-style routers drop their TTL heuristics.
