# Architecture

This document describes how a request moves through `kvrouters`, what the
prefix index looks like under the hood, how the disaggregated flow works, and
what breaks when things go wrong.

## Request lifecycle

```
POST /v1/chat/completions
  │
  ├─ 1. Deserialize (axum Json extractor; 400 on malformed body)
  ├─ 2. Canonical prompt  →  chain_hash(prompt, 512)   [rust, no tokenizer]
  ├─ 3. RouteDecision:
  │       round_robin  → cyclic over healthy `both`-pool workers
  │       cache_aware  → score = matched_blocks·cache_weight − inflight·load_weight
  │       disaggregated → two-phase (see below)
  ├─ 4. InflightGuard::acquire(worker)          [atomic; Drop releases]
  ├─ 5. Proxy to worker (reqwest, body streamed through, never buffered)
  │       streaming → SSE passthrough, guaranteed "data: [DONE]"
  │       non-streaming → JSON passthrough
  ├─ 6. On stream completion: telemetry.finish()
  │       log (request_id, worker, matched blocks, score, ttft_ms, total_ms)
  │       metrics (requests_total, cache_hit_blocks, prompt_blocks_total,
  │                ttft/duration histograms, tokens_generated_total,
  │                per-worker hit stats)         [metrics crate, Prometheus]
  └─ 7. Guard drops when the body is fully drained  → in-flight count released
```

Background loops run beside the request path:

- **Health poller** (every `health_check_interval_secs`): probes
  `{worker}/health`, marks workers up/down, and on a down-transition purges
  the worker from the prefix index (its KV cache died with it).
- **Eviction task** (every `ttl/3`, clamped 1–60s): TTL-expires entries not
  touched for `prefix_index_ttl_secs`, then trims the oldest entries down to
  `prefix_index_max_entries` (global LRU).
- **Engine metrics scraper** (`metrics_scrape: true`): polls each worker's
  `/metrics`, reads the engine-reported prefix-cache hit rate, and emits
  `kvrouter_predicted_cache_hit_rate{worker}` and
  `kvrouter_hit_rate_prediction_error{worker}` = predicted − reported.
- **Worker discovery** (in-cluster, `discovery.enabled: true`): watches the
  EndpointSlices of the headless worker service and reconciles the registry.

## The prefix index

```
PrefixIndex
 └── 64 shards        (shard = block_hash & 63; power of two, no division)
      └── RwLock<HashMap<u64, IndexEntry>>        one shard lock held at a time
           IndexEntry {
             workers: SmallVec<[WorkerId; 4]>,    // who holds this block
             last_touched: Instant,               // TTL + LRU ordering
           }
```

**Operations and complexity** (n = blocks in the chain, m = shard map size):

| Operation | Cost | Notes |
|---|---|---|
| `record(worker, chain)` | O(n) hash + O(n) shard writes | dispatch-time, optimistic |
| `longest_match(worker, chain)` | O(n) reads, O(n) touch writes | breaks at first miss |
| `remove_worker(id)` | O(64·m/64) = O(m) | purge on pod death |
| `evict(ttl, cap)` | O(m log m) | off the request path |
| `len()` | O(64) | scrape-time gauge |

The block hash chain (`h_0 = hash(block_0)`, `h_i = hash(h_{i−1} ‖ block_i)`)
is what makes `longest_match` a prefix walk: because every later hash embeds
every earlier one, a mismatch at block k means no suffix can match either.

**Invariant.** Two prompts share their first k KV-blocks *exactly when* their
first k chain hashes are equal. Everything cache-aware in the router rests on
this.

**Staleness is tolerated by design.** The index is a belief, not a fact:
engine-side eviction or out-of-band traffic make entries wrong in either
direction, and the worst case is a mis-route that pays a prefill — never a
wrong answer. TTL, LRU cap, and purge-on-death keep the belief bounded;
`kvrouter_hit_rate_prediction_error` keeps it honest.

## Disaggregated flow

`routing_policy: disaggregated` splits prefill and decode across pools, with
an explicit KV handoff between them:

```mermaid
sequenceDiagram
    participant C as Client
    participant R as Router
    participant P as Prefill worker
    participant D as Decode worker

    C->>R: POST /v1/chat/completions (stream)
    R->>P: select_in_pool(prefill) · proxy with x-router-phase: prefill
    P-->>R: prefill.result (KV materialized; no tokens yet)
    R->>D: select_in_pool(decode)
    R->>R: KvTransfer.transfer(P, D, blocks)   [mock sleeps; real = NIXL]
    R->>D: proxy with x-router-phase: decode
    D-->>C: token stream (what the client sees)
    R->>R: telemetry(prefill_worker, decode_worker, kv_transfer_ms, ...)
```

Each pool selection uses the same cache-aware scoring: within a pool, the
right worker is the one with the best cache/load tradeoff. The decode worker's
chain is recorded at dispatch (optimistic — the blocks arrive via transfer
milliseconds later), which is a mis-route window, never a correctness bug.

## Failure modes

**Worker death mid-stream.** The health poller marks the worker down on its
next probe (≤ one interval), selection excludes it, and the prefix index
purges its entries. A request already streaming from the dead worker fails at
the transport layer: the client's SSE stream errors out (the router logs it,
counts it as a 502, and releases the in-flight guard). In-flight streams are
not re-routed — replaying a partially-generated completion would require the
engine-side KV state, which we deliberately do not pretend to have.

**Index staleness after eviction.** If the engine evicts blocks the router
still credits, the request routes to a "hit" that is actually a miss: TTFT
degrades to a cold prefill, nothing else. The engine-side hit-rate metric is
the tripwire for this; a persistently positive
`kvrouter_hit_rate_prediction_error` means TTLs should shrink or the engine
cache is undersized.

**Split-brain across router replicas.** This is the honest weak point: each
router replica keeps its own belief index, so N replicas each know ~1/N of
the cache residency, and hit rates drop as the Deployment scales. Two router
replicas also never disagree dangerously — decisions are independent and the
worst case is, again, a mis-route. A shared index fixes the partitioning:

- **Redis-backed index**: store block-hash → worker-set in a Redis hash (or
  sorted set keyed by last-touch for LRU), ~1 round-trip per routing decision
  (read+conditional write), plus a Lua script for the atomic
  match-and-record. Cost: one network hop in the routing hot path (single-digit
  ms on a local Redis, more cross-AZ), Redis as a new failure domain (fall back
  to the per-replica index on Redis outage rather than failing open), and the
  removal of the O(1) in-process shard reads. Worth it past ~2 router
  replicas; not worth it for one.

## Where the seams are

- `KvTransfer` trait (`crates/router-core/src/kv_transfer.rs`) — swap the mock
  for a NIXL connector without touching routing logic.
- `WorkerRegistry::register_worker` / `remove_worker` — discovery-managed
  membership with monotonic identities.
- `metrics_scrape` + `engine_cache_hit_metric` — plug in any engine's
  `/metrics` hit-rate series.
