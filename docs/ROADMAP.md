# Roadmap

Concrete next items, in rough priority order. Each is scoped to the seam that
already exists in the codebase.

## 1. Real KV transfer (NIXL-style)

*Status: trait seam exists (`KvTransfer` + `MockKvTransfer`), transfer is a
flat sleep.*

- Implement a `NixlKvTransfer` against vLLM's kv-transfer APIs: discover the
  block table after prefill (vLLM reports `matched_blocks`/block ids via its
  KV transfer metadata), hand blocks to the decode worker, and only then
  record the chain against the decode worker (remove the dispatch-time
  optimism).
- Backpressure: a bounded transfer queue per worker pair; selection should
  price in queued transfers, not just in-flight requests.
- This flips the disaggregated benchmark row: the current mock-pool floor
  (~0 engine hits, 1.3s TTFT) should become competitive with cache-aware
  single-phase routing.

## 2. Shared index across router replicas

*Status: honest per-replica indexes; replicas > 1 partition cache knowledge
(noted in the Deployment manifest).*

- Redis-backed block-hash index: `block_hash → set of workers` plus a
  last-touch ZSET for LRU. One Lua script for the atomic
  match-and-record so a routing decision is a single round-trip.
- Local shard cache with a short TTL in front of Redis to keep the common
  case in-process; fall back to the per-replica index on Redis outage
  (degrade, don't fail open).
- Measure the crossover: at what request rate does the added hop cost less
  than the partitioned-index hit-rate loss? The benchmark harness can answer
  this by running N router replicas behind a load balancer.

## 3. SLA-aware routing

*Status: load term is a single global coefficient (`load_weight`).*

- Per-worker score shaping from observed tail latencies: discount workers
  whose p99 TTFT exceeds a target, before the cache term applies.
- Priority classes: high-priority traffic may pay cold prefills to avoid a
  queued hot worker; low-priority traffic rides the cache. This is a
  per-request `load_weight` override, not a global one.
- Publish the routing score components per decision (already logged) into a
  histogram so weight tuning becomes data-driven.

## 4. Speculative-decoding awareness

*Status: nothing yet.*

- Speculative decoding makes decode throughput depend on the *draft model's*
  acceptance, which the router cannot see. The cheap win: expose the
  engine's acceptance rate (vLLM reports speculative metrics on `/metrics`)
  through the existing `metrics_scrape` path and fold it into the load term.
- The hard win: route to workers whose draft model state matches the prompt
  (draft KV is a second cache dimension). That doubles the index — worth a
  design note before code.

## Backlog / smaller items

- Migrate `serde_yaml` (unmaintained, ignored in `deny.toml`) to a maintained
  YAML crate.
- Support IPv6 pod addresses in worker discovery (currently IPv4-only, noted
  in `discovery.rs`).
- Named-port selection in discovery (`discovery.port_name`) instead of
  always taking the first endpoint port.
- Drop the `#[ignore]`d vLLM e2e test into a self-hosted GPU runner once one
  exists, with the tolerance from `tests/vllm_e2e.rs` as the gate.
- Weighted HPA: scale the router on in-flight requests as well as CPU.
