//! KV-cache-aware inference routing core.
//!
//! This crate owns the pure decision-making machinery — worker registry,
//! sharded prefix index, cache/load scoring, and the disaggregated
//! prefill/decode flow — while the HTTP layer lives in `router-server` and a
//! Python embedding is provided by `router-py`.
//!
//! The one dependency to be careful about: `tokio` is used only for its
//! task/timer primitives (the eviction task and the mock KV transfer), never
//! for I/O, so the core remains transport-agnostic.

//! docs for public items are enforced; run `cargo doc --no-deps` to check.
#![warn(missing_docs)]

/// Configuration types loaded from `router.yaml`.
pub mod config;
/// Error type shared by the whole core.
pub mod error;
/// Tokenizer-free block prefix hashing (the cache-affinity primitive).
pub mod hashing;
/// Per-worker in-flight request accounting.
pub mod inflight;
/// KV-transfer abstraction for the disaggregated flow.
pub mod kv_transfer;
/// Sharded, TTL/LRU-bounded prefix index.
pub mod prefix;
/// OpenAI-compatible request types plus canonical prompt serialization.
pub mod request;
/// Routing policy implementations and the router itself.
pub mod router;
/// Worker registry: health tracking and identity allocation for backends.
pub mod worker;

pub use config::{DiscoveryConfig, Pool, RouterConfig, RoutingPolicy, WorkerConfig};
pub use error::RouterError;
pub use hashing::chain_hash;
pub use inflight::{InflightGuard, InflightTracker};
pub use kv_transfer::{KvTransfer, KvTransferFuture, MockKvTransfer};
pub use prefix::{EvictionStats, PrefixIndex};
pub use request::{ChatCompletionRequest, ChatMessage};
pub use router::{RouteDecision, Router};
pub use worker::{Worker, WorkerId, WorkerRegistry, WorkerState};
