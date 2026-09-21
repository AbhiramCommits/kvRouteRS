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

pub mod config;
pub mod error;
pub mod hashing;
pub mod inflight;
pub mod kv_transfer;
pub mod prefix;
pub mod request;
pub mod router;
pub mod worker;

pub use config::{Pool, RouterConfig, RoutingPolicy, WorkerConfig};
pub use error::RouterError;
pub use hashing::chain_hash;
pub use inflight::{InflightGuard, InflightTracker};
pub use kv_transfer::{KvTransfer, KvTransferFuture, MockKvTransfer};
pub use prefix::{EvictionStats, PrefixIndex};
pub use request::{ChatCompletionRequest, ChatMessage};
pub use router::{RouteDecision, Router};
pub use worker::{Worker, WorkerId, WorkerRegistry, WorkerState};
