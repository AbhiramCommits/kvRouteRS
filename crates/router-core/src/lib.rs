//! KV-cache-aware inference routing core.
//!
//! This crate is deliberately free of I/O framework dependencies (no tokio, no reqwest):
//! it owns the pure decision-making logic — worker registry, prefix index, and routing
//! policy — while the HTTP layer lives in `router-server`. A Python embedding will
//! wrap this crate via `router-py`.

pub mod config;
pub mod error;
pub mod prefix;
pub mod router;
pub mod worker;

pub use config::{Pool, RouterConfig, RoutingPolicy, WorkerConfig};
pub use error::RouterError;
pub use prefix::PrefixIndex;
pub use router::{RouteDecision, Router};
pub use worker::{Worker, WorkerId, WorkerRegistry, WorkerState};
