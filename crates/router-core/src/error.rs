use thiserror::Error;

use crate::worker::WorkerId;

/// Errors produced by the routing core.
#[derive(Debug, Error)]
pub enum RouterError {
    /// No worker is currently marked healthy.
    #[error("no healthy workers available (policy: {policy})")]
    NoHealthyWorkers { policy: String },

    /// No healthy worker exists in the pool a disaggregated phase needs.
    #[error("no healthy workers in pool `{pool}` (policy: {policy})")]
    NoHealthyWorkersInPool { pool: String, policy: String },

    /// The simulated/real KV transfer between phases failed.
    #[error("KV transfer from worker {from} to worker {to} failed: {reason}")]
    KvTransfer {
        from: WorkerId,
        to: WorkerId,
        reason: String,
    },

    /// `select_for_chat` cannot serve the two-phase disaggregated policy.
    #[error(
        "policy `disaggregated` requires two-phase routing via `select_in_pool`; \
         `select_for_chat` cannot serve disaggregated requests"
    )]
    DisaggregatedRequiresTwoPhase,

    /// The selected routing policy exists but is not implemented yet.
    #[error("routing policy `{0}` is not implemented yet")]
    PolicyNotImplemented(String),

    /// Configuration could not be loaded or is invalid.
    #[error("invalid configuration: {0}")]
    Config(String),
}

impl From<serde_yaml::Error> for RouterError {
    fn from(error: serde_yaml::Error) -> Self {
        RouterError::Config(error.to_string())
    }
}
