use thiserror::Error;

/// Errors produced by the routing core.
#[derive(Debug, Error)]
pub enum RouterError {
    /// No worker is currently marked healthy.
    #[error("no healthy workers available (policy: {policy})")]
    NoHealthyWorkers { policy: String },

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
