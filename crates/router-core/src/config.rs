use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::RouterError;

/// Which inference phase a worker is provisioned for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Pool {
    /// Prefill-heavy workers: long prompts, big KV-cache writes.
    Prefill,
    /// Decode-heavy workers: token generation.
    Decode,
    /// Can serve both phases.
    Both,
}

impl fmt::Display for Pool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Pool::Prefill => "prefill",
            Pool::Decode => "decode",
            Pool::Both => "both",
        };
        f.write_str(s)
    }
}

/// How the router picks a worker for an incoming request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingPolicy {
    RoundRobin,
    /// Reserved for the next milestone; selection currently fails with an explicit
    /// [`RouterError::PolicyNotImplemented`] so there is a clear seam to build on.
    CacheAware,
}

impl fmt::Display for RoutingPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            RoutingPolicy::RoundRobin => "round_robin",
            RoutingPolicy::CacheAware => "cache_aware",
        };
        f.write_str(s)
    }
}

/// A single backend worker as declared in `config/router.yaml`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct WorkerConfig {
    pub url: String,
    pub pool: Pool,
}

/// Top-level router configuration.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct RouterConfig {
    pub workers: Vec<WorkerConfig>,
    pub routing_policy: RoutingPolicy,
    #[serde(default = "default_health_check_interval_secs")]
    pub health_check_interval_secs: u64,
}

fn default_health_check_interval_secs() -> u64 {
    5
}

impl RouterConfig {
    /// Load configuration from a YAML file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RouterError> {
        let raw = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            RouterError::Config(format!("failed to read {}: {e}", path.as_ref().display()))
        })?;
        Self::from_yaml(&raw)
    }

    /// Parse configuration from YAML text.
    pub fn from_yaml(raw: &str) -> Result<Self, RouterError> {
        let config: RouterConfig = serde_yaml::from_str(raw)?;
        if config.workers.is_empty() {
            return Err(RouterError::Config(
                "`workers` must contain at least one entry".to_string(),
            ));
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
routing_policy: round_robin
workers:
  - url: http://127.0.0.1:8001
    pool: both
  - url: http://127.0.0.1:8002
    pool: prefill
"#;

    #[test]
    fn parses_config() {
        let config = RouterConfig::from_yaml(CONFIG).unwrap();
        assert_eq!(config.workers.len(), 2);
        assert_eq!(config.workers[0].url, "http://127.0.0.1:8001");
        assert_eq!(config.workers[0].pool, Pool::Both);
        assert_eq!(config.workers[1].pool, Pool::Prefill);
        assert_eq!(config.routing_policy, RoutingPolicy::RoundRobin);
        assert_eq!(config.health_check_interval_secs, 5);
    }

    #[test]
    fn accepts_cache_aware_policy() {
        let config = RouterConfig::from_yaml(
            "routing_policy: cache_aware\nworkers:\n  - url: http://x\n    pool: decode\n",
        )
        .unwrap();
        assert_eq!(config.routing_policy, RoutingPolicy::CacheAware);
    }

    #[test]
    fn rejects_unknown_pool() {
        let result = RouterConfig::from_yaml(
            "routing_policy: round_robin\nworkers:\n  - url: http://x\n    pool: nope\n",
        );
        assert!(result.is_err());
    }

    #[test]
    fn rejects_empty_workers() {
        let error =
            RouterConfig::from_yaml("routing_policy: round_robin\nworkers: []\n").unwrap_err();
        assert!(matches!(error, RouterError::Config(_)));
    }
}
