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
    /// Strictly cyclic among healthy workers.
    RoundRobin,
    /// Score every healthy worker on cache affinity and load, pick the best.
    CacheAware,
    /// Prefill and decode are served by different pools, with an explicit KV
    /// transfer between the two phases.
    Disaggregated,
}

impl fmt::Display for RoutingPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            RoutingPolicy::RoundRobin => "round_robin",
            RoutingPolicy::CacheAware => "cache_aware",
            RoutingPolicy::Disaggregated => "disaggregated",
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
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct RouterConfig {
    pub workers: Vec<WorkerConfig>,
    pub routing_policy: RoutingPolicy,
    #[serde(default = "default_health_check_interval_secs")]
    pub health_check_interval_secs: u64,
    /// Character block size for prefix-hash chains (vLLM blocks are 16 tokens;
    /// we use 512 characters as a tokenizer-free stand-in).
    #[serde(default = "default_prompt_block_chars")]
    pub prompt_block_chars: usize,
    /// Score multiplier for matched cache blocks:
    /// `score = cache_hit_blocks * cache_weight - in_flight * load_weight`.
    #[serde(default = "default_cache_weight")]
    pub cache_weight: f64,
    /// Score penalty per in-flight request on a worker. A non-zero load term
    /// keeps pure cache affinity from hot-spotting a single replica.
    #[serde(default = "default_load_weight")]
    pub load_weight: f64,
    /// Prefix-index entries expire after this many seconds without being
    /// written or matched.
    #[serde(default = "default_prefix_index_ttl_secs")]
    pub prefix_index_ttl_secs: u64,
    /// Hard cap on prefix-index entries; the LRU sweep trims the oldest.
    #[serde(default = "default_prefix_index_max_entries")]
    pub prefix_index_max_entries: usize,
    /// Simulated KV transfer latency (ms) for the disaggregated mode.
    #[serde(default = "default_kv_transfer_cost_ms")]
    pub kv_transfer_cost_ms: u64,
}

fn default_health_check_interval_secs() -> u64 {
    5
}

fn default_prompt_block_chars() -> usize {
    512
}

fn default_cache_weight() -> f64 {
    1.0
}

fn default_load_weight() -> f64 {
    0.5
}

fn default_prefix_index_ttl_secs() -> u64 {
    300
}

fn default_prefix_index_max_entries() -> usize {
    100_000
}

fn default_kv_transfer_cost_ms() -> u64 {
    20
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
    fn parses_config_with_defaults() {
        let config = RouterConfig::from_yaml(CONFIG).unwrap();
        assert_eq!(config.workers.len(), 2);
        assert_eq!(config.workers[0].url, "http://127.0.0.1:8001");
        assert_eq!(config.workers[0].pool, Pool::Both);
        assert_eq!(config.workers[1].pool, Pool::Prefill);
        assert_eq!(config.routing_policy, RoutingPolicy::RoundRobin);
        assert_eq!(config.health_check_interval_secs, 5);
        assert_eq!(config.prompt_block_chars, 512);
        assert_eq!(config.cache_weight, 1.0);
        assert_eq!(config.load_weight, 0.5);
        assert_eq!(config.prefix_index_ttl_secs, 300);
        assert_eq!(config.prefix_index_max_entries, 100_000);
        assert_eq!(config.kv_transfer_cost_ms, 20);
    }

    #[test]
    fn accepts_all_policies() {
        let cache_aware = RouterConfig::from_yaml(
            "routing_policy: cache_aware\nworkers:\n  - url: http://x\n    pool: decode\n",
        )
        .unwrap();
        assert_eq!(cache_aware.routing_policy, RoutingPolicy::CacheAware);

        let disaggregated = RouterConfig::from_yaml(
            "routing_policy: disaggregated\nworkers:\n  - url: http://x\n    pool: prefill\n",
        )
        .unwrap();
        assert_eq!(disaggregated.routing_policy, RoutingPolicy::Disaggregated);
    }

    #[test]
    fn accepts_explicit_knobs() {
        let config = RouterConfig::from_yaml(
            "routing_policy: cache_aware\n\
             prompt_block_chars: 256\n\
             cache_weight: 2.0\n\
             load_weight: 0.25\n\
             prefix_index_ttl_secs: 60\n\
             prefix_index_max_entries: 1000\n\
             kv_transfer_cost_ms: 42\n\
             workers:\n  - url: http://x\n    pool: both\n",
        )
        .unwrap();
        assert_eq!(config.prompt_block_chars, 256);
        assert_eq!(config.cache_weight, 2.0);
        assert_eq!(config.load_weight, 0.25);
        assert_eq!(config.prefix_index_ttl_secs, 60);
        assert_eq!(config.prefix_index_max_entries, 1000);
        assert_eq!(config.kv_transfer_cost_ms, 42);
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
