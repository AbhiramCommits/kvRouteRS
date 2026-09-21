//! PyO3 bindings for the kvRouteRS routing core, exposed as the `kvrouters`
//! native module (imported as `kvrouters._native`, re-exported with typing by
//! `python/kvrouters/__init__.py`).
//!
//! This mirrors how NVIDIA Dynamo splits its Rust runtime from its Python SDK:
//! the heavy, lock-bound, CPU-bound routing logic stays in Rust; Python calls
//! into it through a thin, typed surface.
//!
//! # GIL discipline
//!
//! Every call that touches the index locks (or does hashing work) runs inside
//! `py.allow_threads`. The prefix index has 64 shards behind independent
//! `RwLock`s, so the *only* way a multi-threaded Python process (e.g. a
//! `ThreadPoolExecutor` serving routes) can overlap work is if the GIL is
//! released while Rust holds the shard locks — otherwise all Python threads
//! serialize on the GIL and the sharding is pointless. Releasing the GIL turns
//! N Python threads into N parallel shard-lock contenders instead of one
//! sequential queue.
//!
//! # Error handling
//!
//! All fallible paths return Python exceptions (`KvRouterError` hierarchy),
//! never panics, so a bad policy string or an unhealthy worker set surfaces as
//! a catchable `kvrouters.PolicyError` / `kvrouters.NoHealthyWorkersError`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

use router_core::{
    chain_hash, ChatCompletionRequest, ChatMessage, PrefixIndex, RouterConfig, RouterError,
    RoutingPolicy, WorkerConfig, WorkerRegistry,
};

create_exception!(kvrouters, KvRouterError, PyException);
create_exception!(kvrouters, NoHealthyWorkersError, KvRouterError);
create_exception!(kvrouters, PolicyError, KvRouterError);

/// Map a core routing error to the Python exception hierarchy.
fn to_py_error(error: RouterError) -> PyErr {
    match error {
        RouterError::NoHealthyWorkers { .. } | RouterError::NoHealthyWorkersInPool { .. } => {
            NoHealthyWorkersError::new_err(error.to_string())
        }
        RouterError::DisaggregatedRequiresTwoPhase | RouterError::PolicyNotImplemented(_) => {
            PolicyError::new_err(error.to_string())
        }
        other => KvRouterError::new_err(other.to_string()),
    }
}

fn parse_policy(policy: &str) -> Result<RoutingPolicy, PyErr> {
    match policy {
        "round_robin" => Ok(RoutingPolicy::RoundRobin),
        "cache_aware" => Ok(RoutingPolicy::CacheAware),
        "disaggregated" => Err(PolicyError::new_err(
            "policy `disaggregated` needs the two-phase server flow (prefill -> KV transfer -> \
             decode); use the HTTP router for it",
        )),
        other => Err(PolicyError::new_err(format!(
            "unknown policy `{other}`; expected `round_robin` or `cache_aware`"
        ))),
    }
}

/// Python-facing view of the router-side prefix index.
///
/// Tracks which workers are believed to hold which prompt blocks (worker set
/// is whatever has been `insert`ed). `ttl_seconds` / `max_entries` bound the
/// index; `evict_stale()` applies both.
#[pyclass(name = "PrefixIndex", module = "kvrouters")]
struct PyPrefixIndex {
    inner: Arc<PrefixIndex>,
    block_size: usize,
    ttl: Duration,
    max_entries: usize,
    workers: Mutex<HashSet<u64>>,
}

#[pymethods]
impl PyPrefixIndex {
    #[new]
    #[pyo3(signature = (block_size = 512, ttl_seconds = 300.0, max_entries = 100_000))]
    fn new(block_size: usize, ttl_seconds: f64, max_entries: usize) -> PyResult<Self> {
        if block_size == 0 {
            return Err(PyValueError::new_err("block_size must be >= 1"));
        }
        if !ttl_seconds.is_finite() || ttl_seconds < 0.0 {
            return Err(PyValueError::new_err(
                "ttl_seconds must be a finite, non-negative number",
            ));
        }
        Ok(Self {
            inner: Arc::new(PrefixIndex::new()),
            block_size,
            ttl: Duration::from_secs_f64(ttl_seconds),
            max_entries,
            workers: Mutex::new(HashSet::new()),
        })
    }

    /// Record that `worker_id` now holds every block of `prompt`.
    fn insert(&self, py: Python<'_>, worker_id: u64, prompt: String) {
        let inner = Arc::clone(&self.inner);
        let workers = &self.workers;
        let block_size = self.block_size;
        // GIL released: hashing + shard-lock writes, see the module docs.
        py.allow_threads(move || {
            let chain = chain_hash(&prompt, block_size);
            inner.record(worker_id, &chain);
            workers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(worker_id);
        });
    }

    /// For every worker that has ever been inserted, return
    /// `(worker_id, matched_blocks)` for `prompt`, sorted by worker id.
    fn lookup(&self, py: Python<'_>, prompt: String) -> Vec<(u64, usize)> {
        let inner = Arc::clone(&self.inner);
        let workers = &self.workers;
        let block_size = self.block_size;
        py.allow_threads(move || {
            let chain = chain_hash(&prompt, block_size);
            let worker_ids: Vec<u64> = workers
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .copied()
                .collect();
            let mut matches: Vec<(u64, usize)> = worker_ids
                .into_iter()
                .map(|worker| (worker, inner.longest_match(worker, &chain)))
                .collect();
            matches.sort_by_key(|(worker, _)| *worker);
            matches
        })
    }

    /// Apply TTL expiry + LRU cap; returns `(expired, lru_evicted)`.
    fn evict_stale(&self, py: Python<'_>) -> (usize, usize) {
        let inner = Arc::clone(&self.inner);
        let ttl = self.ttl;
        let max_entries = self.max_entries;
        py.allow_threads(move || {
            let stats = inner.evict(ttl, max_entries);
            (stats.expired, stats.lru_evicted)
        })
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || inner.len())
    }
}

/// Python-facing router: a worker registry plus one in-process core `Router`
/// per policy (created lazily), so routing decisions are testable and
/// simulatable without running the HTTP server. All workers are considered
/// healthy — there is no health poller in-process; drive health through the
/// server for production.
#[pyclass(name = "Router", module = "kvrouters")]
struct PyRouter {
    registry: Arc<WorkerRegistry>,
    base_config: RouterConfig,
    routers: Mutex<HashMap<RoutingPolicy, Arc<router_core::Router>>>,
}

impl PyRouter {
    fn get_or_create(
        &self,
        policy: RoutingPolicy,
    ) -> Result<Arc<router_core::Router>, RouterError> {
        let mut routers = self
            .routers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(router) = routers.get(&policy) {
            return Ok(Arc::clone(router));
        }
        let mut config = self.base_config.clone();
        config.routing_policy = policy;
        let router = Arc::new(router_core::Router::new(
            Arc::clone(&self.registry),
            &config,
        ));
        routers.insert(policy, Arc::clone(&router));
        Ok(router)
    }
}

#[pymethods]
impl PyRouter {
    #[new]
    #[pyo3(signature = (workers, cache_weight = 1.0, load_weight = 0.5, block_size = 512))]
    fn new(
        workers: Vec<String>,
        cache_weight: f64,
        load_weight: f64,
        block_size: usize,
    ) -> PyResult<Self> {
        if workers.is_empty() {
            return Err(KvRouterError::new_err("`workers` must not be empty"));
        }
        if block_size == 0 {
            return Err(PyValueError::new_err("block_size must be >= 1"));
        }
        let config = RouterConfig {
            workers: workers
                .into_iter()
                .map(|url| WorkerConfig {
                    url,
                    pool: router_core::Pool::Both,
                })
                .collect(),
            routing_policy: RoutingPolicy::RoundRobin,
            health_check_interval_secs: 5,
            prompt_block_chars: block_size,
            cache_weight,
            load_weight,
            prefix_index_ttl_secs: 300,
            prefix_index_max_entries: 100_000,
            kv_transfer_cost_ms: 20,
            metrics_scrape: false,
            engine_metrics_interval_secs: 15,
            engine_cache_hit_metric: "vllm:gpu_prefix_cache_hit_rate".to_string(),
        };
        let registry = Arc::new(WorkerRegistry::from_config(&config.workers));
        // In-process there is no health poller: treat every configured worker
        // as healthy. (The HTTP server's poller owns health in production.)
        for index in 0..config.workers.len() {
            registry.set_healthy(index as u64, true);
        }
        Ok(Self {
            registry,
            base_config: config,
            routers: Mutex::new(HashMap::new()),
        })
    }

    /// Select a worker URL for `prompt` under `policy`
    /// (`"round_robin"` or `"cache_aware"`).
    fn select_worker(&self, py: Python<'_>, prompt: String, policy: String) -> PyResult<String> {
        let decision = self.decide(py, prompt, policy)?;
        Ok(decision.worker.url)
    }

    /// Full routing decision: worker URL, matched blocks, prompt blocks, score.
    fn route(&self, py: Python<'_>, prompt: String, policy: String) -> PyResult<PyRouteDecision> {
        let decision = self.decide(py, prompt, policy)?;
        Ok(PyRouteDecision {
            worker: decision.worker.url,
            matched_blocks: decision.matched_prefix_len,
            prompt_blocks: decision.prompt_blocks,
            score: decision.score,
        })
    }
}

impl PyRouter {
    fn decide(
        &self,
        py: Python<'_>,
        prompt: String,
        policy: String,
    ) -> PyResult<router_core::RouteDecision> {
        let policy = parse_policy(&policy)?;
        let router = self.get_or_create(policy).map_err(to_py_error)?;
        let request = ChatCompletionRequest {
            model: "python-embedding".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt,
            }],
            max_tokens: None,
            temperature: None,
            stream: false,
            extra: Default::default(),
        };
        // GIL released: canonical-prompt hashing, shard locks, scoring.
        py.allow_threads(move || router.select_for_chat(&request))
            .map_err(to_py_error)
    }
}

/// The outcome of one routing decision.
#[pyclass(name = "RouteDecision", module = "kvrouters")]
#[derive(Clone)]
struct PyRouteDecision {
    #[pyo3(get)]
    worker: String,
    #[pyo3(get)]
    matched_blocks: usize,
    #[pyo3(get)]
    prompt_blocks: usize,
    #[pyo3(get)]
    score: f64,
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPrefixIndex>()?;
    module.add_class::<PyRouter>()?;
    module.add_class::<PyRouteDecision>()?;
    module.add("KvRouterError", module.py().get_type::<KvRouterError>())?;
    module.add(
        "NoHealthyWorkersError",
        module.py().get_type::<NoHealthyWorkersError>(),
    )?;
    module.add("PolicyError", module.py().get_type::<PolicyError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
