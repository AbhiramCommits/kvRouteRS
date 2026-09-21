use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;

use crate::hashing::chain_hash;
use crate::inflight::InflightTracker;
use crate::prefix::PrefixIndex;
use crate::request::ChatCompletionRequest;
use crate::worker::{Worker, WorkerId, WorkerRegistry};
use crate::{Pool, RouterConfig, RouterError, RoutingPolicy};

/// The outcome of a routing decision.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub worker: Worker,
    /// Longest prefix of the prompt's block chain the chosen worker is believed
    /// to hold, measured before this request was recorded against it.
    pub matched_prefix_len: usize,
    /// Selection score
    /// `cache_hit_blocks * cache_weight - in_flight * load_weight`.
    /// `0.0` for policies that do not score (round-robin).
    pub score: f64,
}

/// Sort key for cache-aware selection: higher score wins; ties break on fewer
/// in-flight requests, then on lower worker id for determinism.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    score: f64,
    in_flight: u64,
    id: WorkerId,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // NaN is treated as equal-to-anything; scores are finite in practice
        // (finite weights times finite counts), and tie-breaking still keeps
        // the ordering total.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            // Fewer in-flight requests is better.
            .then_with(|| other.in_flight.cmp(&self.in_flight))
            // Lower worker id is better.
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

/// Stateless request router: picks a healthy worker for each request.
///
/// # Policies
///
/// - `round_robin`: strictly cyclic among healthy workers. The prefix index is
///   still maintained so callers can observe cache reuse.
/// - `cache_aware`: scores every healthy worker with
///   `score = cache_hit_blocks * cache_weight - in_flight * load_weight` and
///   picks the max, ties broken by fewest in-flight requests then lowest id.
/// - `disaggregated`: prefill and decode run on different pools; callers drive
///   the two phases explicitly through [`Router::select_in_pool`] plus a
///   [`KvTransfer`](crate::KvTransfer) implementation.
///
/// # Why the load term exists
///
/// A pure cache-affinity router hot-spots one replica: the first request for a
/// prompt lands somewhere, every later request scores that worker higher, and
/// the feedback loop pins the whole workload to a single engine while the rest
/// idle. `queued_requests * load_weight` prices concurrency back in, so traffic
/// sheds to idle replicas once the cache gain stops covering the queueing cost.
#[derive(Debug)]
pub struct Router {
    registry: Arc<WorkerRegistry>,
    policy: RoutingPolicy,
    next_worker: AtomicUsize,
    prefix_index: Arc<PrefixIndex>,
    inflight: Arc<InflightTracker>,
    block_size: usize,
    cache_weight: f64,
    load_weight: f64,
}

impl Router {
    pub fn new(registry: Arc<WorkerRegistry>, config: &RouterConfig) -> Self {
        Self {
            registry,
            policy: config.routing_policy,
            next_worker: AtomicUsize::new(0),
            prefix_index: Arc::new(PrefixIndex::new()),
            inflight: Arc::new(InflightTracker::with_capacity(config.workers.len())),
            block_size: config.prompt_block_chars.max(1),
            cache_weight: config.cache_weight,
            load_weight: config.load_weight,
        }
    }

    pub fn policy(&self) -> RoutingPolicy {
        self.policy
    }

    pub fn registry(&self) -> Arc<WorkerRegistry> {
        Arc::clone(&self.registry)
    }

    pub fn prefix_index(&self) -> Arc<PrefixIndex> {
        Arc::clone(&self.prefix_index)
    }

    pub fn inflight(&self) -> Arc<InflightTracker> {
        Arc::clone(&self.inflight)
    }

    /// Route a full chat-completion request (prefill + decode on one worker).
    ///
    /// The chain is recorded against the chosen worker at dispatch time — the
    /// optimistic belief that the worker caches it after prefill. If the
    /// request later fails, the belief is wrong until the TTL sweeps it; stale
    /// entries cause mis-routes, never incorrect results.
    pub fn select_for_chat(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<RouteDecision, RouterError> {
        let chain = chain_hash(&request.canonical_prompt(), self.block_size);
        let healthy = self.registry.healthy_workers();
        if healthy.is_empty() {
            return Err(RouterError::NoHealthyWorkers {
                policy: self.policy.to_string(),
            });
        }

        // A single proxied request spans prefill and decode, so workers labelled
        // `both` are preferred; fall back to any healthy worker if none exist.
        let candidates: Vec<Worker> = if healthy.iter().any(|w| w.pool == Pool::Both) {
            healthy
                .into_iter()
                .filter(|w| w.pool == Pool::Both)
                .collect()
        } else {
            healthy
        };

        match self.policy {
            RoutingPolicy::RoundRobin => {
                let index =
                    self.next_worker.fetch_add(1, AtomicOrdering::Relaxed) % candidates.len();
                let worker = candidates[index].clone();
                let matched_prefix_len = self.prefix_index.longest_match(worker.id, &chain);
                self.prefix_index.record(worker.id, &chain);
                Ok(RouteDecision {
                    worker,
                    matched_prefix_len,
                    score: 0.0,
                })
            }
            RoutingPolicy::CacheAware => {
                let best = self.pick_best(candidates, &chain).ok_or_else(|| {
                    RouterError::NoHealthyWorkers {
                        policy: self.policy.to_string(),
                    }
                })?;
                self.prefix_index.record(best.worker.id, &chain);
                Ok(RouteDecision {
                    worker: best.worker,
                    matched_prefix_len: best.matched,
                    score: best.candidate.score,
                })
            }
            RoutingPolicy::Disaggregated => Err(RouterError::DisaggregatedRequiresTwoPhase),
        }
    }

    /// Select a worker from a specific pool — one phase of the disaggregated
    /// flow. Always scores cache-aware, regardless of the configured policy:
    /// within a pool, the right worker is the one with the best cache/load
    /// tradeoff, and disaggregated pools exist precisely to exploit KV reuse.
    ///
    /// The chain is recorded optimistically at dispatch; for a decode worker
    /// the KV blocks arrive via transfer milliseconds later, opening a tiny
    /// mis-route window — never a correctness bug.
    pub fn select_in_pool(&self, pool: Pool, chain: &[u64]) -> Result<RouteDecision, RouterError> {
        let candidates: Vec<Worker> = self
            .registry
            .healthy_workers()
            .into_iter()
            .filter(|worker| worker.pool == pool)
            .collect();
        if candidates.is_empty() {
            return Err(RouterError::NoHealthyWorkersInPool {
                pool: pool.to_string(),
                policy: self.policy.to_string(),
            });
        }
        let best = self.pick_best(candidates, chain).ok_or_else(|| {
            RouterError::NoHealthyWorkersInPool {
                pool: pool.to_string(),
                policy: self.policy.to_string(),
            }
        })?;
        self.prefix_index.record(best.worker.id, chain);
        Ok(RouteDecision {
            worker: best.worker,
            matched_prefix_len: best.matched,
            score: best.candidate.score,
        })
    }

    /// Score every candidate and return the winner, if any.
    fn pick_best(&self, candidates: Vec<Worker>, chain: &[u64]) -> Option<ScoredWorker> {
        candidates
            .into_iter()
            .map(|worker| {
                let matched = self.prefix_index.longest_match(worker.id, chain);
                let candidate = Candidate {
                    score: self.score_worker(worker.id, matched),
                    in_flight: self.inflight.count(worker.id),
                    id: worker.id,
                };
                ScoredWorker {
                    worker,
                    matched,
                    candidate,
                }
            })
            .max_by(|a, b| a.candidate.cmp(&b.candidate))
    }

    /// `cache_hit_blocks * cache_weight - queued_requests * load_weight`.
    fn score_worker(&self, worker: WorkerId, matched_blocks: usize) -> f64 {
        matched_blocks as f64 * self.cache_weight
            - self.inflight.count(worker) as f64 * self.load_weight
    }
}

struct ScoredWorker {
    worker: Worker,
    matched: usize,
    candidate: Candidate,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatMessage, InflightGuard, WorkerConfig};

    fn setup(
        policy: RoutingPolicy,
        pools: [Pool; 2],
        healthy: &[u64],
    ) -> (Router, Arc<WorkerRegistry>) {
        let config = RouterConfig {
            workers: vec![
                WorkerConfig {
                    url: "http://a".to_string(),
                    pool: pools[0],
                },
                WorkerConfig {
                    url: "http://b".to_string(),
                    pool: pools[1],
                },
            ],
            routing_policy: policy,
            health_check_interval_secs: 5,
            prompt_block_chars: 512,
            cache_weight: 1.0,
            load_weight: 0.5,
            prefix_index_ttl_secs: 300,
            prefix_index_max_entries: 100_000,
            kv_transfer_cost_ms: 20,
        };
        let registry = Arc::new(WorkerRegistry::from_config(&config.workers));
        for id in healthy {
            registry.set_healthy(*id, true);
        }
        (Router::new(Arc::clone(&registry), &config), registry)
    }

    fn request(prompt: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "mock".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
            max_tokens: None,
            temperature: None,
            stream: false,
            extra: Default::default(),
        }
    }

    /// 1000 chars => 2 blocks at the default 512-char block size.
    const TWO_BLOCKS: &str = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

    #[test]
    fn candidate_ordering_prefers_score_then_load_then_id() {
        let high_score = Candidate {
            score: 2.0,
            in_flight: 5,
            id: 1,
        };
        let low_score = Candidate {
            score: 1.0,
            in_flight: 0,
            id: 0,
        };
        assert!(high_score > low_score, "score dominates load");

        let idle = Candidate {
            score: 1.0,
            in_flight: 0,
            id: 1,
        };
        let busy = Candidate {
            score: 1.0,
            in_flight: 3,
            id: 0,
        };
        assert!(idle > busy, "equal score breaks ties on load");

        let first = Candidate {
            score: 1.0,
            in_flight: 0,
            id: 0,
        };
        let second = Candidate {
            score: 1.0,
            in_flight: 0,
            id: 1,
        };
        assert!(first > second, "equal score and load breaks ties on id");
    }

    #[test]
    fn round_robin_cycles_through_healthy_workers() {
        let (router, _) = setup(RoutingPolicy::RoundRobin, [Pool::Both; 2], &[0, 1]);
        let first = router.select_for_chat(&request("hello")).unwrap();
        let second = router.select_for_chat(&request("hello")).unwrap();
        let third = router.select_for_chat(&request("hello")).unwrap();
        assert_eq!(
            (first.worker.id, second.worker.id, third.worker.id),
            (0, 1, 0)
        );
    }

    #[test]
    fn never_routes_to_a_down_worker() {
        let (router, _) = setup(RoutingPolicy::RoundRobin, [Pool::Both; 2], &[0]);
        for _ in 0..16 {
            assert_eq!(router.select_for_chat(&request("x")).unwrap().worker.id, 0);
        }
    }

    #[test]
    fn no_healthy_workers_is_an_error() {
        let (router, _) = setup(RoutingPolicy::RoundRobin, [Pool::Both; 2], &[]);
        assert!(matches!(
            router.select_for_chat(&request("x")),
            Err(RouterError::NoHealthyWorkers { .. })
        ));
    }

    #[test]
    fn falls_back_to_other_pools_when_no_both_workers_are_healthy() {
        let (router, _) = setup(RoutingPolicy::RoundRobin, [Pool::Both, Pool::Prefill], &[1]);
        assert_eq!(
            router.select_for_chat(&request("x")).unwrap().worker.pool,
            Pool::Prefill
        );
    }

    #[test]
    fn matched_prefix_length_is_tracked_per_worker() {
        let (router, _) = setup(RoutingPolicy::RoundRobin, [Pool::Both; 2], &[0, 1]);

        let first = router.select_for_chat(&request(TWO_BLOCKS)).unwrap();
        assert_eq!((first.worker.id, first.matched_prefix_len), (0, 0));

        // Round-robin moves to worker 1, which has its own (empty) prefix view.
        let second = router.select_for_chat(&request(TWO_BLOCKS)).unwrap();
        assert_eq!((second.worker.id, second.matched_prefix_len), (1, 0));

        // Back on worker 0, both blocks are now known.
        let third = router.select_for_chat(&request(TWO_BLOCKS)).unwrap();
        assert_eq!((third.worker.id, third.matched_prefix_len), (0, 2));
    }

    #[test]
    fn cache_aware_prefers_the_worker_that_holds_the_chain() {
        let (router, _) = setup(RoutingPolicy::CacheAware, [Pool::Both; 2], &[0, 1]);
        // Simulate worker 1 having served this prompt before. Note the chain
        // must be computed over the canonical prompt (role prefixes included).
        let chain = chain_hash(&request(TWO_BLOCKS).canonical_prompt(), 512);
        router.prefix_index().record(1, &chain);

        let decision = router.select_for_chat(&request(TWO_BLOCKS)).unwrap();
        assert_eq!(decision.worker.id, 1);
        assert_eq!(decision.matched_prefix_len, 2);
        assert_eq!(decision.score, 2.0, "2 blocks * cache_weight 1.0");
    }

    #[test]
    fn load_term_redirects_traffic_away_from_a_hot_replica() {
        let (router, _) = setup(RoutingPolicy::CacheAware, [Pool::Both; 2], &[0, 1]);
        // Worker 0 is drowning in 10 in-flight requests; with empty caches on
        // both sides its score is -5 vs worker 1's 0, so the idle replica wins.
        let guards: Vec<InflightGuard> = (0..10)
            .map(|_| InflightGuard::acquire(&router.inflight(), 0))
            .collect();
        let decision = router.select_for_chat(&request("fresh prompt")).unwrap();
        assert_eq!(decision.worker.id, 1);
        assert_eq!(decision.score, 0.0);
        drop(guards);
    }

    #[test]
    fn cache_bonus_outweighs_a_small_queue() {
        let (router, _) = setup(RoutingPolicy::CacheAware, [Pool::Both; 2], &[0, 1]);
        let chain = chain_hash(&request(TWO_BLOCKS).canonical_prompt(), 512);
        router.prefix_index().record(0, &chain);
        // Worker 0: 2 cached blocks (score 2.0) minus 3 queued * 0.5 = 0.5.
        // Worker 1: 0. Cache affinity wins despite the queue.
        let guards: Vec<InflightGuard> = (0..3)
            .map(|_| InflightGuard::acquire(&router.inflight(), 0))
            .collect();
        let decision = router.select_for_chat(&request(TWO_BLOCKS)).unwrap();
        assert_eq!(decision.worker.id, 0);
        assert_eq!(decision.score, 0.5);
        drop(guards);
    }

    #[test]
    fn ties_break_on_fewest_in_flight() {
        let (router, _) = setup(RoutingPolicy::CacheAware, [Pool::Both; 2], &[0, 1]);
        // Empty caches: first pick goes to the lowest id (0)...
        let first = router
            .select_for_chat(&request("brand new prompt"))
            .unwrap();
        assert_eq!(first.worker.id, 0);
        // ...but with worker 0 busy, the tie flips to worker 1.
        let guard = InflightGuard::acquire(&router.inflight(), 0);
        let second = router
            .select_for_chat(&request("another new prompt"))
            .unwrap();
        assert_eq!(second.worker.id, 1);
        drop(guard);
    }

    #[test]
    fn disaggregated_selects_per_pool() {
        let (router, _) = setup(
            RoutingPolicy::Disaggregated,
            [Pool::Prefill, Pool::Decode],
            &[0, 1],
        );
        let chain = chain_hash("hello", 512);
        let prefill = router.select_in_pool(Pool::Prefill, &chain).unwrap();
        assert_eq!(prefill.worker.id, 0);
        assert_eq!(prefill.worker.pool, Pool::Prefill);
        let decode = router.select_in_pool(Pool::Decode, &chain).unwrap();
        assert_eq!(decode.worker.id, 1);
        assert_eq!(decode.worker.pool, Pool::Decode);
    }

    #[test]
    fn disaggregated_missing_pool_is_an_error() {
        let (router, _) = setup(
            RoutingPolicy::Disaggregated,
            [Pool::Prefill, Pool::Prefill],
            &[0, 1],
        );
        let chain = chain_hash("hello", 512);
        assert!(matches!(
            router.select_in_pool(Pool::Decode, &chain),
            Err(RouterError::NoHealthyWorkersInPool { .. })
        ));
    }

    #[test]
    fn disaggregated_rejects_single_phase_routing() {
        let (router, _) = setup(RoutingPolicy::Disaggregated, [Pool::Both; 2], &[0, 1]);
        assert!(matches!(
            router.select_for_chat(&request("x")),
            Err(RouterError::DisaggregatedRequiresTwoPhase)
        ));
    }
}
