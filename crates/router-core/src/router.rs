use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::prefix::{tokenize, PrefixIndex};
use crate::{Pool, RouterError, RoutingPolicy, Worker, WorkerRegistry};

/// The outcome of a routing decision.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub worker: Worker,
    /// Length of the longest prompt prefix the chosen worker is known to have
    /// cached, measured before this request is recorded.
    pub matched_prefix_len: usize,
}

/// Stateless request router: picks a healthy worker for each request.
///
/// The prefix index is maintained regardless of policy so that (a) requests can
/// always report matched-prefix length and (b) the cache-aware policy in the next
/// milestone has data ready to score workers on.
#[derive(Debug)]
pub struct Router {
    registry: Arc<WorkerRegistry>,
    policy: RoutingPolicy,
    next_worker: AtomicUsize,
    prefix_index: PrefixIndex,
}

impl Router {
    pub fn new(registry: Arc<WorkerRegistry>, policy: RoutingPolicy) -> Self {
        Self {
            registry,
            policy,
            next_worker: AtomicUsize::new(0),
            prefix_index: PrefixIndex::default(),
        }
    }

    pub fn policy(&self) -> RoutingPolicy {
        self.policy
    }

    pub fn registry(&self) -> Arc<WorkerRegistry> {
        Arc::clone(&self.registry)
    }

    /// Select a worker to serve a chat-completion request.
    ///
    /// Pool policy for now: a single proxied request spans prefill and decode, so
    /// workers labelled `both` are preferred; if none are healthy we fall back to
    /// any healthy worker (prefill or decode).
    pub fn select_for_chat(&self, prompt: &str) -> Result<RouteDecision, RouterError> {
        match self.policy {
            RoutingPolicy::CacheAware => {
                // Seam for the next milestone: prefix-hash scoring across workers
                // plugs in here. Failing fast keeps cache-aware routing from
                // silently degrading to round-robin.
                return Err(RouterError::PolicyNotImplemented(self.policy.to_string()));
            }
            RoutingPolicy::RoundRobin => {}
        }

        let healthy = self.registry.healthy_workers();
        if healthy.is_empty() {
            return Err(RouterError::NoHealthyWorkers {
                policy: self.policy.to_string(),
            });
        }

        let candidates: Vec<Worker> = if healthy.iter().any(|worker| worker.pool == Pool::Both) {
            healthy
                .into_iter()
                .filter(|worker| worker.pool == Pool::Both)
                .collect()
        } else {
            healthy
        };
        let index = self.next_worker.fetch_add(1, Ordering::Relaxed) % candidates.len();
        let worker = candidates[index].clone();

        let tokens = tokenize(prompt);
        let matched_prefix_len = self.prefix_index.matched_prefix_len(worker.id, &tokens);
        self.prefix_index.record(worker.id, &tokens);

        Ok(RouteDecision {
            worker,
            matched_prefix_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WorkerConfig;

    fn registry(healthy: &[u64]) -> Arc<WorkerRegistry> {
        let registry = Arc::new(WorkerRegistry::from_config(&[
            WorkerConfig {
                url: "http://a".to_string(),
                pool: Pool::Both,
            },
            WorkerConfig {
                url: "http://b".to_string(),
                pool: Pool::Both,
            },
        ]));
        for id in healthy {
            registry.set_healthy(*id, true);
        }
        registry
    }

    #[test]
    fn round_robin_cycles_through_healthy_workers() {
        let router = Router::new(registry(&[0, 1]), RoutingPolicy::RoundRobin);
        let first = router.select_for_chat("hello").unwrap();
        let second = router.select_for_chat("hello").unwrap();
        let third = router.select_for_chat("hello").unwrap();
        assert_eq!(first.worker.id, 0);
        assert_eq!(second.worker.id, 1);
        assert_eq!(third.worker.id, 0);
    }

    #[test]
    fn never_routes_to_a_down_worker() {
        let router = Router::new(registry(&[0]), RoutingPolicy::RoundRobin);
        for _ in 0..16 {
            assert_eq!(router.select_for_chat("x").unwrap().worker.id, 0);
        }
    }

    #[test]
    fn no_healthy_workers_is_an_error() {
        let router = Router::new(registry(&[]), RoutingPolicy::RoundRobin);
        assert!(matches!(
            router.select_for_chat("x"),
            Err(RouterError::NoHealthyWorkers { .. })
        ));
    }

    #[test]
    fn cache_aware_policy_is_explicitly_unimplemented() {
        let router = Router::new(registry(&[0, 1]), RoutingPolicy::CacheAware);
        assert!(matches!(
            router.select_for_chat("x"),
            Err(RouterError::PolicyNotImplemented(policy)) if policy == "cache_aware"
        ));
    }

    #[test]
    fn falls_back_to_other_pools_when_no_both_workers_are_healthy() {
        let registry = Arc::new(WorkerRegistry::from_config(&[
            WorkerConfig {
                url: "http://a".to_string(),
                pool: Pool::Both,
            },
            WorkerConfig {
                url: "http://b".to_string(),
                pool: Pool::Prefill,
            },
        ]));
        registry.set_healthy(1, true); // the prefill worker is the only healthy one
        let router = Router::new(registry, RoutingPolicy::RoundRobin);
        assert_eq!(
            router.select_for_chat("x").unwrap().worker.pool,
            Pool::Prefill
        );
    }

    #[test]
    fn matched_prefix_length_is_tracked_per_worker() {
        let router = Router::new(registry(&[0, 1]), RoutingPolicy::RoundRobin);

        let first = router.select_for_chat("the quick brown fox").unwrap();
        assert_eq!((first.worker.id, first.matched_prefix_len), (0, 0));

        // Round-robin moves to worker 1, which has its own (empty) prefix view.
        let second = router.select_for_chat("the quick brown fox jumps").unwrap();
        assert_eq!((second.worker.id, second.matched_prefix_len), (1, 0));

        // Back on worker 0, the first four tokens are now known.
        let third = router
            .select_for_chat("the quick brown fox jumps over")
            .unwrap();
        assert_eq!((third.worker.id, third.matched_prefix_len), (0, 4));
    }
}
