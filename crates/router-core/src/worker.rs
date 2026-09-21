use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::{Pool, WorkerConfig};

/// Stable identifier for a worker, assigned in configuration order.
pub type WorkerId = u64;

/// Static description of a backend worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Worker {
    pub id: WorkerId,
    pub url: String,
    pub pool: Pool,
}

/// A worker plus its runtime health state.
#[derive(Debug, Clone)]
pub struct WorkerState {
    pub worker: Worker,
    pub healthy: bool,
}

/// Shared, concurrently-updated health registry for all configured workers.
///
/// Implementation note: we use `std::sync::RwLock<HashMap<WorkerId, WorkerState>>`
/// rather than `arc-swap`.
///
/// - The worker set is tiny (a handful of endpoints) and mutations are rare — one
///   health update per worker every `health_check_interval_secs` — so lock
///   contention is negligible next to the milliseconds each proxied request spends
///   on I/O.
/// - `arc-swap` would force a full `Arc<HashMap>` swap on every health flip and a
///   snapshot copy on every read, while the `RwLock` allows in-place per-entry
///   mutation and lets future per-worker metadata (prefix caches, request counts)
///   live behind the same lock.
/// - Read sections are short and never await, so the blocking std lock is safe
///   under the tokio runtime.
#[derive(Debug, Default)]
pub struct WorkerRegistry {
    workers: RwLock<HashMap<WorkerId, WorkerState>>,
    /// Monotonic id allocator for runtime-registered workers. Ids are never
    /// reused: a replacement worker must never inherit the cache affinity
    /// (or in-flight accounting) keyed to a removed one.
    next_id: AtomicU64,
}

impl WorkerRegistry {
    /// Build a registry from configuration. All workers start `healthy: false`
    /// until the health poller marks them up.
    pub fn from_config(workers: &[WorkerConfig]) -> Self {
        let entries = workers
            .iter()
            .enumerate()
            .map(|(index, config)| {
                let id = index as WorkerId;
                let worker = Worker {
                    id,
                    url: config.url.clone(),
                    pool: config.pool,
                };
                (
                    id,
                    WorkerState {
                        worker,
                        healthy: false,
                    },
                )
            })
            .collect();
        Self {
            workers: RwLock::new(entries),
            next_id: AtomicU64::new(workers.len() as u64),
        }
    }

    /// Register a worker discovered at runtime (Kubernetes discovery) and
    /// return its freshly allocated id. Starts unhealthy until the poller
    /// marks it up.
    pub fn register_worker(&self, url: String, pool: Pool) -> WorkerId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.workers.write().unwrap_or_else(|p| p.into_inner());
        let worker = Worker { id, url, pool };
        guard.insert(
            id,
            WorkerState {
                worker: worker.clone(),
                healthy: false,
            },
        );
        id
    }

    /// Remove a worker entirely (its endpoints disappeared). Callers are
    /// responsible for purging the worker from the prefix index: its KV cache
    /// died with the pod, and a replacement pod must not inherit its affinity.
    pub fn remove_worker(&self, id: WorkerId) -> Option<Worker> {
        let mut guard = self.workers.write().unwrap_or_else(|p| p.into_inner());
        guard.remove(&id).map(|state| state.worker)
    }

    /// Update a worker's health state. Returns `true` if the state actually changed.
    pub fn set_healthy(&self, id: WorkerId, healthy: bool) -> bool {
        let mut guard = self.workers.write().unwrap_or_else(|p| p.into_inner());
        match guard.get_mut(&id) {
            Some(state) => {
                let changed = state.healthy != healthy;
                state.healthy = healthy;
                changed
            }
            None => false,
        }
    }

    /// Look up a worker by id.
    pub fn get(&self, id: WorkerId) -> Option<Worker> {
        let guard = self.workers.read().unwrap_or_else(|p| p.into_inner());
        guard.get(&id).map(|state| state.worker.clone())
    }

    /// Whether the worker is currently healthy.
    pub fn is_healthy(&self, id: WorkerId) -> bool {
        let guard = self.workers.read().unwrap_or_else(|p| p.into_inner());
        guard.get(&id).is_some_and(|state| state.healthy)
    }

    /// Snapshot of every registered worker, ordered by id (configuration order).
    /// Ordering matters: round-robin selection and "first healthy worker" lookups
    /// must be deterministic.
    pub fn all_workers(&self) -> Vec<Worker> {
        let guard = self.workers.read().unwrap_or_else(|p| p.into_inner());
        let mut workers: Vec<Worker> = guard.values().map(|state| state.worker.clone()).collect();
        workers.sort_by_key(|worker| worker.id);
        workers
    }

    /// Snapshot of the workers currently marked healthy, ordered by id.
    pub fn healthy_workers(&self) -> Vec<Worker> {
        let guard = self.workers.read().unwrap_or_else(|p| p.into_inner());
        let mut workers: Vec<Worker> = guard
            .values()
            .filter(|state| state.healthy)
            .map(|state| state.worker.clone())
            .collect();
        workers.sort_by_key(|worker| worker.id);
        workers
    }

    pub fn healthy_count(&self) -> usize {
        let guard = self.workers.read().unwrap_or_else(|p| p.into_inner());
        guard.values().filter(|state| state.healthy).count()
    }

    pub fn any_healthy(&self) -> bool {
        let guard = self.workers.read().unwrap_or_else(|p| p.into_inner());
        guard.values().any(|state| state.healthy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_workers() -> Vec<WorkerConfig> {
        vec![
            WorkerConfig {
                url: "http://a".to_string(),
                pool: Pool::Both,
            },
            WorkerConfig {
                url: "http://b".to_string(),
                pool: Pool::Both,
            },
        ]
    }

    #[test]
    fn starts_with_everything_down() {
        let registry = WorkerRegistry::from_config(&two_workers());
        assert!(!registry.any_healthy());
        assert_eq!(registry.healthy_count(), 0);
        assert_eq!(registry.all_workers().len(), 2);
    }

    #[test]
    fn health_updates_report_changes_only() {
        let registry = WorkerRegistry::from_config(&two_workers());
        assert!(registry.set_healthy(0, true));
        assert!(!registry.set_healthy(0, true));
        assert!(registry.is_healthy(0));
        assert!(!registry.is_healthy(1));
        assert_eq!(registry.healthy_workers(), vec![registry.get(0).unwrap()]);
    }

    #[test]
    fn unknown_worker_id_is_ignored() {
        let registry = WorkerRegistry::from_config(&two_workers());
        assert!(!registry.set_healthy(42, true));
        assert!(registry.get(42).is_none());
    }

    #[test]
    fn registers_and_removes_runtime_workers() {
        let registry = WorkerRegistry::from_config(&two_workers());
        let id = registry.register_worker("http://discovered".to_string(), Pool::Both);
        assert_eq!(id, 2);
        assert_eq!(registry.all_workers().len(), 3);
        assert!(!registry.is_healthy(id));

        registry.set_healthy(id, true);
        assert!(registry.is_healthy(id));

        let removed = registry.remove_worker(id);
        assert_eq!(removed.unwrap().url, "http://discovered");
        assert_eq!(registry.all_workers().len(), 2);

        // Ids are not reused: a replacement worker must never inherit the
        // cache affinity keyed to the removed one.
        let next = registry.register_worker("http://replacement".to_string(), Pool::Both);
        assert_eq!(next, 3);
    }
}
