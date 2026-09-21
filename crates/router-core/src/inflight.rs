use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::worker::WorkerId;

/// Lock-free per-worker in-flight request counts.
///
/// Worker ids are dense `0..n` by construction ([`WorkerRegistry::from_config`]
/// assigns them in configuration order), so a `Vec<AtomicU64>` gives us a fully
/// lock-free tracker the routing hot path can read without contention.
#[derive(Debug)]
pub struct InflightTracker {
    counts: Vec<AtomicU64>,
}

impl InflightTracker {
    pub fn with_capacity(workers: usize) -> Self {
        Self {
            counts: (0..workers).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// In-flight requests currently being served by `worker`.
    pub fn count(&self, worker: WorkerId) -> u64 {
        self.counts
            .get(worker as usize)
            .map(|count| count.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Total in-flight requests across all workers (for the metrics gauge).
    pub fn total(&self) -> u64 {
        self.counts
            .iter()
            .map(|count| count.load(Ordering::Relaxed))
            .sum()
    }

    fn increment(&self, worker: WorkerId) {
        if let Some(count) = self.counts.get(worker as usize) {
            count.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn decrement(&self, worker: WorkerId) {
        if let Some(count) = self.counts.get(worker as usize) {
            count.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// RAII guard for one in-flight request against a worker.
///
/// Increments on construction and decrements on `Drop` — including when the
/// proxied upstream call fails — so in-flight counts can never leak, and the
/// load term of the cache-aware score stays correct even under error storms.
#[derive(Debug)]
pub struct InflightGuard {
    tracker: Arc<InflightTracker>,
    worker: WorkerId,
}

impl InflightGuard {
    pub fn acquire(tracker: &Arc<InflightTracker>, worker: WorkerId) -> Self {
        tracker.increment(worker);
        Self {
            tracker: Arc::clone(tracker),
            worker,
        }
    }

    pub fn worker(&self) -> WorkerId {
        self.worker
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.tracker.decrement(self.worker);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashing::chain_hash;
    use crate::prefix::PrefixIndex;
    use crate::RouterError;

    #[test]
    fn guard_increments_and_decrements() {
        let tracker = Arc::new(InflightTracker::with_capacity(2));
        assert_eq!(tracker.count(0), 0);
        let guard = InflightGuard::acquire(&tracker, 0);
        assert_eq!(tracker.count(0), 1);
        assert_eq!(tracker.total(), 1);
        assert_eq!(guard.worker(), 0);
        drop(guard);
        assert_eq!(tracker.count(0), 0);
        assert_eq!(tracker.total(), 0);
    }

    fn fallible_upstream_call(
        tracker: &Arc<InflightTracker>,
        worker: WorkerId,
    ) -> Result<(), RouterError> {
        // The guard is acquired before the upstream call; when the call errors,
        // unwinding drops it and the count must return to zero.
        let _guard = InflightGuard::acquire(tracker, worker);
        Err(RouterError::NoHealthyWorkers {
            policy: "round_robin".to_string(),
        })
    }

    #[test]
    fn guard_released_when_upstream_call_errors() {
        let tracker = Arc::new(InflightTracker::with_capacity(1));
        for _ in 0..5 {
            assert!(fallible_upstream_call(&tracker, 0).is_err());
            assert_eq!(tracker.count(0), 0);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_hammering_leaves_counts_at_zero() {
        const TASKS: usize = 64;
        let tracker = Arc::new(InflightTracker::with_capacity(TASKS));
        let index = Arc::new(PrefixIndex::new());

        let mut handles = Vec::new();
        for task in 0..TASKS {
            let tracker = Arc::clone(&tracker);
            let index = Arc::clone(&index);
            handles.push(tokio::spawn(async move {
                let chain = chain_hash(&format!("concurrent prompt number {task}"), 512);
                for _ in 0..50 {
                    let _guard = InflightGuard::acquire(&tracker, task as WorkerId);
                    index.record(task as WorkerId, &chain);
                    let _ = index.longest_match(task as WorkerId, &chain);
                    tokio::time::sleep(std::time::Duration::from_micros(100)).await;
                }
            }));
        }

        // Completing without deadlocking is most of the test; the 30s timeout
        // turns a lock-order bug into a failure instead of a hang.
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            for handle in handles {
                handle.await.unwrap();
            }
        })
        .await
        .expect("index hammering deadlocked or exceeded 30s");

        assert_eq!(tracker.total(), 0, "in-flight counts leaked");
        assert_eq!(index.len(), TASKS, "one distinct chain per task");
    }
}
