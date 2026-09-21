use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use smallvec::SmallVec;

use crate::worker::WorkerId;

/// Number of shards backing the index.
///
/// Justification: a power of two lets a block hash map to a shard with
/// `hash & 63` (no division), and 64 is an order of magnitude more than the
/// number of workers a deployment realistically has — a burst of concurrent
/// requests touching different prompts rarely contends on the same shard lock.
/// Every operation holds at most one shard lock at a time, so the shard count
/// is purely a contention-tuning knob, never a correctness parameter.
const NUM_SHARDS: usize = 64;

/// Inline capacity of the per-block worker set: a block is usually resident on
/// 1-4 replicas; `SmallVec` keeps that allocation-free and spills to the heap
/// gracefully when a popular prefix fans out further.
const INLINE_WORKERS: usize = 4;

#[derive(Debug, Clone)]
struct IndexEntry {
    /// Workers believed to hold this block in their KV cache.
    workers: SmallVec<[WorkerId; INLINE_WORKERS]>,
    /// Last time this entry was written (`record`) or matched (`longest_match`
    /// touches on hit). Drives both TTL expiry and LRU eviction ordering.
    last_touched: Instant,
}

#[derive(Debug)]
struct Shard {
    entries: RwLock<HashMap<u64, IndexEntry>>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EvictionStats {
    /// Entries removed because they exceeded the TTL.
    pub expired: usize,
    /// Entries removed because the index exceeded the max-entries cap.
    pub lru_evicted: usize,
}

/// Router-side belief about which workers hold which KV blocks.
///
/// Entries are a *belief*, not a guarantee: a worker may evict its real KV
/// cache at any time (memory pressure), and the index only notices once the TTL
/// expires. Stale entries cause mis-routes — a request lands on a worker that
/// no longer has the prefix and pays the prefill cost again — never crashes and
/// never incorrect responses. That asymmetry (cheap reads, eventual cleanup) is
/// what makes the index safe to consult on every request.
///
/// The index is sharded ([`NUM_SHARDS`] shards behind one `RwLock` each) and
/// bounded by a background eviction task ([`PrefixIndex::spawn_eviction_task`])
/// applying TTL expiry plus a global LRU cap.
#[derive(Debug)]
pub struct PrefixIndex {
    shards: Vec<Shard>,
}

impl Default for PrefixIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl PrefixIndex {
    pub fn new() -> Self {
        Self {
            shards: (0..NUM_SHARDS)
                .map(|_| Shard {
                    entries: RwLock::new(HashMap::new()),
                })
                .collect(),
        }
    }

    #[inline]
    fn shard(&self, hash: u64) -> &Shard {
        &self.shards[(hash as usize) & (NUM_SHARDS - 1)]
    }

    /// Record that `worker` now holds every block in `chain` (it just prefilled
    /// this prompt, or received the blocks via KV transfer).
    pub fn record(&self, worker: WorkerId, chain: &[u64]) {
        let now = Instant::now();
        for &hash in chain {
            let mut entries = self
                .shard(hash)
                .entries
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let entry = entries.entry(hash).or_insert_with(|| IndexEntry {
                workers: SmallVec::new(),
                last_touched: now,
            });
            if !entry.workers.contains(&worker) {
                entry.workers.push(worker);
            }
            entry.last_touched = now;
        }
    }

    /// Longest prefix of `chain` (in blocks) that `worker` is believed to hold.
    ///
    /// Walks the chain from the front and stops at the first block the worker
    /// is not recorded against — the chain invariant makes later blocks
    /// irrelevant after a miss. Matching refreshes the entries' `last_touched`
    /// (TTL + LRU hybrid). The touch pass runs after the read pass so no shard
    /// lock is ever held while acquiring another one.
    pub fn longest_match(&self, worker: WorkerId, chain: &[u64]) -> usize {
        let mut matched_hashes = Vec::new();
        for &hash in chain {
            let entries = self
                .shard(hash)
                .entries
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match entries.get(&hash) {
                Some(entry) if entry.workers.contains(&worker) => matched_hashes.push(hash),
                _ => break,
            }
        }

        let now = Instant::now();
        for hash in &matched_hashes {
            if let Some(entry) = self
                .shard(*hash)
                .entries
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .get_mut(hash)
            {
                entry.last_touched = now;
            }
        }
        matched_hashes.len()
    }

    /// Drop a worker from the index entirely: its KV cache died with it (worker
    /// went down and will restart empty), so cache-affinity scoring must not
    /// prefer it until it has genuinely rebuilt that cache.
    pub fn remove_worker(&self, worker: WorkerId) {
        for shard in &self.shards {
            let mut entries = shard
                .entries
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            entries.retain(|_, entry| {
                entry.workers.retain(|w| *w != worker);
                !entry.workers.is_empty()
            });
        }
    }

    /// Number of block-hash entries currently in the index.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .entries
                    .read()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .len()
            })
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evict entries not touched within `ttl`, then trim the index down to
    /// `max_entries` by LRU age (oldest `last_touched` first).
    ///
    /// O(n log n) in the index size because of the global LRU sort; intended
    /// for the background eviction task, never the request path.
    pub fn evict(&self, ttl: Duration, max_entries: usize) -> EvictionStats {
        let now = Instant::now();
        let mut expired = 0usize;
        for shard in &self.shards {
            let mut entries = shard
                .entries
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let before = entries.len();
            entries.retain(|_, entry| now.duration_since(entry.last_touched) <= ttl);
            expired += before - entries.len();
        }

        let total = self.len();
        if total <= max_entries {
            return EvictionStats {
                expired,
                lru_evicted: 0,
            };
        }
        let overflow = total - max_entries;

        // Global LRU sweep: collect (age, hash, shard), evict the oldest.
        let mut oldest: Vec<(Instant, u64, usize)> = Vec::with_capacity(total);
        for (shard_index, shard) in self.shards.iter().enumerate() {
            let entries = shard
                .entries
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for (&hash, entry) in entries.iter() {
                oldest.push((entry.last_touched, hash, shard_index));
            }
        }
        oldest.sort_by_key(|(touched, _, _)| *touched);

        let mut lru_evicted = 0usize;
        for (_, hash, shard_index) in oldest.into_iter().take(overflow) {
            if self.shards[shard_index]
                .entries
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .remove(&hash)
                .is_some()
            {
                lru_evicted += 1;
            }
        }
        EvictionStats {
            expired,
            lru_evicted,
        }
    }

    /// Spawn a background tokio task that applies TTL + LRU-cap eviction every
    /// `ttl / 3` (clamped to [1s, 60s]). Must be called from within a tokio
    /// runtime. The task is detached; it runs for the lifetime of the runtime.
    pub fn spawn_eviction_task(
        index: Arc<PrefixIndex>,
        ttl: Duration,
        max_entries: usize,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let period = (ttl / 3).clamp(Duration::from_secs(1), Duration::from_secs(60));
            let mut ticker = tokio::time::interval(period);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                let stats = index.evict(ttl, max_entries);
                if stats.expired > 0 || stats.lru_evicted > 0 {
                    tracing::debug!(
                        expired = stats.expired,
                        lru_evicted = stats.lru_evicted,
                        index_entries = index.len(),
                        "prefix index eviction sweep"
                    );
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hashing::chain_hash;
    use proptest::prelude::*;

    /// Small block size keeps test prompts short.
    fn chain(text: &str) -> Vec<u64> {
        chain_hash(text, 4)
    }

    #[test]
    fn record_then_match() {
        let index = PrefixIndex::new();
        let chain = chain("the quick brown fox");
        assert_eq!(index.longest_match(0, &chain), 0);
        index.record(0, &chain);
        assert_eq!(index.longest_match(0, &chain), chain.len());
        assert_eq!(index.len(), chain.len());
    }

    #[test]
    fn tracking_is_per_worker() {
        let index = PrefixIndex::new();
        let chain = chain("cache aware routing");
        index.record(0, &chain);
        assert_eq!(index.longest_match(0, &chain), chain.len());
        assert_eq!(index.longest_match(1, &chain), 0);
    }

    #[test]
    fn partial_prefix_matches_in_blocks() {
        let index = PrefixIndex::new();
        // 3 blocks: "abcd", "efgh", "ijkl".
        let longer = chain("abcdefghijkl");
        index.record(0, &longer);
        // Shares only the first block: "abcd" == "abcd", but the second block
        // is "ef" vs "efgh" — a partial block is not a cache hit.
        let shorter = chain("abcdef");
        assert_eq!(index.longest_match(0, &shorter), 1);
        // No overlap at all.
        assert_eq!(index.longest_match(0, &chain("zzzz")), 0);
    }

    #[test]
    fn ttl_expiry_removes_stale_entries() {
        let index = PrefixIndex::new();
        let chain = chain("hello world");
        index.record(0, &chain);
        std::thread::sleep(Duration::from_millis(5));
        let stats = index.evict(Duration::from_millis(1), usize::MAX);
        assert_eq!(stats.expired, chain.len());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn lru_cap_evicts_oldest_entries_first() {
        let index = PrefixIndex::new();
        let old_chain = chain("aaaa");
        let new_chain = chain("bbbb");
        index.record(0, &old_chain);
        std::thread::sleep(Duration::from_millis(2));
        index.record(0, &new_chain);

        let stats = index.evict(Duration::from_secs(300), 1);
        assert_eq!(stats.lru_evicted, 1);
        assert_eq!(index.len(), 1);
        assert_eq!(index.longest_match(0, &new_chain), new_chain.len());
        assert_eq!(index.longest_match(0, &old_chain), 0);
    }

    #[test]
    fn matching_touches_an_entry_and_extends_its_life() {
        let index = PrefixIndex::new();
        let chain_a = chain("aaaa");
        let chain_b = chain("bbbb");
        index.record(0, &chain_a);
        std::thread::sleep(Duration::from_millis(2));
        index.record(0, &chain_b);

        // Touching A makes it newer than B, so the cap evicts B instead.
        assert_eq!(index.longest_match(0, &chain_a), chain_a.len());
        let stats = index.evict(Duration::from_secs(300), 1);
        assert_eq!(stats.lru_evicted, 1);
        assert_eq!(index.longest_match(0, &chain_a), chain_a.len());
        assert_eq!(index.longest_match(0, &chain_b), 0);
    }

    #[test]
    fn remove_worker_clears_its_entries_but_keeps_others() {
        let index = PrefixIndex::new();
        let chain = chain("shared prompt");
        index.record(0, &chain);
        index.record(1, &chain);
        index.remove_worker(0);
        assert_eq!(index.longest_match(0, &chain), 0);
        assert_eq!(index.longest_match(1, &chain), chain.len());
        assert_eq!(index.len(), chain.len());
    }

    proptest! {
        #[test]
        fn longest_match_never_exceeds_shorter_prompt(
            prompt_a in "[a-z ]{0,200}",
            prompt_b in "[a-z ]{0,200}",
            block_size in 1usize..64,
        ) {
            let index = PrefixIndex::new();
            let chain_a = chain_hash(&prompt_a, block_size);
            let chain_b = chain_hash(&prompt_b, block_size);
            index.record(0, &chain_a);
            let matched = index.longest_match(0, &chain_b);
            prop_assert!(matched <= chain_a.len());
            prop_assert!(matched <= chain_b.len());
            // Hashing is deterministic: recomputing the chain must agree.
            prop_assert_eq!(chain_hash(&prompt_a, block_size), chain_a);
        }
    }
}
