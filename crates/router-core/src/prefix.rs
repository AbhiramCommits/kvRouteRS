use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::RwLock;

use crate::worker::WorkerId;

/// Split a prompt into cheap, model-agnostic tokens: lowercased runs of
/// alphanumeric characters. This is a stand-in for a real tokenizer — good enough
/// to measure prefix reuse in demos, and it keeps the core free of heavy
/// tokenizer dependencies. A production deployment would use the engine's own
/// tokenizer (or per-worker cache metadata reported by the engine).
pub fn tokenize(prompt: &str) -> Vec<String> {
    prompt
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| word.to_lowercase())
        .collect()
}

/// Prefix hashes for every non-empty prefix of `tokens`. Prefix i hashes
/// `tokens[..i]` (plus separators so "a" + "bc" != "ab" + "c").
///
/// Hashes use `std::hash::DefaultHasher` (SipHash-1-3 with fixed keys). The index
/// is process-local and rebuilt from nothing on restart, so cross-version
/// determinism is irrelevant.
fn prefix_hashes(tokens: &[String]) -> Vec<u64> {
    (1..=tokens.len())
        .map(|len| {
            let mut hasher = DefaultHasher::new();
            tokens[..len].hash(&mut hasher);
            hasher.finish()
        })
        .collect()
}

/// Per-worker memory of which prompt prefixes have been routed there.
///
/// This mirrors, at the router level, the KV-cache residency tracked by each
/// engine: once a worker has served a prompt, we assume its prefixes stay cached
/// (in reality they may be evicted — a future step can model eviction with the
/// same LRU semantics the mock backend uses).
#[derive(Debug, Default)]
pub struct PrefixIndex {
    worker_prefixes: RwLock<HashMap<WorkerId, HashSet<u64>>>,
}

impl PrefixIndex {
    /// Longest prefix of `tokens` this worker is known to have cached.
    pub fn matched_prefix_len(&self, worker: WorkerId, tokens: &[String]) -> usize {
        if tokens.is_empty() {
            return 0;
        }
        let guard = self
            .worker_prefixes
            .read()
            .unwrap_or_else(|p| p.into_inner());
        let Some(set) = guard.get(&worker) else {
            return 0;
        };
        for (index, hash) in prefix_hashes(tokens).iter().enumerate().rev() {
            if set.contains(hash) {
                return index + 1;
            }
        }
        0
    }

    /// Record that this worker has now seen every prefix of `tokens`.
    pub fn record(&self, worker: WorkerId, tokens: &[String]) {
        if tokens.is_empty() {
            return;
        }
        let hashes = prefix_hashes(tokens);
        let mut guard = self
            .worker_prefixes
            .write()
            .unwrap_or_else(|p| p.into_inner());
        guard.entry(worker).or_default().extend(hashes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_into_lowercase_words() {
        assert_eq!(tokenize("Hello, World!"), vec!["hello", "world"]);
        assert_eq!(
            tokenize("  mixed-CASE\tinput\n"),
            vec!["mixed", "case", "input"]
        );
        assert_eq!(tokenize(""), Vec::<String>::new());
    }

    #[test]
    fn matched_length_grows_with_recording() {
        let index = PrefixIndex::default();
        let tokens = tokenize("the quick brown fox");

        assert_eq!(index.matched_prefix_len(0, &tokens), 0);
        index.record(0, &tokens);
        assert_eq!(index.matched_prefix_len(0, &tokens), 4);

        let longer = tokenize("the quick brown fox jumps");
        assert_eq!(index.matched_prefix_len(0, &longer), 4);
        index.record(0, &longer);
        assert_eq!(index.matched_prefix_len(0, &longer), 5);
    }

    #[test]
    fn prefix_tracking_is_per_worker() {
        let index = PrefixIndex::default();
        let tokens = tokenize("cache aware routing");

        index.record(0, &tokens);
        assert_eq!(index.matched_prefix_len(0, &tokens), 3);
        assert_eq!(index.matched_prefix_len(1, &tokens), 0);
    }

    #[test]
    fn partial_prefix_matches() {
        let index = PrefixIndex::default();
        index.record(0, &tokenize("the quick brown fox"));

        // Only the first two words overlap.
        let overlap = tokenize("the quick lazy dog");
        assert_eq!(index.matched_prefix_len(0, &overlap), 2);

        // No overlap at all.
        let none = tokenize("totally different prompt");
        assert_eq!(index.matched_prefix_len(0, &none), 0);
    }
}
