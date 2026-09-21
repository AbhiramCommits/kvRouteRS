use std::hash::{DefaultHasher, Hash, Hasher};

/// Compute the prefix-hash chain of a canonical prompt string.
///
/// The prompt is split into fixed-size blocks of `block_size` characters and
/// each block is hashed into its predecessor:
///
/// ```text
/// h_0 = hash(block_0)
/// h_i = hash(h_{i-1} || block_i)
/// ```
///
/// The returned `Vec<u64>` is the prompt's prefix-hash chain. Character blocks
/// stand in for vLLM's 16-token blocks in this tokenizer-free design; a
/// production deployment would replace the block split with the engine's own
/// block table without touching anything else in this crate.
///
/// # Invariant
///
/// Two prompts share their first `k` KV-cache blocks **exactly when** the first
/// `k` hashes of their chains are equal. Everything cache-aware in this crate —
/// longest-prefix matching, scoring, eviction — rests on that property: a
/// worker holds prompt `P`'s first `k` blocks iff
/// `PrefixIndex::longest_match(worker, chain(P)) >= k`.
///
/// # Caveats
///
/// - Blocks count characters, not tokens, so the "cache" being modeled is the
///   character-block cache — deliberately cheap and dependency-free.
/// - `DefaultHasher` (SipHash-1-3 with fixed keys) has no stability guarantee
///   across Rust versions. The index is in-memory only and rebuilt from
///   nothing on restart, so cross-version stability is irrelevant.
pub fn chain_hash(prompt: &str, block_size: usize) -> Vec<u64> {
    let block_size = block_size.max(1);
    let chars: Vec<char> = prompt.chars().collect();
    let mut chain = Vec::with_capacity(chars.len().div_ceil(block_size));
    let mut previous: Option<u64> = None;
    for block in chars.chunks(block_size) {
        let mut hasher = DefaultHasher::new();
        if let Some(prev) = previous {
            prev.hash(&mut hasher);
        }
        block.hash(&mut hasher);
        let hash = hasher.finish();
        chain.push(hash);
        previous = Some(hash);
    }
    chain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_prompts_produce_identical_chains() {
        let a = chain_hash("the same prompt text", 4);
        let b = chain_hash("the same prompt text", 4);
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn shared_prefix_produces_shared_chain_head() {
        let base = "once upon a time";
        let chain_base = chain_hash(base, 4);
        let chain_longer = chain_hash(&format!("{base} there was a cache"), 4);
        assert!(chain_longer.len() > chain_base.len());
        assert_eq!(&chain_longer[..chain_base.len()], &chain_base[..]);
    }

    #[test]
    fn single_char_change_at_start_diverges_the_whole_chain() {
        // Same length, only the first character differs: h_0 differs, so every
        // subsequent h_i inherits a different predecessor and the chains share
        // nothing. (64-bit hashes make collisions effectively impossible here.)
        let a = chain_hash("aaaaaaaaaaaaaaaa", 4);
        let b = chain_hash("baaaaaaaaaaaaaaa", 4);
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(&b).all(|(x, y)| x != y));
    }

    #[test]
    fn block_count_is_ceil_chars_over_block_size() {
        assert_eq!(chain_hash("12345", 2).len(), 3);
        assert_eq!(chain_hash("1234", 2).len(), 2);
        assert_eq!(chain_hash("abc", 512).len(), 1);
        assert_eq!(chain_hash("", 512).len(), 0);
    }

    #[test]
    fn multi_byte_chars_count_as_single_characters() {
        // Four 4-byte UTF-8 characters with block size 2 => 2 blocks, not 8.
        assert_eq!(
            chain_hash("\u{1F600}\u{1F600}\u{1F600}\u{1F600}", 2).len(),
            2
        );
    }

    #[test]
    fn zero_block_size_clamps_to_one() {
        let a = chain_hash("hello", 0);
        let b = chain_hash("hello", 1);
        assert_eq!(a, b);
    }
}
