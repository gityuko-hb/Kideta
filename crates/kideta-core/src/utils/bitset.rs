//! `FixedBitset` — dynamically-sized, heap-backed packed bitset.
//!
//! Storage is a `Vec<u64>` so each word holds 64 bits.  All operations are
//! O(1) except `reset` (O(words)).  Designed for tracking visited nodes in the
//! HNSW search loop with zero per-query allocation (just call `reset`).
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::bitset::FixedBitset;
//!
//! // Create a bitset with capacity for 200 bits
//! let mut bs = FixedBitset::new(200);
//!
//! // Set some bits
//! bs.set(0);
//! bs.set(63);
//! bs.set(64);
//! bs.set(127);
//!
//! // Check if bits are set
//! assert!(bs.get(0));
//! assert!(bs.get(63));
//! assert!(!bs.get(1)); // not set
//!
//! // Count set bits
//! assert_eq!(bs.count_ones(), 4);
//!
//! // Clear a bit
//! bs.clear(63);
//! assert!(!bs.get(63));
//!
//! // Reset all bits (O(words) = O(200/64) = 4 operations)
//! bs.reset();
//! assert_eq!(bs.count_ones(), 0);
//! ```
//!
//! ## Use Case: Tracking visited nodes in graph traversal
//!
//! ```
//! use kideta_core::utils::bitset::FixedBitset;
//!
//! // For HNSW search, track visited nodes without per-query allocation
//! let mut visited = FixedBitset::new(10000); // capacity for 10k nodes
//!
//! // Check if already visited (helper function marks visited when processed)
//! fn try_process_node(node_id: usize, visited: &mut FixedBitset) -> bool {
//!     if visited.get(node_id) {
//!         return false; // already visited, skip
//!     }
//!     visited.set(node_id);
//!     true // new node, process it
//! }
//!
//! // First visit succeeds (not yet marked)
//! assert!(try_process_node(42, &mut visited));
//! // Second visit is rejected (already marked)
//! assert!(!try_process_node(42, &mut visited));
//! // But different nodes can be processed
//! assert!(try_process_node(100, &mut visited));
//!
//! // Reset for next query (no allocation)
//! visited.reset();
//! // Now we can visit 42 again after reset
//! assert!(try_process_node(42, &mut visited));
//! ```

/// A heap-allocated, packed bitset with a fixed capacity chosen at construction.
pub struct FixedBitset {
    /// Packed storage: bit `i` lives in `words[i / 64]` at position `i % 64`.
    words: Vec<u64>,
    /// Total number of bits (capacity).
    nbits: usize,
}

impl FixedBitset {
    /// Create a new bitset that can hold `nbits` bits, all cleared.
    #[inline]
    pub fn new(nbits: usize) -> Self {
        let nwords = nbits.div_ceil(64);
        Self {
            words: vec![0u64; nwords],
            nbits,
        }
    }

    /// Number of bits this bitset can hold.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.nbits
    }

    /// Set bit `i` to 1.
    ///
    /// # Panics
    /// Panics (debug) if `i >= self.capacity()`.
    #[inline]
    pub fn set(
        &mut self,
        i: usize,
    ) {
        debug_assert!(i < self.nbits, "bit index {i} out of range {}", self.nbits);
        self.words[i / 64] |= 1u64 << (i % 64);
    }

    /// Return `true` if bit `i` is set.
    ///
    /// # Panics
    /// Panics (debug) if `i >= self.capacity()`.
    #[inline]
    pub fn get(
        &self,
        i: usize,
    ) -> bool {
        debug_assert!(i < self.nbits, "bit index {i} out of range {}", self.nbits);
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Clear bit `i` (set to 0).
    #[inline]
    pub fn clear(
        &mut self,
        i: usize,
    ) {
        debug_assert!(i < self.nbits, "bit index {i} out of range {}", self.nbits);
        self.words[i / 64] &= !(1u64 << (i % 64));
    }

    /// Clear all bits in O(words) time.
    #[inline]
    pub fn reset(&mut self) {
        for w in &mut self.words {
            *w = 0;
        }
    }

    /// Count the number of set bits (popcount).
    #[inline]
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    /// Iterate over the indices of all set bits.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        self.words
            .iter()
            .enumerate()
            .flat_map(|(wi, &word)| BitIter(word).map(move |bit| wi * 64 + bit))
            .filter(|&i| i < self.nbits)
    }
}

/// Iterator over set-bit positions within a single u64 word.
struct BitIter(u64);

impl Iterator for BitIter {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            return None;
        }
        let pos = self.0.trailing_zeros() as usize;
        self.0 &= self.0 - 1; // clear lowest set bit
        Some(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_all_clear() {
        let bs = FixedBitset::new(200);
        assert_eq!(bs.count_ones(), 0);
        for i in 0..200 {
            assert!(!bs.get(i));
        }
    }

    #[test]
    fn set_get_clear() {
        let mut bs = FixedBitset::new(128);
        bs.set(0);
        bs.set(63);
        bs.set(64);
        bs.set(127);
        assert!(bs.get(0));
        assert!(bs.get(63));
        assert!(bs.get(64));
        assert!(bs.get(127));
        assert!(!bs.get(1));
        assert!(!bs.get(65));
        bs.clear(63);
        assert!(!bs.get(63));
        assert_eq!(bs.count_ones(), 3);
    }

    #[test]
    fn reset_clears_all() {
        let mut bs = FixedBitset::new(64);
        for i in 0..64 {
            bs.set(i);
        }
        assert_eq!(bs.count_ones(), 64);
        bs.reset();
        assert_eq!(bs.count_ones(), 0);
    }

    #[test]
    fn iter_set_correct() {
        let mut bs = FixedBitset::new(130);
        let bits = [0usize, 5, 63, 64, 129];
        for &b in &bits {
            bs.set(b);
        }
        let collected: Vec<usize> = bs.iter_set().collect();
        assert_eq!(collected, bits);
    }

    #[test]
    fn capacity_1() {
        let mut bs = FixedBitset::new(1);
        assert_eq!(bs.capacity(), 1);
        bs.set(0);
        assert!(bs.get(0));
    }
}
