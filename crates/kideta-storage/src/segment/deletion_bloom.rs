//! Deletion Bloom Filter for fast "not deleted" checks in delta log.
//!
//! Specialized bloom filter for tracking deleted IDs in segments.
//! Provides O(1) probabilistic test to avoid disk reads for IDs
//! that definitely don't exist in the delta log.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

#[allow(dead_code)]
const DELETION_BLOOM_MAGIC: u32 = 0x_44_45_4C_42; // "DELB"
#[allow(dead_code)]
const DELETION_BLOOM_VERSION: u32 = 1;

/// Bloom filter specialized for deletion tracking.
#[derive(Clone)]
pub struct DeletionBloomFilter {
    bits: Vec<u64>,
    num_hashes: usize,
    num_bits: usize,
    count: usize,
    created_at: Instant,
}

impl DeletionBloomFilter {
    /// Create a new deletion bloom filter.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of deleted IDs.
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%).///
    pub fn new(
        expected_items: usize,
        false_positive_rate: f32,
    ) -> Self {
        let ln2 = std::f32::consts::LN_2;
        let num_bits =
            (-(expected_items as f32) * false_positive_rate.ln() / (ln2 * ln2)).ceil() as usize;
        let num_bits = num_bits.max(64);

        let num_hashes =
            ((num_bits as f32 / (expected_items as f32).max(1.0)) * ln2).ceil() as usize;
        let num_hashes = num_hashes.clamp(1, 10);

        let num_words = num_bits.div_ceil(64);

        Self {
            bits: vec![0u64; num_words],
            num_hashes,
            num_bits,
            count: 0,
            created_at: Instant::now(),
        }
    }

    /// Create from a deleted bitmap.
    pub fn from_deleted_bitmap(deleted_bitmap: &[u64]) -> Self {
        // Count ones in bitmap
        let count: usize = deleted_bitmap
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum();

        let mut filter = Self::new(count.max(1), 0.01);

        // Iterate through set bits
        for (word_idx, &word) in deleted_bitmap.iter().enumerate() {
            let base_id = (word_idx * 64) as u32;
            for bit_idx in 0..64 {
                if (word & (1u64 << bit_idx)) != 0 {
                    filter.insert(base_id + bit_idx as u32);
                }
            }
        }

        filter
    }

    /// Hash a value with a seed.
    fn hash_with_seed(
        value: u32,
        seed: usize,
    ) -> usize {
        let mut hasher = DefaultHasher::new();
        (value as u64).hash(&mut hasher);
        (seed as u64).hash(&mut hasher);
        hasher.finish() as usize
    }

    /// Insert a deleted ID.
    pub fn insert(
        &mut self,
        id: u32,
    ) {
        for seed in 0..self.num_hashes {
            let hash = Self::hash_with_seed(id, seed) % self.num_bits;
            let word_idx = hash / 64;
            let bit_idx = hash % 64;
            self.bits[word_idx] |= 1u64 << bit_idx;
        }
        self.count += 1;
    }

    /// Insert multiple deleted IDs.
    pub fn insert_batch(
        &mut self,
        ids: &[u32],
    ) {
        for id in ids {
            self.insert(*id);
        }
    }

    /// Check if an ID MIGHT be deleted.
    /// Returns `true` if the ID might be deleted (may be false positive).
    /// Returns `false` if the ID is definitely NOT deleted (useful fast path).
    pub fn may_be_deleted(
        &self,
        id: u32,
    ) -> bool {
        for seed in 0..self.num_hashes {
            let hash = Self::hash_with_seed(id, seed) % self.num_bits;
            let word_idx = hash / 64;
            let bit_idx = hash % 64;
            if (self.bits[word_idx] & (1u64 << bit_idx)) == 0 {
                return false;
            }
        }
        true
    }

    /// Fast "probably not deleted" check.
    /// Returns `true` if the ID is probably NOT deleted.
    /// Returns `false` if the ID might be deleted (need to check bitmap).
    #[inline]
    pub fn probably_not_deleted(
        &self,
        id: u32,
    ) -> bool {
        !self.may_be_deleted(id)
    }

    /// Clear the filter.
    pub fn clear(&mut self) {
        self.bits.fill(0);
        self.count = 0;
    }

    /// Rebuild from deleted bitmap.
    pub fn rebuild_from_bitmap(
        &mut self,
        deleted_bitmap: &[u64],
    ) {
        self.clear();

        for (word_idx, &word) in deleted_bitmap.iter().enumerate() {
            let base_id = (word_idx * 64) as u32;
            for bit_idx in 0..64 {
                if (word & (1u64 << bit_idx)) != 0 {
                    self.insert(base_id + bit_idx as u32);
                }
            }
        }
    }

    /// Get the number of deleted IDs inserted.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get age in seconds.
    pub fn age_secs(&self) -> f32 {
        self.created_at.elapsed().as_secs_f32()
    }

    /// Estimate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.bits.len() * std::mem::size_of::<u64>()
    }

    /// Merge with another bloom filter (OR operation).
    pub fn merge(
        &mut self,
        other: &DeletionBloomFilter,
    ) {
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= b;
        }
        self.count += other.count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deletion_bloom_basic() {
        let mut filter = DeletionBloomFilter::new(100, 0.01);

        filter.insert(42);
        filter.insert(100);
        filter.insert(200);

        assert!(filter.may_be_deleted(42));
        assert!(filter.may_be_deleted(100));
        assert!(filter.may_be_deleted(200));

        // Probably not deleted
        assert!(filter.probably_not_deleted(1));
        assert!(filter.probably_not_deleted(999));
    }

    #[test]
    fn test_deletion_bloom_from_bitmap() {
        let mut bitmap = vec![0u64; 4];
        bitmap[0] = 0b1010; // IDs 1 and 3
        bitmap[1] = 1 << 10; // ID 74

        let filter = DeletionBloomFilter::from_deleted_bitmap(&bitmap);

        assert!(filter.may_be_deleted(1));
        assert!(filter.may_be_deleted(3));
        assert!(filter.may_be_deleted(74));
        assert!(!filter.may_be_deleted(0));
        assert!(!filter.may_be_deleted(2));
    }

    #[test]
    fn test_deletion_bloom_memory() {
        let filter = DeletionBloomFilter::new(1000, 0.01);

        // Memory should be reasonable (a fewKB for 1000 items at 1% FP rate)
        let mem = filter.memory_bytes();
        assert!(mem > 0);
        assert!(mem < 20000); // Less than 20KB
    }

    #[test]
    fn test_deletion_bloom_merge() {
        let mut filter1 = DeletionBloomFilter::new(100, 0.01);
        filter1.insert(1);
        filter1.insert(2);

        let mut filter2 = DeletionBloomFilter::new(100, 0.01);
        filter2.insert(3);
        filter2.insert(4);

        filter1.merge(&filter2);

        assert!(filter1.may_be_deleted(1));
        assert!(filter1.may_be_deleted(3));
        assert_eq!(filter1.count(), 4);
    }
}
