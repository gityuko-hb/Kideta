//! `BloomFilter` — probabilistic membership test.
//!
//! A Bloom filter provides fast membership testing with a configurable false-positive
//! rate. It never returns false negatives (if it says "not present", it's definitely
//! not present), but may return false positives (it says "possibly present" when it's
//! actually not).
//!
//! ## Design
//! * Configurable false-positive rate `fp` and expected item count `n`.
//! * Backing storage: `FixedBitset`.
//! * Hash functions: double-hashing trick — `h_i(x) = h1(x) + i·h2(x)`
//!   using two independent `xxhash64` calls with different seeds.
//!
//! ## Optimal parameters
//! ```text
//! m (bits)   = ceil(-n · ln(fp) / ln(2)²)
//! k (hashes) = round((m / n) · ln(2))
//! ```
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::bloom::BloomFilter;
//!
//! // Create a filter for 1000 items with 1% false-positive rate
//! let mut filter = BloomFilter::new(1000, 0.01);
//!
//! // Insert items
//! filter.insert(b"hello");
//! filter.insert(b"world");
//! filter.insert(b"test");
//!
//! // Check membership - no false negatives
//! assert!(filter.contains(b"hello"));
//! assert!(filter.contains(b"world"));
//! assert!(filter.contains(b"test"));
//!
//! // Items not inserted will return false (with high probability)
//! assert!(!filter.contains(b"not present"));
//!
//! // Fill ratio tells you how "full" the filter is
//! let ratio = filter.fill_ratio();
//! println!("Filter fill ratio: {:.2}", ratio);
//!
//! // Clear and reuse
//! filter.clear();
//! assert!(!filter.contains(b"hello")); // after clear, probably gone
//! ```
//!
//! ## Use Case: Fast Pre-Filter Before Expensive Lookup
//!
//! Use a Bloom filter as a first-pass filter before doing expensive lookups
//! (like disk reads or network calls):
//!
//! ```
//! use kideta_core::utils::bloom::BloomFilter;
//!
//! // Setup: Pre-populate filter with all known IDs
//! let mut cache_filter = BloomFilter::new(10_000, 0.001);
//! for id in 1000..2000 {
//!     cache_filter.insert(id.to_string().as_bytes());
//! }
//!
//! // Query: Check before expensive lookup
//! fn query_with_filter(id: u32, filter: &BloomFilter) -> &'static str {
//!     if filter.contains(id.to_string().as_bytes()) {
//!         // Possibly in cache - do expensive lookup
//!         // (in real code, this would be a disk/network call)
//!         "found_in_expensive_lookup"
//!     } else {
//!         // Definitely not in cache - skip expensive lookup
//!         "not_in_cache"
//!     }
//! }
//!
//! assert!(query_with_filter(1500, &cache_filter).contains("found"));
//! assert!(query_with_filter(999, &cache_filter).contains("not"));
//! ```

use super::bitset::FixedBitset;
use super::hash::xxhash64;

/// Bloom filter with configurable false-positive rate.
pub struct BloomFilter {
    bits: FixedBitset,
    k: u32,   // number of hash functions
    m: usize, // number of bits
}

impl BloomFilter {
    /// Create a filter for `n` expected items and a false-positive rate `fp`
    /// (e.g. `0.01` for 1 %).
    ///
    /// # Panics
    /// Panics if `n == 0` or `fp` is not in `(0, 1)`.
    pub fn new(
        n: usize,
        fp: f64,
    ) -> Self {
        assert!(n > 0, "n must be > 0");
        assert!(fp > 0.0 && fp < 1.0, "fp must be in (0, 1)");

        let ln2 = std::f64::consts::LN_2;
        let m = (-(n as f64) * fp.ln() / (ln2 * ln2)).ceil() as usize;
        let m = m.max(64); // floor at 64 bits
        let k = ((m as f64 / n as f64) * ln2).round() as u32;
        let k = k.clamp(1, 32);

        Self {
            bits: FixedBitset::new(m),
            k,
            m,
        }
    }

    /// Number of bits in the filter.
    #[inline]
    pub fn num_bits(&self) -> usize {
        self.m
    }

    /// Number of hash functions.
    #[inline]
    pub fn num_hashes(&self) -> u32 {
        self.k
    }

    /// Insert `item` into the filter.
    pub fn insert(
        &mut self,
        item: &[u8],
    ) {
        let (h1, h2) = self.hashes(item);
        for i in 0..self.k as u64 {
            let idx = (h1.wrapping_add(i.wrapping_mul(h2))) as usize % self.m;
            self.bits.set(idx);
        }
    }

    /// Return `true` if `item` **might** be in the set (false positives
    /// possible).  Returns `false` only when `item` is definitely absent.
    pub fn contains(
        &self,
        item: &[u8],
    ) -> bool {
        let (h1, h2) = self.hashes(item);
        for i in 0..self.k as u64 {
            let idx = (h1.wrapping_add(i.wrapping_mul(h2))) as usize % self.m;
            if !self.bits.get(idx) {
                return false;
            }
        }
        true
    }

    /// Reset the filter — all bits cleared.
    pub fn clear(&mut self) {
        self.bits.reset();
    }

    /// Estimated fill ratio (fraction of bits set).
    pub fn fill_ratio(&self) -> f64 {
        self.bits.count_ones() as f64 / self.m as f64
    }

    // ── Double-hashing (3.16.1) ───────────────────────────────────────────────

    /// Produce two independent 64-bit hashes from `item` using different seeds.
    #[inline]
    fn hashes(
        &self,
        item: &[u8],
    ) -> (u64, u64) {
        let h1 = xxhash64(item, 0);
        let h2 = xxhash64(item, 0x9747b28c); // second independent seed
        // Ensure h2 is odd so the probe sequence covers all m positions
        // when gcd(h2, m) = 1 (guaranteed when m is a power of 2; good enough otherwise).
        (h1, h2 | 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_false_negatives() {
        let mut bf = BloomFilter::new(1000, 0.01);
        let keys: Vec<String> = (0..1000).map(|i| format!("key-{i}")).collect();
        for k in &keys {
            bf.insert(k.as_bytes());
        }
        for k in &keys {
            assert!(bf.contains(k.as_bytes()), "false negative for {k}");
        }
    }

    #[test]
    fn false_positive_rate_rough() {
        // Insert 1000 items and test 10 000 items not in the set.
        // FP rate should be well under 5 % (configured at 1 %).
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0u64..1000 {
            bf.insert(&i.to_le_bytes());
        }
        let mut fp = 0usize;
        for i in 1000u64..11000 {
            if bf.contains(&i.to_le_bytes()) {
                fp += 1;
            }
        }
        let rate = fp as f64 / 10_000.0;
        assert!(rate < 0.05, "FP rate too high: {rate:.3}");
    }

    #[test]
    fn clear_resets() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert(b"hello");
        assert!(bf.contains(b"hello"));
        bf.clear();
        // After clear most bits should be 0; "hello" very likely gone.
        // (Not guaranteed due to hash collisions but statistically certain.)
        assert_eq!(bf.fill_ratio(), 0.0);
    }

    #[test]
    fn params_reasonable() {
        let bf = BloomFilter::new(10_000, 0.001);
        // m > 0, k in [1, 32]
        assert!(bf.num_bits() > 0);
        assert!(bf.num_hashes() >= 1);
        assert!(bf.num_hashes() <= 32);
    }

    #[test]
    fn definitely_absent_returns_false() {
        let bf = BloomFilter::new(100, 0.01);
        // Nothing inserted, every query must return false.
        assert!(!bf.contains(b"anything"));
        assert!(!bf.contains(b"not-here"));
    }
}
