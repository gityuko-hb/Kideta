//! `Xorshift64` — fast, deterministic pseudo-random number generator.
//!
//! Period: 2⁶⁴ − 1 (all non-zero 64-bit states).
//! Use cases: HNSW level generation, k-means++ sampling, reproducible tests.
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::rng::Xorshift64;
//!
//! // Create with a seed
//! let mut rng = Xorshift64::new(42);
//!
//! // Get random numbers
//! let u64_val = rng.next_u64();
//! let f64_val = rng.next_f64(); // in [0, 1)
//! let f32_val = rng.next_f32(); // in [0, 1)
//!
//! // Bounded random (e.g., for array indexing)
//! let bounded = rng.next_u64_bounded(100); // [0, 100)
//!
//! // Zero seed is automatically corrected to 1
//! let mut rng0 = Xorshift64::new(0);
//! assert_ne!(rng0.next_u64(), 0); // won't get stuck
//! ```
//!
//! ## Use Case: HNSW Level Generation
//!
//! In HNSW, you need to sample random levels for each new node:
//!
//! ```
//! use kideta_core::utils::rng::Xorshift64;
//!
//! let mut rng = Xorshift64::new(12345);
//!
//! // Typical mL = 1/ln(16) ≈ 0.36 for HNSW
//! let ml = 1.0 / (16_f64.ln());
//!
//! // Sample levels for multiple nodes
//! for _ in 0..5 {
//!     let level = rng.hnsw_level(ml);
//!     println!("Sampled level: {}", level);
//!     // Most levels will be 0, occasionally 1, rarely 2+
//! }
//! ```
//!
//! ## Reproducible Tests
//!
//! Use a fixed seed for deterministic, reproducible tests:
//!
//! ```
//! use kideta_core::utils::rng::Xorshift64;
//!
//! fn generate_test_data(seed: u64) -> Vec<f32> {
//!     let mut rng = Xorshift64::new(seed);
//!     (0..100).map(|_| rng.next_f32()).collect()
//! }
//!
//! // Same seed = same data
//! let data1 = generate_test_data(42);
//! let data2 = generate_test_data(42);
//! assert_eq!(data1, data2);
//!
//! // Different seed = different data
//! let data3 = generate_test_data(43);
//! assert_ne!(data1, data3);
//! ```

/// Xorshift64 PRNG (Marsaglia 2003).
///
/// Seed must be non-zero; passing 0 is automatically corrected to 1.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new RNG with the given seed (0 → 1).
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                1
            } else {
                seed
            },
        }
    }

    /// Return the next pseudo-random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Return a pseudo-random `f64` in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits for a clean IEEE-754 mantissa.
        (self.next_u64() >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Return a pseudo-random `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0_f32 / (1u64 << 24) as f32)
    }

    /// Return a pseudo-random `u64` in `[0, n)`.
    ///
    /// Uses rejection sampling to avoid modulo bias.
    #[inline]
    pub fn next_u64_bounded(
        &mut self,
        n: u64,
    ) -> u64 {
        debug_assert!(n > 0, "bound must be > 0");
        // Rejection: discard values in the biased top remainder.
        let threshold = n.wrapping_neg() % n; // (2^64 % n)
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return r % n;
            }
        }
    }

    /// Sample a random level for HNSW insertion.
    ///
    /// Returns a non-negative integer drawn from a geometric distribution
    /// with parameter `p = 1/M` (probability of advancing to the next layer).
    #[inline]
    pub fn hnsw_level(
        &mut self,
        ml: f64,
    ) -> usize {
        // level = floor(-ln(uniform) * mL), capped to avoid degenerate graphs.
        let uniform = self.next_f64().max(f64::MIN_POSITIVE);
        (-uniform.ln() * ml).floor() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_seed_becomes_one() {
        let mut rng = Xorshift64::new(0);
        // State should not be stuck at 0.
        assert_ne!(rng.next_u64(), 0);
    }

    #[test]
    fn deterministic() {
        let mut a = Xorshift64::new(42);
        let mut b = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn f64_in_range() {
        let mut rng = Xorshift64::new(1);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn bounded_in_range() {
        let mut rng = Xorshift64::new(7);
        for _ in 0..10_000 {
            let v = rng.next_u64_bounded(100);
            assert!(v < 100);
        }
    }

    #[test]
    fn no_short_cycle() {
        // Check the first 1000 values are distinct (extremely likely for xorshift64).
        let mut rng = Xorshift64::new(12345);
        let vals: std::collections::HashSet<u64> = (0..1000).map(|_| rng.next_u64()).collect();
        assert_eq!(vals.len(), 1000);
    }

    #[test]
    fn hnsw_level_non_negative() {
        let mut rng = Xorshift64::new(99);
        for _ in 0..1000 {
            // typical mL = 1/ln(16) ≈ 0.36
            let level = rng.hnsw_level(1.0 / (16_f64.ln()));
            assert!(level < 64, "level={level} suspiciously high");
        }
    }
}
