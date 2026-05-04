//! Distance metrics — SIMD-accelerated with scalar fallback.
//!
//! This module provides high-performance distance functions for vector similarity
//! search. All functions automatically select the fastest available implementation
//! based on your CPU's SIMD capabilities.
//!
//! ## Runtime Dispatch
//!
//! The module detects CPU features at runtime and selects the best implementation:
//! 1. **AVX-512** — 16 floats per cycle (Intel Skylake-X and newer)
//! 2. **AVX2 + FMA** — 8 floats per cycle (Intel Haswell and newer, AMD Ryzen)
//! 3. **SSE4.1** — 4 floats per cycle (Intel Nehalem and newer, AMDBulldozer)
//! 4. **NEON** — 4 floats per cycle (ARM AArch64)
//! 5. **Scalar** — 1 float per cycle (fallback for all platforms)
//!
//! ## Quick Start
//!
//! ```
//! use kideta_core::distance::{
//!     l2_f32, l2_squared_f32, cosine_similarity_f32, cosine_f32,
//!     dot_f32, manhattan_f32, jaccard_u8, hamming_u8, hamming_packed_u8,
//! };
//!
//! // L2 distance between two vectors
//! let a = vec![0.0, 0.0];
//! let b = vec![3.0, 4.0];
//! let dist = l2_f32(&a, &b);
//! assert!((dist - 5.0).abs() < 1e-6);
//!
//! // L2 squared (faster, no sqrt)
//! let dist_sq = l2_squared_f32(&a, &b);
//! assert!((dist_sq - 25.0).abs() < 1e-6);
//!
//! // Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
//! let a = vec![1.0, 0.0];
//! let b = vec![1.0, 0.0];
//! assert!((cosine_similarity_f32(&a, &b) - 1.0).abs() < 1e-6);
//!
//! let a = vec![1.0, 0.0];
//! let b = vec![0.0, 1.0];
//! assert!(cosine_similarity_f32(&a, &b).abs() < 1e-6); // orthogonal = 0
//!
//! // Cosine distance (1 - similarity)
//! let dist = cosine_f32(&a, &b);
//! assert!((dist - 1.0).abs() < 1e-6);
//!
//! // Dot product
//! let a = vec![1.0, 2.0, 3.0];
//! let b = vec![4.0, 5.0, 6.0];
//! let dot = dot_f32(&a, &b);
//! assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
//!
//! // Manhattan distance
//! let a = vec![1.0, 2.0, 3.0];
//! let b = vec![4.0, 6.0, 3.0];
//! let manhattan = manhattan_f32(&a, &b);
//! assert!((manhattan - 7.0).abs() < 1e-6); // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
//! ```
//!
//! ## Binary Vector Distances
//!
//! For binary/quantized vectors:
//!
//! ```
//! use kideta_core::distance::{jaccard_u8, jaccard_distance_u8, hamming_u8, hamming_packed_u8};
//!
//! // Jaccard similarity for packed binary vectors (packed bits = 8 dims per byte)
//! let a = vec![0b1111_0000_u8];
//! let b = vec![0b0000_1111_u8];
//! let jaccard = jaccard_u8(&a, &b);
//! assert_eq!(jaccard, 0.0); // disjoint sets
//!
//! let a = vec![0b1111_1111_u8];
//! let b = vec![0b1111_1111_u8];
//! assert!((jaccard_u8(&a, &b) - 1.0).abs() < 1e-6); // identical
//!
//! // Jaccard distance = 1 - similarity
//! let dist = jaccard_distance_u8(&a, &b);
//! assert!(dist.abs() < 1e-6);
//!
//! // Hamming distance (count differing bytes)
//! let a = vec![1_u8, 2, 3];
//! let b = vec![1_u8, 0, 3];
//! assert_eq!(hamming_u8(&a, &b), 1); // only position 1 differs
//!
//! // Hamming distance with popcount (count differing BITS)
//! let a = vec![0b1111_1111_u8];
//! let b = vec![0b0000_0000_u8];
//! assert_eq!(hamming_packed_u8(&a, &b), 8); // 8 bits differ
//! ```
//!
//! ## Runtime Dispatch with `get_distance_fn`
//!
//! Get a function pointer to the fastest available implementation:
//!
//! ```
//! use kideta_core::distance::{get_distance_fn, Metric, l2_f32};
//!
//! // Get the fastest L2 distance function for this CPU
//! let l2_fn = get_distance_fn(Metric::L2);
//!
//! // Use it like any other function
//! let a = [3.0_f32, 0.0];
//! let b = [0.0_f32, 4.0];
//! let dist = l2_fn(&a, &b);
//! assert!((dist - 5.0).abs() < 1e-5);
//!
//! // Other metrics
//! let cosine_fn = get_distance_fn(Metric::Cosine);
//! let dot_fn = get_distance_fn(Metric::Dot);
//! let manhattan_fn = get_distance_fn(Metric::Manhattan);
//! ```
//!
//! ## Safe Variants with Validation
//!
//! Use the `*_safe` variants when you need to handle errors gracefully:
//!
//! ```
//! use kideta_core::distance::{
//!     l2_f32_safe, cosine_similarity_f32_safe, dot_f32_safe, manhattan_f32_safe
//! };
//!
//! let a = vec![1.0_f32, 2.0];
//! let b = vec![3.0_f32, 4.0];
//!
//! // These should all succeed
//! assert!(l2_f32_safe(&a, &b).is_ok());
//! assert!(cosine_similarity_f32_safe(&a, &b).is_ok());
//! assert!(dot_f32_safe(&a, &b).is_ok());
//! assert!(manhattan_f32_safe(&a, &b).is_ok());
//!
//! // Different lengths return errors
//! let short = vec![1.0_f32];
//! assert!(l2_f32_safe(&a, &short).is_err());
//!
//! // Empty vectors return errors
//! let empty: Vec<f32> = vec![];
//! assert!(l2_f32_safe(&empty, &empty).is_err());
//! ```
//!
//! ## CPU Feature Detection
//!
//! Check which SIMD features are available on your system:
//!
//! ```
//! use kideta_core::distance::detection::best_simd;
//!
//! let simd = best_simd();
//! println!("Best SIMD available: {simd}");
//! // Prints something like: "avx512f", "avx2+fma", "sse4.1", "neon", or "scalar"
//! ```

pub mod detection;
pub mod hamming;
pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "x86_64")]
pub mod sse41;

// ── Prefetch hint (1.17) ─────────────────────────────────────────────────────

/// Prefetch a cache line into L1 before distance computation.
///
/// A no-op on non-x86_64 targets.
#[inline(always)]
pub fn prefetch(ptr: *const f32) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` is safe to call with any pointer.
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

/// Prefetch a `u8` cache line into L1 (for PQ codes, binary vectors, etc.).
#[inline(always)]
pub fn prefetch_u8(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

/// Non-temporal prefetch — load into L1 without polluting L2/L3 caches.
///
/// Use for sequential scans (FlatIndex) where data is touched once
/// and then never accessed again.
#[inline(always)]
pub fn prefetch_nta(ptr: *const f32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::{_MM_HINT_NTA, _mm_prefetch};
        _mm_prefetch(ptr as *const i8, _MM_HINT_NTA);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

// ── Metric enum + function-pointer type (1.16) ───────────────────────────────

/// Supported distance metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    L2,
    Cosine,
    Dot,
    Manhattan,
    Hamming,
}

/// Signature shared by all `f32` distance functions.
pub type DistanceFn = fn(&[f32], &[f32]) -> f32;

/// Returns the fastest available implementation for the given metric.
///
/// The returned function pointer is valid for the lifetime of the process
/// (the CPU feature set doesn't change at runtime).
///
/// Note: For dot product, returns `dot_distance_f32` (negated raw dot)
/// so that lower scores = more similar, consistent with L2 and cosine.
pub fn get_distance_fn(metric: Metric) -> DistanceFn {
    match metric {
        Metric::L2 => l2_f32,
        Metric::Cosine => cosine_f32,
        Metric::Dot => dot_distance_f32,
        Metric::Manhattan => manhattan_f32,
        Metric::Hamming => {
            // Hamming operates on binary vectors, not f32
            // Return a wrapper that converts f32 to binary
            hamming_f32_wrapper
        },
    }
}

/// Wrapper to adapt Hamming distance for f32 vectors
/// Treats positive values as 1, non-positive as 0
fn hamming_f32_wrapper(
    a: &[f32],
    b: &[f32],
) -> f32 {
    hamming::hamming_distance_f32(a, b) as f32
}

// Re-export Hamming items
pub use hamming::{
    DistanceError, get_hamming_fn, hamming_distance, hamming_distance_avx2, hamming_distance_f32,
    hamming_distance_safe, hamming_distance_sse42,
};

// ── L2 (Euclidean) ───────────────────────────────────────────────────────────

#[inline]
pub fn l2_squared_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if detection::has_avx512f() {
            // SAFETY: CPU feature detected via `detection::has_avx512f`.
            return unsafe { avx512::l2_squared_f32(a, b) };
        }
        if detection::has_avx2() && detection::has_fma() {
            // SAFETY: CPU features detected via `detection::has_avx2` and `has_fma`.
            return unsafe { avx2::l2_squared_f32(a, b) };
        }
        if detection::has_sse4_1() {
            // SAFETY: CPU feature detected via `detection::has_sse4_1`.
            return unsafe { sse41::l2_squared_f32(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if detection::has_neon() {
            // SAFETY: CPU feature detected via `detection::has_neon`.
            return unsafe { neon::l2_squared_f32(a, b) };
        }
    }
    scalar::l2_squared_f32(a, b)
}

#[inline]
pub fn l2_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    l2_squared_f32(a, b).sqrt()
}

#[inline]
pub fn l2_squared_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    scalar::l2_squared_f64(a, b)
}

#[inline]
pub fn l2_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    l2_squared_f64(a, b).sqrt()
}

// ── Dot product ──────────────────────────────────────────────────────────────

#[inline]
pub fn dot_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if detection::has_avx512f() {
            // SAFETY: CPU feature detected via `detection::has_avx512f`.
            return unsafe { avx512::dot_f32(a, b) };
        }
        if detection::has_avx2() && detection::has_fma() {
            // SAFETY: CPU features detected via `detection::has_avx2` and `has_fma`.
            return unsafe { avx2::dot_f32(a, b) };
        }
        if detection::has_sse4_1() {
            // SAFETY: CPU feature detected via `detection::has_sse4_1`.
            return unsafe { sse41::dot_f32(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if detection::has_neon() {
            // SAFETY: CPU feature detected via `detection::has_neon`.
            return unsafe { neon::dot_f32(a, b) };
        }
    }
    scalar::dot_f32(a, b)
}

/// Dot-product **distance** (negated raw dot product).
///
/// Returns `-dot_f32(a, b)` so that lower values = more similar,
/// consistent with L2 and cosine distance conventions.
/// Use this as the distance function for dot-product similarity search.
#[inline]
pub fn dot_distance_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    -dot_f32(a, b)
}

#[inline]
pub fn dot_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    scalar::dot_f64(a, b)
}

// ── Cosine ────────────────────────────────────────────────────────────────────

#[inline]
pub fn cosine_similarity_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if detection::has_avx512f() {
            // SAFETY: CPU feature detected via `detection::has_avx512f`.
            return unsafe { avx512::cosine_similarity_f32(a, b) };
        }
        if detection::has_avx2() && detection::has_fma() {
            // SAFETY: CPU features detected via `detection::has_avx2` and `has_fma`.
            return unsafe { avx2::cosine_similarity_f32(a, b) };
        }
        if detection::has_sse4_1() {
            // SAFETY: CPU feature detected via `detection::has_sse4_1`.
            return unsafe { sse41::cosine_similarity_f32(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if detection::has_neon() {
            // SAFETY: CPU feature detected via `detection::has_neon`.
            return unsafe { neon::cosine_similarity_f32(a, b) };
        }
    }
    scalar::cosine_similarity_f32(a, b)
}

#[inline]
pub fn cosine_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    1.0 - cosine_similarity_f32(a, b)
}

#[inline]
pub fn cosine_similarity_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    scalar::cosine_similarity_f64(a, b)
}

#[inline]
pub fn cosine_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    scalar::cosine_f64(a, b)
}

// ── Manhattan (L1) (1.18) ─────────────────────────────────────────────────────

#[inline]
pub fn manhattan_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if detection::has_avx512f() {
            return unsafe { avx512::manhattan_f32(a, b) };
        }
        if detection::has_avx2() {
            return unsafe { avx2::manhattan_f32(a, b) };
        }
        if detection::has_sse4_1() {
            return unsafe { sse41::manhattan_f32(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if detection::has_neon() {
            return unsafe { neon::manhattan_f32(a, b) };
        }
    }
    scalar::manhattan_f32(a, b)
}

#[inline]
pub fn manhattan_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    scalar::manhattan_f64(a, b)
}

// ── Jaccard (1.19) ────────────────────────────────────────────────────────────

/// Jaccard similarity for packed-bit `u8` slices: |A∩B| / |A∪B| ∈ [0, 1].
#[inline]
pub fn jaccard_u8(
    a: &[u8],
    b: &[u8],
) -> f32 {
    scalar::jaccard_u8(a, b)
}

/// Jaccard distance = 1 − Jaccard similarity.
#[inline]
pub fn jaccard_distance_u8(
    a: &[u8],
    b: &[u8],
) -> f32 {
    scalar::jaccard_distance_u8(a, b)
}

// ── Hamming ───────────────────────────────────────────────────────────────────

#[inline]
pub fn hamming_u8(
    a: &[u8],
    b: &[u8],
) -> u64 {
    scalar::hamming_u8(a, b)
}

#[inline]
pub fn hamming_packed_u8(
    a: &[u8],
    b: &[u8],
) -> u64 {
    scalar::hamming_packed_u8(a, b)
}

#[inline]
pub fn hamming_f32(
    a: &[f32],
    b: &[f32],
) -> u64 {
    scalar::hamming_f32(a, b)
}

// ── Safe Function Alternatives ────────────────────────────────────────────────

/// Safe version of L2 distance that returns Result instead of panicking
///
/// Validates input before computing distance
pub fn l2_f32_safe(
    a: &[f32],
    b: &[f32],
) -> Result<f32, DistanceError> {
    if a.len() != b.len() {
        return Err(DistanceError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    if a.is_empty() {
        return Err(DistanceError::EmptyVector);
    }
    // Check for NaN
    for (i, &v) in a.iter().enumerate() {
        if v.is_nan() {
            return Err(DistanceError::InvalidValue {
                index: i,
                reason: "NaN value".to_string(),
            });
        }
    }
    for (i, &v) in b.iter().enumerate() {
        if v.is_nan() {
            return Err(DistanceError::InvalidValue {
                index: i,
                reason: "NaN value".to_string(),
            });
        }
    }
    Ok(l2_f32(a, b))
}

/// Safe version of cosine similarity that returns Result
pub fn cosine_similarity_f32_safe(
    a: &[f32],
    b: &[f32],
) -> Result<f32, DistanceError> {
    if a.len() != b.len() {
        return Err(DistanceError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    if a.is_empty() {
        return Err(DistanceError::EmptyVector);
    }
    Ok(cosine_similarity_f32(a, b))
}

/// Safe version of dot product that returns Result
pub fn dot_f32_safe(
    a: &[f32],
    b: &[f32],
) -> Result<f32, DistanceError> {
    if a.len() != b.len() {
        return Err(DistanceError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    if a.is_empty() {
        return Err(DistanceError::EmptyVector);
    }
    Ok(dot_f32(a, b))
}

/// Safe version of Manhattan distance that returns Result
pub fn manhattan_f32_safe(
    a: &[f32],
    b: &[f32],
) -> Result<f32, DistanceError> {
    if a.len() != b.len() {
        return Err(DistanceError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    if a.is_empty() {
        return Err(DistanceError::EmptyVector);
    }
    Ok(manhattan_f32(a, b))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_a() -> Vec<f32> {
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0,
        ]
    }
    fn vec_b() -> Vec<f32> {
        vec![
            8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
        ]
    }

    #[test]
    fn dispatch_l2_matches_scalar() {
        let a = vec_a();
        let b = vec_b();
        let expected = scalar::l2_squared_f32(&a, &b);
        assert!(
            (l2_squared_f32(&a, &b) - expected).abs() < 1e-3,
            "dispatch={} scalar={}",
            l2_squared_f32(&a, &b),
            expected
        );
    }

    #[test]
    fn dispatch_dot_matches_scalar() {
        let a = vec_a();
        let b = vec_b();
        let expected = scalar::dot_f32(&a, &b);
        assert!(
            (dot_f32(&a, &b) - expected).abs() < 1e-3,
            "dispatch={} scalar={}",
            dot_f32(&a, &b),
            expected
        );
    }

    #[test]
    fn dispatch_cosine_matches_scalar() {
        let a = vec_a();
        let b = vec_b();
        let expected = scalar::cosine_similarity_f32(&a, &b);
        assert!(
            (cosine_similarity_f32(&a, &b) - expected).abs() < 1e-5,
            "dispatch={} scalar={}",
            cosine_similarity_f32(&a, &b),
            expected
        );
    }

    #[test]
    fn get_distance_fn_l2() {
        let f = get_distance_fn(Metric::L2);
        let a = [3.0_f32, 0.0];
        let b = [0.0_f32, 4.0];
        assert!((f(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn get_distance_fn_dot() {
        let f = get_distance_fn(Metric::Dot);
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        assert!((f(&a, &b) - (-32.0)).abs() < 1e-5);
    }

    #[test]
    fn get_distance_fn_cosine_identical() {
        let f = get_distance_fn(Metric::Cosine);
        let a = [1.0_f32, 2.0, 3.0];
        assert!(f(&a, &a).abs() < 1e-5);
    }

    #[test]
    fn get_distance_fn_manhattan() {
        let f = get_distance_fn(Metric::Manhattan);
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 6.0, 3.0];
        // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        assert!((f(&a, &b) - 7.0).abs() < 1e-5);
    }

    #[test]
    fn best_simd_non_empty() {
        let s = detection::best_simd();
        assert!(!s.is_empty());
        println!("SIMD tier: {s}");
    }

    #[test]
    fn manhattan_known() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 6.0, 3.0];
        assert!((manhattan_f32(&a, &b) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_identical() {
        let a = [0xFF_u8, 0xAA];
        assert!((jaccard_u8(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_disjoint() {
        let a = [0xF0_u8];
        let b = [0x0F_u8];
        assert_eq!(jaccard_u8(&a, &b), 0.0);
        assert!((jaccard_distance_u8(&a, &b) - 1.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::collection::vec as sv;
    use proptest::prelude::*;

    fn approx_eq(
        actual: f32,
        expected: f32,
        rel_epsilon: f32,
    ) -> bool {
        if actual.is_nan() && expected.is_nan() {
            true
        } else if actual.is_infinite() && expected.is_infinite() {
            actual == expected
        } else {
            let diff = (actual - expected).abs();
            let scale = actual.abs().max(expected.abs()).max(1.0);
            diff <= rel_epsilon * scale
        }
    }

    proptest! {
        #[test]
        fn l2_simd_matches_scalar(a in sv(0.1f32..100.0, 128..=128), b in sv(0.1f32..100.0, 128..=128)) {
            let expected = scalar::l2_squared_f32(&a, &b);
            let actual = l2_squared_f32(&a, &b);
            prop_assert!(approx_eq(actual, expected, 1e-3), "l2: dispatch={} scalar={}", actual, expected);
        }

        #[test]
        fn dot_simd_matches_scalar(a in sv(0.1f32..10.0, 128..=128), b in sv(0.1f32..10.0, 128..=128)) {
            let expected = scalar::dot_f32(&a, &b);
            let actual = dot_f32(&a, &b);
            prop_assert!(approx_eq(actual, expected, 1e-3), "dot: dispatch={} scalar={}", actual, expected);
        }

        #[test]
        fn cosine_simd_matches_scalar(a in sv(0.1f32..10.0, 128..=128), b in sv(0.1f32..10.0, 128..=128)) {
            let expected = scalar::cosine_similarity_f32(&a, &b);
            let actual = cosine_similarity_f32(&a, &b);
            prop_assert!(approx_eq(actual, expected, 1e-3), "cosine: dispatch={} scalar={}", actual, expected);
        }

        #[test]
        fn manhattan_simd_matches_scalar(a in sv(0.1f32..100.0, 128..=128), b in sv(0.1f32..100.0, 128..=128)) {
            let expected = scalar::manhattan_f32(&a, &b);
            let actual = manhattan_f32(&a, &b);
            prop_assert!(approx_eq(actual, expected, 1e-3), "manhattan: dispatch={} scalar={}", actual, expected);
        }
    }
}
