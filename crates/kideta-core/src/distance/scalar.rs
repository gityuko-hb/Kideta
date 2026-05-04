//! Scalar (non-SIMD) implementations of distance functions.
//!
//! These are the fallback implementations used when SIMD instructions are not
//! available on the target platform. They are also used for correctness testing
//! against SIMD implementations.
//!
//! ## Usage
//!
//! ```
//! use kideta_core::distance::scalar::{
//!     l2_squared_f32, l2_f32, dot_f32, cosine_similarity_f32,
//!     manhattan_f32, jaccard_u8, hamming_u8, hamming_packed_u8
//! };
//!
//! // L2 distance
//! let a = [1.0_f32, 2.0, 3.0];
//! let b = [4.0_f32, 5.0, 6.0];
//! // diff = [-3, -3, -3], squared = [9, 9, 9], sum = 27
//! assert!((l2_squared_f32(&a, &b) - 27.0).abs() < 1e-6);
//! assert!((l2_f32(&a, &b) - 5.196152422706632).abs() < 1e-5);
//!
//! // Dot product
//! assert_eq!(dot_f32(&[1.0, 2.0], &[3.0, 4.0]), 11.0);
//!
//! // Cosine similarity
//! assert!((cosine_similarity_f32(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
//! assert!(cosine_similarity_f32(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
//!
//! // Manhattan distance
//! assert!((manhattan_f32(&[1.0, 2.0], &[4.0, 3.0]) - 4.0).abs() < 1e-6);
//!
//! // Jaccard similarity
//! assert!((jaccard_u8(&[0b1111_0000_u8], &[0b1111_0000_u8]) - 1.0).abs() < 1e-6);
//!
//! // Hamming distance
//! assert_eq!(hamming_u8(&[1_u8, 2], &[1_u8, 3]), 1);
//! assert_eq!(hamming_packed_u8(&[0xFF_u8], &[0x00_u8]), 8);
//! ```

// ── Dot product ──────────────────────────────────────────────────────────────

#[inline]
pub fn l2_squared_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
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
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

#[inline]
pub fn l2_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    l2_squared_f64(a, b).sqrt()
}

// ── Dot product ──────────────────────────────────────────────────────────────

/// `Σ aᵢ·bᵢ`
#[inline]
pub fn dot_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `Σ aᵢ·bᵢ`
#[inline]
pub fn dot_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Cosine distance ───────────────────────────────────────────────────────────
//
// cosine_similarity = dot(a, b) / (‖a‖ · ‖b‖)
// cosine_distance   = 1 − cosine_similarity   (∈ [0, 2])
//
// Returns 0.0 similarity (distance = 1.0) when either vector is a zero-vector.

/// Cosine similarity between two `f32` vectors (∈ [−1, 1]).
#[inline]
pub fn cosine_similarity_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    let (mut dot, mut norm_a, mut norm_b) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Cosine distance = `1 − cosine_similarity` (∈ [0, 2]).
#[inline]
pub fn cosine_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    1.0 - cosine_similarity_f32(a, b)
}

/// Cosine similarity between two `f64` vectors.
#[inline]
pub fn cosine_similarity_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    let (mut dot, mut norm_a, mut norm_b) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Cosine distance = `1 − cosine_similarity` (∈ [0, 2]).
#[inline]
pub fn cosine_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    1.0 - cosine_similarity_f64(a, b)
}

// ── Manhattan (L1) distance ────────────────────────────────────────────
//
// L1(a, b) = Σ |aᵢ − bᵢ|

/// Manhattan (L1) distance between two `f32` vectors.
#[inline]
pub fn manhattan_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

/// Manhattan (L1) distance between two `f64` vectors.
#[inline]
pub fn manhattan_f64(
    a: &[f64],
    b: &[f64],
) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

// ── Jaccard similarity for binary vectors (1.19) ──────────────────────────────
//
// Jaccard(A, B) = |A ∩ B| / |A ∪ B|
//               = popcount(A & B) / popcount(A | B)
//
// Returns 0.0 when both vectors are all-zero (empty set).

/// Jaccard similarity for packed-bit `u8` slices (popcount via hardware).
///
/// Each byte holds 8 binary dimensions.
/// Returns a value in [0.0, 1.0]; 1.0 means identical, 0.0 means disjoint.
#[inline]
pub fn jaccard_u8(
    a: &[u8],
    b: &[u8],
) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    let (mut inter, mut union) = (0u64, 0u64);
    for (x, y) in a.iter().zip(b.iter()) {
        inter += (x & y).count_ones() as u64;
        union += (x | y).count_ones() as u64;
    }
    if union == 0 {
        0.0
    } else {
        inter as f32 / union as f32
    }
}

/// Jaccard distance = 1 − Jaccard similarity.
#[inline]
pub fn jaccard_distance_u8(
    a: &[u8],
    b: &[u8],
) -> f32 {
    1.0 - jaccard_u8(a, b)
}

// ── Hamming distance ──────────────────────────────────────────────────────────
//
// Counts positions where the two slices differ.
// For binary/quantised vectors use the `u8` variant; for dense float vectors
// use the `f32`/`f64` variants (element != check via bit pattern).

/// Number of positions where `a[i] != b[i]` (integer / binary vectors).
#[inline]
pub fn hamming_u8(
    a: &[u8],
    b: &[u8],
) -> u64 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x != y)
        .count() as u64
}

/// Popcount-based Hamming distance for packed-bit `u8` slices.
///
/// Each byte holds 8 bits; returns the total number of differing **bits**.
#[inline]
pub fn hamming_packed_u8(
    a: &[u8],
    b: &[u8],
) -> u64 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as u64)
        .sum()
}

/// Number of positions where `a[i] != b[i]` for `f32` vectors.
#[inline]
pub fn hamming_f32(
    a: &[f32],
    b: &[f32],
) -> u64 {
    debug_assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared_f32_zero() {
        let a = [1.0_f32, 2.0, 3.0];
        assert_eq!(l2_squared_f32(&a, &a), 0.0);
    }

    #[test]
    fn test_l2_squared_f32_known() {
        // (3, 0) vs (0, 4) => 9 + 16 = 25
        let a = [3.0_f32, 0.0];
        let b = [0.0_f32, 4.0];
        assert_eq!(l2_squared_f32(&a, &b), 25.0);
    }

    #[test]
    fn test_l2_f32_known() {
        let a = [3.0_f32, 0.0];
        let b = [0.0_f32, 4.0];
        assert!((l2_f32(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_squared_f64_known() {
        let a = [3.0_f64, 0.0];
        let b = [0.0_f64, 4.0];
        assert_eq!(l2_squared_f64(&a, &b), 25.0);
    }

    #[test]
    fn test_l2_f64_known() {
        let a = [3.0_f64, 0.0];
        let b = [0.0_f64, 4.0];
        assert!((l2_f64(&a, &b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_l2_symmetry() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [4.0_f32, 3.0, 2.0, 1.0];
        assert_eq!(l2_squared_f32(&a, &b), l2_squared_f32(&b, &a));
    }

    #[test]
    fn test_l2_single_element() {
        let a = [5.0_f32];
        let b = [2.0_f32];
        assert_eq!(l2_squared_f32(&a, &b), 9.0);
        assert!((l2_f32(&a, &b) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_empty() {
        let a: &[f32] = &[];
        let b: &[f32] = &[];
        assert_eq!(l2_squared_f32(a, b), 0.0);
        assert_eq!(l2_f32(a, b), 0.0);
    }

    // ── dot product ───────────────────────────────────────────────────────────

    #[test]
    fn test_dot_f32_orthogonal() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert_eq!(dot_f32(&a, &b), 0.0);
    }

    #[test]
    fn test_dot_f32_known() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 5.0, 6.0];
        assert_eq!(dot_f32(&a, &b), 32.0); // 4+10+18
    }

    #[test]
    fn test_dot_f64_known() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        assert_eq!(dot_f64(&a, &b), 32.0);
    }

    // ── cosine ────────────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_identical() {
        let a = [1.0_f32, 2.0, 3.0];
        assert!((cosine_f32(&a, &a)).abs() < 1e-6); // distance = 0
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!((cosine_f32(&a, &b) - 1.0).abs() < 1e-6); // distance = 1
    }

    #[test]
    fn test_cosine_opposite() {
        let a = [1.0_f32, 0.0];
        let b = [-1.0_f32, 0.0];
        assert!((cosine_f32(&a, &b) - 2.0).abs() < 1e-6); // distance = 2
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0_f32, 0.0];
        let b = [1.0_f32, 0.0];
        // zero-vector → similarity defined as 0, distance = 1
        assert_eq!(cosine_similarity_f32(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_f64_known() {
        let a = [3.0_f64, 0.0];
        let b = [0.0_f64, 4.0];
        assert!((cosine_similarity_f64(&a, &b)).abs() < 1e-12);
    }

    // ── hamming ───────────────────────────────────────────────────────────────

    #[test]
    fn test_hamming_u8_identical() {
        let a = [1_u8, 2, 3];
        assert_eq!(hamming_u8(&a, &a), 0);
    }

    #[test]
    fn test_hamming_u8_known() {
        let a = [0_u8, 1, 2, 3];
        let b = [0_u8, 0, 2, 0];
        assert_eq!(hamming_u8(&a, &b), 2); // positions 1 and 3 differ
    }

    #[test]
    fn test_hamming_packed_popcount() {
        // 0b0000_0001 ^ 0b0000_0011 = 0b0000_0010 → 1 bit
        let a = [0b0000_0001_u8];
        let b = [0b0000_0011_u8];
        assert_eq!(hamming_packed_u8(&a, &b), 1);
    }

    #[test]
    fn test_hamming_packed_u8_known() {
        // 0xFF ^ 0x00 = 0xFF → 8 bits set
        let a = [0xFF_u8, 0x00];
        let b = [0x00_u8, 0xFF];
        assert_eq!(hamming_packed_u8(&a, &b), 16);
    }

    #[test]
    fn test_hamming_f32_known() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [1.0_f32, 0.0, 3.0];
        assert_eq!(hamming_f32(&a, &b), 1);
    }

    // ── manhattan ─────────────────────────────────────────────────────────────

    #[test]
    fn test_manhattan_f32_known() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 6.0, 3.0];
        // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        assert!((manhattan_f32(&a, &b) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_f32_identical() {
        let a = [1.0_f32, 2.0, 3.0];
        assert_eq!(manhattan_f32(&a, &a), 0.0);
    }

    #[test]
    fn test_manhattan_f64_known() {
        let a = [0.0_f64, 0.0];
        let b = [3.0_f64, 4.0];
        assert!((manhattan_f64(&a, &b) - 7.0).abs() < 1e-12);
    }

    // ── jaccard ───────────────────────────────────────────────────────────────

    #[test]
    fn test_jaccard_identical() {
        let a = [0b1111_0000_u8, 0b0000_1111];
        assert!((jaccard_u8(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = [0b1111_0000_u8];
        let b = [0b0000_1111_u8];
        assert_eq!(jaccard_u8(&a, &b), 0.0);
    }

    #[test]
    fn test_jaccard_half_overlap() {
        // A = 0b1100, B = 0b0011 → inter=0 bits, union=4 bits → 0.0
        // Use A = 0b1100, B = 0b1010 → inter=0b1000=1 bit, union=0b1110=3 bits → 1/3
        let a = [0b1100_u8];
        let b = [0b1010_u8];
        let j = jaccard_u8(&a, &b);
        assert!((j - 1.0 / 3.0).abs() < 1e-6, "jaccard={j}");
    }

    #[test]
    fn test_jaccard_empty() {
        let a: &[u8] = &[0, 0];
        let b: &[u8] = &[0, 0];
        assert_eq!(jaccard_u8(a, b), 0.0); // both zero-vectors
    }

    #[test]
    fn test_jaccard_distance_disjoint() {
        let a = [0xFF_u8];
        let b = [0x00_u8];
        assert!((jaccard_distance_u8(&a, &b) - 1.0).abs() < 1e-6);
    }
}
