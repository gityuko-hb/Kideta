//! Hamming distance implementation for binary vectors
//!
//! Hamming distance counts the number of bit positions at which two binary vectors differ.
//! This is commonly used for binary quantized vectors where each dimension is a single bit.
//!
//! # Examples
//!
//! ```
//! use kideta_core::distance::hamming_distance;
//!
//! let a = vec![0b1010_1010, 0b1111_0000];
//! let b = vec![0b1010_0000, 0b1111_1111];
//! let dist = hamming_distance(&a, &b);
//! assert_eq!(dist, 6); // 6 bits differ (2 in first byte, 4 in second)
//! ```

/// Compute Hamming distance between two binary vectors (scalar implementation)
///
/// # Arguments
/// * `a` - First binary vector (packed bytes)
/// * `b` - Second binary vector (same length as `a`)
///
/// # Returns
/// Number of differing bits
///
/// # Panics
/// Panics if `a` and `b` have different lengths
///
/// # Examples
///
/// ```
/// use kideta_core::distance::hamming_distance;
///
/// let a = vec![0b1111_1111, 0b0000_0000];
/// let b = vec![0b1111_1111, 0b1111_1111];
/// assert_eq!(hamming_distance(&a, &b), 8); // Only second byte differs
/// ```
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    assert_eq!(a.len(), b.len(), "Binary vectors must have the same length");

    a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Compute Hamming distance with validation (safe version)
///
/// Returns an error instead of panicking on invalid input
///
/// # Arguments
/// * `a` - First binary vector
/// * `b` - Second binary vector
///
/// # Returns
/// * `Ok(distance)` - The Hamming distance
/// * `Err(DistanceError)` - If vectors have different lengths
///
/// # Examples
///
/// ```
/// use kideta_core::distance::hamming_distance_safe;
///
/// let a = vec![0b1010_1010];
/// let b = vec![0b1111_0000];
/// match hamming_distance_safe(&a, &b) {
///     Ok(dist) => println!("Distance: {}", dist),
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
pub fn hamming_distance_safe(a: &[u8], b: &[u8]) -> Result<u32, DistanceError> {
    if a.len() != b.len() {
        return Err(DistanceError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    Ok(hamming_distance(a, b))
}

/// Error type for distance computation failures
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceError {
    /// Vectors have different dimensions
    DimensionMismatch { expected: usize, actual: usize },
    /// Vector contains invalid values (NaN, Inf)
    InvalidValue { index: usize, reason: String },
    /// Empty vector
    EmptyVector,
}

impl std::fmt::Display for DistanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DistanceError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, actual)
            }
            DistanceError::InvalidValue { index, reason } => {
                write!(f, "Invalid value at index {}: {}", index, reason)
            }
            DistanceError::EmptyVector => {
                write!(f, "Empty vector")
            }
        }
    }
}

impl std::error::Error for DistanceError {}

/// SIMD-accelerated Hamming distance using SSE4.2 POPCNT
///
/// Processes 16 bytes at a time using SSE4.2 intrinsics
///
/// # Safety
/// This function uses SSE4.2 intrinsics. The caller must ensure SSE4.2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
pub unsafe fn hamming_distance_sse42(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut sum = 0u32;

    // Process 16 bytes at a time
    let chunks = len / 16;
    for i in 0..chunks {
        let a_chunk = unsafe { _mm_loadu_si128(a.as_ptr().add(i * 16) as *const __m128i) };
        let b_chunk = unsafe { _mm_loadu_si128(b.as_ptr().add(i * 16) as *const __m128i) };
        let xor = _mm_xor_si128(a_chunk, b_chunk);

        // Extract 64-bit chunks and use native popcount
        let low = _mm_extract_epi64(xor, 0) as u64;
        let high = _mm_extract_epi64(xor, 1) as u64;
        sum += low.count_ones() + high.count_ones();
    }

    // Process remaining bytes
    for i in (chunks * 16)..len {
        sum += (a[i] ^ b[i]).count_ones();
    }

    sum
}

/// SIMD-accelerated Hamming distance using AVX2
///
/// Processes 32 bytes at a time using AVX2 intrinsics
///
/// # Safety
/// This function uses AVX2 intrinsics. The caller must ensure AVX2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn hamming_distance_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    assert_eq!(a.len(), b.len());

    let len = a.len();
    let mut sum = 0u32;

    // Process 32 bytes at a time
    let chunks = len / 32;
    for i in 0..chunks {
        let a_chunk = unsafe { _mm256_loadu_si256(a.as_ptr().add(i * 32) as *const __m256i) };
        let b_chunk = unsafe { _mm256_loadu_si256(b.as_ptr().add(i * 32) as *const __m256i) };
        let xor = { _mm256_xor_si256(a_chunk, b_chunk) };

        // Extract and count bits
        let low = { _mm256_extract_epi64(xor, 0) as u64 };
        let mid1 = { _mm256_extract_epi64(xor, 1) as u64 };
        let mid2 = { _mm256_extract_epi64(xor, 2) as u64 };
        let high = { _mm256_extract_epi64(xor, 3) as u64 };

        sum += low.count_ones() + mid1.count_ones() + mid2.count_ones() + high.count_ones();
    }

    // Process remaining bytes with scalar code
    for i in (chunks * 32)..len {
        sum += (a[i] ^ b[i]).count_ones();
    }

    sum
}

/// Runtime dispatcher for Hamming distance
///
/// Returns the best available implementation based on CPU features
///
/// # Examples
///
/// ```
/// use kideta_core::distance::get_hamming_fn;
///
/// let hamming_fn = get_hamming_fn();
/// let a = vec![0b1010_1010, 0b1111_0000];
/// let b = vec![0b1010_0000, 0b1111_1111];
/// let dist = hamming_fn(&a, &b);
/// ```
pub fn get_hamming_fn() -> fn(&[u8], &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return |a, b| unsafe { hamming_distance_avx2(a, b) };
        }
        if is_x86_feature_detected!("sse4.2") {
            return |a, b| unsafe { hamming_distance_sse42(a, b) };
        }
    }

    hamming_distance
}

/// Compute Hamming distance between two f32 vectors by first binarizing them
///
/// This is useful when you have f32 vectors but want to use Hamming distance
/// by treating positive values as 1 and negative/zero as 0.
///
/// # Arguments
/// * `a` - First f32 vector
/// * `b` - Second f32 vector (same length as `a`)
///
/// # Returns
/// Hamming distance between binarized versions of `a` and `b`
///
/// # Panics
/// Panics if `a` and `b` have different lengths
pub fn hamming_distance_f32(a: &[f32], b: &[f32]) -> u32 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let x_bit = if *x > 0.0 { 1u8 } else { 0u8 };
            let y_bit = if *y > 0.0 { 1u8 } else { 0u8 };
            (x_bit ^ y_bit) as u32
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_hamming_distance_basic() {
        let a = vec![0b1010_1010];
        let b = vec![0b1010_0000];
        assert_eq!(hamming_distance(&a, &b), 2); // Bits 4 and 5 differ
    }

    #[test]
    fn test_hamming_distance_same() {
        let a = vec![0b1111_1111, 0b0000_0000];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let a = vec![0b0000_0000];
        let b = vec![0b1111_1111];
        assert_eq!(hamming_distance(&a, &b), 8);
    }

    #[test]
    fn test_hamming_distance_multi_byte() {
        let a = vec![0b1010_1010, 0b1111_0000, 0b0000_1111];
        let b = vec![0b0101_0101, 0b0000_1111, 0b1111_0000];
        // Byte 0: 8 bits differ
        // Byte 1: 8 bits differ
        // Byte 2: 8 bits differ
        assert_eq!(hamming_distance(&a, &b), 24);
    }

    #[test]
    fn test_hamming_distance_empty() {
        let a: Vec<u8> = vec![];
        let b: Vec<u8> = vec![];
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_hamming_distance_different_lengths() {
        let a = vec![0b1010_1010];
        let b = vec![0b1010_1010, 0b1111_0000];
        let _ = hamming_distance(&a, &b);
    }

    #[test]
    fn test_hamming_distance_safe_success() {
        let a = vec![0b1010_1010];
        let b = vec![0b1010_0000];
        assert_eq!(hamming_distance_safe(&a, &b).unwrap(), 2);
    }

    #[test]
    fn test_hamming_distance_safe_error() {
        let a = vec![0b1010_1010];
        let b = vec![0b1010_1010, 0b1111_0000];
        let result = hamming_distance_safe(&a, &b);
        assert!(result.is_err());
        match result.unwrap_err() {
            DistanceError::DimensionMismatch { expected: 1, actual: 2 } => {}
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_hamming_distance_f32() {
        let a = vec![1.0, -1.0, 0.5, -0.5];
        let b = vec![0.5, -0.5, 1.0, -1.0];
        // Binarized: a = [1, 0, 1, 0], b = [1, 0, 1, 0]
        // Distance: 0
        assert_eq!(hamming_distance_f32(&a, &b), 0);

        let c = vec![-1.0, 1.0, -0.5, 0.5];
        // Binarized: c = [0, 1, 0, 1]
        // Distance from a: 4
        assert_eq!(hamming_distance_f32(&a, &c), 4);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_simd_matches_scalar() {
        // Generate random test data
        let test_cases: Vec<(Vec<u8>, Vec<u8>)> = (0..100)
            .map(|i| {
                let len = 100 + i; // Various lengths
                let a: Vec<u8> = (0..len).map(|j| ((i + j) % 256) as u8).collect();
                let b: Vec<u8> = (0..len).map(|j| ((i * 2 + j) % 256) as u8).collect();
                (a, b)
            })
            .collect();

        for (a, b) in test_cases {
            let scalar = hamming_distance(&a, &b);

            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let avx2 = unsafe { hamming_distance_avx2(&a, &b) };
                    assert_eq!(scalar, avx2, "AVX2 mismatch for len={}", a.len());
                }

                if is_x86_feature_detected!("sse4.2") {
                    let sse42 = unsafe { hamming_distance_sse42(&a, &b) };
                    assert_eq!(scalar, sse42, "SSE4.2 mismatch for len={}", a.len());
                }
            }
        }
    }

    #[test]
    fn test_dispatcher_returns_correct_fn() {
        let hamming_fn = get_hamming_fn();
        let a = vec![0b1010_1010, 0b1111_0000];
        let b = vec![0b1010_0000, 0b1111_1111];
        // Byte 0: 0b1010_1010 ^ 0b1010_0000 = 0b0000_1010 = 2 bits
        // Byte 1: 0b1111_0000 ^ 0b1111_1111 = 0b0000_1111 = 4 bits
        // Total: 6 bits
        let dist = hamming_fn(&a, &b);
        assert_eq!(dist, 6);
    }

    proptest! {
        #[test]
        fn test_hamming_symmetry(len in 0usize..1000) {
            // Generate vectors of same length
            let a: Vec<u8> = (0..len).map(|i| ((i * 7) % 256) as u8).collect();
            let b: Vec<u8> = (0..len).map(|i| ((i * 13 + 42) % 256) as u8).collect();

            let d1 = hamming_distance(&a, &b);
            let d2 = hamming_distance(&b, &a);
            prop_assert_eq!(d1, d2);
        }

        #[test]
        fn test_hamming_triangle_inequality(a in prop::collection::vec(0u8..255, 10),
                                           b in prop::collection::vec(0u8..255, 10),
                                           c in prop::collection::vec(0u8..255, 10)) {
            let ab = hamming_distance(&a, &b);
            let bc = hamming_distance(&b, &c);
            let ac = hamming_distance(&a, &c);
            prop_assert!(ac <= ab + bc);
        }

        #[test]
        fn test_hamming_same_vector(a in prop::collection::vec(0u8..255, 0..1000)) {
            let dist = hamming_distance(&a, &a);
            prop_assert_eq!(dist, 0);
        }
    }
}
