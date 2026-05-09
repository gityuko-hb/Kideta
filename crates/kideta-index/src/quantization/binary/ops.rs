//! Binary quantization operations — encode, Hamming distance with SIMD popcount.
//!
//! # Encoding
//!
//! ```ignore
//! bit[i / 8] |= (vector[i] >= 0 ? 1 : 0) << (i % 8)
//! ```
//!
//! # Hamming Distance
//!
//! XOR two binary vectors → popcount of result = Hamming distance.
//!
//! # SIMD Paths
//!
//! - AVX-512: 512-bit popcnt (16 × 64-bit at once)
//! - AVX2: 256-bit popcnt (4 × 64-bit at once)
//! - SSE4.2: `_mm_popcnt_u64` scalar
//! - Fallback: software popcount loop

use crate::quantization::config::BinaryConfig;

/// Binary quantization operations.
pub struct BinaryOps;

impl BinaryOps {
    /// Encode f32 vector → packed bits.
    ///
    /// `bit = 1` if `vector[i] >= 0`, else `bit = 0`.
    #[inline]
    pub fn encode(vector: &[f32]) -> Vec<u8> {
        let dimension = vector.len();
        let num_bytes = dimension.div_ceil(8);
        let mut bits = vec![0u8; num_bytes];
        Self::encode_to_slice(vector, &mut bits);
        bits
    }

    /// Encode f32 vector directly into a pre-allocated byte slice.
    #[inline]
    pub fn encode_to_slice(
        vector: &[f32],
        bits: &mut [u8],
    ) {
        let dimension = vector.len();
        let num_bytes = dimension.div_ceil(8);
        assert!(bits.len() >= num_bytes);

        for i in 0..dimension {
            let bit = if vector[i] >= 0.0 {
                1u8
            } else {
                0u8
            };
            bits[i / 8] |= bit << (i % 8);
        }
    }

    /// Decode packed bits → f32 (±1 per dimension).
    ///
    /// Returns `±1.0` per dimension (for rescore, not exact reconstruction).
    #[inline]
    pub fn decode(
        bits: &[u8],
        dimension: usize,
    ) -> Vec<f32> {
        let mut vector = vec![0.0_f32; dimension];
        Self::decode_to_slice(bits, &mut vector);
        vector
    }

    /// Decode directly into a pre-allocated f32 slice.
    #[inline]
    pub fn decode_to_slice(
        bits: &[u8],
        vector: &mut [f32],
    ) {
        let dimension = vector.len();
        for i in 0..dimension {
            let bit = (bits[i / 8] >> (i % 8)) & 1u8;
            vector[i] = if bit == 1 {
                1.0
            } else {
                -1.0
            };
        }
    }

    /// Compute Hamming distance between two binary vectors (stored as bytes).
    ///
    /// Uses SIMD popcount when available.
    #[inline]
    pub fn hamming(
        a: &[u8],
        b: &[u8],
    ) -> u64 {
        assert_eq!(a.len(), b.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("popcnt") {
                return unsafe { Self::hamming_popcnt_impl(a, b) };
            }
        }

        Self::hamming_scalar(a, b)
    }

    /// Scalar fallback Hamming distance.
    #[inline]
    pub fn hamming_scalar(
        a: &[u8],
        b: &[u8],
    ) -> u64 {
        let mut dist = 0u64;
        for i in 0..a.len() {
            dist += (a[i] ^ b[i]).count_ones() as u64;
        }
        dist
    }

    /// Hamming distance using SIMD popcount (AVX2).
    ///
    /// # Safety
    ///
    /// Must be called only when POPCNT and AVX2 features are detected.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "popcnt")]
    #[target_feature(enable = "avx2")]
    unsafe fn hamming_popcnt_impl(
        a: &[u8],
        b: &[u8],
    ) -> u64 {
        use std::arch::x86_64::*;

        let mut total: u64 = 0;
        let len = a.len();

        let mut i = 0usize;

        while i + 32 <= len {
            let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i) as *const _) };
            let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i) as *const _) };
            let xored = _mm256_xor_si256(va, vb);
            let cnt = unsafe { _mm256_popcnt_epi64(xored) };
            let reduced = (_mm256_extract_epi64(cnt, 0) as u64)
                + (_mm256_extract_epi64(cnt, 1) as u64)
                + (_mm256_extract_epi64(cnt, 2) as u64)
                + (_mm256_extract_epi64(cnt, 3) as u64);
            total += reduced;
            i += 32;
        }

        while i < len {
            total += (a[i] ^ b[i]).count_ones() as u64;
            i += 1;
        }

        total
    }

    /// Compute Hamming distance between a query (f32) and a stored binary code.
    ///
    /// First encodes the query to binary, then computes Hamming distance.
    #[inline]
    pub fn hamming_distance(
        config: &BinaryConfig,
        query: &[f32],
        code: &[u8],
    ) -> u64 {
        assert_eq!(query.len(), config.dimension);
        let query_bits = Self::encode(query);
        Self::hamming(&query_bits, code)
    }

    /// Estimate cosine similarity from Hamming distance for normalized vectors.
    ///
    /// For normalized vectors, `cosine ≈ 1 - 2 * hamming / dim`.
    /// This is faster than decoding and computing dot/norm.
    #[inline]
    pub fn hamming_to_cosine_estimate(
        hamming: u64,
        dimension: usize,
    ) -> f32 {
        if dimension == 0 {
            return 1.0;
        }
        1.0 - 2.0 * (hamming as f32) / (dimension as f32)
    }

    /// Compute approximate cosine similarity using Hamming distance.
    #[inline]
    pub fn approx_cosine(
        query: &[f32],
        code: &[u8],
    ) -> f32 {
        let dimension = query.len();
        let query_bits = Self::encode(query);
        let hamming = Self::hamming(&query_bits, code);
        Self::hamming_to_cosine_estimate(hamming, dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let v1 = vec![1.0_f32, -2.0, 3.0, -4.0, 0.5, -0.5];
        let bits = BinaryOps::encode(&v1);

        let decoded = BinaryOps::decode(&bits, 6);
        assert_eq!(decoded.len(), 6);
        assert_eq!(decoded[0], 1.0);
        assert_eq!(decoded[1], -1.0);
        assert_eq!(decoded[2], 1.0);
        assert_eq!(decoded[3], -1.0);
        assert_eq!(decoded[4], 1.0);
        assert_eq!(decoded[5], -1.0);
    }

    #[test]
    fn test_encode_to_slice() {
        let v = vec![1.0_f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let mut bits = [0u8; 1];
        BinaryOps::encode_to_slice(&v, &mut bits);
        assert_eq!(bits[0], 0b01010101);
    }

    #[test]
    fn test_hamming_identical() {
        let a = vec![1.0_f32, -1.0, 1.0, -1.0];
        let b = vec![1.0_f32, -1.0, 1.0, -1.0];
        let bits_a = BinaryOps::encode(&a);
        let bits_b = BinaryOps::encode(&b);
        assert_eq!(BinaryOps::hamming(&bits_a, &bits_b), 0);
    }

    #[test]
    fn test_hamming_all_different() {
        let a = vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let b = vec![-1.0_f32, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let bits_a = BinaryOps::encode(&a);
        let bits_b = BinaryOps::encode(&b);
        assert_eq!(BinaryOps::hamming(&bits_a, &bits_b), 8);
    }

    #[test]
    fn test_hamming_partial() {
        let a = vec![1.0_f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let b = vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let bits_a = BinaryOps::encode(&a);
        let bits_b = BinaryOps::encode(&b);
        assert_eq!(BinaryOps::hamming(&bits_a, &bits_b), 4);
    }

    #[test]
    fn test_hamming_scalar_fallback() {
        let a = vec![1.0_f32, -1.0, 1.0, -1.0];
        let b = vec![-1.0_f32, 1.0, -1.0, 1.0];
        let bits_a = BinaryOps::encode(&a);
        let bits_b = BinaryOps::encode(&b);
        let scalar = BinaryOps::hamming_scalar(&bits_a, &bits_b);
        assert_eq!(scalar, 4);
    }

    #[test]
    fn test_hamming_distance_with_config() {
        let config = BinaryConfig::new(8);
        let query = vec![1.0_f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let code = BinaryOps::encode(&query);
        let dist = BinaryOps::hamming_distance(&config, &query, &code);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_hamming_to_cosine() {
        assert!((BinaryOps::hamming_to_cosine_estimate(0, 128) - 1.0).abs() < 1e-6);
        assert!((BinaryOps::hamming_to_cosine_estimate(64, 128) - 0.0).abs() < 1e-6);
        assert!((BinaryOps::hamming_to_cosine_estimate(128, 128) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_approx_cosine() {
        let v = vec![1.0_f32, -1.0, 1.0, -1.0];
        let bits = BinaryOps::encode(&v);
        let cosine = BinaryOps::approx_cosine(&v, &bits);
        assert!((cosine - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_edge_cases() {
        let zero = vec![0.0_f32, 0.0, 0.0, 0.0];
        let bits = BinaryOps::encode(&zero);
        assert_eq!(bits[0], 0b00001111);

        let empty: Vec<f32> = vec![];
        let empty_bits = BinaryOps::encode(&empty);
        assert!(empty_bits.is_empty());
    }
}
