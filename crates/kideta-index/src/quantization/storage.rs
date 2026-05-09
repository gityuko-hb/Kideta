//! Quantized vector storage — stores compressed vectors and provides approximate distance.
//!
//! `QuantizedStorage` holds the compressed byte codes for all vectors in a segment
//! along with the quantization configuration. It provides `approx_distance()` which
//! computes distance entirely in the compressed domain without decoding.
//!
//! # Storage Layout
//!
//! ```ignore
//! QuantizedStorage {
//!     config: QuantizationConfig,
//!     codes: Vec<u8>,     // [code_0, code_1, ...] packed consecutively
//!     num_vectors: usize,
//!     bytes_per_vector: usize,
//! }
//! ```
//!
//! # Two-Phase Search
//!
//! 1. **Approximate phase**: `approx_distance()` on compressed codes — O(1) per vector
//! 2. **Rescore phase**: decode top-N candidates to f32, compute exact distance
//!
//! The rescore factor (default 10×) controls how many extra candidates are fetched
//! before rescore. Higher factor = better recall, more compute.

use crate::quantization::binary::BinaryOps;
use crate::quantization::config::QuantizationConfig;
use crate::quantization::pq::PqOps;
use crate::quantization::sq4::Sq4Ops;
use crate::quantization::sq8::Sq8Ops;

/// Stores compressed vectors and provides approximate distance computation.
///
/// Does not store original vectors — those live in the index (FlatIndex/HnswIndex).
/// QuantizedStorage is created from training data, then used for fast approximate search.
pub struct QuantizedStorage {
    pub config: QuantizationConfig,
    codes: Vec<u8>,
    num_vectors: usize,
    bytes_per_vector: usize,
}

impl QuantizedStorage {
    pub fn new(
        codes: Vec<u8>,
        config: &QuantizationConfig,
        num_vectors: usize,
    ) -> Self {
        let bytes_per_vector = config.bytes_per_vector().unwrap_or(0);
        Self {
            config: config.clone(),
            codes,
            num_vectors,
            bytes_per_vector,
        }
    }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vector
    }

    pub fn total_bytes(&self) -> usize {
        self.codes.len()
    }

    pub fn get_code(
        &self,
        idx: usize,
    ) -> Option<&[u8]> {
        if idx >= self.num_vectors {
            return None;
        }
        let start = idx * self.bytes_per_vector;
        Some(&self.codes[start..start + self.bytes_per_vector])
    }

    pub fn codes_slice(&self) -> &[u8] {
        &self.codes
    }

    /// Computes approximate distance between `query` (f32) and stored vector `idx`.
    ///
    /// Returns `None` if quantization type doesn't support approximate distance
    /// (e.g., when `QuantizationConfig::None` is used).
    pub fn approx_distance(
        &self,
        query: &[f32],
        idx: usize,
    ) -> Option<f32> {
        let code = self.get_code(idx)?;
        match &self.config {
            QuantizationConfig::Sq8(cfg) => Some(Sq8Ops::approx_l2_distance(cfg, query, code)),
            QuantizationConfig::Sq4(cfg) => Some(Sq4Ops::approx_l2_distance(cfg, query, code)),
            QuantizationConfig::Binary(cfg) => {
                Some(BinaryOps::hamming_distance(cfg, query, code) as f32)
            },
            QuantizationConfig::PQ(cfg) => Some(PqOps::approx_l2_distance(cfg, query, code)),
            QuantizationConfig::None => None,
        }
    }

    /// Returns all codes as a contiguous slice.
    pub fn as_codes(&self) -> &[u8] {
        &self.codes
    }
}

impl std::fmt::Debug for QuantizedStorage {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("QuantizedStorage")
            .field("config", &self.config)
            .field("num_vectors", &self.num_vectors)
            .field("bytes_per_vector", &self.bytes_per_vector)
            .field("total_bytes", &self.codes.len())
            .finish()
    }
}

/// Encodes a batch of vectors using the given quantization config.
pub fn encode_vectors(
    config: &QuantizationConfig,
    vectors: &[&[f32]],
) -> QuantizedStorage {
    let num_vectors = vectors.len();
    let bytes_per_vector = config.bytes_per_vector().unwrap_or(0);
    let mut codes = vec![0u8; num_vectors * bytes_per_vector];

    for (i, vector) in vectors.iter().enumerate() {
        let start = i * bytes_per_vector;
        let slice = &mut codes[start..start + bytes_per_vector];
        match config {
            QuantizationConfig::Sq8(cfg) => Sq8Ops::encode_to_slice(cfg, vector, slice),
            QuantizationConfig::Sq4(cfg) => Sq4Ops::encode_to_slice(cfg, vector, slice),
            QuantizationConfig::Binary(_cfg) => BinaryOps::encode_to_slice(vector, slice),
            QuantizationConfig::PQ(cfg) => PqOps::encode_to_slice(cfg, vector, slice),
            QuantizationConfig::None => {},
        }
    }

    QuantizedStorage::new(codes, config, num_vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::config::BinaryConfig;
    #[allow(unused_imports)]
    use crate::quantization::config::Sq8Config;
    use crate::quantization::sq8::Sq8Stats;

    fn make_test_vectors() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![-1.0, -2.0, 3.0, 4.0],
        ]
    }

    #[test]
    fn test_sq8_encode_and_approx() {
        let test_vectors = make_test_vectors();
        let vectors: Vec<_> = test_vectors
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let stats = Sq8Stats::compute(&vectors);
        let config = stats.into_config();
        let storage = encode_vectors(&QuantizationConfig::Sq8(config), &vectors);

        assert_eq!(storage.num_vectors(), 3);
        assert_eq!(storage.bytes_per_vector(), 4);

        let query = vec![1.0, 2.0, 3.0, 4.0];
        let dist = storage.approx_distance(&query, 0);
        assert!(dist.is_some());
        assert!(dist.unwrap() < 0.1);
    }

    #[test]
    fn test_binary_encode() {
        let test_vectors = make_test_vectors();
        let vectors: Vec<_> = test_vectors
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let config = BinaryConfig::new(4);
        let storage = encode_vectors(&QuantizationConfig::Binary(config), &vectors);

        assert_eq!(storage.num_vectors(), 3);
        assert_eq!(storage.bytes_per_vector(), 1);

        let query = vec![1.0, 2.0, 3.0, 4.0];
        let dist = storage.approx_distance(&query, 0);
        assert!(dist.is_some());
    }

    #[test]
    fn test_get_code() {
        let test_vectors = make_test_vectors();
        let vectors: Vec<_> = test_vectors
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let config = BinaryConfig::new(4);
        let storage = encode_vectors(&QuantizationConfig::Binary(config), &vectors);

        let code0 = storage.get_code(0).unwrap();
        assert_eq!(code0.len(), 1);

        assert!(storage.get_code(100).is_none());
    }
}
