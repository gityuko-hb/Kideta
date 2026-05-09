//! Quantization configuration types — serializable, stored in segment metadata.
//!
//! Each quantization type has its own config struct containing all parameters
//! needed to encode vectors and compute approximate distances. Configs are
//! serde-serializable so they can be stored in segment metadata and reloaded
//! at startup without retraining.

use serde::{Deserialize, Serialize};

/// Quantization type — determines which compression scheme is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
#[derive(Default)]
pub enum QuantizationType {
    #[default]
    None,
    Sq8,
    Sq4,
    Binary,
    PQ,
}

/// Quantization configuration — stored in segment metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[derive(Default)]
pub enum QuantizationConfig {
    #[default]
    None,
    Sq8(Sq8Config),
    Sq4(Sq4Config),
    Binary(BinaryConfig),
    PQ(PqConfig),
}

impl QuantizationConfig {
    pub fn dimension(&self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::Sq8(c) => Some(c.dimension),
            Self::Sq4(c) => Some(c.dimension),
            Self::Binary(c) => Some(c.dimension),
            Self::PQ(c) => Some(c.dimension),
        }
    }

    pub fn quantization_type(&self) -> QuantizationType {
        match self {
            Self::None => QuantizationType::None,
            Self::Sq8(_) => QuantizationType::Sq8,
            Self::Sq4(_) => QuantizationType::Sq4,
            Self::Binary(_) => QuantizationType::Binary,
            Self::PQ(_) => QuantizationType::PQ,
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    pub fn bytes_per_vector(&self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::Sq8(c) => Some(c.dimension),
            Self::Sq4(c) => Some(c.dimension.div_ceil(2)),
            Self::Binary(c) => Some(c.dimension.div_ceil(8)),
            Self::PQ(c) => Some(c.num_subspaces),
        }
    }
}

/// SQ8 (Scalar Quantization 8-bit) configuration.
///
/// Each dimension is independently quantized to u8 [0, 255] using per-dimension
/// min/max scaling. Compression ratio: 4× (f32 → u8).
///
/// Memory: `dimension` bytes per vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Sq8Config {
    pub dimension: usize,
    pub min_vals: Vec<f32>,
    pub max_vals: Vec<f32>,
    pub scale: Vec<f32>,
    pub offset: Vec<f32>,
}

impl Sq8Config {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            min_vals: vec![f32::MAX; dimension],
            max_vals: vec![f32::MIN; dimension],
            scale: vec![0.0_f32; dimension],
            offset: vec![0.0_f32; dimension],
        }
    }

    pub fn with_stats(
        min_vals: Vec<f32>,
        max_vals: Vec<f32>,
    ) -> Self {
        let dimension = min_vals.len();
        assert_eq!(dimension, max_vals.len());
        let mut scale = vec![0.0_f32; dimension];
        let mut offset = vec![0.0_f32; dimension];
        for i in 0..dimension {
            offset[i] = -min_vals[i];
            let range = max_vals[i] - min_vals[i];
            scale[i] = if range > 0.0 {
                255.0 / range
            } else {
                0.0
            };
        }
        Self {
            dimension,
            min_vals,
            max_vals,
            scale,
            offset,
        }
    }

    pub fn compression_ratio(&self) -> f64 {
        4.0
    }
}

/// SQ4 (Scalar Quantization 4-bit) configuration.
///
/// Each dimension is quantized to 4 bits [0, 15], with 2 dimensions packed
/// into each byte. Compression ratio: 8× (f32 → 4-bit).
///
/// Memory: `(dimension + 1) / 2` bytes per vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Sq4Config {
    pub dimension: usize,
    pub min_vals: Vec<f32>,
    pub max_vals: Vec<f32>,
    pub scale: Vec<f32>,
    pub offset: Vec<f32>,
}

impl Sq4Config {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            min_vals: vec![f32::MAX; dimension],
            max_vals: vec![f32::MIN; dimension],
            scale: vec![0.0_f32; dimension],
            offset: vec![0.0_f32; dimension],
        }
    }

    pub fn with_stats(
        min_vals: Vec<f32>,
        max_vals: Vec<f32>,
    ) -> Self {
        let dimension = min_vals.len();
        assert_eq!(dimension, max_vals.len());
        let mut scale = vec![0.0_f32; dimension];
        let mut offset = vec![0.0_f32; dimension];
        for i in 0..dimension {
            offset[i] = -min_vals[i];
            let range = max_vals[i] - min_vals[i];
            scale[i] = if range > 0.0 {
                15.0 / range
            } else {
                0.0
            };
        }
        Self {
            dimension,
            min_vals,
            max_vals,
            scale,
            offset,
        }
    }

    pub fn compression_ratio(&self) -> f64 {
        8.0
    }
}

/// Binary quantization configuration.
///
/// Each dimension is encoded as a single bit (sign(x): 1 if x >= 0 else 0).
/// Compression ratio: 32× (f32 → 1 bit). Works best with normalized vectors.
///
/// Memory: `(dimension + 7) / 8` bytes per vector.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BinaryConfig {
    pub dimension: usize,
}

impl BinaryConfig {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    pub fn bytes_per_vector(&self) -> usize {
        self.dimension.div_ceil(8)
    }

    pub fn compression_ratio(&self) -> f64 {
        32.0
    }
}

/// Product Quantization (PQ) configuration.
///
/// Vectors are split into M sub-spaces, each independently quantized via k-means
/// into 256 centroids. Compression ratio: d / M bytes per vector.
///
/// # Example
///
/// For a 768-dimensional vector with `bytes_per_subvec = 8`:
/// - M = 768 / 8 = 96 sub-spaces
/// - Each sub-space: 8 bytes → 1 byte (256 centroids)
/// - Total: 96 bytes per vector (vs 3072 bytes for f32)
/// - Compression ratio: 3072 / 96 = 32×
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqConfig {
    pub dimension: usize,
    pub num_subspaces: usize,
    pub subspace_dim: usize,
    pub bytes_per_subvec: usize,
    pub codebook: Vec<f32>,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            dimension: 0,
            num_subspaces: 16,
            subspace_dim: 0,
            bytes_per_subvec: 0,
            codebook: vec![],
        }
    }
}

impl PqConfig {
    pub fn new(
        dimension: usize,
        bytes_per_subvec: usize,
    ) -> Self {
        let subspace_dim = bytes_per_subvec;
        let num_subspaces = dimension / subspace_dim;
        let codebook_size = num_subspaces * 256 * subspace_dim;
        Self {
            dimension,
            num_subspaces,
            subspace_dim,
            bytes_per_subvec,
            codebook: vec![0.0_f32; codebook_size],
        }
    }

    pub fn codebook_len(&self) -> usize {
        self.codebook.len()
    }

    pub fn num_centroids(&self) -> usize {
        256
    }

    pub fn compression_ratio(&self) -> f64 {
        (self.dimension as f64 * 4.0) / (self.num_subspaces as f64 * self.bytes_per_subvec as f64)
    }

    pub fn get_centroid(
        &self,
        subspace: usize,
        centroid_id: usize,
    ) -> &[f32] {
        let start = subspace * 256 * self.subspace_dim + centroid_id * self.subspace_dim;
        &self.codebook[start..start + self.subspace_dim]
    }

    pub fn centroid_mut(
        &mut self,
        subspace: usize,
        centroid_id: usize,
    ) -> &mut [f32] {
        let start = subspace * 256 * self.subspace_dim + centroid_id * self.subspace_dim;
        &mut self.codebook[start..start + self.subspace_dim]
    }

    pub fn bytes_per_vector(&self) -> usize {
        self.num_subspaces
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_config() {
        let min = vec![0.0_f32, -10.0_f32];
        let max = vec![100.0_f32, 50.0_f32];
        let config = Sq8Config::with_stats(min, max);
        assert_eq!(config.dimension, 2);
        assert!((config.scale[0] - 2.55).abs() < 1e-5);
        assert!((config.scale[1] - 4.25).abs() < 1e-5);
    }

    #[test]
    fn test_pq_config() {
        let config = PqConfig::new(768, 8);
        assert_eq!(config.num_subspaces, 96);
        assert_eq!(config.subspace_dim, 8);
        assert_eq!(config.bytes_per_vector(), 96);
        assert_eq!(config.codebook.len(), 96 * 256 * 8);
    }

    #[test]
    fn test_binary_config_bytes() {
        let config = BinaryConfig::new(100);
        assert_eq!(config.bytes_per_vector(), 13);

        let config2 = BinaryConfig::new(64);
        assert_eq!(config2.bytes_per_vector(), 8);
    }

    #[test]
    fn test_quantization_config_dimension() {
        assert!(QuantizationConfig::None.dimension().is_none());
        assert_eq!(Sq8Config::with_stats(vec![0.0], vec![1.0]).dimension, 1);
        assert_eq!(BinaryConfig::new(128).dimension, 128);
        assert_eq!(PqConfig::new(256, 8).dimension, 256);
    }

    #[test]
    fn test_compression_ratios() {
        assert!(
            (Sq8Config::with_stats(vec![0.0], vec![1.0]).compression_ratio() - 4.0).abs() < 1e-6
        );
        assert!((BinaryConfig::new(128).compression_ratio() - 32.0).abs() < 1e-6);
        let pq = PqConfig::new(128, 8);
        assert!((pq.compression_ratio() - 4.0).abs() < 1e-6);
    }
}
