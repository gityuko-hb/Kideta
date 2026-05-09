//! SQ8 trainer — per-dimension min/max statistics collection.
//!
//! Computes per-dimension min and max values from a set of training vectors
//! in a single pass. O(n × d) time, O(d) space.
//!
//! # Usage
//!
//! ```ignore
//! let vectors: Vec<&[f32]> = ...;
//! let stats = Sq8Stats::compute(&vectors);
//! let config = stats.into_config();
//! ```

use crate::quantization::config::Sq8Config;

/// Per-dimension min/max statistics for SQ8 quantization.
///
/// Computed in a single pass over training vectors.
#[derive(Debug, Clone)]
pub struct Sq8Stats {
    pub dimension: usize,
    pub min_vals: Vec<f32>,
    pub max_vals: Vec<f32>,
}

impl Sq8Stats {
    /// Compute per-dimension min/max from a slice of vectors.
    ///
    /// Single pass: O(n × d) time, O(d) space.
    ///
    /// # Panics
    ///
    /// Panics if `vectors` is empty or if vectors have inconsistent dimensions.
    pub fn compute(vectors: &[&[f32]]) -> Self {
        if vectors.is_empty() {
            panic!("Sq8Stats::compute requires at least one vector");
        }

        let dimension = vectors[0].len();
        let mut min_vals = vec![f32::MAX; dimension];
        let mut max_vals = vec![f32::MIN; dimension];

        for vector in vectors {
            assert_eq!(
                vector.len(),
                dimension,
                "All vectors must have the same dimension"
            );
            for (i, &val) in vector.iter().enumerate() {
                if val < min_vals[i] {
                    min_vals[i] = val;
                }
                if val > max_vals[i] {
                    max_vals[i] = val;
                }
            }
        }

        Self {
            dimension,
            min_vals,
            max_vals,
        }
    }

    /// Convert stats into a serializable `Sq8Config`.
    pub fn into_config(self) -> Sq8Config {
        Sq8Config::with_stats(self.min_vals, self.max_vals)
    }
}

impl Sq8Config {
    /// Returns the effective scale for dimension `i`.
    ///
    /// Returns `1.0` if all values in this dimension are identical (zero range),
    /// which means this dimension will quantize to a constant value.
    #[inline]
    pub fn get_scale(
        &self,
        i: usize,
    ) -> f32 {
        self.scale[i]
    }

    /// Returns the effective offset for dimension `i`.
    #[inline]
    pub fn get_offset(
        &self,
        i: usize,
    ) -> f32 {
        self.offset[i]
    }

    /// Validates that this config matches the given dimension.
    pub fn validate_dimension(
        &self,
        dim: usize,
    ) -> Result<(), String> {
        if self.dimension != dim {
            return Err(format!(
                "Sq8Config dimension mismatch: expected {}, got {}",
                self.dimension, dim
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_basic() {
        let vectors = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![3.0_f32, 2.0, 1.0],
            vec![2.0_f32, 2.0, 2.0],
        ];
        let vectors_refs: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let stats = Sq8Stats::compute(&vectors_refs);

        assert_eq!(stats.dimension, 3);
        assert_eq!(stats.min_vals, vec![1.0, 2.0, 1.0]);
        assert_eq!(stats.max_vals, vec![3.0, 2.0, 3.0]);
    }

    #[test]
    fn test_compute_single_vector() {
        let vectors = vec![vec![10.0_f32, -5.0]];
        let vectors_refs: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let stats = Sq8Stats::compute(&vectors_refs);

        assert_eq!(stats.min_vals, vec![10.0, -5.0]);
        assert_eq!(stats.max_vals, vec![10.0, -5.0]);
    }

    #[test]
    fn test_into_config() {
        let vectors = vec![vec![0.0_f32, 0.0_f32], vec![100.0_f32, 100.0_f32]];
        let vectors_refs: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let stats = Sq8Stats::compute(&vectors_refs);
        let config = stats.into_config();

        assert_eq!(config.dimension, 2);
        assert!((config.scale[0] - 2.55).abs() < 1e-3);
        assert!((config.scale[1] - 2.55).abs() < 1e-3);
    }

    #[test]
    #[should_panic(expected = "requires at least one vector")]
    fn test_compute_empty() {
        let vectors: Vec<&[f32]> = vec![];
        Sq8Stats::compute(&vectors);
    }

    #[test]
    #[should_panic(expected = "same dimension")]
    fn test_compute_mismatched_dims() {
        let vectors = vec![vec![1.0_f32, 2.0, 3.0], vec![1.0_f32, 2.0]];
        let vectors_refs: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        Sq8Stats::compute(&vectors_refs);
    }

    #[test]
    fn test_zero_range_dimension() {
        let vectors = vec![vec![5.0_f32, 10.0_f32], vec![5.0_f32, 20.0_f32]];
        let vectors_refs: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let config = Sq8Stats::compute(&vectors_refs).into_config();

        assert_eq!(config.scale[0], 0.0);
        assert!(config.scale[1] > 0.0);
    }
}
