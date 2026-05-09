//! SQ4 trainer — compute min/max statistics for quantization.
//!
//! Single-pass training to compute per-dimension min/max values.

use crate::quantization::config::Sq4Config;

/// Statistics collector for SQ4 training.
pub struct Sq4Stats {
    min_vals: Vec<f32>,
    max_vals: Vec<f32>,
    count: usize,
}

impl Sq4Stats {
    /// Create a new stats collector for the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            min_vals: vec![f32::MAX; dimension],
            max_vals: vec![f32::MIN; dimension],
            count: 0,
        }
    }

    /// Train on a single vector (update min/max).
    pub fn train(
        &mut self,
        vector: &[f32],
    ) {
        assert_eq!(
            vector.len(),
            self.min_vals.len(),
            "Vector dimension mismatch"
        );

        for (i, &val) in vector.iter().enumerate() {
            self.min_vals[i] = self.min_vals[i].min(val);
            self.max_vals[i] = self.max_vals[i].max(val);
        }
        self.count += 1;
    }

    /// Train on multiple vectors.
    pub fn train_batch<'a, I>(
        &mut self,
        vectors: I,
    ) where
        I: Iterator<Item = &'a [f32]>,
    {
        for vector in vectors {
            self.train(vector);
        }
    }

    /// Get the number of vectors trained on.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Convert collected statistics into an Sq4Config.
    ///
    /// This finalizes the training and computes per-dimension scale/offset.
    pub fn into_config(self) -> Sq4Config {
        Sq4Config::with_stats(self.min_vals, self.max_vals)
    }

    /// Get a reference to the min values.
    pub fn min_vals(&self) -> &[f32] {
        &self.min_vals
    }

    /// Get a reference to the max values.
    pub fn max_vals(&self) -> &[f32] {
        &self.max_vals
    }

    /// Reset statistics (start fresh training).
    pub fn reset(&mut self) {
        for min in self.min_vals.iter_mut() {
            *min = f32::MAX;
        }
        for max in self.max_vals.iter_mut() {
            *max = f32::MIN;
        }
        self.count = 0;
    }

    /// Merge statistics from another trainer.
    ///
    /// Useful for parallel training where each thread trains on a subset.
    pub fn merge(
        &mut self,
        other: &Sq4Stats,
    ) {
        assert_eq!(
            self.min_vals.len(),
            other.min_vals.len(),
            "Dimension mismatch"
        );

        for i in 0..self.min_vals.len() {
            self.min_vals[i] = self.min_vals[i].min(other.min_vals[i]);
            self.max_vals[i] = self.max_vals[i].max(other.max_vals[i]);
        }
        self.count += other.count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq4_stats_basic() {
        let mut stats = Sq4Stats::new(4);

        stats.train(&[0.0_f32, 10.0, 20.0, 30.0]);
        stats.train(&[5.0_f32, 15.0, 25.0, 35.0]);
        stats.train(&[2.0_f32, 12.0, 22.0, 32.0]);

        assert_eq!(stats.count(), 3);
        assert_eq!(stats.min_vals(), &[0.0, 10.0, 20.0, 30.0]);
        assert_eq!(stats.max_vals(), &[5.0, 15.0, 25.0, 35.0]);
    }

    #[test]
    fn test_sq4_stats_into_config() {
        let mut stats = Sq4Stats::new(2);

        stats.train(&[0.0_f32, 10.0]);
        stats.train(&[10.0_f32, 20.0]);

        let config = stats.into_config();

        assert_eq!(config.dimension, 2);
        assert_eq!(config.min_vals, vec![0.0, 10.0]);
        assert_eq!(config.max_vals, vec![10.0, 20.0]);

        // Scale should be 15/ (max - min)
        assert!((config.scale[0] - 1.5).abs() < 0.001); // 15/10
        assert!((config.scale[1] - 1.5).abs() < 0.001); //15/10

        // Offset should be -min
        assert_eq!(config.offset, vec![0.0, -10.0]);
    }

    #[test]
    fn test_sq4_stats_merge() {
        let mut stats1 = Sq4Stats::new(2);
        stats1.train(&[0.0_f32, 10.0]);
        stats1.train(&[5.0_f32, 15.0]);

        let mut stats2 = Sq4Stats::new(2);
        stats2.train(&[2.0_f32, 12.0]);
        stats2.train(&[8.0_f32, 18.0]);

        stats1.merge(&stats2);

        assert_eq!(stats1.count(), 4);
        assert_eq!(stats1.min_vals(), &[0.0, 10.0]);
        assert_eq!(stats1.max_vals(), &[8.0, 18.0]);
    }

    #[test]
    fn test_sq4_stats_reset() {
        let mut stats = Sq4Stats::new(2);
        stats.train(&[1.0_f32, 2.0]);
        stats.train(&[3.0_f32, 4.0]);

        assert_eq!(stats.count(), 2);

        stats.reset();

        assert_eq!(stats.count(), 0);
        assert_eq!(stats.min_vals(), &[f32::MAX, f32::MAX]);
        assert_eq!(stats.max_vals(), &[f32::MIN, f32::MIN]);
    }

    #[test]
    fn test_sq4_stats_batch() {
        let vectors: Vec<Vec<f32>> = vec![vec![0.0, 1.0], vec![2.0, 3.0], vec![4.0, 5.0]];

        let mut stats = Sq4Stats::new(2);
        stats.train_batch(vectors.iter().map(|v| v.as_slice()));

        assert_eq!(stats.count(), 3);
        assert_eq!(stats.min_vals(), &[0.0, 1.0]);
        assert_eq!(stats.max_vals(), &[4.0, 5.0]);
    }
}
