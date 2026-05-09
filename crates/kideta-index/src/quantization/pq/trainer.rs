//! PQ Trainer — k-means++ initialization and Lloyd iterations.
//!
//! Training is embarrassingly parallel across sub-spaces. Each sub-space's
//! codebook is independent, so we can train all M sub-spaces in parallel
//! using rayon.

use crate::quantization::config::PqConfig;

/// Product Quantization trainer.
///
/// Trains PQ codebooks from a set of training vectors using k-means++
/// initialization followed by Lloyd iterations.
///
/// # Usage
///
/// ```ignore
/// let trainer = PqTrainer::new(8, 256, 20);
/// let config = trainer.train(&training_vectors);
/// ```
pub struct PqTrainer {
    pub bytes_per_subvec: usize,
    pub num_centroids: usize,
    pub iterations: usize,
    pub seed: u64,
}

impl PqTrainer {
    pub fn new(
        bytes_per_subvec: usize,
        num_centroids: usize,
        iterations: usize,
    ) -> Self {
        Self {
            bytes_per_subvec,
            num_centroids,
            iterations,
            seed: 42,
        }
    }

    pub fn with_seed(
        mut self,
        seed: u64,
    ) -> Self {
        self.seed = seed;
        self
    }

    /// Train all sub-space codebooks from training vectors.
    ///
    /// # Arguments
    ///
    /// * `vectors` — slice of vector slices, all same dimension
    ///
    /// # Returns
    ///
    /// `PqConfig` containing the trained codebook.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `vectors` is empty
    /// - `dimension % bytes_per_subvec != 0`
    /// - Any vector has wrong dimension
    pub fn train(
        &self,
        vectors: &[&[f32]],
    ) -> PqConfig {
        if vectors.is_empty() {
            panic!("PqTrainer::train requires at least one vector");
        }

        let dimension = vectors[0].len();
        if !dimension.is_multiple_of(self.bytes_per_subvec) {
            panic!(
                "Dimension {} must be divisible by bytes_per_subvec {}",
                dimension, self.bytes_per_subvec
            );
        }

        let mut config = PqConfig::new(dimension, self.bytes_per_subvec);

        self.train_subspaces(vectors, &mut config);
        config
    }

    fn train_subspaces(
        &self,
        vectors: &[&[f32]],
        config: &mut PqConfig,
    ) {
        let subspace_dim = config.subspace_dim;
        let num_subspaces = config.num_subspaces;
        let num_restarts = 3usize;

        for m in 0..num_subspaces {
            let mut subspace_data = Vec::with_capacity(vectors.len() * subspace_dim);
            for v in vectors {
                subspace_data.extend_from_slice(&v[m * subspace_dim..(m + 1) * subspace_dim]);
            }

            let mut best_centroids = Vec::new();
            let mut best_error = f32::INFINITY;

            for restart in 0..num_restarts {
                let seed = self.seed.wrapping_add(restart as u64);
                let mut rng = Xorshift64::new(seed);
                let mut centroids =
                    self.kmeanspp_init_with_rng(&subspace_data, subspace_dim, &mut rng);
                let error = self.lloyd_iterations(&subspace_data, subspace_dim, &mut centroids);

                if error < best_error {
                    best_error = error;
                    best_centroids = centroids;
                }
            }

            for (k, centroid) in best_centroids
                .chunks_exact(subspace_dim)
                .enumerate()
            {
                config
                    .centroid_mut(m, k)
                    .copy_from_slice(centroid);
            }
        }
    }

    /// K-means++ initialization — D² sampling for better initial centroids.
    ///
    /// Unlike random init, k-means++ spreads centroids to cover the data space,
    /// reducing the number of Lloyd iterations needed to converge.
    fn kmeanspp_init_with_rng(
        &self,
        data: &[f32],
        dim: usize,
        rng: &mut Xorshift64,
    ) -> Vec<f32> {
        let n = data.len() / dim;
        let k = self.num_centroids;

        let mut centroids = Vec::with_capacity(k * dim);

        let first_idx = rng.next_u64() as usize % n;
        centroids.extend_from_slice(&data[first_idx * dim..(first_idx + 1) * dim]);

        let mut distances = vec![0.0_f32; n];
        let mut weights = vec![0.0_f32; n];

        for _ in 1..k {
            let mut total_weight = 0.0_f32;
            for (i, dist) in distances.iter_mut().enumerate() {
                let centroid_idx = i * dim;
                let d = Self::_l2_squared(
                    &data[centroid_idx..centroid_idx + dim],
                    &centroids[centroids.len() - dim..],
                );
                *dist = d;
                weights[i] = d;
                total_weight += d;
            }

            if total_weight == 0.0 {
                let random_idx = rng.next_u64() as usize % n;
                centroids.extend_from_slice(&data[random_idx * dim..(random_idx + 1) * dim]);
                continue;
            }

            let threshold = rng.next_f64() * total_weight as f64;
            let mut cumsum = 0.0_f64;
            let mut selected_idx = 0;
            for (i, &w) in weights.iter().enumerate() {
                cumsum += w as f64;
                if cumsum >= threshold {
                    selected_idx = i;
                    break;
                }
            }
            centroids.extend_from_slice(&data[selected_idx * dim..(selected_idx + 1) * dim]);
        }

        centroids
    }

    /// Lloyd iterations — assign vectors to nearest centroid, update centroids.
    /// Returns final reconstruction error (sum of squared distances).
    fn lloyd_iterations(
        &self,
        data: &[f32],
        dim: usize,
        centroids: &mut [f32],
    ) -> f32 {
        let n = data.len() / dim;
        let k = centroids.len() / dim;
        let mut assignments = vec![0u32; n];
        let mut counts = vec![0usize; k];
        let mut new_centroids = vec![0.0_f32; centroids.len()];

        let mut total_error = 0.0f32;

        for _ in 0..self.iterations {
            counts.fill(0);
            new_centroids.fill(0.0);
            total_error = 0.0;

            for (i, slice) in data.chunks_exact(dim).enumerate() {
                let (best_k, best_dist) = Self::_find_nearest(slice, centroids, dim, k);
                assignments[i] = best_k as u32;
                counts[best_k] += 1;
                total_error += best_dist;

                let nc_start = best_k * dim;
                for j in 0..dim {
                    new_centroids[nc_start + j] += slice[j];
                }
            }

            for (k_idx, &count) in counts.iter().enumerate() {
                if count > 0 {
                    let start = k_idx * dim;
                    let count_f = count as f32;
                    for j in 0..dim {
                        centroids[start + j] = new_centroids[start + j] / count_f;
                    }
                }
            }
        }

        total_error
    }

    #[inline]
    fn _l2_squared(
        a: &[f32],
        b: &[f32],
    ) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    #[inline]
    fn _find_nearest(
        vector: &[f32],
        centroids: &[f32],
        dim: usize,
        k: usize,
    ) -> (usize, f32) {
        let mut best_k = 0;
        let mut best_dist = f32::MAX;

        for ki in 0..k {
            let start = ki * dim;
            let dist = Self::_l2_squared(vector, &centroids[start..start + dim]);
            if dist < best_dist {
                best_dist = dist;
                best_k = ki;
            }
        }

        (best_k, best_dist)
    }
}

/// Xorshift64 PRNG — fast, deterministic, no external deps.
#[derive(Debug, Clone)]
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            1
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vectors(
        dim: usize,
        count: usize,
    ) -> Vec<Vec<f32>> {
        (0..count)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * dim + j) as f32) * 0.01)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_train_simple() {
        let vectors = make_test_vectors(64, 100);
        let vectors_ref: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let trainer = PqTrainer::new(8, 256, 5);
        let config = trainer.train(&vectors_ref);

        assert_eq!(config.dimension, 64);
        assert_eq!(config.num_subspaces, 8);
        assert_eq!(config.subspace_dim, 8);
        assert_eq!(config.codebook.len(), 8 * 256 * 8);
    }

    #[test]
    fn test_train_large_dim() {
        let vectors = make_test_vectors(768, 500);
        let vectors_ref: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let trainer = PqTrainer::new(8, 256, 10);
        let config = trainer.train(&vectors_ref);

        assert_eq!(config.num_subspaces, 96);
        assert_eq!(config.bytes_per_vector(), 96);
    }

    #[test]
    fn test_train_bytes_per_subvec() {
        let vectors = make_test_vectors(128, 50);
        let vectors_ref: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();

        let trainer4 = PqTrainer::new(4, 256, 5);
        let config4 = trainer4.train(&vectors_ref);
        assert_eq!(config4.num_subspaces, 32);

        let trainer8 = PqTrainer::new(8, 256, 5);
        let config8 = trainer8.train(&vectors_ref);
        assert_eq!(config8.num_subspaces, 16);
    }

    #[test]
    #[should_panic(expected = "requires at least one vector")]
    fn test_train_empty() {
        let vectors: Vec<&[f32]> = vec![];
        let trainer = PqTrainer::new(8, 256, 5);
        trainer.train(&vectors);
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn test_train_wrong_dim() {
        let vectors = vec![vec![1.0_f32, 2.0, 3.0]];
        let vectors_ref: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let trainer = PqTrainer::new(8, 256, 5);
        trainer.train(&vectors_ref);
    }

    #[test]
    fn test_centroids_change_after_training() {
        let vectors = make_test_vectors(64, 100);
        let vectors_ref: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let trainer = PqTrainer::new(8, 256, 5);
        let config = trainer.train(&vectors_ref);

        let centroid = config.get_centroid(0, 0);
        assert!(!centroid.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_compression_ratio() {
        let vectors = make_test_vectors(768, 100);
        let vectors_ref: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let trainer = PqTrainer::new(8, 256, 5);
        let config = trainer.train(&vectors_ref);

        let ratio = config.compression_ratio();
        assert!((ratio - 4.0).abs() < 0.1);
    }
}
