//! IVF Trainer — k-means++ initialization and Lloyd iterations for centroid computation.
//!
//! This trainer computes cluster centroids for the IVF index. It uses:
//! - **k-means++** for smart initialization (D² sampling)
//! - **Lloyd iterations** for refinement
//!
//! The algorithm is:
//! 1. Pick first centroid randomly from data points
//! 2. For each remaining centroid, sample proportional to D² (distance squared)
//! 3. Repeat Lloyd: assign points to nearest centroid, recompute centroids as means

use crate::ivf::params::IvfParams;
use kideta_core::utils::rng::Xorshift64;
use rayon::prelude::*;

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("vectors array is empty")]
    EmptyVectors,

    #[error("vector dimension mismatch")]
    DimensionMismatch,
}

#[derive(Debug)]
pub struct IvfTrainer {
    params: IvfParams,
    dimension: usize,
}

impl IvfTrainer {
    pub fn new(
        params: IvfParams,
        dimension: usize,
    ) -> Self {
        Self { params, dimension }
    }

    pub fn train(
        &self,
        vectors: &[&[f32]],
    ) -> Result<(Vec<f32>, Vec<usize>), TrainError> {
        if vectors.is_empty() {
            return Err(TrainError::EmptyVectors);
        }

        let n = vectors.len();
        let k = self.params.num_clusters.min(n);

        if vectors.iter().any(|v| v.len() != self.dimension) {
            return Err(TrainError::DimensionMismatch);
        }

        let mut rng = Xorshift64::new(self.params.seed);

        let sampled_vectors = if n > 50_000 {
            Some(self.stratified_training_sample(vectors, 50_000))
        } else {
            None
        };
        let init_vectors = sampled_vectors.as_deref().unwrap_or(vectors);
        let mut centroids = self.kmeanspp_init(init_vectors, k, &mut rng);

        let mut assignments = vec![0usize; n];

        for _ in 0..self.params.iterations {
            let changed = self.lloyd_iteration(vectors, &mut centroids, &mut assignments, k);
            if !changed {
                break;
            }
        }

        Ok((centroids, assignments))
    }

    fn stratified_training_sample<'a>(
        &self,
        vectors: &'a [&[f32]],
        target: usize,
    ) -> Vec<&'a [f32]> {
        if vectors.len() <= target {
            return vectors.to_vec();
        }
        let stride = vectors.len().div_ceil(target).max(1);
        let mut sample = Vec::with_capacity(target);
        let mut cursor = 0usize;
        while sample.len() < target && cursor < vectors.len() {
            sample.push(vectors[cursor]);
            cursor += stride;
        }
        sample
    }

    fn kmeanspp_init(
        &self,
        vectors: &[&[f32]],
        k: usize,
        rng: &mut Xorshift64,
    ) -> Vec<f32> {
        let n = vectors.len();
        let dim = self.dimension;

        let first_idx = rng.next_u64() as usize % n;
        let mut centroids = vectors[first_idx].to_vec();

        let mut distances = vec![0.0_f32; n];
        let mut weights = vec![0.0_f32; n];

        for _ in 1..k {
            let mut total_weight = 0.0_f64;
            let num_centroids = centroids.len() / dim;

            for (i, vector) in vectors.iter().enumerate() {
                let mut min_dist = f32::MAX;
                for ki in 0..num_centroids {
                    let dist = self.l2_squared(vector, &centroids[ki * dim..]);
                    min_dist = min_dist.min(dist);
                }
                distances[i] = min_dist;
                weights[i] = min_dist;
                total_weight += min_dist as f64;
            }

            if total_weight == 0.0 {
                let random_idx = rng.next_u64() as usize % n;
                centroids.extend_from_slice(vectors[random_idx]);
                continue;
            }

            let threshold = rng.next_f64() * total_weight;
            let mut cumsum = 0.0_f64;
            let mut selected_idx = 0;

            for (i, &w) in weights.iter().enumerate() {
                cumsum += w as f64;
                if cumsum >= threshold {
                    selected_idx = i;
                    break;
                }
            }

            centroids.extend_from_slice(vectors[selected_idx]);
        }

        centroids
    }

    fn lloyd_iteration(
        &self,
        vectors: &[&[f32]],
        centroids: &mut [f32],
        assignments: &mut [usize],
        k: usize,
    ) -> bool {
        let dim = self.dimension;
        let _n = vectors.len();

        let mut counts = vec![0usize; k];
        let mut new_centroids = vec![0.0_f32; centroids.len()];
        let mut changed = false;

        let centroids_ref = centroids.to_vec();

        let new_assignments: Vec<usize> = vectors
            .par_iter()
            .map(|vector| Self::find_nearest(vector, &centroids_ref, dim, k).0)
            .collect();

        for (i, &new_a) in new_assignments.iter().enumerate() {
            if assignments[i] != new_a {
                changed = true;
                assignments[i] = new_a;
            }
            counts[new_a] += 1;
            let nc_start = new_a * dim;
            for j in 0..dim {
                new_centroids[nc_start + j] += vectors[i][j];
            }
        }

        for cluster in 0..k {
            if counts[cluster] != 0 {
                continue;
            }

            let donor = counts
                .iter()
                .enumerate()
                .max_by_key(|(_, count)| **count)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let donor_member = assignments
                .iter()
                .position(|assignment| *assignment == donor)
                .unwrap_or(0);

            assignments[donor_member] = cluster;
            counts[donor] = counts[donor].saturating_sub(1);
            counts[cluster] = 1;

            let start = cluster * dim;
            new_centroids[start..start + dim].copy_from_slice(vectors[donor_member]);
            changed = true;
        }

        for (ki, &count_val) in counts.iter().enumerate() {
            if count_val > 0 {
                let start = ki * dim;
                let count_f = count_val as f32;
                for j in 0..dim {
                    centroids[start + j] = new_centroids[start + j] / count_f;
                }
            }
        }

        changed
    }

    #[inline]
    fn l2_squared(
        &self,
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
    fn find_nearest(
        vector: &[f32],
        centroids: &[f32],
        dim: usize,
        k: usize,
    ) -> (usize, f32) {
        let mut best_k = 0;
        let mut best_dist = f32::MAX;

        for ki in 0..k {
            let start = ki * dim;
            let dist = vector
                .iter()
                .zip(&centroids[start..start + dim])
                .map(|(x, y)| {
                    let d = x - y;
                    d * d
                })
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_k = ki;
            }
        }

        (best_k, best_dist)
    }
}
