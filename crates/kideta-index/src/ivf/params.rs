//! IVF (Inverted File Index) parameters.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IvfParams {
    pub num_clusters: usize,
    pub nprobe: usize,
    pub iterations: usize,
    pub seed: u64,
    pub pq_bytes_per_subvec: Option<usize>,
}

impl Default for IvfParams {
    fn default() -> Self {
        Self {
            num_clusters: 1024,
            nprobe: 10,
            iterations: 20,
            seed: 42,
            pq_bytes_per_subvec: None,
        }
    }
}

impl IvfParams {
    pub fn new(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            ..Default::default()
        }
    }

    pub fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_pq(mut self, bytes_per_subvec: usize) -> Self {
        self.pq_bytes_per_subvec = Some(bytes_per_subvec);
        self
    }

    pub fn for_scale(n_vectors: usize, dim: usize, metric: kideta_core::metric::DistanceMetric) -> Self {
        let base_clusters = ((n_vectors as f64).sqrt() as usize).clamp(32, 16_384);
        let dim_boost = if dim >= 512 { 2 } else { 1 };
        let metric_boost = if matches!(metric, kideta_core::metric::DistanceMetric::Cosine) {
            2
        } else {
            1
        };
        let num_clusters = (base_clusters * dim_boost).min(n_vectors.max(1));
        let nprobe = ((num_clusters as f64).sqrt() as usize * metric_boost).clamp(8, num_clusters.max(1));

        Self::new(num_clusters)
            .with_nprobe(nprobe)
            .with_iterations(50)
            .with_seed(42)
    }
}

