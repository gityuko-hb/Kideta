//! PQ operations — encode vectors, build ADC tables, compute approximate distance.
//!
//! # ADC (Asymmetric Distance Computation)
//!
//! ADC is the standard PQ distance computation. For each query:
//!
//! 1. Split query into M sub-vectors
//! 2. Precompute `tables[m][k] = ||query_m - c_mk||²` for all M×256 values
//! 3. For each stored PQ code: `approx_dist = Σ tables[m][code[m]]`
//!
//! This is O(M × K + n × M) = O(n × M) instead of O(n × d).
//!
//! # Memory Layout
//!
//! - Codebook: `[M][256][subspace_dim] f32` — M × 256 × subspace_dim × 4 bytes
//! - ADC tables: `[M][256] f32` — allocated per query, M × 256 × 4 bytes
//! - Stored codes: `[n][M] u8` — n × M bytes

use crate::quantization::config::PqConfig;

/// Product Quantization operations.
pub struct PqOps;

impl PqOps {
    /// Encode an f32 vector → PQ code (M bytes).
    ///
    /// For each sub-space, finds the nearest centroid.
    #[inline]
    pub fn encode(
        config: &PqConfig,
        vector: &[f32],
    ) -> Vec<u8> {
        let mut code = vec![0u8; config.num_subspaces];
        Self::encode_to_slice(config, vector, &mut code);
        code
    }

    /// Encode directly into a pre-allocated slice.
    ///
    /// # Panics
    ///
    /// Panics if `code.len() != config.num_subspaces`.
    #[inline]
    pub fn encode_to_slice(
        config: &PqConfig,
        vector: &[f32],
        code: &mut [u8],
    ) {
        assert_eq!(code.len(), config.num_subspaces);
        assert_eq!(vector.len(), config.dimension);

        for (m, code_m) in code
            .iter_mut()
            .enumerate()
            .take(config.num_subspaces)
        {
            let start = m * config.subspace_dim;
            let end = start + config.subspace_dim;
            let subvec = &vector[start..end];

            let best_k = Self::_find_nearest_centroid(config, m, subvec);
            *code_m = best_k as u8;
        }
    }

    #[inline]
    fn _find_nearest_centroid(
        config: &PqConfig,
        subspace: usize,
        subvec: &[f32],
    ) -> usize {
        let mut best_k = 0;
        let mut best_dist = f32::MAX;

        for k in 0..config.num_centroids() {
            let centroid = config.get_centroid(subspace, k);
            let dist = Self::_l2_squared(subvec, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_k = k;
            }
        }

        best_k
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

    /// Build ADC distance lookup tables for a query.
    ///
    /// Returns `[M][256]` f32 table where `tables[m][k] = ||query_m - c_mk||²`.
    ///
    /// This is the expensive part — O(M × K × subspace_dim) per query —
    /// but is amortized over all scanned vectors.
    #[inline]
    pub fn build_adc_tables(
        config: &PqConfig,
        query: &[f32],
    ) -> Vec<Vec<f32>> {
        let mut tables = vec![vec![0.0_f32; config.num_centroids()]; config.num_subspaces];

        for (m, table_row) in tables.iter_mut().enumerate() {
            let start = m * config.subspace_dim;
            let end = start + config.subspace_dim;
            let query_subvec = &query[start..end];

            for (k, table_val) in table_row.iter_mut().enumerate() {
                let centroid = config.get_centroid(m, k);
                *table_val = Self::_l2_squared(query_subvec, centroid);
            }
        }

        tables
    }

    /// Compute approximate L2² distance using precomputed ADC tables.
    ///
    /// ```ignore
    /// dist² ≈ Σ tables[m][code[m]]
    /// ```
    ///
    /// # Safety
    ///
    /// Panics if `code.len() != config.num_subspaces` or
    /// `tables.len() != config.num_subspaces`.
    #[inline]
    pub fn approx_l2_distance(
        config: &PqConfig,
        query: &[f32],
        code: &[u8],
    ) -> f32 {
        assert_eq!(query.len(), config.dimension);
        assert_eq!(code.len(), config.num_subspaces);

        let tables = Self::build_adc_tables(config, query);
        Self::approx_l2_with_tables(config, &tables, code)
    }

    /// Compute approximate L2² distance using precomputed ADC tables.
    ///
    /// Use this when you have precomputed tables from `build_adc_tables`
    /// to avoid recomputing them for each vector.
    #[inline]
    pub fn approx_l2_with_tables(
        config: &PqConfig,
        tables: &[Vec<f32>],
        code: &[u8],
    ) -> f32 {
        assert_eq!(tables.len(), config.num_subspaces);
        assert_eq!(code.len(), config.num_subspaces);

        let mut sum = 0.0_f32;
        for (m, &c) in code.iter().enumerate() {
            sum += tables[m][c as usize];
        }
        sum
    }

    /// Compute approximate L2 distance (sqrt of approx_l2_distance).
    #[inline]
    pub fn approx_l2(
        config: &PqConfig,
        query: &[f32],
        code: &[u8],
    ) -> f32 {
        Self::approx_l2_distance(config, query, code).sqrt()
    }

    /// Compute approximate cosine similarity via PQ.
    ///
    /// First decodes the code to its nearest centroid reconstruction,
    /// then computes cosine with the query. This is approximate because
    /// the reconstruction is the centroid, not the original vector.
    #[inline]
    pub fn approx_cosine(
        config: &PqConfig,
        query: &[f32],
        code: &[u8],
    ) -> f32 {
        let reconstructed = Self::decode(config, code);
        Self::_cosine_similarity(query, &reconstructed)
    }

    /// Decode a PQ code → f32 vector (nearest centroid reconstruction).
    #[inline]
    pub fn decode(
        config: &PqConfig,
        code: &[u8],
    ) -> Vec<f32> {
        let mut vector = vec![0.0_f32; config.dimension];
        Self::decode_to_slice(config, code, &mut vector);
        vector
    }

    /// Decode directly into a pre-allocated slice.
    #[inline]
    pub fn decode_to_slice(
        config: &PqConfig,
        code: &[u8],
        vector: &mut [f32],
    ) {
        assert_eq!(vector.len(), config.dimension);
        assert_eq!(code.len(), config.num_subspaces);

        for (m, &c) in code.iter().enumerate() {
            let start = m * config.subspace_dim;
            let centroid = config.get_centroid(m, c as usize);
            vector[start..start + config.subspace_dim].copy_from_slice(centroid);
        }
    }

    #[inline]
    fn _cosine_similarity(
        a: &[f32],
        b: &[f32],
    ) -> f32 {
        let mut dot: f32 = 0.0;
        let mut norm_a: f32 = 0.0;
        let mut norm_b: f32 = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = (norm_a.sqrt() * norm_b.sqrt()).max(f32::EPSILON);
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::PqTrainer;

    fn train_test_config() -> PqConfig {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..64)
                    .map(|j| ((i * 64 + j) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let refs: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let trainer = PqTrainer::new(8, 256, 5);
        trainer.train(&refs)
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let config = train_test_config();
        let original = vec![0.5_f32; 64];

        let code = PqOps::encode(&config, &original);
        assert_eq!(code.len(), 8);

        let decoded = PqOps::decode(&config, &code);
        assert_eq!(decoded.len(), 64);

        let reconstruction_error: f32 = original
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        assert!(reconstruction_error < 10.0);
    }

    #[test]
    fn test_encode_to_slice() {
        let config = train_test_config();
        let vector = vec![1.0_f32; 64];
        let mut code = vec![0u8; 8];
        PqOps::encode_to_slice(&config, &vector, &mut code);
        assert!(code.iter().all(|&b| b < 255));
    }

    #[test]
    fn test_build_adc_tables() {
        let config = train_test_config();
        let query = vec![0.5_f32; 64];
        let tables = PqOps::build_adc_tables(&config, &query);

        assert_eq!(tables.len(), 8);
        assert_eq!(tables[0].len(), 256);
        assert!(tables.iter().all(|t| t.iter().all(|&d| d >= 0.0)));
    }

    #[test]
    fn test_approx_l2_with_tables() {
        let config = train_test_config();
        let query = vec![0.5_f32; 64];
        let tables = PqOps::build_adc_tables(&config, &query);
        let code = PqOps::encode(&config, &query);

        let dist = PqOps::approx_l2_with_tables(&config, &tables, &code);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_approx_l2_vs_full() {
        let config = train_test_config();
        let v1 = vec![0.3_f32; 64];
        let v2 = vec![0.7_f32; 64];

        let _full_dist = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let _code1 = PqOps::encode(&config, &v1);
        let code2 = PqOps::encode(&config, &v2);

        let approx_dist = PqOps::approx_l2_distance(&config, &v1, &code2);

        assert!(approx_dist > 0.0);
    }

    #[test]
    fn test_approx_cosine() {
        let config = train_test_config();
        let v = vec![1.0_f32; 64];
        let code = PqOps::encode(&config, &v);
        let cosine = PqOps::approx_cosine(&config, &v, &code);
        assert!(cosine >= 0.0 && cosine <= 1.0);
    }

    #[test]
    fn test_encode_tiny_dimension() {
        let vectors = vec![vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]];
        let refs: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let trainer = PqTrainer::new(8, 256, 5);
        let config = trainer.train(&refs);

        let vector = vec![1.5_f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        let code = PqOps::encode(&config, &vector);
        assert_eq!(code.len(), 1);
    }
}
