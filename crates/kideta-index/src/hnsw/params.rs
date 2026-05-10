//! HNSW configuration parameters.
//!
//! # Parameters
//!
//! - `m`: Maximum number of connections per node at layer 0.
//!   Higher values improve recall but increase memory and build time.
//!   Typical values: 16-64. Default: 16.
//!
//! - `ef_construction`: Beam width during index construction.
//!   Higher values improve graph quality but increase build time.
//!   Typical values: 100-400. Default: 200.
//!
//! - `ef_search`: Beam width during search.
//!   Higher values improve recall but increase search latency.
//!   Typical values: 50-200. Default: 50.
//!
//! - `ml` (level sampling parameter): Controls the probability of
//!   nodes appearing at higher levels. Derived from M as ml = 1/ln(M).
//!   Higher ml means more levels, which can improve search efficiency
//!   but increases memory usage.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub ml: f64,
    pub max_level: usize,
}

impl HnswParams {
    pub fn new(
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Self {
        let ml = 1.0 / (m as f64).ln();
        let max_level = Self::compute_max_level(m);
        Self {
            m,
            ef_construction,
            ef_search,
            ml,
            max_level,
        }
    }

    #[allow(dead_code)]
    pub fn default_for_dimension(_dimension: usize) -> Self {
        Self::new(16, 200, 50)
    }

    fn compute_max_level(m: usize) -> usize {
        let ml = 1.0 / (m as f64).ln();
        let ln_max_nodes = 30.0_f64.ln();
        (ln_max_nodes / ml).ceil() as usize
    }
}

impl Default for HnswParams {
    fn default() -> Self {
        Self::new(16, 200, 50)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_derivation() {
        let params = HnswParams::new(16, 200, 50);
        let expected_ml = 1.0 / 16_f64.ln();
        assert!((params.ml - expected_ml).abs() < 1e-6);
    }

    #[test]
    fn test_max_level_for_m16() {
        let params = HnswParams::new(16, 200, 50);
        assert!(params.max_level >= 6);
    }

    #[test]
    fn test_default_params() {
        let params = HnswParams::default();
        assert_eq!(params.m, 16);
        assert_eq!(params.ef_construction, 200);
        assert_eq!(params.ef_search, 50);
    }
}
