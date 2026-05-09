//! Product Quantization (PQ) — configurable compression for high-dim vectors.
//!
//! Vectors are split into M sub-spaces, each independently quantized via k-means
//! into 256 centroids. Memory reduction: d × 4 / M bytes per vector.
//!
//! # Algorithm
//!
//! ```ignore
//! For each vector v of dimension d:
//!   Split into M sub-vectors: v = [v_0, v_1, ..., v_{M-1}]
//!   Each v_m has dimension d / M
//!   Encode: code[m] = argmin_k ||v_m - c_mk||²
//!     where c_mk is centroid k of sub-space m
//!
//! Memory: M bytes per vector (instead of d × 4 bytes f32)
//! ```
//!
//! # ADC (Asymmetric Distance Computation)
//!
//! Precompute distance from query to all centroids once per query:
//!
//! ```ignore
//! tables[m][k] = ||query_m - c_mk||²  for all m, k
//! approx_dist(code) = Σ tables[m][code[m]]
//! ```
//!
//! This is O(M) per vector instead of O(d), where M << d.

pub mod codebook;
pub mod ops;
pub mod trainer;

pub use codebook::CodebookError;
pub use ops::PqOps;
pub use trainer::PqTrainer;
