//! Binary Quantization — 32× compression.
//!
//! Each dimension is encoded as a single bit: `1` if x >= 0, else `0`.
//! This is the most aggressive compression scheme.
//!
//! # Best Use Cases
//!
//! - Normalized vectors (cosine similarity works well with binarized signs)
//! - High-dimensional vectors where memory is the bottleneck
//! - Fast candidate filtering before full precision rescore
//!
//! # Distance Metric
//!
//! Hamming distance is the natural metric for binary vectors:
//! `dist = popcount(a XOR b)` — number of differing bits.
//!
//! For cosine similarity on normalized vectors, Hamming distance on binary
//! codes correlates well: `cosine ≈ 1 - 2 * hamming / dim`.
//!
//! # SIMD Acceleration
//!
//! | Architecture | Instruction | Throughput |
//! |--------------|-------------|------------|
//! | AVX-512 | `_mm512_popcnt_epi64` | 16 u64/cycle |
//! | AVX2 | `_mm256_popcnt_epi64` | 8 u64/cycle |
//! | SSE4.2 | `_mm_popcnt_u64` | 2 u64/cycle |
//! | Scalar | software fallback | 1 u64/~3 cycles |

pub mod ops;

pub use ops::BinaryOps;
