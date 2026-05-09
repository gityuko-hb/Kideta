//! Scalar Quantization 4-bit (SQ4) —8× compression.
//!
//! Each dimension is quantized to 4 bits [0, 15], with 2 dimensions packed
//! into each byte. This provides higher compression than SQ8 (8× vs 4×)
//! at the cost of precision.
//!
//! # Memory Layout
//!
//! For dimension D:
//! - Bytes per vector: `(D + 1) / 2`
//! - Byte[i/2] = `(value[2i] << 4) | value[2i+1]`
//!
//! # SIMD Optimization
//!
//! AVX2 and NEON implementations provide significant speedup:
//! - Process 32 nibbles per iteration (16 bytes)
//! - Runtime dispatch selects best implementation
//!
//! # Usage
//!
//! ```ignore
//! use kideta_index::quantization::{Sq4Config, Sq4Stats};
//! use kideta_index::quantization::sq4::Sq4Ops;
//!
//! // Create sample vectors for training
//! let vec1 = vec![0.1_f32; 128];
//! let vec2 = vec![0.2_f32; 128];
//! let vec3 = vec![0.3_f32; 128];
//!
//! // Train on sample data
//! let mut stats = Sq4Stats::new(128);
//! stats.train(&vec1);
//! stats.train(&vec2);
//! stats.train(&vec3);
//! let config = stats.into_config();
//!
//! // Encode vectors
//! let vector = vec![0.15_f32; 128];
//! let code = Sq4Ops::encode(&config, &vector);
//!
//! // Compute approximate distance
//! let query = vec![0.2_f32; 128];
//! let dist = Sq4Ops::approx_l2_distance(&config, &query, &code);
//!
//! // Verify the code length
//! assert_eq!(code.len(), 64); // (128 + 1) / 2 = 64 bytes
//! ```

mod ops;
mod trainer;

pub use ops::{Sq4Ops, Sq4Simd};
pub use trainer::Sq4Stats;
