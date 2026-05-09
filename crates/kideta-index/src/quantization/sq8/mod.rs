//! Scalar Quantization 8-bit (SQ8) — 4× compression.
//!
//! Each dimension is independently quantized to u8 [0, 255] using per-dimension
//! min/max scaling. This is the simplest quantization scheme — no training
//! beyond min/max statistics.
//!
//! # Algorithm
//!
//! ```ignore
//! scale[i]  = 255.0 / (max[i] - min[i])
//! offset[i] = -min[i]
//! code[i]   = round((value[i] + offset[i]) * scale[i]) as u8
//! ```
//!
//! # Approximate Distance
//!
//! To avoid full decode, quantize query the same way:
//!
//! ```ignore
//! q[i] = round((query[i] + offset[i]) * scale[i])
//! dist² ≈ Σ (q[i] - code[i])²
//! ```
//!
//! This is much faster than decoding all vectors to f32 and computing exact L2.

pub mod ops;
pub mod trainer;

pub use ops::Sq8Ops;
pub use trainer::Sq8Stats;
