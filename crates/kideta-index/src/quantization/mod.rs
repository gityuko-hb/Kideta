//! Quantization module — lossy vector compression for memory reduction and fast approximate search.
//!
//! # Overview
//!
//! Quantization compresses floating-point vectors into compact byte representations,
//! enabling 4×–32× memory reduction with acceptable recall loss. All quantization
//! types support two-phase search: fast approximate distance on compressed vectors,
//! followed by full-precision rescore of top candidates.
//!
//! # Types
//!
//! | Type | Compression | Speed | Recall | Best For |
//! |------|-------------|-------|--------|----------|
//! | SQ8 | 4× (f32→u8) | Fast | High | General purpose |
//! | Binary | 32× (f32→1bit) | Very Fast | Medium | Normalized embeddings |
//! | PQ | ~4–16× configurable | Fastest | High | High-dim (768d+) |
//!
//! # Usage
//!
//! ```ignore
//! use kideta_index::quantization::{QuantizationConfig, QuantizedStorage, Sq8Stats, PqTrainer};
//!
//! // SQ8: train on vectors, encode, search with rescore
//! let stats = Sq8Stats::compute(vectors.iter().map(|v| v.as_slice()).collect::<Vec<_>>());
//! let config = stats.into_config();
//! let storage = QuantizedStorage::new(vectors, &config);
//!
//! // Search: phase 1 (approx) + phase 2 (rescore top candidates)
//! let candidates = storage.approx_search_candidates(&query, k * 10);
//! let results = rescore_top_candidates(candidates, &query, k);
//! ```
//!
//! # Design
//!
//! - **QuantizationConfig** is serializable (serde) and stored in segment metadata
//! - **QuantizedStorage** stores compressed codes and provides approx_distance()
//! - Training is done offline (see `Sq8Stats`, `PqTrainer`) before indexing
//! - Integration via `VectorIndex::search_quantized()` on FlatIndex and HnswIndex

pub mod binary;
pub mod config;
pub mod pq;
pub mod sq4;
pub mod sq8;
pub mod storage;

pub use config::{BinaryConfig, PqConfig, QuantizationConfig, Sq4Config, Sq8Config};
pub use pq::PqTrainer;
pub use sq4::Sq4Stats;
pub use sq8::Sq8Stats;
pub use storage::{QuantizedStorage, encode_vectors};
