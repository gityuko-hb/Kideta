//! Index and quantization type enumerations.
//!
//! This module provides enums for specifying the type of index to build
//! and the quantization scheme to use for compressing vectors.
//!
//! # Quantization Types
//!
//! Quantization reduces memory footprint by compressing vectors. See [`QuantizationType`].
//!
//! | Type | Compression | Speed | Recall | Use Case |
//! |------|------------|-------|--------|----------|
//! | [`None`](QuantizationType::None) | 1x (none) | Slowest | Highest | <100K vectors |
//! | [`Sq8`](QuantizationType::Sq8) | 4x | Medium | Good | General purpose |
//! | [`Sq4`](QuantizationType::Sq4) | 8x | Fast | Lower | Memory-constrained |
//! | [`Binary`](QuantizationType::Binary) | 32x | Fastest | Lowest | Binary fingerprints |
//! | [`PQ`](QuantizationType::PQ) | 4-64x | Fast | Good | Billion-scale |
//!
//! # Index Types
//!
//! Index types determine the ANN algorithm used for search. See [`IndexType`].
//!
//! | Type | Build Speed | Search Speed | Memory | Best For |
//! |------|-------------|--------------|--------|----------|
//! | [`Flat`](IndexType::Flat) | O(n) | O(n) | High | <10K vectors, benchmarks |
//! | [`Hnsw`](IndexType::Hnsw) | O(n log n) | O(log n) | High | General ANN |
//! | [`Ivf`](IndexType::Ivf) | O(n log n) | O(n/k) | Medium | Clustered data |
//! | [`IvfPQ`](IndexType::IvfPQ) | O(n log n) | O(n/k) | Low | Billion-scale |
//! | [`Vamana`](IndexType::Vamana) | O(n log n) | O(log n) | Medium | Disk ANN |
//! | [`Paged`](IndexType::Paged) | Variable | Variable | Medium | Hierarchical docs |
//!
//! ## Usage Examples
//!
//! ### Compression Ratio
//!
//! ```
//! use kideta_core::enums::QuantizationType;
//!
//! // Check compression ratios
//! assert_eq!(QuantizationType::None.compression_ratio(), 1.0);
//! assert_eq!(QuantizationType::Sq8.compression_ratio(), 4.0);
//! assert_eq!(QuantizationType::Sq4.compression_ratio(), 8.0);
//! assert_eq!(QuantizationType::Binary.compression_ratio(), 32.0);
//!
//! // Calculate memory savings for 1M vectors at 768 dimensions
//! let vectors = 1_000_000_f64;
//! let dim = 768_f64;
//! let bytes_per_float = 4_f64;
//!
//! let uncompressed = vectors * dim * bytes_per_float;
//! let sq8 = uncompressed / QuantizationType::Sq8.compression_ratio();
//! let sq4 = uncompressed / QuantizationType::Sq4.compression_ratio();
//! let binary = uncompressed / QuantizationType::Binary.compression_ratio();
//!
//! println!("Uncompressed: {:.2} GB", uncompressed / 1e9);
//! println!("SQ8: {:.2} GB", sq8 / 1e9);
//! println!("SQ4: {:.2} GB", sq4 / 1e9);
//! println!("Binary: {:.2} GB", binary / 1e9);
//! ```
//!
//! ### Index Quantization Support
//!
//! ```
//! use kideta_core::enums::IndexType;
//!
//! // Check which indexes support quantization
//! assert!(!IndexType::Flat.supports_quantization());
//! assert!(IndexType::Hnsw.supports_quantization());
//! assert!(IndexType::Ivf.supports_quantization());
//! assert!(IndexType::IvfPQ.supports_quantization());
//! assert!(IndexType::Vamana.supports_quantization());
//! assert!(!IndexType::Paged.supports_quantization());
//! ```
//!
//! ### Serialization
//!
//! ```
//! use kideta_core::enums::{QuantizationType, IndexType};
//!
//! // JSON serialization/deserialization
//! let qt = QuantizationType::Sq8;
//! let json = serde_json::to_string(&qt).unwrap();
//! assert_eq!(json, "\"sq8\"");
//! let parsed: QuantizationType = serde_json::from_str(&json).unwrap();
//! assert_eq!(parsed, qt);
//!
//! let it = IndexType::Hnsw;
//! let json = serde_json::to_string(&it).unwrap();
//! assert_eq!(json, "\"hnsw\"");
//! ```

use serde::{Deserialize, Serialize};

/// Quantization type for compressing vectors.
///
/// Quantization trades memory for speed and is essential for large-scale
/// deployments. The choice depends on your memory constraints and
/// quality requirements.
///
/// # No Quantization
///
/// Using `None` keeps vectors in full float32 precision. This provides
/// the best recall but uses the most memory (4 bytes per dimension).
///
/// ```
/// use kideta_core::enums::QuantizationType;
///
/// let qt: QuantizationType = Default::default();
/// assert_eq!(qt, QuantizationType::None);
/// ```
///
/// # Scalar Quantization (SQ8)
///
/// SQ8 maps each f32 dimension to a u8 (1 byte), achieving 4x compression.
/// Each dimension is scaled from [-127, 127] to [0, 255].
///
/// SQ8 provides good speedup with minimal recall loss (typically >95%).
///
/// # SQ4
///
/// SQ4 packs 2 dimensions into 1 byte, achieving 8x compression.
/// Requires careful handling of the narrow range [-7, 7] per value.
///
/// # Binary Quantization
///
/// Binary quantization reduces each dimension to 1 bit (sign of the value),
/// achieving 32x compression. This is extremely fast but can significantly
/// reduce recall unless vectors are pre-normalized.
///
/// # Product Quantization (PQ)
///
/// PQ divides vectors into M sub-spaces and quantizes each independently
/// using a codebook. Compression ratio is M * 256 / original_bytes.
///
/// PQ is the go-to choice for billion-scale deployments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum QuantizationType {
    /// No quantization — full float32 precision.
    #[default]
    None,

    /// Scalar quantization to 8-bit unsigned integers (4x compression).
    Sq8,

    /// Scalar quantization to 4-bit signed integers (8x compression).
    Sq4,

    /// Binary quantization — 1 bit per dimension (32x compression).
    Binary,

    /// Product quantization with learned codebooks.
    PQ,
}

impl QuantizationType {
    /// Returns the string representation of this quantization type.
    pub fn as_str(&self) -> &'static str {
        match self {
            QuantizationType::None => "none",
            QuantizationType::Sq8 => "sq8",
            QuantizationType::Sq4 => "sq4",
            QuantizationType::Binary => "binary",
            QuantizationType::PQ => "pq",
        }
    }

    /// Returns the compression ratio achieved by this quantization type.
    ///
    /// This is an approximation; actual compression depends on the vector
    /// dimension and PQ codebook size.
    pub fn compression_ratio(&self) -> f64 {
        match self {
            QuantizationType::None => 1.0,
            QuantizationType::Sq8 => 4.0,
            QuantizationType::Sq4 => 8.0,
            QuantizationType::Binary => 32.0,
            QuantizationType::PQ => 4.0, // Approximate; varies by configuration
        }
    }
}

impl std::fmt::Display for QuantizationType {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Index type for approximate nearest neighbor search.
///
/// Each index type implements a different ANN algorithm with its own
/// tradeoffs. Choose based on your dataset size, latency requirements,
/// and memory constraints.
///
/// # Flat Index
///
/// The flat (brute-force) index performs linear scan through all vectors.
/// While slow, it provides 100% recall and is useful for:
/// - Small datasets (<10K vectors)
/// - Benchmarking other index types
/// - Scenarios where recall is critical
///
/// # HNSW
///
/// Hierarchical Navigable Small World (HNSW) is the most popular ANN algorithm.
/// It builds a multi-layer graph with skip connections, enabling O(log n) search.
///
/// HNSW provides excellent recall/speed tradeoff for in-memory datasets.
/// It's the default choice for most use cases.
///
/// # IVF (Inverted File Index)
///
/// IVF partitions vectors into k clusters using k-means. Search scans only
/// the nearest clusters (nprobe parameter).
///
/// IVF is efficient when vectors are naturally clustered and works well
/// with quantization.
///
/// # IVF-PQ
///
/// Combines IVF clustering with PQ compression. This is the standard
/// configuration for billion-scale vector search (e.g., FAISS IVFPQ).
///
/// # Vamana / DiskANN
///
/// Vamana is designed for disk-based ANN search. It minimizes random
/// I/O by building a graph with predictable access patterns.
///
/// Ideal for datasets that don't fit in RAM but need fast queries.
///
/// # Paged Index
///
/// PagedIndex is designed for hierarchical documents (PDFs, articles).
/// It navigates document structure to find relevant passages efficiently.
///
/// Best for RAG (Retrieval-Augmented Generation) applications.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum IndexType {
    /// Flat (brute-force) index — linear scan.
    ///
    /// 100% recall, O(n) search, no memory overhead.
    Flat,

    /// Hierarchical Navigable Small World graph.
    #[default]
    Hnsw,

    /// Inverted File Index (IVF).
    Ivf,

    /// IVF with Product Quantization (IVF-PQ).
    IvfPQ,

    /// Vamana / DiskANN — disk-optimized graph index.
    Vamana,

    /// Paged hierarchical index for documents.
    Paged,
}

impl IndexType {
    /// Returns the string representation of this index type.
    pub fn as_str(&self) -> &'static str {
        match self {
            IndexType::Hnsw => "hnsw",
            IndexType::Vamana => "vamana",
            IndexType::Ivf => "ivf",
            IndexType::IvfPQ => "ivf_pq",
            IndexType::Flat => "flat",
            IndexType::Paged => "paged",
        }
    }

    /// Returns whether this index type supports quantization.
    pub fn supports_quantization(&self) -> bool {
        matches!(
            self,
            IndexType::Ivf | IndexType::IvfPQ | IndexType::Vamana | IndexType::Hnsw
        )
    }
}

impl std::fmt::Display for IndexType {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type_as_str() {
        assert_eq!(QuantizationType::None.as_str(), "none");
        assert_eq!(QuantizationType::Sq8.as_str(), "sq8");
        assert_eq!(QuantizationType::Sq4.as_str(), "sq4");
        assert_eq!(QuantizationType::Binary.as_str(), "binary");
        assert_eq!(QuantizationType::PQ.as_str(), "pq");
    }

    #[test]
    fn test_index_type_as_str() {
        assert_eq!(IndexType::Hnsw.as_str(), "hnsw");
        assert_eq!(IndexType::Vamana.as_str(), "vamana");
        assert_eq!(IndexType::Ivf.as_str(), "ivf");
        assert_eq!(IndexType::IvfPQ.as_str(), "ivf_pq");
        assert_eq!(IndexType::Flat.as_str(), "flat");
        assert_eq!(IndexType::Paged.as_str(), "paged");
    }

    #[test]
    fn test_quantization_type_default() {
        let qt: QuantizationType = Default::default();
        assert_eq!(qt, QuantizationType::None);
    }

    #[test]
    fn test_index_type_default() {
        let it: IndexType = Default::default();
        assert_eq!(it, IndexType::Hnsw);
    }

    #[test]
    fn test_quantization_type_serde() {
        let qt = QuantizationType::Sq8;
        let serialized = serde_json::to_string(&qt).unwrap();
        assert_eq!(serialized, "\"sq8\"");
        let deserialized: QuantizationType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, qt);
    }

    #[test]
    fn test_index_type_serde() {
        let it = IndexType::Hnsw;
        let serialized = serde_json::to_string(&it).unwrap();
        assert_eq!(serialized, "\"hnsw\"");
        let deserialized: IndexType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, it);
    }

    #[test]
    fn test_index_type_supports_quantization() {
        assert!(!IndexType::Flat.supports_quantization());
        assert!(IndexType::Hnsw.supports_quantization());
        assert!(IndexType::Ivf.supports_quantization());
        assert!(IndexType::IvfPQ.supports_quantization());
        assert!(IndexType::Vamana.supports_quantization());
        assert!(!IndexType::Paged.supports_quantization());
    }

    #[test]
    fn test_compression_ratios() {
        assert_eq!(QuantizationType::None.compression_ratio(), 1.0);
        assert_eq!(QuantizationType::Sq8.compression_ratio(), 4.0);
        assert_eq!(QuantizationType::Sq4.compression_ratio(), 8.0);
        assert_eq!(QuantizationType::Binary.compression_ratio(), 32.0);
    }
}
