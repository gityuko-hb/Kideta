//! Distance metrics for vector similarity search.
//!
//! This module defines [`DistanceMetric`], the type used to specify which
//! distance function should be used when searching or building indexes
//! in a collection.
//!
//! # Available Metrics
//!
//! | Metric | Description | Range | Use Case |
//! |--------|-------------|-------|----------|
//! | [`Cosine`](DistanceMetric::Cosine) | Cosine similarity | [-1, 1] | Text embeddings, normalized vectors |
//! | [`L2`](DistanceMetric::L2) | Euclidean (L2) distance | [0, ∞) | General purpose, images |
//! | [`DotProduct`](DistanceMetric::DotProduct) | Inner product | (-∞, ∞) | Unnormalized embeddings, MaxSIM |
//! | [`Hamming`](DistanceMetric::Hamming) | Hamming distance | [0, dim] | Binary vectors, fingerprints |
//!
//! ## Usage
//!
//! ```
//! use kideta_core::metric::DistanceMetric;
//!
//! // Get string representation
//! assert_eq!(DistanceMetric::Cosine.as_str(), "cosine");
//! assert_eq!(DistanceMetric::L2.as_str(), "l2");
//! assert_eq!(DistanceMetric::DotProduct.as_str(), "dot");
//! assert_eq!(DistanceMetric::Hamming.as_str(), "hamming");
//!
//! // Display format
//! assert_eq!(format!("{}", DistanceMetric::Cosine), "cosine");
//! assert_eq!(format!("{}", DistanceMetric::L2), "l2");
//! ```
//!
//! # Cosine Similarity
//!
//! Cosine similarity measures the angle between two vectors, ignoring magnitude:
//!
//! ```text
//! cosine(a, b) = (a · b) / (||a|| × ||b||)
//! ```
//!
//! Values close to 1 mean the vectors point in the same direction,
//! 0 means they are orthogonal, and -1 means they point in opposite directions.
//!
//! Cosine is ideal for:
//! - Text embeddings (word2vec, BERT, etc.)
//! - When vector magnitude doesn't matter
//! - Normalized deep learning features
//!
//! # L2 (Euclidean) Distance
//!
//! L2 distance is the straight-line distance between two points in Euclidean space:
//!
//! ```text
//! L2(a, b) = sqrt(sum((a_i - b_i)²))
//! ```
//!
//! L2 is ideal for:
//! - General purpose similarity
//! - Image features (e.g., CNN embeddings)
//! - When magnitude differences are meaningful
//!
//! # Dot Product
//!
//! The dot product (inner product) is:
//!
//! ```text
//! dot(a, b) = sum(a_i × b_i)
//! ```
//!
//! For normalized vectors, dot product is proportional to cosine similarity.
//! For unnormalized vectors, it encodes both direction and magnitude.
//!
//! Dot product is ideal for:
//! - Maximum Inner Product Search (MIPS)
//! - Unnormalized neural network embeddings
//! - When computation speed is critical (no sqrt needed)
//!
//! # Hamming Distance
//!
//! Hamming distance counts the number of positions where two vectors differ:
//!
//! ```text
//! hamming(a, b) = count(i where a_i != b_i)
//! ```
//!
//! Hamming is ideal for:
//! - Binary vectors (0/1 or bit-packed)
//! - Fingerprint comparison
//! - DNA sequence comparison
//!
//! # Example
//!
//! ```
//! use kideta_core::metric::DistanceMetric;
//!
//! // Create from string (useful for API parsing)
//! let cosine: DistanceMetric = serde_json::from_str("\"cosine\"").unwrap();
//! assert!(matches!(cosine, DistanceMetric::Cosine));
//!
//! // Convert back to string
//! let s = cosine.as_str();
//! assert_eq!(s, "cosine");
//!
//! // Display format
//! assert_eq!(format!("{}", cosine), "cosine");
//! ```
//!
//! #serde
//!
//! Serializes to/from JSON strings using the snake_case format:
//!
//! ```rust
//! use kideta_core::metric::DistanceMetric;
//!
//! let metric = DistanceMetric::L2;
//! let json = serde_json::to_string(&metric).unwrap();
//! assert_eq!(json, "\"l2\"");
//!
//! let parsed: DistanceMetric = serde_json::from_str(&json).unwrap();
//! assert_eq!(parsed, metric);
//! ```

use serde::{Deserialize, Serialize};

/// Specifies the distance metric to use for a collection or query.
///
/// See the [module-level documentation](self) for descriptions of each metric.
///
/// #serde
///
/// Serializes to/from JSON using snake_case strings:
/// - `"cosine"` → `Cosine`
/// - `"l2"` → `L2`
/// - `"dot"` → `DotProduct`
/// - `"hamming"` → `Hamming`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistanceMetric {
    /// Cosine similarity.
    ///
    /// Measures the cosine of the angle between two vectors.
    /// Range: [-1, 1], where 1 means identical direction.
    Cosine,

    /// Euclidean (L2) distance.
    ///
    /// The straight-line distance between two points.
    /// Range: [0, ∞), where 0 means identical vectors.
    L2,

    /// Dot product (inner product).
    ///
    /// The sum of element-wise products.
    /// Range: (-∞, ∞)
    DotProduct,

    /// Hamming distance.
    ///
    /// The number of positions where vectors differ.
    /// Range: [0, dimension], where 0 means identical.
    Hamming,
}

impl DistanceMetric {
    /// Returns the string representation of this metric.
    ///
    /// This string is used for JSON serialization and API responses.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::metric::DistanceMetric;
    ///
    /// assert_eq!(DistanceMetric::Cosine.as_str(), "cosine");
    /// assert_eq!(DistanceMetric::L2.as_str(), "l2");
    /// assert_eq!(DistanceMetric::DotProduct.as_str(), "dot");
    /// assert_eq!(DistanceMetric::Hamming.as_str(), "hamming");
    /// ```
    pub fn as_str(&self) -> &'static str {
        match self {
            DistanceMetric::Cosine => "cosine",
            DistanceMetric::L2 => "l2",
            DistanceMetric::DotProduct => "dot",
            DistanceMetric::Hamming => "hamming",
        }
    }
}

impl std::fmt::Display for DistanceMetric {
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
    fn test_distance_metric_as_str() {
        assert_eq!(DistanceMetric::Cosine.as_str(), "cosine");
        assert_eq!(DistanceMetric::L2.as_str(), "l2");
        assert_eq!(DistanceMetric::DotProduct.as_str(), "dot");
        assert_eq!(DistanceMetric::Hamming.as_str(), "hamming");
    }

    #[test]
    fn test_distance_metric_display() {
        assert_eq!(format!("{}", DistanceMetric::L2), "l2");
        assert_eq!(format!("{}", DistanceMetric::Cosine), "cosine");
        assert_eq!(format!("{}", DistanceMetric::DotProduct), "dot");
        assert_eq!(format!("{}", DistanceMetric::Hamming), "hamming");
    }

    #[test]
    fn test_distance_metric_serde() {
        let metric = DistanceMetric::Cosine;
        let serialized = serde_json::to_string(&metric).unwrap();
        assert_eq!(serialized, "\"cosine\"");
        let deserialized: DistanceMetric = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, metric);
    }

    #[test]
    fn test_all_metrics_serde_roundtrip() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::L2,
            DistanceMetric::DotProduct,
            DistanceMetric::Hamming,
        ] {
            let json = serde_json::to_string(&metric).unwrap();
            let parsed: DistanceMetric = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, metric, "Roundtrip failed for {:?}", metric);
        }
    }
}
