//! Schema definitions for collections.
//!
//! This module defines field types for describing the structure of
//! collections. Each collection has vector fields (for embeddings) and
//! scalar fields (for filtering metadata).
//!
//! # Vector Fields
//!
//! Vector fields contain the dense embeddings that will be indexed and
//! searched. Each vector field has its own dimension and distance metric.
//!
//! # Scalar Fields
//!
//! Scalar fields contain typed metadata that can be used for filtering
//! during search. They are not indexed for similarity search but enable
//! efficient filtered queries.
//!
//! # Example: Creating a Multi-Vector Collection
//!
//! ```
//! use kideta_core::schema::{ScalarField, ScalarFieldType, VectorField};
//! use kideta_core::metric::DistanceMetric;
//! use kideta_core::enums::{IndexType, QuantizationType};
//!
//! // Create a vector field for text embeddings
//! let text_embedding = VectorField::new("text", 768, DistanceMetric::Cosine);
//!
//! // Create a vector field for image embeddings with PQ
//! let image_embedding = VectorField::new("image", 512, DistanceMetric::L2)
//!     .with_index_type(IndexType::IvfPQ)
//!     .with_quantization(QuantizationType::PQ);
//!
//! // Create scalar fields for filtering
//! let category = ScalarField::new("category", ScalarFieldType::Str).indexed(true);
//! let price = ScalarField::new("price", ScalarFieldType::Float).indexed(true);
//! let in_stock = ScalarField::new("in_stock", ScalarFieldType::Bool).indexed(true);
//! ```

use crate::enums::{IndexType, QuantizationType};
use crate::metric::DistanceMetric;
use serde::{Deserialize, Serialize};

/// A field containing vector embeddings.
///
/// Each vector field has:
/// - A unique name within the collection
/// - A dimension (number of floats)
/// - A distance metric for similarity search
/// - An index type for ANN search
/// - A quantization type for compression
///
/// # Creating a Vector Field
///
/// ```
/// use kideta_core::schema::VectorField;
/// use kideta_core::metric::DistanceMetric;
///
/// // Simple creation with defaults
/// let field = VectorField::new("embedding", 768, DistanceMetric::Cosine);
/// assert_eq!(field.dimension, 768);
/// assert_eq!(field.metric, DistanceMetric::Cosine);
/// ```
///
/// # Customizing Index and Quantization
///
/// ```
/// use kideta_core::schema::VectorField;
/// use kideta_core::metric::DistanceMetric;
/// use kideta_core::enums::{IndexType, QuantizationType};
///
/// let field = VectorField::new("embedding", 768, DistanceMetric::L2)
///     .with_index_type(IndexType::IvfPQ)
///     .with_quantization(QuantizationType::PQ);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorField {
    /// Unique name identifying this vector field within a collection.
    pub name: String,

    /// The dimensionality of the vectors (number of floats per vector).
    pub dimension: usize,

    /// The distance metric to use for similarity search.
    pub metric: DistanceMetric,

    /// The ANN index type to use (defaults to HNSW).
    pub index_type: IndexType,

    /// The quantization scheme to use (defaults to None).
    pub quantization: QuantizationType,
}

impl VectorField {
    /// Creates a new vector field with default index and quantization settings.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for this field within the collection
    /// * `dimension` - Number of dimensions in each vector
    /// * `metric` - Distance metric for similarity search
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::schema::VectorField;
    /// use kideta_core::metric::DistanceMetric;
    ///
    /// let field = VectorField::new("embedding", 768, DistanceMetric::Cosine);
    /// ```
    pub fn new(
        name: impl Into<String>,
        dimension: usize,
        metric: DistanceMetric,
    ) -> Self {
        VectorField {
            name: name.into(),
            dimension,
            metric,
            index_type: IndexType::default(),
            quantization: QuantizationType::default(),
        }
    }

    /// Sets the index type for this vector field.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::schema::VectorField;
    /// use kideta_core::metric::DistanceMetric;
    /// use kideta_core::enums::IndexType;
    ///
    /// let field = VectorField::new("embedding", 768, DistanceMetric::Cosine)
    ///     .with_index_type(IndexType::Ivf);
    /// ```
    pub fn with_index_type(
        mut self,
        index_type: IndexType,
    ) -> Self {
        self.index_type = index_type;
        self
    }

    /// Sets the quantization type for this vector field.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::schema::VectorField;
    /// use kideta_core::metric::DistanceMetric;
    /// use kideta_core::enums::QuantizationType;
    ///
    /// let field = VectorField::new("embedding", 768, DistanceMetric::Cosine)
    ///     .with_quantization(QuantizationType::PQ);
    /// ```
    pub fn with_quantization(
        mut self,
        quantization: QuantizationType,
    ) -> Self {
        self.quantization = quantization;
        self
    }
}

/// Data type for scalar fields.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarFieldType {
    /// UTF-8 string values.
    Str,
    /// 64-bit signed integer values.
    Int,
    /// 64-bit floating point values.
    Float,
    /// Boolean values.
    Bool,
}

/// Index type for scalar fields.
///
/// Determines which index implementation is used for filtering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ScalarIndexType {
    /// Auto-select based on data type (default).
    #[default]
    Auto,
    /// Sorted array index for integers/floats.
    SortedArray,
    /// B-tree index for frequent updates.
    BTree,
    /// Hash index for exact string match.
    Hash,
    /// Full-text inverted index.
    FullText,
    /// Geohash-based spatial index.
    Geo,
}

impl ScalarFieldType {
    /// Returns the string representation of this scalar field type.
    pub fn as_str(&self) -> &'static str {
        match self {
            ScalarFieldType::Str => "str",
            ScalarFieldType::Int => "int",
            ScalarFieldType::Float => "float",
            ScalarFieldType::Bool => "bool",
        }
    }
}

/// A scalar metadata field for filtering.
///
/// Scalar fields store typed metadata that can be used to filter
/// search results. They are indexed for efficient filtering but
/// are not used for vector similarity search.
///
/// # Creating a Scalar Field
///
/// ```
/// use kideta_core::schema::{ScalarField, ScalarFieldType};
///
/// // Basic field (not indexed)
/// let category = ScalarField::new("category", ScalarFieldType::Str);
///
/// // Indexed field (for filtering)
/// let category_indexed = ScalarField::new("category", ScalarFieldType::Str).indexed(true);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScalarField {
    /// Unique name identifying this field within the collection.
    pub name: String,

    /// The data type of values in this field.
    pub data_type: ScalarFieldType,

    /// Whether this field is indexed for filtering.
    ///
    /// Indexed fields can be used in filter expressions during search.
    /// Non-indexed fields are stored but cannot be used for filtering.
    pub indexed: bool,

    /// The index type to use for this field.
    ///
    /// Only used when `indexed` is true. Default is Auto which selects
    /// based on the data type.
    #[serde(default)]
    pub index_type: ScalarIndexType,
}

impl ScalarField {
    /// Creates a new scalar field with the given name and data type.
    ///
    /// The field is not indexed by default.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::schema::{ScalarField, ScalarFieldType};
    ///
    /// let field = ScalarField::new("price", ScalarFieldType::Float);
    /// assert!(!field.indexed);
    /// ```
    pub fn new(
        name: impl Into<String>,
        data_type: ScalarFieldType,
    ) -> Self {
        ScalarField {
            name: name.into(),
            data_type,
            indexed: false,
            index_type: ScalarIndexType::default(),
        }
    }

    /// Sets whether this field should be indexed for filtering.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::schema::{ScalarField, ScalarFieldType};
    ///
    /// let field = ScalarField::new("category", ScalarFieldType::Str).indexed(true);
    /// assert!(field.indexed);
    /// ```
    pub fn indexed(
        mut self,
        indexed: bool,
    ) -> Self {
        self.indexed = indexed;
        self
    }

    /// Sets the index type for this field.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::schema::{ScalarField, ScalarFieldType, ScalarIndexType};
    ///
    /// let field = ScalarField::new("price", ScalarFieldType::Float)
    ///     .indexed(true)
    ///     .index_type(ScalarIndexType::SortedArray);
    /// assert!(field.indexed);
    /// ```
    pub fn index_type(
        mut self,
        index_type: ScalarIndexType,
    ) -> Self {
        self.index_type = index_type;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_field_creation() {
        let field = VectorField::new("embedding", 768, DistanceMetric::Cosine);
        assert_eq!(field.name, "embedding");
        assert_eq!(field.dimension, 768);
        assert_eq!(field.metric, DistanceMetric::Cosine);
        assert_eq!(field.index_type, IndexType::Hnsw);
        assert_eq!(field.quantization, QuantizationType::None);
    }

    #[test]
    fn test_vector_field_with_options() {
        let field = VectorField::new("embedding", 768, DistanceMetric::L2)
            .with_index_type(IndexType::IvfPQ)
            .with_quantization(QuantizationType::PQ);
        assert_eq!(field.index_type, IndexType::IvfPQ);
        assert_eq!(field.quantization, QuantizationType::PQ);
    }

    #[test]
    fn test_scalar_field_creation() {
        let field = ScalarField::new("age", ScalarFieldType::Int);
        assert_eq!(field.name, "age");
        assert_eq!(field.data_type, ScalarFieldType::Int);
        assert!(!field.indexed);
    }

    #[test]
    fn test_scalar_field_indexed() {
        let field = ScalarField::new("age", ScalarFieldType::Int).indexed(true);
        assert!(field.indexed);
    }

    #[test]
    fn test_vector_field_serde() {
        let field = VectorField::new("embedding", 768, DistanceMetric::Cosine);
        let serialized = serde_json::to_string(&field).unwrap();
        let deserialized: VectorField = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, field);
    }

    #[test]
    fn test_scalar_field_serde() {
        let field = ScalarField::new("category", ScalarFieldType::Str).indexed(true);
        let serialized = serde_json::to_string(&field).unwrap();
        let deserialized: ScalarField = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, field);
    }
}
