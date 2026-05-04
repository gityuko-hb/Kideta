//! Vector record types for storing and transporting vectors.
//!
//! This module provides types for representing vectors and their associated
//! metadata in Kideta.
//!
//! # VectorRecord
//!
//! A [`VectorRecord`] is the fundamental unit of storage in Kideta. It contains:
//!
//! - A unique vector ID
//! - One or more named vectors (for multi-vector collections)
//! - Arbitrary metadata (payload)
//! - Version information for MVCC
//! - Timestamps for auditing
//!
//! # SparseVector
//!
//! A [`SparseVector`] represents a vector where most dimensions are zero.
//! Only non-zero indices and values are stored, saving memory for sparse data
//! like TF-IDF or BM25 vectors.
//!
//! # Example: Creating a VectorRecord
//!
//! ```
//! use kideta_core::record::VectorRecord;
//! use kideta_core::payload::{Payload, PayloadValue};
//! use kideta_core::types::VectorId;
//! use std::collections::HashMap;
//!
//! let mut payload_data = HashMap::new();
//! payload_data.insert("title".to_string(), PayloadValue::Str("Example".to_string()));
//! payload_data.insert("score".to_string(), PayloadValue::Float(0.95));
//!
//! let record = VectorRecord::new(VectorId::new(1))
//!     .add_vector("embedding", vec![0.1, 0.2, 0.3, 0.4])
//!     .set_payload(Payload::from(payload_data));
//!
//! assert_eq!(record.num_vectors(), 1);
//! assert_eq!(record.get_vector("embedding"), Some(&vec![0.1, 0.2, 0.3, 0.4]));
//! ```
//!
//! # Example: Using SparseVector
//!
//! ```
//! use kideta_core::record::SparseVector;
//!
//! // A sparse vector with 3 non-zero values in a 1000-dim space
//! let sparse = SparseVector::new(
//!     vec![10, 25, 300],    // indices
//!     vec![1.5, 2.0, 0.5]  // values at those indices
//! );
//!
//! assert_eq!(sparse.len(), 3);        // 3 non-zero elements
//! assert_eq!(sparse.dimension(), 301); // max index + 1
//! ```

use crate::payload::Payload;
use crate::types::VectorId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A record containing one or more vectors and their metadata.
///
/// VectorRecord is the fundamental unit of storage in Kideta. Each record:
/// - Has a unique [`VectorId`]
/// - Contains one or more named vectors (supports multi-vector collections)
/// - Has an optional payload for metadata
/// - Has version info for MVCC
/// - Has timestamps for auditing
///
/// # Multi-Vector Records
///
/// Collections can have multiple vector fields (e.g., text + image embeddings).
/// Each vector is stored under its field name:
///
/// ```
/// use kideta_core::record::VectorRecord;
/// use kideta_core::types::VectorId;
///
/// let record = VectorRecord::new(VectorId::new(1))
///     .add_vector("text", vec![0.1, 0.2, 0.3])
///     .add_vector("image", vec![0.4, 0.5, 0.6]);
///
/// assert_eq!(record.num_vectors(), 2);
/// assert!(record.get_vector("text").is_some());
/// assert!(record.get_vector("image").is_some());
/// ```
///
/// # Timestamps and Versioning
///
/// Records track creation and update times (Unix timestamps) and a version
/// number for optimistic concurrency control:
///
/// ```
/// use kideta_core::record::VectorRecord;
/// use kideta_core::types::VectorId;
///
/// let record = VectorRecord::new(VectorId::new(1));
///
/// // Version starts at 1
/// assert_eq!(record.version, 1);
///
/// // Timestamps are set automatically
/// assert!(record.created_at > 0);
/// assert_eq!(record.created_at, record.updated_at);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    /// Unique identifier for this vector.
    pub id: VectorId,

    /// Named vectors for this record (one per vector field).
    /// The key is the vector field name.
    pub vectors: HashMap<String, Vec<f32>>,

    /// Arbitrary metadata associated with this vector.
    pub payload: Payload,

    /// Version number for MVCC (starts at 1).
    pub version: u64,

    /// Unix timestamp when this record was created.
    pub created_at: u64,

    /// Unix timestamp when this record was last updated.
    pub updated_at: u64,
}

impl VectorRecord {
    /// Creates a new VectorRecord with the given ID.
    ///
    /// The record is created with:
    /// - Empty vectors (use [`add_vector`](VectorRecord::add_vector) to add)
    /// - Empty payload (use [`set_payload`](VectorRecord::set_payload) to set)
    /// - Version 1
    /// - Current timestamp for created_at and updated_at
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::VectorRecord;
    /// use kideta_core::types::VectorId;
    ///
    /// let record = VectorRecord::new(VectorId::new(42));
    /// assert_eq!(record.id.as_u64(), 42);
    /// assert!(record.vectors.is_empty());
    /// assert_eq!(record.version, 1);
    /// ```
    pub fn new(id: VectorId) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        VectorRecord {
            id,
            vectors: HashMap::new(),
            payload: Payload::new(),
            version: 1,
            created_at: now,
            updated_at: now,
        }
    }

    /// Adds a named vector to this record.
    ///
    /// Returns a new `VectorRecord` with the vector added.
    /// This allows for a fluent builder pattern.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::VectorRecord;
    /// use kideta_core::types::VectorId;
    ///
    /// let record = VectorRecord::new(VectorId::new(1))
    ///     .add_vector("embedding", vec![0.1, 0.2, 0.3]);
    ///
    /// assert_eq!(record.get_vector("embedding"), Some(&vec![0.1, 0.2, 0.3]));
    /// ```
    pub fn add_vector(
        mut self,
        name: impl Into<String>,
        vector: Vec<f32>,
    ) -> Self {
        self.vectors.insert(name.into(), vector);
        self
    }

    /// Sets the payload for this record.
    ///
    /// Returns a new `VectorRecord` with the payload set.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::VectorRecord;
    /// use kideta_core::types::VectorId;
    /// use kideta_core::payload::{Payload, PayloadValue};
    /// use std::collections::HashMap;
    ///
    /// let mut payload_data = HashMap::new();
    /// payload_data.insert("title".to_string(), PayloadValue::Str("Hello".to_string()));
    ///
    /// let record = VectorRecord::new(VectorId::new(1))
    ///     .set_payload(Payload::from(payload_data));
    ///
    /// assert_eq!(record.payload.get("title").and_then(|v| v.as_str()), Some("Hello"));
    /// ```
    pub fn set_payload(
        mut self,
        payload: Payload,
    ) -> Self {
        self.payload = payload;
        self
    }

    /// Gets a vector by field name.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::VectorRecord;
    /// use kideta_core::types::VectorId;
    ///
    /// let record = VectorRecord::new(VectorId::new(1))
    ///     .add_vector("embedding", vec![1.0, 2.0, 3.0]);
    ///
    /// assert_eq!(record.get_vector("embedding"), Some(&vec![1.0, 2.0, 3.0]));
    /// assert_eq!(record.get_vector("missing"), None);
    /// ```
    pub fn get_vector(
        &self,
        name: &str,
    ) -> Option<&Vec<f32>> {
        self.vectors.get(name)
    }

    /// Returns the number of vector fields in this record.
    pub fn num_vectors(&self) -> usize {
        self.vectors.len()
    }

    /// Returns the sum of all vector dimensions.
    ///
    /// This is useful for validating that all vectors match the
    /// collection schema dimensions.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::VectorRecord;
    /// use kideta_core::types::VectorId;
    ///
    /// let record = VectorRecord::new(VectorId::new(1))
    ///     .add_vector("text", vec![0.1; 768])
    ///     .add_vector("image", vec![0.2; 512]);
    ///
    /// assert_eq!(record.total_dimension(), 1280);
    /// ```
    pub fn total_dimension(&self) -> usize {
        self.vectors.values().map(|v| v.len()).sum()
    }
}

/// A sparse vector storing only non-zero indices and values.
///
/// SparseVector is designed for high-dimensional vectors where most
/// values are zero, such as:
/// - TF-IDF vectors
/// - BM25 vectors
/// - One-hot encoded features
///
/// By storing only non-zero indices and values, sparse vectors
/// can dramatically reduce memory usage for sparse data.
///
/// # Storage Format
///
/// - `indices`: Sorted list of dimension indices that have non-zero values
/// - `values`: The corresponding non-zero values at each index
///
/// Both vectors must have the same length.
///
/// # Example
///
/// ```
/// use kideta_core::record::SparseVector;
///
/// // A document with TF-IDF scores at specific terms
/// let doc = SparseVector::new(
///     vec![0, 5, 12, 45, 100],   // term indices
///     vec![0.5, 0.8, 0.3, 0.9, 0.1]  // TF-IDF scores
/// );
///
/// assert_eq!(doc.len(), 5);           // 5 non-zero terms
/// assert_eq!(doc.dimension(), 101);    // highest index + 1
/// ```
///
/// # Memory Savings
///
/// For a 100,000-dimension vector with only 10 non-zero values:
/// - Dense: 100,000 × 4 bytes = 400 KB
/// - Sparse: 10 × 4 bytes (indices) + 10 × 4 bytes (values) = 80 bytes
///
/// That's a 5000x reduction in memory usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Sorted indices of non-zero elements (must be in ascending order).
    pub indices: Vec<u32>,

    /// Values at the corresponding indices.
    pub values: Vec<f32>,

    /// Explicit dimension of the full vector space.
    ///
    /// When `None`, dimension is computed as `max(indices) + 1`.
    /// When `Some(d)`, `d` is used directly (preserves original dimension
    /// from `from_dense` where the full length may exceed max index + 1).
    dimension: Option<usize>,
}

impl PartialEq for SparseVector {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.indices == other.indices && self.values == other.values
    }
}

impl SparseVector {
    /// Creates a new sparse vector from indices and values.
    ///
    /// Both vectors must have the same length. The indices should be
    /// sorted in ascending order for efficient operations.
    /// Dimension will be computed as `max(indices) + 1`.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::SparseVector;
    ///
    /// let sparse = SparseVector::new(
    ///     vec![1, 5, 10],
    ///     vec![1.0, 2.0, 3.0]
    /// );
    /// ```
    pub fn new(
        indices: Vec<u32>,
        values: Vec<f32>,
    ) -> Self {
        SparseVector {
            indices,
            values,
            dimension: None,
        }
    }

    /// Creates a new sparse vector with an explicit dimension.
    ///
    /// Use this when the dimension should be preserved exactly (e.g., from `from_dense`).
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::SparseVector;
    ///
    /// let sparse = SparseVector::with_dimension(
    ///     vec![1, 4],
    ///     vec![1.0, 0.5],
    ///     6,  // exact original dimension
    /// );
    /// assert_eq!(sparse.dimension(), 6);
    /// ```
    pub fn with_dimension(
        indices: Vec<u32>,
        values: Vec<f32>,
        dimension: usize,
    ) -> Self {
        SparseVector {
            indices,
            values,
            dimension: Some(dimension),
        }
    }

    /// Returns the dimension of the sparse vector.
    ///
    /// If an explicit dimension was set (via `with_dimension` or `from_dense`),
    /// that value is returned. Otherwise, computed as `max(indices) + 1`,
    /// or 0 if the vector is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::SparseVector;
    ///
    /// let sparse = SparseVector::new(vec![10, 20, 30], vec![1.0, 2.0, 3.0]);
    /// assert_eq!(sparse.dimension(), 31);  // max index (30) + 1
    /// ```
    pub fn dimension(&self) -> usize {
        self.dimension.unwrap_or_else(|| {
            self.indices
                .iter()
                .max()
                .copied()
                .map(|i| i as usize + 1)
                .unwrap_or(0)
        })
    }

    /// Returns the number of non-zero elements in this sparse vector.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::SparseVector;
    ///
    /// let sparse = SparseVector::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]);
    /// assert_eq!(sparse.len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns `true` if this sparse vector has no non-zero elements.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Computes the dot product of this sparse vector with another.
    ///
    /// Both vectors must have sorted indices in ascending order.
    /// Time complexity: O(n + m) where n = self.len() and m = other.len().
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::SparseVector;
    ///
    /// let a = SparseVector::new(vec![1, 3, 5], vec![1.0, 2.0, 3.0]);
    /// let b = SparseVector::new(vec![1, 2, 3], vec![0.5, 1.0, 1.5]);
    ///
    /// // dot = 1.0*0.5 + 2.0*1.5 = 0.5 + 3.0 = 3.5
    /// assert!((a.dot(&b) - 3.5).abs() < 1e-6);
    /// ```
    pub fn dot(
        &self,
        other: &SparseVector,
    ) -> f32 {
        let mut sum = 0.0f32;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    sum += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                },
                std::cmp::Ordering::Less => {
                    i += 1;
                },
                std::cmp::Ordering::Greater => {
                    j += 1;
                },
            }
        }

        sum
    }

    /// Computes the L2 (Euclidean) norm of this sparse vector.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::SparseVector;
    ///
    /// let v = SparseVector::new(vec![0, 2], vec![3.0, 4.0]);
    /// assert!((v.l2_norm() - 5.0).abs() < 1e-6);
    /// ```
    pub fn l2_norm(&self) -> f32 {
        self.values
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
    }

    /// Validates that this sparse vector is well-formed.
    ///
    /// Returns `Ok(())` if the vector is valid, or an error describing
    /// the problem.
    ///
    /// # Validation Rules
    ///
    /// - `indices` and `values` must have the same length
    /// - `indices` must be strictly ascending (no duplicates)
    /// - All indices must be non-negative
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::SparseVector;
    ///
    /// let valid = SparseVector::new(vec![1, 5, 10], vec![1.0, 2.0, 3.0]);
    /// assert!(valid.validate().is_ok());
    ///
    /// // Duplicate indices
    /// let invalid = SparseVector::new(vec![1, 1, 5], vec![1.0, 2.0, 3.0]);
    /// assert!(invalid.validate().is_err());
    /// ```
    pub fn validate(&self) -> crate::error::Result<()> {
        use crate::error::KidetaError;

        if self.indices.len() != self.values.len() {
            return Err(KidetaError::InvalidInput {
                reason: format!(
                    "SparseVector indices and values length mismatch: {} vs {}",
                    self.indices.len(),
                    self.values.len()
                ),
            });
        }

        for window in self.indices.windows(2) {
            if window[1] <= window[0] {
                return Err(KidetaError::InvalidInput {
                    reason: format!(
                        "SparseVector indices are not strictly ascending: indices[{}]={} >= indices[{}]={}",
                        window[0], window[0], window[1], window[1]
                    ),
                });
            }
        }

        for (i, idx) in self.indices.iter().enumerate() {
            if (*idx as i64) < 0 {
                return Err(KidetaError::InvalidInput {
                    reason: format!(
                        "SparseVector contains negative index at position {}: {}",
                        i, idx
                    ),
                });
            }
        }

        Ok(())
    }

    /// Creates a sparse vector from a dense f32 slice.
    ///
    /// Only non-zero values are stored, making this efficient for
    /// vectors with few non-zero elements. The original dimension is
    /// stored so `dimension()` returns the full space size.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::SparseVector;
    ///
    /// let dense = vec![0.0, 1.0, 0.0, 0.0, 0.5, 0.0];
    /// let sparse = SparseVector::from_dense(&dense);
    ///
    /// assert_eq!(sparse.len(), 2);
    /// assert_eq!(sparse.indices, &[1, 4]);
    /// assert_eq!(sparse.values, &[1.0, 0.5]);
    /// assert_eq!(sparse.dimension(), 6);
    /// ```
    pub fn from_dense(dense: &[f32]) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &v) in dense.iter().enumerate() {
            if v != 0.0 {
                indices.push(i as u32);
                values.push(v);
            }
        }

        SparseVector::with_dimension(indices, values, dense.len())
    }

    /// Creates a normalized copy of this sparse vector (L2 normalized).
    ///
    /// Returns `None` if the vector has zero norm.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::record::SparseVector;
    ///
    /// let v = SparseVector::new(vec![0, 1], vec![3.0, 4.0]);
    /// let normalized = v.normalize().unwrap();
    ///
    /// assert!((normalized.l2_norm() - 1.0).abs() < 1e-6);
    /// ```
    pub fn normalize(&self) -> Option<Self> {
        let norm = self.l2_norm();
        if norm == 0.0 || norm.is_nan() {
            return None;
        }

        let inv_norm = 1.0 / norm;
        Some(SparseVector {
            indices: self.indices.clone(),
            values: self.values.iter().map(|v| v * inv_norm).collect(),
            dimension: self.dimension,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_record_creation() {
        let record =
            VectorRecord::new(VectorId::new(1)).add_vector("embedding", vec![0.1, 0.2, 0.3]);

        assert_eq!(record.id.as_u64(), 1);
        assert_eq!(record.num_vectors(), 1);
        assert_eq!(record.get_vector("embedding"), Some(&vec![0.1, 0.2, 0.3]));
        assert_eq!(record.version, 1);
    }

    #[test]
    fn test_vector_record_dimension() {
        let record = VectorRecord::new(VectorId::new(1)).add_vector("embedding", vec![0.1; 768]);

        assert_eq!(record.total_dimension(), 768);
    }

    #[test]
    fn test_sparse_vector_creation() {
        let sparse = SparseVector::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]);
        assert_eq!(sparse.len(), 3);
        assert_eq!(sparse.dimension(), 11);
    }

    #[test]
    fn test_sparse_vector_empty() {
        let sparse: SparseVector = SparseVector::new(vec![], vec![]);
        assert!(sparse.is_empty());
        assert_eq!(sparse.dimension(), 0);
    }

    #[test]
    fn test_vector_record_serde() {
        let record = VectorRecord::new(VectorId::new(42)).add_vector("embedding", vec![0.1, 0.2]);

        let serialized = serde_json::to_string(&record).unwrap();
        let deserialized: VectorRecord = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.id, record.id);
    }

    #[test]
    fn test_sparse_vector_serde() {
        let sparse = SparseVector::new(vec![1, 5, 10], vec![1.0, 2.0, 3.0]);
        let serialized = serde_json::to_string(&sparse).unwrap();
        let deserialized: SparseVector = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, sparse);
    }

    #[test]
    fn test_sparse_vector_dot() {
        let a = SparseVector::new(vec![1, 3, 5], vec![1.0, 2.0, 3.0]);
        let b = SparseVector::new(vec![1, 2, 3], vec![0.5, 1.0, 1.5]);

        let dot = a.dot(&b);
        assert!((dot - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_dot_no_overlap() {
        let a = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]);
        let b = SparseVector::new(vec![4, 5, 6], vec![1.0, 1.0, 1.0]);

        assert!((a.dot(&b)).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_dot_full_overlap() {
        let a = SparseVector::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]);
        let b = SparseVector::new(vec![1, 2, 3], vec![1.0, 1.0, 1.0]);

        let dot = a.dot(&b);
        assert!((dot - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_l2_norm() {
        let v = SparseVector::new(vec![0, 2], vec![3.0, 4.0]);
        assert!((v.l2_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_l2_norm_empty() {
        let v: SparseVector = SparseVector::new(vec![], vec![]);
        assert!((v.l2_norm() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_vector_validate_ok() {
        let v = SparseVector::new(vec![1, 5, 10], vec![1.0, 2.0, 3.0]);
        assert!(v.validate().is_ok());
    }

    #[test]
    fn test_sparse_vector_validate_empty() {
        let v: SparseVector = SparseVector::new(vec![], vec![]);
        assert!(v.validate().is_ok());
    }

    #[test]
    fn test_sparse_vector_validate_duplicate_indices() {
        let v = SparseVector::new(vec![1, 1, 5], vec![1.0, 2.0, 3.0]);
        assert!(v.validate().is_err());
    }

    #[test]
    fn test_sparse_vector_validate_non_ascending() {
        let v = SparseVector::new(vec![1, 5, 3], vec![1.0, 2.0, 3.0]);
        assert!(v.validate().is_err());
    }

    #[test]
    fn test_sparse_vector_validate_length_mismatch() {
        let v = SparseVector::new(vec![1, 5], vec![1.0]);
        assert!(v.validate().is_err());
    }

    #[test]
    fn test_sparse_vector_from_dense() {
        let dense = vec![0.0, 1.0, 0.0, 0.0, 0.5, 0.0];
        let sparse = SparseVector::from_dense(&dense);

        assert_eq!(sparse.len(), 2);
        assert_eq!(sparse.indices, &[1, 4]);
        assert_eq!(sparse.values, &[1.0, 0.5]);
        assert_eq!(sparse.dimension(), 6);
    }

    #[test]
    fn test_sparse_vector_from_dense_all_zeros() {
        let dense = vec![0.0, 0.0, 0.0];
        let sparse = SparseVector::from_dense(&dense);

        assert!(sparse.is_empty());
        assert_eq!(sparse.dimension(), 3);
    }

    #[test]
    fn test_sparse_vector_normalize() {
        let v = SparseVector::new(vec![0, 2], vec![3.0, 4.0]);
        let norm = v.normalize().unwrap();

        assert!((norm.l2_norm() - 1.0).abs() < 1e-6);
        assert_eq!(norm.indices, v.indices);
    }

    #[test]
    fn test_sparse_vector_normalize_zero() {
        let v: SparseVector = SparseVector::new(vec![], vec![]);
        assert!(v.normalize().is_none());
    }
}
