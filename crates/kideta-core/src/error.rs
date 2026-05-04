//! Centralized error types for Kideta vector database.
//!
//! This module defines [`KidetaError`], the canonical error type used throughout
//! the Kideta codebase. All operations that can fail return a `Result<T, KidetaError>`
//! rather than using raw `std::io::Error` or other low-level errors.
//!
//! # Error Categories
//!
//! The error types are organized into the following categories:
//!
//! - **IO Errors** ([`Io`](KidetaError::Io)) — Filesystem, network, and system I/O failures
//! - **Data Corruption** ([`Corruption`](KidetaError::Corruption)) — Detected data corruption
//! - **Not Found** ([`NotFound`](KidetaError::NotFound), [`VectorNotFound`](KidetaError::VectorNotFound), etc.) — Missing resources
//! - **Invalid Input** ([`InvalidInput`](KidetaError::InvalidInput), [`InvalidDimension`](KidetaError::InvalidDimension), etc.) — User-provided invalid data
//! - **Storage Errors** ([`Storage`](KidetaError::Storage), [`Wal`](KidetaError::Wal)) — Storage layer failures
//! - **Serialization** ([`Serialization`](KidetaError::Serialization), [`Deserialization`](KidetaError::Deserialization)) — Encoding/decoding failures
//! - **Internal Errors** ([`Internal`](KidetaError::Internal)) — Unexpected programming errors
//!
//! # Usage
//!
//! ```
//! use kideta_core::error::{KidetaError, Result};
//!
//! // Basic error creation
//! let err = KidetaError::NotFound {
//!     what: "collection 'my_collection'".to_string()
//! };
//! assert!(err.to_string().contains("my_collection"));
//!
//! // Dimension mismatch error
//! let err = KidetaError::DimensionMismatch {
//!     expected: 128,
//!     actual: 256,
//! };
//! assert!(err.to_string().contains("128"));
//! assert!(err.to_string().contains("256"));
//!
//! // Using with Result
//! fn example() -> Result<Vec<u8>> {
//!     Err(KidetaError::NotFound {
//!         what: "collection 'my_collection'".to_string()
//!     })
//! }
//! ```
//!
//! # Using `KidetaError` with `?`
//!
//! The standard library's `std::io::Error` can be automatically converted
//! into `KidetaError` using the `?` operator:
//!
//! ```
//! use kideta_core::error::{KidetaError, Result};
//! use std::fs;
//!
//! fn read_file(path: &str) -> Result<String> {
//!     // std::io::Error automatically converted to KidetaError::Io
//!     let contents = fs::read_to_string(path)?;
//!     Ok(contents)
//! }
//! ```
//!
//! # Display Format
//!
//! Each error variant has a human-readable display format:
//!
//! ```
//! use kideta_core::error::KidetaError;
//!
//! let err = KidetaError::DimensionMismatch {
//!     expected: 128,
//!     actual: 256,
//! };
//! assert!(err.to_string().contains("128"));
//! assert!(err.to_string().contains("256"));
//! ```

use thiserror::Error;

/// The central error type for all Kideta operations.
///
/// This enum uses [`thiserror`] to provide exhaustive pattern matching
/// and automatic `Display` and `From` implementations.
///
/// # When to Use Which Variant
///
/// | Variant | Use When |
/// |---------|----------|
/// | [`Io`](KidetaError::Io) | File system, network, or OS-level errors |
/// | [`Corruption`](KidetaError::Corruption) | Data is detected as corrupted |
/// | [`NotFound`](KidetaError::NotFound) | A generic resource is not found |
/// | [`CollectionNotFound`](KidetaError::CollectionNotFound) | A collection does not exist |
/// | [`VectorNotFound`](KidetaError::VectorNotFound) | A specific vector ID is not found |
/// | [`InvalidInput`](KidetaError::InvalidInput) | User-provided data is invalid |
/// | [`DimensionMismatch`](KidetaError::DimensionMismatch) | Vector dimensions don't match |
/// | [`InvalidDimension`](KidetaError::InvalidDimension) | A dimension value is invalid (e.g., 0) |
/// | [`InvalidMetric`](KidetaError::InvalidMetric) | Unknown or unsupported distance metric |
/// | [`InvalidIndexType`](KidetaError::InvalidIndexType) | Unknown or unsupported index type |
/// | [`Storage`](KidetaError::Storage) | Storage layer (KidetaStore) errors |
/// | [`Wal`](KidetaError::Wal) | Write-Ahead Log errors |
/// | [`Serialization`](KidetaError::Serialization) | JSON/binary serialization failures |
/// | [`Deserialization`](KidetaError::Deserialization) | JSON/binary deserialization failures |
/// | [`Internal`](KidetaError::Internal) | Unexpected bugs in Kideta itself |
#[derive(Debug, Error)]
pub enum KidetaError {
    /// I/O error from the operating system or filesystem.
    ///
    /// This wraps standard library [`std::io::Error`] with additional context
    /// about what operation was being performed.
    #[error("IO error: {context}{source}")]
    Io {
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
        /// Additional context about what operation was being performed.
        context: String,
    },

    /// Data corruption detected (e.g., checksum mismatch, malformed data).
    ///
    /// This indicates that data was found to be corrupted when reading
    /// or validating stored data.
    #[error("data corruption: {reason}")]
    Corruption {
        /// Human-readable explanation of what was corrupted.
        reason: String,
    },

    /// A generic "not found" error with a description.
    #[error("not found: {what}")]
    NotFound {
        /// What was not found (e.g., "config file", "index").
        what: String,
    },

    /// User-provided input was invalid.
    ///
    /// This is used for semantic validation errors, such as malformed
    /// vector data or invalid query parameters.
    #[error("invalid input: {reason}")]
    InvalidInput {
        /// Human-readable explanation of what was invalid.
        reason: String,
    },

    /// Vector dimensions don't match.
    ///
    /// This occurs when performing operations on vectors with incompatible
    /// dimensions (e.g., computing distance between 128-dim and 256-dim vectors).
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// The expected dimension.
        expected: usize,
        /// The actual dimension encountered.
        actual: usize,
    },

    /// A collection with the given name does not exist.
    #[error("collection not found: {name}")]
    CollectionNotFound {
        /// The name of the collection that was not found.
        name: String,
    },

    /// A storage segment with the given ID does not exist.
    #[error("segment not found: {id}")]
    SegmentNotFound {
        /// The ID of the segment that was not found.
        id: u64,
    },

    /// A vector with the given ID does not exist in the collection.
    #[error("vector not found: {id}")]
    VectorNotFound {
        /// The ID of the vector that was not found.
        id: u64,
    },

    /// A collection with this name already exists.
    #[error("collection already exists: {name}")]
    CollectionAlreadyExists {
        /// The conflicting collection name.
        name: String,
    },

    /// An invalid dimension value was provided.
    ///
    /// Dimensions must be positive integers (> 0).
    #[error("invalid dimension: {value} (must be > 0)")]
    InvalidDimension {
        /// The invalid dimension value that was provided.
        value: usize,
    },

    /// An unsupported or invalid distance metric was specified.
    #[error("invalid metric: {metric}")]
    InvalidMetric {
        /// The metric string that was provided.
        metric: String,
    },

    /// An unsupported or invalid index type was specified.
    #[error("invalid index type: {index_type}")]
    InvalidIndexType {
        /// The index type string that was provided.
        index_type: String,
    },

    /// An unsupported or invalid quantization type was specified.
    #[error("invalid quantization type: {quant_type}")]
    InvalidQuantizationType {
        /// The quantization type string that was provided.
        quant_type: String,
    },

    /// Encoding (serialization) failed.
    #[error("encoding error: {reason}")]
    Encoding {
        /// Human-readable explanation of the encoding failure.
        reason: String,
    },

    /// Decoding (deserialization) failed.
    #[error("decoding error: {reason}")]
    Decoding {
        /// Human-readable explanation of the decoding failure.
        reason: String,
    },

    /// Storage layer (KidetaStore) error.
    #[error("storage error: {reason}")]
    Storage {
        /// Human-readable explanation of the storage error.
        reason: String,
    },

    /// Write-Ahead Log (WAL) error.
    #[error("WAL error: {reason}")]
    Wal {
        /// Human-readable explanation of the WAL error.
        reason: String,
    },

    /// Serialization error (JSON, binary, etc.).
    #[error("serialization error: {reason}")]
    Serialization {
        /// Human-readable explanation of the serialization failure.
        reason: String,
    },

    /// Deserialization error (JSON, binary, etc.).
    #[error("deserialization error: {reason}")]
    Deserialization {
        /// Human-readable explanation of the deserialization failure.
        reason: String,
    },

    /// Internal error indicating a bug in Kideta.
    ///
    /// This variant is used for unexpected errors that shouldn't occur
    /// if the code is correct, such as invariant violations or reaching
    /// unreachable code paths.
    #[error("internal error: {message}")]
    Internal {
        /// A description of the internal error.
        message: String,
    },
}

/// A result type specialized for Kideta operations that can fail.
pub type Result<T> = std::result::Result<T, KidetaError>;

impl From<std::io::Error> for KidetaError {
    fn from(source: std::io::Error) -> Self {
        KidetaError::Io {
            source,
            context: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = KidetaError::NotFound {
            what: "collection 'test'".to_string(),
        };
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = KidetaError::DimensionMismatch {
            expected: 128,
            actual: 256,
        };
        let msg = err.to_string();
        assert!(msg.contains("128"));
        assert!(msg.contains("256"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: KidetaError = io_err.into();
        assert!(matches!(err, KidetaError::Io { .. }));
    }

    #[test]
    fn test_collection_not_found() {
        let err = KidetaError::CollectionNotFound {
            name: "my_collection".to_string(),
        };
        assert!(err.to_string().contains("my_collection"));
    }

    #[test]
    fn test_internal_error() {
        let err = KidetaError::Internal {
            message: "unexpected invariant violation".to_string(),
        };
        assert!(err.to_string().contains("internal error"));
    }
}
