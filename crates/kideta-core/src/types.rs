//! Core type definitions for Kideta vector database.
//!
//! This module provides fundamental types used throughout the Kideta codebase.
//! All ID types are newtype wrappers around `u64` to prevent accidental mixing
//! of different identifier types.
//!
//! # Example
//!
//! ```
//! use kideta_core::types::{VectorId, SegmentId, CollectionId};
//!
//! let vid = VectorId::new(1);
//! let sid = SegmentId::new(100);
//! let cid = CollectionId::new(999);
//!
//! // Each type is distinct and cannot be accidentally mixed
//! assert_eq!(vid.as_u64(), 1);
//! assert_eq!(sid.as_u64(), 100);
//! assert_eq!(cid.as_u64(), 999);
//! ```

use serde::{Deserialize, Serialize};

/// A unique identifier for a vector in the database.
///
/// `VectorId` is a newtype wrapper around `u64` that provides compile-time
/// type safety to prevent accidentally mixing vector IDs with other types
/// of IDs (e.g., segment IDs or collection IDs).
///
/// # Why a newtype?
///
/// In a vector database, you might have many different kinds of IDs:
/// - Vector IDs (the vectors you search)
/// - Segment IDs (internal storage segments)
/// - Collection IDs (top-level organizational units)
///
/// Using raw `u64` for all of them would allow bugs like:
/// ```ignore
/// let segment = SegmentId::new(42);
/// let vector = VectorId::new(42);  // Oops! These are different things
/// ```
///
/// With newtypes, the compiler prevents this mistake.
///
/// # Usage
///
/// ```
/// use kideta_core::types::VectorId;
///
/// let id = VectorId::new(12345);
/// assert_eq!(id.as_u64(), 12345);
///
/// // From raw u64
/// let id2 = VectorId::from(12345u64);
/// assert_eq!(id, id2);
///
/// // Convert back to u64
/// let raw: u64 = id.into();
/// assert_eq!(raw, 12345);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct VectorId(u64);

impl VectorId {
    /// Creates a new `VectorId` from a `u64` value.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::types::VectorId;
    /// let id = VectorId::new(42);
    /// ```
    #[inline]
    pub const fn new(id: u64) -> Self {
        VectorId(id)
    }

    /// Returns the underlying `u64` value.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for VectorId {
    /// Creates a `VectorId` from a raw `u64` value.
    fn from(id: u64) -> Self {
        VectorId(id)
    }
}

impl From<VectorId> for u64 {
    /// Converts a `VectorId` back to a raw `u64` value.
    fn from(id: VectorId) -> Self {
        id.0
    }
}

/// A unique identifier for a storage segment.
///
/// Segments are the basic unit of storage in Kideta. Each segment
/// contains a subset of vectors and is immutable once sealed.
///
/// See [`Segment`] for more details on how segments are used.
///
/// [`Segment`]: crate::storage::Segment
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct SegmentId(u64);

impl SegmentId {
    /// Creates a new `SegmentId` from a `u64` value.
    #[inline]
    pub const fn new(id: u64) -> Self {
        SegmentId(id)
    }

    /// Returns the underlying `u64` value.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for SegmentId {
    fn from(id: u64) -> Self {
        SegmentId(id)
    }
}

impl From<SegmentId> for u64 {
    fn from(id: SegmentId) -> Self {
        id.0
    }
}

/// A unique identifier for a collection.
///
/// Collections are the top-level organizational unit in Kideta.
/// They contain vectors with associated metadata and can be
/// queried independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct CollectionId(u64);

impl CollectionId {
    /// Creates a new `CollectionId` from a `u64` value.
    #[inline]
    pub const fn new(id: u64) -> Self {
        CollectionId(id)
    }

    /// Returns the underlying `u64` value.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl From<u64> for CollectionId {
    fn from(id: u64) -> Self {
        CollectionId(id)
    }
}

impl From<CollectionId> for u64 {
    fn from(id: CollectionId) -> Self {
        id.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_id() {
        let id = VectorId::new(42);
        assert_eq!(id.as_u64(), 42);
        assert_eq!(VectorId::from(42u64), id);
        assert_eq!(u64::from(id), 42);
    }

    #[test]
    fn test_segment_id() {
        let id = SegmentId::new(100);
        assert_eq!(id.as_u64(), 100);
        let id2 = SegmentId::from(100u64);
        assert_eq!(id, id2);
    }

    #[test]
    fn test_collection_id() {
        let id = CollectionId::new(999);
        assert_eq!(id.as_u64(), 999);
    }

    #[test]
    fn test_no_accidental_mixing() {
        let vid = VectorId::new(1);
        let sid = SegmentId::new(2);
        let cid = CollectionId::new(3);
        assert_eq!(vid.as_u64(), 1);
        assert_eq!(sid.as_u64(), 2);
        assert_eq!(cid.as_u64(), 3);
        let vid2 = VectorId::new(1);
        assert_eq!(vid, vid2);
        let sid2 = SegmentId::new(2);
        assert_eq!(sid, sid2);
        let cid2 = CollectionId::new(3);
        assert_eq!(cid, cid2);
    }
}
