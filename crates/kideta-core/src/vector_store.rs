//! Abstract vector storage interface for zero-copy vector access.
//!
//! The [`VectorStore`] trait provides a uniform read-only interface for
//! accessing vectors regardless of their backing storage:
//!
//! - **In-memory** via [`VecVectorStore`] — a `Vec<f32>` wrapper with dimension tracking
//! - **Memory-mapped** via `MmapVectorStorage` in `kideta-storage` — zero-copy reads from disk
//!
//! This trait is the key abstraction that allows HNSW, Vamana, and other index
//! types to operate on both in-RAM (small datasets, testing) and disk-backed
//! (large datasets, production) vector data without code changes.
//!
//! # Usage
//!
//! ```rust
//! use kideta_core::vector_store::{VectorStore, VecVectorStore};
//!
//! let mut store = VecVectorStore::new(3, 10);
//! store.push(&[1.0, 2.0, 3.0]);
//! store.push(&[4.0, 5.0, 6.0]);
//!
//! assert_eq!(store.len(), 2);
//! assert_eq!(store.dimension(), 3);
//! assert_eq!(store.get_vector(0), Some(&[1.0, 2.0, 3.0][..]));
//! assert_eq!(store.get_vector(1), Some(&[4.0, 5.0, 6.0][..]));
//! ```

use std::fmt;

/// Read-only interface for random-access vector storage.
///
/// Implementations must provide:
/// - `len()` — number of stored vectors
/// - `dimension()` — dimensionality of each vector
/// - `get_vector(i)` — zero-copy access to the `i`-th vector
///
/// The optional `data_ptr()` method enables CPU prefetching optimizations
/// in graph search algorithms by exposing the base address of the flat
/// f32 data array (if available).
pub trait VectorStore: Send + Sync {
    /// Returns the number of vectors in this store.
    fn len(&self) -> usize;

    /// Returns `true` if the store contains no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the dimensionality of each vector.
    fn dimension(&self) -> usize;

    /// Returns a reference to the `i`-th vector, or `None` if out of bounds.
    ///
    /// This is a zero-copy operation — the returned slice borrows from
    /// the underlying storage (either heap or mmap'd region).
    fn get_vector(
        &self,
        i: usize,
    ) -> Option<&[f32]>;

    /// Returns a raw pointer to the start of the flat f32 data array,
    /// if the implementation stores vectors contiguously.
    ///
    /// This enables prefetch optimizations in graph search hot paths.
    /// Returns `None` if the store does not expose a contiguous f32 array.
    fn data_ptr(&self) -> Option<*const f32> {
        None
    }
}

impl fmt::Debug for dyn VectorStore {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "VectorStore(len={}, dim={})",
            self.len(),
            self.dimension()
        )
    }
}

/// In-memory vector store backed by a flat `Vec<f32>`.
///
/// Vectors are stored contiguously: vectors `[dim]` elements each are
/// stored as a single flat array of `len * dim` f32 values. This is the
/// default store used during index construction and by `InMemoryCollection`.
///
/// # Construction
///
/// ```rust
/// use kideta_core::vector_store::{VecVectorStore, VectorStore};
///
/// let mut store: VecVectorStore = VecVectorStore::new(128, 1000);
/// store.push(&[0.5; 128]);
/// assert_eq!(store.len(), 1);
/// ```
#[derive(Clone)]
pub struct VecVectorStore {
    /// Flat f32 vector data: `[vec_0_0, vec_0_1, ..., vec_0_{dim-1}, vec_1_0, ...]`
    pub data: Vec<f32>,
    /// Dimensionality of each vector
    pub dim: usize,
}

impl VecVectorStore {
    /// Creates a new in-memory vector store with the given dimension
    /// and pre-allocated capacity.
    ///
    /// The underlying `Vec<f32>` is pre-allocated to hold `dim * capacity`
    /// elements, avoiding reallocation during initial insertions.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is 0.
    pub fn new(
        dim: usize,
        capacity: usize,
    ) -> Self {
        Self {
            data: Vec::with_capacity(dim * capacity),
            dim,
        }
    }

    /// Appends a single vector to the store.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if `v.len() != self.dim`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kideta_core::vector_store::{VecVectorStore, VectorStore};
    ///
    /// let mut store = VecVectorStore::new(2, 10);
    /// store.push(&[1.0, 2.0]);
    /// assert_eq!(store.len(), 1);
    /// ```
    pub fn push(
        &mut self,
        v: &[f32],
    ) {
        debug_assert_eq!(v.len(), self.dim);
        self.data.extend_from_slice(v);
    }
}

impl VectorStore for VecVectorStore {
    fn len(&self) -> usize {
        if self.dim == 0 {
            return 0;
        }
        self.data.len() / self.dim
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn get_vector(
        &self,
        i: usize,
    ) -> Option<&[f32]> {
        if self.dim == 0 || i >= self.len() {
            return None;
        }
        let start = i * self.dim;
        self.data.get(start..start + self.dim)
    }

    fn data_ptr(&self) -> Option<*const f32> {
        Some(self.data.as_ptr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_store() {
        let store = VecVectorStore::new(4, 0);
        assert_eq!(store.len(), 0);
        assert!(store.get_vector(0).is_none());
    }

    #[test]
    fn test_push_and_retrieve() {
        let mut store = VecVectorStore::new(3, 10);
        store.push(&[1.0, 2.0, 3.0]);
        store.push(&[4.0, 5.0, 6.0]);
        assert_eq!(store.len(), 2);
        assert_eq!(store.get_vector(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(store.get_vector(1), Some(&[4.0, 5.0, 6.0][..]));
    }

    #[test]
    fn test_out_of_bounds() {
        let mut store = VecVectorStore::new(2, 5);
        store.push(&[0.0; 2]);
        assert!(store.get_vector(0).is_some());
        assert!(store.get_vector(1).is_none());
        assert!(store.get_vector(usize::MAX).is_none());
    }

    #[test]
    fn test_dimension_zero() {
        let store = VecVectorStore::new(0, 10);
        assert_eq!(store.len(), 0);
        assert!(store.get_vector(0).is_none());
    }

    #[test]
    fn test_data_ptr() {
        let mut store = VecVectorStore::new(2, 5);
        store.push(&[1.0, 2.0]);
        let ptr = store.data_ptr();
        assert!(ptr.is_some());
        unsafe {
            assert_eq!(*ptr.unwrap(), 1.0);
        }
    }

    #[test]
    fn test_debug_dyn() {
        let store: Box<dyn VectorStore> = Box::new(VecVectorStore::new(4, 0));
        let s = format!("{:?}", store);
        assert!(s.contains("VectorStore"));
        assert!(s.contains("dim=4"));
    }
}
