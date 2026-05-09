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

    /// Returns the dimensionality of each vector.
    fn dimension(&self) -> usize;

    /// Returns a reference to the `i`-th vector, or `None` if out of bounds.
    ///
    /// This is a zero-copy operation — the returned slice borrows from
    /// the underlying storage (either heap or mmap'd region).
    fn get_vector(&self, i: usize) -> Option<&[f32]>;

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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorStore(len={}, dim={})", self.len(), self.dimension())
    }
}