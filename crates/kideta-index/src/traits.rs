//! VectorIndex trait — common interface for all index types.
//!
//! This trait defines the operations that all index implementations
//! (Flat, HNSW, IVF, Vamana, etc.) must support, allowing the
//! query planner to use them polymorphically.
//!
//! # Mmap Integration
//!
//! The [`VectorIndex::load_mmap`] method provides a way to load an index
//! from memory-mapped files on disk (vectors + graph adjacency) without
//! copying all data into RAM. The [`VectorIndex::set_vector_store`] method
//! allows replacing an in-memory vector store with an mmap'd store after
//! index construction, freeing RAM while keeping the index searchable.
//!
//! These work together with:
//! - [`kideta_core::vector_store::VectorStore`] — abstract vector access
//! - `kideta_storage::vector_storage::MmapVectorStorage` — mmap'd vector file
//! - [`crate::hnsw::MmapLayer`] — mmap'd HNSW graph adjacency

use std::cmp::Ordering;

use kideta_core::{metric::DistanceMetric, payload::Payload, types::VectorId};

use crate::search_params::SearchParams;

/// Common error type for index operations.

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("vector not found: {0:?}")]
    NotFound(VectorId),

    #[error("duplicate vector id: {0:?}")]
    DuplicateId(VectorId),

    #[error("index not ready")]
    IndexNotReady,

    #[error("quantization error: {0}")]
    Quantization(String),

    #[error("empty index")]
    EmptyIndex,

    #[error("internal error: {0}")]
    Internal(String),
}

/// A scored vector ID, used for search results with associated scores.
#[derive(Debug, Clone, Copy)]
pub struct ScoredVectorId {
    pub id: VectorId,
    pub score: f32,
}

impl ScoredVectorId {
    pub fn new(
        id: VectorId,
        score: f32,
    ) -> Self {
        Self { id, score }
    }
}

impl PartialEq for ScoredVectorId {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        if self.score.is_nan() && other.score.is_nan() {
            true
        } else {
            self.score == other.score
        }
    }
}

impl Eq for ScoredVectorId {}

impl PartialOrd for ScoredVectorId {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredVectorId {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        if self.score.is_nan() && other.score.is_nan() {
            Ordering::Equal
        } else if self.score.is_nan() {
            Ordering::Greater
        } else if other.score.is_nan() {
            Ordering::Less
        } else {
            self.score
                .partial_cmp(&other.score)
                .unwrap_or(Ordering::Equal)
        }
    }
}

/// A filter function type for filtering search results based on payloads.
pub type FilterFn = dyn Fn(VectorId, &Payload) -> bool + Send + Sync;

/// The `VectorIndex` trait defines the common interface for all vector index types.
pub trait VectorIndex: Send + Sync {
    fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<ScoredVectorId>;

    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterFn>,
    ) -> Vec<ScoredVectorId>;

    fn search_with_params(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Vec<ScoredVectorId>;

    fn search_with_params_and_filter(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        filter: Option<&FilterFn>,
    ) -> Vec<ScoredVectorId>;

    fn insert(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError>;

    fn delete(
        &mut self,
        id: VectorId,
    ) -> Result<(), IndexError>;

    fn update(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn dimension(&self) -> usize;

    fn metric(&self) -> DistanceMetric;

    fn size_bytes(&self) -> usize;
}
