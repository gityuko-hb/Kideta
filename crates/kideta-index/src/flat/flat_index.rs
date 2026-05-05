//! FlatIndex — brute-force linear scan index.
//!
//! This is the simplest index implementation: it stores all vectors
//! in a contiguous memory buffer and performs exact nearest-neighbor
//! search via linear scan.
//!
//! # Storage Layout
//!
//! Vectors are stored in a flat contiguous buffer:
//!
//! ```text
//! vectors: [v0_0, v0_1, ..., v0_d-1, v1_0, v1_1, ..., v1_d-1, ...]
//! ```
//!
//! Parallel arrays track IDs and payloads:
//!
//! - `ids[i]` → VectorId of the vector at offset `i * dim`
//! - `payloads[i]` → Payload of the vector at offset `i * dim`
//!
//! This layout provides O(1) random access to any vector and is
//! cache-friendly for sequential scans.
//!
//! # Delete Handling
//!
//! Deletes are soft — vectors are marked as deleted in a RoaringBitmap
//! but the underlying storage is not reclaimed. This avoids expensive
//! data movement during active writes.
//!
//! # Performance
//!
//! For small collections (< 10K vectors), FlatIndex is often faster
//! than ANN indexes due to zero index overhead. It's also the reference
//! implementation used to measure ANN recall accuracy.

use kideta_core::distance::{get_distance_fn, prefetch_nta, Metric};
use kideta_core::metric::DistanceMetric;
use kideta_core::payload::Payload;
use kideta_core::types::VectorId;
use kideta_core::utils::heap::BoundedMaxHeap;
use kideta_core::utils::roaring::RoaringBitmap;

use crate::traits::{FilterFn, IndexError, ScoredVectorId, VectorIndex};

/// Threshold for switching to parallel search. Collections smaller than this
const PARALLEL_THRESHOLD: usize = 10_000; // 10 * 1000 vectors 

/// Wrapper to adapt Hamming distance for f32 vectors
fn hamming_distance_wrapper(a: &[f32], b: &[f32]) -> f32 {
    kideta_core::distance::hamming_distance_f32(a, b) as f32
}

pub struct FlatIndex {
    vectors: Vec<f32>,
    ids: Vec<VectorId>,
    payloads: Vec<Payload>,
    deleted: RoaringBitmap,
    dimension: usize,
    metric: DistanceMetric,
    distance_fn: fn(&[f32], &[f32]) -> f32,
}

impl FlatIndex {
    pub fn new(dimension: usize, metric: DistanceMetric) -> Self {
        let distance_fn = match metric {
            DistanceMetric::Cosine => get_distance_fn(Metric::Cosine),
            DistanceMetric::L2 => get_distance_fn(Metric::L2),
            DistanceMetric::DotProduct => get_distance_fn(Metric::Dot),
            DistanceMetric::Hamming => {
                // Hamming operates on binary vectors - use wrapper
                hamming_distance_wrapper
            }
        };

        Self {
            vectors: Vec::new(),
            ids: Vec::new(),
            payloads: Vec::new(),
            deleted: RoaringBitmap::new(),
            dimension,
            metric,
            distance_fn,
        }
    }

    #[inline]
    fn vector_offset(&self, local_id: usize) -> usize {
        local_id * self.dimension
    }

    #[inline]
    pub fn get_vector(&self, local_id: usize) -> Option<&[f32]> {
        if local_id >= self.len() {
            return None;
        }
        let offset = self.vector_offset(local_id);
        Some(&self.vectors[offset..offset + self.dimension])
    }

    pub fn insert(
        &mut self,
        id: VectorId,
        vector: &[f32], payload: Payload
    ) -> Result<(), IndexError> {
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            })
        }

        self.vectors.extend_from_slice(vector);
        self.ids.push(id);
        self.payloads.push(payload);
        
        Ok(())
    }

    pub fn delete(&mut self, id: VectorId) -> Result<(), IndexError> {
        let local_id = self.ids
            .iter()
            .position(|&x| x == id)
            .ok_or(IndexError::NotFound(id))?;

        self.deleted.insert(local_id as u32);
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<ScoredVectorId> {
        todo!()
    }

    fn search_serial(&self, query: &[f32], k: usize) -> Vec<ScoredVectorId> {
        todo!()
    }

    fn search_parallel(&self, query: &[f32], k: usize) -> Vec<ScoredVectorId> {
        todo!()
    }

    pub fn search_with_filter<F: Fn(VectorId, &Payload) -> bool + Send + Sync>(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&F>,
    ) -> Vec<ScoredVectorId> {
        todo!()
    }

    pub fn len(&self) -> usize {
        self.ids.len() - self.deleted.len()
    }

    pub fn total_len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn distance_fn(&self) -> fn(&[f32], &[f32]) -> f32 {
        self.distance_fn
    }

    pub fn vectors(&self) -> &Vec<f32> {
        &self.vectors
    }

    pub fn ids(&self) -> &Vec<VectorId> {
        &self.ids
    }

    pub fn size_bytes(&self) -> usize {
        let vector_bytes = self.vectors.capacity() * std::mem::size_of::<f32>();
        let id_bytes = self.ids.capacity() * std::mem::size_of::<VectorId>();
        let payload_bytes = self.payloads.capacity() * std::mem::size_of::<Payload>();
        let deleted_bytes = self.deleted.serialize().len();

        vector_bytes + id_bytes + payload_bytes + deleted_bytes
    }

    pub fn deleted_count(&self) -> usize {
        self.deleted.len()
    }

    pub fn deleted(&self) -> &RoaringBitmap {
        &self.deleted
    }
}

impl VectorIndex for FlatIndex {
    fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<ScoredVectorId>
    {
        todo!()
    }

    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterFn>,
    ) -> Vec<ScoredVectorId>
    {
        todo!()
    }

    fn search_with_params(
        &self,
        query: &[f32],
        k: usize,
        params: &crate::search_params::SearchParams,
    ) -> Vec<ScoredVectorId> {
        todo!()
    }
    
    fn search_with_params_and_filter(
        &self,
        query: &[f32],
        k: usize,
        params: &crate::search_params::SearchParams,
        filter: Option<&FilterFn>,
    ) -> Vec<ScoredVectorId> {
        todo!()
    }

    fn insert(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError>
    {
        todo!()
    }

    fn delete(
        &mut self,
        id: VectorId,
    ) -> Result<(), IndexError>
    {
        todo!()
    }

    fn update(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError>
    {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn is_empty(&self) -> bool {
        todo!()
    }

    fn dimension(&self) -> usize {
        todo!()
    }

    fn metric(&self) -> DistanceMetric {
        todo!()
    }

    fn size_bytes(&self) -> usize {
        todo!()
    }

}