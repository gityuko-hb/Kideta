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

use kideta_core::distance::{Metric, get_distance_fn, prefetch_nta};
use kideta_core::metric::DistanceMetric;
use kideta_core::payload::Payload;
use kideta_core::types::VectorId;
use kideta_core::utils::heap::BoundedMaxHeap;
use kideta_core::utils::roaring::RoaringBitmap;

use crate::quantization::{QuantizationConfig, QuantizedStorage, Sq8Stats, encode_vectors};
use crate::traits::{FilterFn, IndexError, ScoredVectorId, VectorIndex};

/// Threshold for switching to parallel search. Collections smaller than this
const PARALLEL_THRESHOLD: usize = 10_000; // 10 * 1000 vectors 

/// Wrapper to adapt Hamming distance for f32 vectors
fn hamming_distance_wrapper(
    a: &[f32],
    b: &[f32],
) -> f32 {
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
    quantization_config: Option<QuantizationConfig>,
    quantized_storage: Option<QuantizedStorage>,
}

impl FlatIndex {
    pub fn new(
        dimension: usize,
        metric: DistanceMetric,
    ) -> Self {
        let distance_fn = match metric {
            DistanceMetric::Cosine => get_distance_fn(Metric::Cosine),
            DistanceMetric::L2 => get_distance_fn(Metric::L2),
            DistanceMetric::DotProduct => get_distance_fn(Metric::Dot),
            DistanceMetric::Hamming => {
                // Hamming operates on binary vectors - use wrapper
                hamming_distance_wrapper
            },
        };

        Self {
            vectors: Vec::new(),
            ids: Vec::new(),
            payloads: Vec::new(),
            deleted: RoaringBitmap::new(),
            dimension,
            metric,
            distance_fn,
            quantization_config: None,
            quantized_storage: None,
        }
    }

    #[inline]
    fn vector_offset(
        &self,
        local_id: usize,
    ) -> usize {
        local_id * self.dimension
    }

    #[inline]
    pub fn get_vector(
        &self,
        local_id: usize,
    ) -> Option<&[f32]> {
        if local_id >= self.len() {
            return None;
        }
        let offset = self.vector_offset(local_id);
        Some(&self.vectors[offset..offset + self.dimension])
    }

    pub fn insert(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError> {
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        self.vectors.extend_from_slice(vector);
        self.ids.push(id);
        self.payloads.push(payload);

        Ok(())
    }

    pub fn delete(
        &mut self,
        id: VectorId,
    ) -> Result<(), IndexError> {
        let local_id = self
            .ids
            .iter()
            .position(|&x| x == id)
            .ok_or(IndexError::NotFound(id))?;

        self.deleted.insert(local_id as u32);
        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<ScoredVectorId> {
        if query.len() != self.dimension {
            return Vec::new();
        }

        if self.is_empty() {
            return Vec::new();
        }

        if self.total_len() > PARALLEL_THRESHOLD {
            return self.search_parallel(query, k);
        }

        self.search_serial(query, k)
    }

    fn search_serial(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<ScoredVectorId> {
        let distance_fn = self.distance_fn;
        let dim = self.dimension;
        let total_len = self.total_len();

        let mut results = BoundedMaxHeap::new(k);

        for i in 0..total_len {
            if self.deleted.contains(i as u32) {
                continue;
            }

            // Prefetch vector 4 positions ahead
            let pf = i + 4;
            if pf < total_len {
                let pf_off = pf * dim;
                prefetch_nta(&self.vectors[pf_off] as *const f32);
            }

            let offset = i * dim;
            let vector = &self.vectors[offset..offset + dim];
            let score = distance_fn(query, vector);
            results.push(ScoredVectorId::new(self.ids[i], score));
        }

        let mut sorted = results.into_sorted_vec();
        sorted.reverse();
        sorted
    }

    fn search_parallel(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<ScoredVectorId> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                crate::flat::parallel::parallel_search_async(self, query, k).await
            })
        })
    }

    pub fn search_with_filter<F: Fn(VectorId, &Payload) -> bool + Send + Sync>(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&F>,
    ) -> Vec<ScoredVectorId> {
        if query.len() != self.dimension {
            return Vec::new();
        }

        if self.total_len() == 0 {
            return Vec::new();
        }

        let distance_fn = self.distance_fn;
        let dim = self.dimension;
        let total_len = self.total_len();

        let mut results = BoundedMaxHeap::new(k);

        for i in 0..total_len {
            if self.deleted.contains(i as u32) {
                continue;
            }

            // Prefetch vector 4 positions ahead
            let pf = i + 4;
            if pf < total_len {
                let pf_off = pf * dim;
                prefetch_nta(&self.vectors[pf_off] as *const f32);
            }

            let offset = i * dim;
            let vector = &self.vectors[offset..offset + dim];

            if let Some(f) = filter
                && !f(self.ids[i], &self.payloads[i])
            {
                continue;
            }

            let score = distance_fn(query, vector);
            results.push(ScoredVectorId::new(self.ids[i], score));
        }

        let mut sorted = results.into_sorted_vec();
        sorted.reverse();
        sorted
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

    pub fn quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    pub fn quantized_storage(&self) -> Option<&QuantizedStorage> {
        self.quantized_storage.as_ref()
    }

    pub fn train_quantization(
        &mut self,
        config: &QuantizationConfig,
    ) -> Result<(), IndexError> {
        if self.ids.is_empty() {
            return Err(IndexError::Internal(
                "cannot train quantization on empty index".into(),
            ));
        }

        let dim = self.dimension;
        config
            .dimension()
            .and_then(|d| {
                if d == dim {
                    Some(())
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                IndexError::Quantization(format!(
                    "dimension mismatch: index={}, config={:?}",
                    dim,
                    config.dimension()
                ))
            })?;

        let all_vectors: Vec<&[f32]> = (0..self.total_len())
            .map(|i| {
                let start = i * dim;
                &self.vectors[start..start + dim]
            })
            .collect();

        let storage = encode_vectors(config, &all_vectors);
        self.quantized_storage = Some(storage);
        self.quantization_config = Some(config.clone());

        Ok(())
    }

    pub fn train_sq8(&mut self) -> Result<(), IndexError> {
        if self.total_len() == 0 {
            return Err(IndexError::EmptyIndex);
        }
        let all_vectors: Vec<&[f32]> = (0..self.total_len())
            .map(|i| {
                let start = i * self.dimension;
                &self.vectors[start..start + self.dimension]
            })
            .collect();

        let stats = Sq8Stats::compute(&all_vectors);
        let config = QuantizationConfig::Sq8(stats.into_config());
        self.train_quantization(&config)
    }

    pub fn search_quantized(
        &self,
        query: &[f32],
        k: usize,
        rescore_factor: usize,
    ) -> Vec<ScoredVectorId> {
        let storage = match &self.quantized_storage {
            Some(s) => s,
            None => return self.search(query, k),
        };

        let distance_fn = self.distance_fn;
        let dim = self.dimension;
        let total_len = self.total_len();
        let target = k * rescore_factor;

        let mut candidates = BoundedMaxHeap::new(target);

        for i in 0..total_len {
            if self.deleted.contains(i as u32) {
                continue;
            }

            if let Some(approx_dist) = storage.approx_distance(query, i) {
                candidates.push(ScoredVectorId::new(self.ids[i], approx_dist));
            }
        }

        let candidate_ids: Vec<_> = candidates
            .into_sorted_vec()
            .into_iter()
            .rev()
            .map(|s| s.id)
            .collect();

        let mut results = BoundedMaxHeap::new(k);
        for id in candidate_ids {
            if let Some(local_idx) = self.ids.iter().position(|&x| x == id) {
                let start = local_idx * dim;
                let vector = &self.vectors[start..start + dim];
                let score = distance_fn(query, vector);
                results.push(ScoredVectorId::new(id, score));
            }
        }

        let mut sorted = results.into_sorted_vec();
        sorted.reverse();
        sorted
    }
}

impl std::fmt::Debug for FlatIndex {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("FlatIndex")
            .field("dimension", &self.dimension)
            .field("total_vectors", &self.total_len())
            .field("deleted_count", &self.deleted_count())
            .field("active_vectors", &self.len())
            .field("metric", &self.metric)
            .field("size_bytes", &self.size_bytes())
            .field("quantized", &self.quantization_config.is_some())
            .finish()
    }
}

impl VectorIndex for FlatIndex {
    fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<ScoredVectorId> {
        FlatIndex::search(self, query, k)
    }

    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterFn>,
    ) -> Vec<ScoredVectorId> {
        match filter {
            Some(f) => {
                let filter_fn = |id: VectorId, payload: &Payload| f(id, payload);
                FlatIndex::search_with_filter(self, query, k, Some(&filter_fn))
            },
            None => FlatIndex::search_with_filter(
                self,
                query,
                k,
                None::<&fn(VectorId, &Payload) -> bool>,
            ),
        }
    }

    fn insert(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError> {
        FlatIndex::insert(self, id, vector, payload)
    }

    fn delete(
        &mut self,
        id: VectorId,
    ) -> Result<(), IndexError> {
        FlatIndex::delete(self, id)
    }

    fn update(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError> {
        FlatIndex::delete(self, id)?;
        FlatIndex::insert(self, id, vector, payload)
    }

    fn len(&self) -> usize {
        FlatIndex::len(self)
    }

    fn is_empty(&self) -> bool {
        FlatIndex::is_empty(self)
    }

    fn dimension(&self) -> usize {
        FlatIndex::dimension(self)
    }

    fn metric(&self) -> DistanceMetric {
        FlatIndex::metric(self)
    }

    fn size_bytes(&self) -> usize {
        FlatIndex::size_bytes(self)
    }

    fn quantization_config(&self) -> Option<&QuantizationConfig> {
        FlatIndex::quantization_config(self)
    }

    fn quantized_storage(&self) -> Option<&QuantizedStorage> {
        FlatIndex::quantized_storage(self)
    }

    fn search_quantized(
        &self,
        query: &[f32],
        k: usize,
        rescore_factor: usize,
    ) -> Vec<ScoredVectorId> {
        FlatIndex::search_quantized(self, query, k, rescore_factor)
    }

    fn train_quantization(
        &mut self,
        config: &QuantizationConfig,
    ) -> Result<(), IndexError> {
        FlatIndex::train_quantization(self, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> FlatIndex {
        FlatIndex::new(4, DistanceMetric::L2)
    }

    #[test]
    fn test_new_index() {
        let index = create_test_index();
        assert_eq!(index.dimension(), 4);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_insert() {
        let mut index = create_test_index();
        let id = VectorId::new(1);
        let vector = vec![1.0, 2.0, 3.0, 4.0];

        index.insert(id, &vector, Payload::new()).unwrap();

        assert_eq!(index.len(), 1);
        assert_eq!(index.total_len(), 1);
    }

    #[test]
    fn test_insert_dimension_mismatch() {
        let mut index = create_test_index();
        let id = VectorId::new(1);
        let vector = vec![1.0, 2.0, 3.0];

        let result = index.insert(id, &vector, Payload::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_search_single_vector() {
        let mut index = create_test_index();
        let id = VectorId::new(1);
        let vector = vec![0.0, 0.0, 0.0, 0.0];

        index.insert(id, &vector, Payload::new()).unwrap();

        let query = vec![1.0, 1.0, 1.0, 1.0];
        let results = index.search(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn test_search_returns_nearest() {
        let mut index = FlatIndex::new(2, DistanceMetric::L2);

        index
            .insert(VectorId::new(1), &[0.0, 0.0], Payload::new())
            .unwrap();
        index
            .insert(VectorId::new(2), &[10.0, 10.0], Payload::new())
            .unwrap();
        index
            .insert(VectorId::new(3), &[1.0, 1.0], Payload::new())
            .unwrap();

        let query = vec![1.0, 1.0];
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, VectorId::new(3));
        assert_eq!(results[1].id, VectorId::new(1));
    }

    #[test]
    fn test_delete() {
        let mut index = create_test_index();
        let id = VectorId::new(1);
        let vector = vec![1.0, 2.0, 3.0, 4.0];

        index.insert(id, &vector, Payload::new()).unwrap();
        assert_eq!(index.len(), 1);

        index.delete(id).unwrap();
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_delete_skips_in_search() {
        let mut index = create_test_index();

        index
            .insert(VectorId::new(1), &[0.0, 0.0, 0.0, 0.0], Payload::new())
            .unwrap();
        index
            .insert(VectorId::new(2), &[1.0, 1.0, 1.0, 1.0], Payload::new())
            .unwrap();
        index
            .insert(VectorId::new(3), &[2.0, 2.0, 2.0, 2.0], Payload::new())
            .unwrap();

        index.delete(VectorId::new(2)).unwrap();

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 3);

        assert_eq!(results.len(), 2);
        let ids: Vec<_> = results.iter().map(|r| r.id).collect();
        assert!(!ids.contains(&VectorId::new(2)));
    }

    #[test]
    fn test_search_with_filter() {
        let mut index = FlatIndex::new(4, DistanceMetric::L2);

        let mut payload1 = Payload::new();
        payload1.insert("category", kideta_core::payload::PayloadValue::Int(1));

        let mut payload2 = Payload::new();
        payload2.insert("category", kideta_core::payload::PayloadValue::Int(2));

        index
            .insert(VectorId::new(1), &[0.0, 0.0, 0.0, 0.0], payload1.clone())
            .unwrap();
        index
            .insert(VectorId::new(2), &[1.0, 1.0, 1.0, 1.0], payload2.clone())
            .unwrap();
        index
            .insert(VectorId::new(3), &[2.0, 2.0, 2.0, 2.0], payload1.clone())
            .unwrap();

        let filter = |_id: VectorId, payload: &Payload| {
            payload
                .get("category")
                .and_then(|v| v.as_int())
                .map(|v| v == 1)
                .unwrap_or(false)
        };

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.search_with_filter(&query, 10, Some(&filter));

        assert_eq!(results.len(), 2);
        let ids: Vec<_> = results.iter().map(|r| r.id).collect();
        assert!(ids.contains(&VectorId::new(1)));
        assert!(ids.contains(&VectorId::new(3)));
        assert!(!ids.contains(&VectorId::new(2)));
    }

    #[test]
    fn test_empty_index_search() {
        let index = create_test_index();
        let query = vec![1.0, 1.0, 1.0, 1.0];
        let results = index.search(&query, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_size_bytes() {
        let mut index = create_test_index();
        let vector = vec![1.0, 2.0, 3.0, 4.0];

        index
            .insert(VectorId::new(1), &vector, Payload::new())
            .unwrap();

        let size = index.size_bytes();
        assert!(size > 0);
    }

    #[test]
    fn test_recall_100_percent() {
        let mut index = FlatIndex::new(128, DistanceMetric::L2);

        for i in 0u32..1000 {
            let mut vector = vec![0.0; 128];
            vector[i as usize % 128] = i as f32;
            index
                .insert(VectorId::new(i as u64), &vector, Payload::new())
                .unwrap();
        }

        let mut query = vec![0.0; 128];
        query[0] = 500.0;

        let results = index.search(&query, 1);

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut index = create_test_index();
        let result = index.delete(VectorId::new(999));
        assert!(result.is_err());
    }

    #[test]
    fn test_top_k_respects_limit() {
        let mut index = FlatIndex::new(2, DistanceMetric::L2);

        for i in 0..100 {
            let vector = vec![i as f32, i as f32];
            index
                .insert(VectorId::new(i), &vector, Payload::new())
                .unwrap();
        }

        let query = vec![50.0, 50.0];
        let results: Vec<ScoredVectorId> = index.search(&query, 10);

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_train_sq8_quantization() {
        let mut index = FlatIndex::new(4, DistanceMetric::L2);
        for i in 0u32..50 {
            let vector = (0..4)
                .map(|j| (i * 4 + j) as f32 * 0.1)
                .collect::<Vec<_>>();
            index
                .insert(VectorId::new(i as u64), &vector, Payload::new())
                .unwrap();
        }

        index.train_sq8().unwrap();

        assert!(index.quantization_config().is_some());
        assert!(index.quantized_storage().is_some());
    }

    #[test]
    fn test_search_quantized_vs_exact() {
        let mut index = FlatIndex::new(4, DistanceMetric::L2);
        for i in 0u32..100 {
            let vector = (0..4)
                .map(|j| ((i * 4 + j) % 100) as f32)
                .collect::<Vec<_>>();
            index
                .insert(VectorId::new(i as u64), &vector, Payload::new())
                .unwrap();
        }

        index.train_sq8().unwrap();

        let query = vec![50.0, 51.0, 52.0, 53.0];
        let exact_results = index.search(&query, 10);
        let quantized_results = index.search_quantized(&query, 10, 10);

        let exact_ids: Vec<_> = exact_results.iter().map(|r| r.id).collect();
        let quantized_ids: Vec<_> = quantized_results.iter().map(|r| r.id).collect();

        let recall = exact_ids
            .iter()
            .filter(|id| quantized_ids.contains(id))
            .count() as f32
            / exact_ids.len() as f32;

        assert!(recall >= 0.8, "recall {} < 0.8", recall);
    }

    #[test]
    fn test_search_quantized_without_training() {
        let mut index = FlatIndex::new(4, DistanceMetric::L2);
        index
            .insert(VectorId::new(1), &[1.0, 2.0, 3.0, 4.0], Payload::new())
            .unwrap();

        let query = vec![1.0, 2.0, 3.0, 4.0];
        let results = index.search_quantized(&query, 1, 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, VectorId::new(1));
    }

    #[test]
    fn test_train_empty_index_fails() {
        let mut index = FlatIndex::new(4, DistanceMetric::L2);
        let result = index.train_sq8();
        assert!(result.is_err());
    }

    #[test]
    fn test_quantization_config_debug() {
        let mut index = FlatIndex::new(4, DistanceMetric::L2);
        for i in 0u32..10 {
            let vector = vec![i as f32; 4];
            index
                .insert(VectorId::new(i as u64), &vector, Payload::new())
                .unwrap();
        }

        let debug_str = format!("{:?}", index);
        assert!(debug_str.contains("quantized"));
    }
}
