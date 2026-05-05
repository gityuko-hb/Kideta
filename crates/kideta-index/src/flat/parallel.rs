//! Parallel scan using tokio's thread pool.
//!
//! This module provides parallel search functionality using tokio's
//! `spawn_blocking` for CPU-bound work. The approach is abstracted
//! so that the underlying executor can be swapped if needed.
//!
//! # Design
//!
//! - Uses `tokio::task::JoinSet` for managing parallel tasks
//! - `spawn_blocking` moves CPU-bound work to a dedicated thread pool
//! - Results are merged using a top-k reduction
//!

use std::cmp::Ordering;
use std::sync::Arc;
use tokio::task::JoinSet;

use crate::flat::flat_index::FlatIndex;
use crate::traits::ScoredVectorId;
use kideta_core::distance::prefetch_nta;
use kideta_core::types::VectorId;
use kideta_core::utils::roaring::RoaringBitmap;

fn get_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(16)
}

fn scan_chunk(
    vectors: &[f32],
    ids: &[u64],
    deleted: &RoaringBitmap,
    query: &[f32],
    distance_fn: fn(&[f32], &[f32]) -> f32,
    k: usize,
) -> Vec<ScoredVectorId> {
    use kideta_core::utils::heap::BoundedMaxHeap;

    let dim = query.len();
    let chunk_len = vectors.len() / dim;
    let mut results = BoundedMaxHeap::new(k);

    for (i, &id) in ids.iter().enumerate().take(chunk_len) {
        if deleted.contains(i as u32) {
            continue;
        }

        // Prefetch vector 4 positions ahead
        let pf = i + 4;
        if pf < chunk_len {
            let pf_off = pf * dim;
            prefetch_nta(&vectors[pf_off] as *const f32);
        }

        let offset = i * dim;
        let vector = &vectors[offset..offset + dim];
        let score = distance_fn(query, vector);
        let id = kideta_core::types::VectorId::new(id);
        results.push(ScoredVectorId::new(id, score));
    }

    let mut sorted = results.into_sorted_vec();
    sorted.reverse();
    sorted
}

pub async fn parallel_search_async(
    index: &FlatIndex,
    query: &[f32],
    k: usize,
) -> Vec<ScoredVectorId> {
    let num_threads = get_num_threads();
    let total_len = index.total_len();

    if total_len == 0 {
        return Vec::new();
    }

    if total_len < num_threads * 1000 {
        return index.search(query, k);
    }

    let chunk_size = total_len.div_ceil(num_threads);
    let dim = index.dimension();
    let distance_fn = index.distance_fn();

    // Clone Arc references to the underlying data. This is very cheap (just increments ref count)
    let vectors: Arc<[f32]> = index.vectors().clone().into();
    let ids: Arc<[u64]> = index
        .ids()
        .iter()
        .map(|id: &VectorId| id.as_u64())
        .collect::<Vec<_>>()
        .into();
    let deleted: Arc<RoaringBitmap> = Arc::new(index.deleted().clone());
    let query_arc: Arc<[f32]> = query.to_vec().into();

    let mut join_set = JoinSet::new();

    for thread_idx in 0..num_threads {
        let start_idx = thread_idx * chunk_size;
        let end_idx = (start_idx + chunk_size).min(total_len);

        if start_idx >= total_len {
            break;
        }

        // Clone Arc references for this thread's closure
        let vectors_clone = Arc::clone(&vectors);
        let ids_clone = Arc::clone(&ids);
        let deleted_clone = Arc::clone(&deleted);
        let query_clone = Arc::clone(&query_arc);

        // Spawn a blocking task for this chunk of data
        join_set.spawn_blocking(move || {
            let chunk_vectors = &vectors_clone[start_idx * dim..end_idx * dim];
            let chunk_ids = &ids_clone[start_idx..end_idx];

            scan_chunk(
                chunk_vectors,
                chunk_ids,
                &deleted_clone,
                &query_clone,
                distance_fn,
                k,
            )
        });
    }

    let mut all_results: Vec<ScoredVectorId> = Vec::new();

    // Collect results as they complete
    while let Some(res) = join_set.join_next().await {
        if let Ok(chunk_results) = res {
            all_results.extend(chunk_results);
        }
    }

    // Final top-k reduction across all chunks
    all_results.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(Ordering::Less)
    });
    all_results.truncate(k);
    all_results.reverse();
    all_results
}
