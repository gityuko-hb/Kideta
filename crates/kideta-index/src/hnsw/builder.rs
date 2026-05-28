//! HNSW index builder and main struct.
//!
//! This module provides the `HnswIndex` struct which implements the
//! `VectorIndex` trait and provides the main HNSW algorithm implementation.
//!
//! # Construction
//!
//! ```rust
//! use kideta_index::hnsw::HnswParams;
//! use kideta_index::hnsw::HnswIndex;
//! use kideta_core::metric::DistanceMetric;
//!
//! let params = HnswParams::new(16, 200, 50);
//! let index = HnswIndex::new(128, params, DistanceMetric::L2);
//! ```
//!
//! # Insert & Search
//!
//! Vectors are inserted one at a time with `insert()`. The index builds
//! incrementally - no batch rebuild is required. Search with `search()`.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use kideta_core::metric::DistanceMetric;
use kideta_core::payload::Payload;
use kideta_core::types::VectorId;
use kideta_core::utils::rng::Xorshift64;
use kideta_core::utils::roaring::RoaringBitmap;

use crate::hnsw::graph::{HnswGraph, HnswStats};
use crate::hnsw::neighbors::select_neighbors_heuristic;
use crate::hnsw::params::HnswParams;
use crate::hnsw::searcher::HnswSearcher;
use crate::quantization::{QuantizationConfig, QuantizedStorage, encode_vectors};
use crate::search_params::SearchParams;
use crate::traits::{FilterFn, IndexError, ScoredVectorId, VectorIndex};

#[derive(Debug, Clone, PartialEq)]
pub enum BuildState {
    NotBuilt,
    Building,
    Built,
    Failed,
}

pub struct HnswIndex {
    graph: RwLock<HnswGraph>,
    graph_sealed: Option<Arc<HnswGraph>>,
    build_state: BuildState,
    params: HnswParams,
    metric: DistanceMetric,
    distance_fn: fn(&[f32], &[f32]) -> f32,
    rng: RwLock<Xorshift64>,
    deleted: RwLock<RoaringBitmap>,
    payloads: RwLock<Vec<Option<Payload>>>,
    id_to_node: RwLock<HashMap<VectorId, usize>>,
    node_to_id: RwLock<HashMap<usize, VectorId>>,
    operation_count: RwLock<usize>,
    quantization_config: Option<QuantizationConfig>,
    quantized_storage: Option<QuantizedStorage>,
}

impl HnswIndex {
    pub fn new(
        dimension: usize,
        params: HnswParams,
        metric: DistanceMetric,
    ) -> Self {
        let distance_fn = match metric {
            DistanceMetric::Cosine => kideta_core::distance::cosine_f32,
            DistanceMetric::L2 => kideta_core::distance::l2_f32,
            DistanceMetric::DotProduct => kideta_core::distance::dot_distance_f32,
            DistanceMetric::Hamming => {
                // Hamming operates on binary vectors - use wrapper that converts f32 to binary
                hamming_distance_wrapper
            },
        };

        Self {
            graph: RwLock::new(HnswGraph::new(dimension, params.m, 1000)),
            graph_sealed: None,
            build_state: BuildState::NotBuilt,
            params,
            metric,
            distance_fn,
            rng: RwLock::new(Xorshift64::new(42)),
            deleted: RwLock::new(RoaringBitmap::new()),
            payloads: RwLock::new(Vec::new()),
            id_to_node: RwLock::new(HashMap::new()),
            node_to_id: RwLock::new(HashMap::new()),
            operation_count: RwLock::new(0),
            quantization_config: None,
            quantized_storage: None,
        }
    }

    pub fn with_capacity(
        dimension: usize,
        params: HnswParams,
        metric: DistanceMetric,
        expected_count: usize,
    ) -> Self {
        let distance_fn = match metric {
            DistanceMetric::Cosine => kideta_core::distance::cosine_f32,
            DistanceMetric::L2 => kideta_core::distance::l2_f32,
            DistanceMetric::DotProduct => kideta_core::distance::dot_distance_f32,
            DistanceMetric::Hamming => {
                // Hamming operates on binary vectors - use wrapper that converts f32 to binary
                hamming_distance_wrapper
            },
        };

        Self {
            graph: RwLock::new(HnswGraph::new(dimension, params.m, expected_count)),
            graph_sealed: None,
            build_state: BuildState::NotBuilt,
            params,
            metric,
            distance_fn,
            rng: RwLock::new(Xorshift64::new(42)),
            deleted: RwLock::new(RoaringBitmap::new()),
            payloads: RwLock::new(Vec::with_capacity(expected_count)),
            id_to_node: RwLock::new(HashMap::with_capacity(expected_count)),
            node_to_id: RwLock::new(HashMap::with_capacity(expected_count)),
            operation_count: RwLock::new(0),
            quantization_config: None,
            quantized_storage: None,
        }
    }

    pub fn stats(&self) -> HnswStats {
        let graph = self.get_readable_graph();
        graph.stats()
    }

    fn generate_level(&self) -> usize {
        let mut rng = self.rng.write().unwrap();
        rng.hnsw_level(self.params.ml)
    }

    fn insert_internal(
        &self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError> {
        if vector.len() != self.dimension() {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension(),
                got: vector.len(),
            });
        }

        let mut graph = self.graph.write().unwrap();
        let node_id = graph.add_node(vector);

        {
            let mut payloads = self.payloads.write().unwrap();
            payloads.push(Some(payload));
        }

        {
            let mut id_to_node = self.id_to_node.write().unwrap();
            id_to_node.insert(id, node_id);
        }

        {
            let mut node_to_id = self.node_to_id.write().unwrap();
            node_to_id.insert(node_id, id);
        }

        let level = self.generate_level().min(self.params.max_level);

        graph.update_entry_point(node_id, level);

        let entry_point = match graph.get_entry_point() {
            Some(ep) => ep,
            None => return Ok(()),
        };

        while graph.num_layers() <= entry_point.level.max(level) {
            graph.add_layer();
        }

        let query_vector = vector;
        let mut _current_node = entry_point.node_id;
        let mut current_level = entry_point.level;

        while current_level > level + 1 {
            let searcher = HnswSearcher::new(&graph, &self.params, self.metric);
            let results =
                searcher.search_layer(query_vector, 1, current_level, Some(_current_node));
            if let Some(first) = results.first() {
                _current_node = first.id.as_u64() as usize;
            }
            current_level -= 1;
        }

        for l in (0..=level).rev() {
            let searcher = HnswSearcher::new(&graph, &self.params, self.metric);
            let candidates = if l == 0 {
                searcher.search_layer_0(
                    query_vector,
                    self.params.ef_construction,
                    self.params.ef_construction,
                    Some(_current_node),
                )
            } else {
                searcher.search_layer(query_vector, self.params.m, l, Some(_current_node))
            };
            let candidate_ids: Vec<usize> = candidates
                .iter()
                .map(|s| s.id.as_u64() as usize)
                .collect();

            let neighbors = select_neighbors_heuristic(
                &graph,
                self.distance_fn,
                query_vector,
                &candidate_ids,
                self.params.m,
                &mut Xorshift64::new(42),
            );

            for &neighbor_id in &neighbors {
                graph.add_edge(neighbor_id, node_id, l);
                graph.add_edge(node_id, neighbor_id, l);
            }
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn load(
        _dimension: usize,
        params: HnswParams,
        metric: DistanceMetric,
        graph: HnswGraph,
    ) -> Self {
        let distance_fn = match metric {
            DistanceMetric::Cosine => kideta_core::distance::cosine_f32,
            DistanceMetric::L2 => kideta_core::distance::l2_f32,
            DistanceMetric::DotProduct => kideta_core::distance::dot_distance_f32,
            DistanceMetric::Hamming => {
                // Hamming operates on binary vectors - use wrapper that converts f32 to binary
                hamming_distance_wrapper
            },
        };

        let node_count = graph.len();
        Self {
            graph: RwLock::new(graph),
            graph_sealed: None,
            build_state: BuildState::NotBuilt,
            params,
            metric,
            distance_fn,
            rng: RwLock::new(Xorshift64::new(42)),
            deleted: RwLock::new(RoaringBitmap::new()),
            payloads: RwLock::new(vec![None; node_count]),
            id_to_node: RwLock::new(HashMap::new()),
            node_to_id: RwLock::new(HashMap::new()),
            operation_count: RwLock::new(0),
            quantization_config: None,
            quantized_storage: None,
        }
    }

    pub fn build_state(&self) -> BuildState {
        self.build_state.clone()
    }

    pub fn is_index_ready(&self) -> bool {
        self.build_state == BuildState::Built
    }

    pub fn seal_for_reading(&mut self) {
        if self.build_state == BuildState::Built {
            let graph = self.graph.read().unwrap();
            self.graph_sealed = Some(Arc::new(graph.clone()));
        }
    }

    fn get_readable_graph(&self) -> Arc<HnswGraph> {
        if let Some(ref sealed) = self.graph_sealed {
            return sealed.clone();
        }
        self.graph.read().unwrap().clone().into()
    }

    fn build_internal(
        dimension: usize,
        params: HnswParams,
        metric: DistanceMetric,
        ids: Vec<VectorId>,
        vectors: Vec<Vec<f32>>,
        payloads: Vec<Option<Payload>>,
    ) -> Result<HnswGraph, IndexError> {
        let n = ids.len();
        if n == 0 {
            return Ok(HnswGraph::new(dimension, params.m, 0));
        }

        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let new_index = HnswIndex::with_capacity(dimension, params, metric, n);

            for ((id, vector), payload) in ids.into_iter().zip(vectors).zip(payloads) {
                let payload = payload.unwrap_or(Payload::default());
                if let Err(e) = new_index.insert_internal(id, &vector, payload) {
                    eprintln!("Error during background build: {:?}", e);
                }
            }

            let new_graph = new_index.graph.write().unwrap().clone();
            tx.send(new_graph).ok();
        });

        let rebuilt_graph = rx.recv().map_err(|_| IndexError::IndexNotReady)?;
        Ok(rebuilt_graph)
    }

    pub fn build_from_vectors(
        &mut self,
        ids: Vec<VectorId>,
        vectors: Vec<Vec<f32>>,
        payloads: Vec<Payload>,
    ) -> Result<(), IndexError> {
        if self.build_state == BuildState::Building {
            return Err(IndexError::IndexNotReady);
        }

        let dimension = self.dimension();
        let params = self.params;
        let metric = self.metric;

        if vectors.is_empty() {
            self.build_state = BuildState::Built;
            return Ok(());
        }

        self.build_state = BuildState::Building;

        let payloads: Vec<Option<Payload>> = payloads.into_iter().map(Some).collect();
        // Clone ids để sau build còn reconstruct node_to_id mapping
        let ids_clone = ids.clone();
        let rebuilt_graph =
            Self::build_internal(dimension, params, metric, ids, vectors, payloads)?;

        // ⚠ QUAN TRỌNG: rebuild node_to_id / id_to_node mapping
        // build_internal tạo index mới rồi chỉ lấy graph, mappings bị mất
        let mut node_to_id = self.node_to_id.write().unwrap();
        let mut id_to_node = self.id_to_node.write().unwrap();
        node_to_id.clear();
        id_to_node.clear();
        for (node_id, id) in ids_clone.iter().enumerate() {
            node_to_id.insert(node_id, *id);
            id_to_node.insert(*id, node_id);
        }
        drop(node_to_id);
        drop(id_to_node);

        self.graph_sealed = Some(Arc::new(rebuilt_graph));
        self.build_state = BuildState::Built;

        Ok(())
    }

    pub fn build_index(&mut self) -> Result<(), IndexError> {
        if self.build_state == BuildState::Building {
            return Err(IndexError::IndexNotReady);
        }

        let dimension = self.dimension();
        let params = self.params;
        let metric = self.metric;

        let (vectors, ids, payloads): (Vec<_>, Vec<_>, Vec<_>) = {
            let graph = self.graph.read().unwrap();
            let node_to_id = self.node_to_id.read().unwrap();
            let payloads = self.payloads.read().unwrap();

            let mut vectors = Vec::with_capacity(graph.len());
            let mut ids = Vec::with_capacity(graph.len());
            let mut payloads_out = Vec::with_capacity(graph.len());

            for node_id in 0..graph.len() {
                if let Some(vector) = graph.get_vector(node_id)
                    && let Some(&id) = node_to_id.get(&node_id)
                {
                    vectors.push(vector.to_vec());
                    ids.push(id);
                    payloads_out.push(payloads.get(node_id).cloned().flatten());
                }
            }
            (vectors, ids, payloads_out)
        };

        self.build_state = BuildState::Building;

        let ids_clone = ids.clone();
        let rebuilt_graph =
            Self::build_internal(dimension, params, metric, ids, vectors, payloads)?;

        // ⚠ Rebuild node_to_id mapping (build_internal không giữ mapping)
        let mut node_to_id = self.node_to_id.write().unwrap();
        let mut id_to_node = self.id_to_node.write().unwrap();
        node_to_id.clear();
        id_to_node.clear();
        for (node_id, id) in ids_clone.iter().enumerate() {
            node_to_id.insert(node_id, *id);
            id_to_node.insert(*id, node_id);
        }
        drop(node_to_id);
        drop(id_to_node);

        self.graph_sealed = Some(Arc::new(rebuilt_graph));
        self.build_state = BuildState::Built;

        Ok(())
    }

    pub fn check_connectivity(&self) -> bool {
        let graph = self.get_readable_graph();
        let entry_point = match graph.get_entry_point() {
            Some(ep) => ep.node_id,
            None => return true,
        };

        let deleted = self.deleted.read().unwrap();
        let total_nodes = graph.len();
        let mut visited = vec![false; total_nodes];
        let mut queue = vec![entry_point];
        visited[entry_point] = true;
        let mut visited_count = 1;

        while let Some(node) = queue.pop() {
            let neighbors = graph.get_neighbors_unchecked(node, 0);
            for &neighbor in neighbors {
                if !visited[neighbor as usize] {
                    visited[neighbor as usize] = true;
                    visited_count += 1;
                    queue.push(neighbor as usize);
                }
            }
        }

        let non_deleted_count = total_nodes.saturating_sub(deleted.len());
        visited_count >= non_deleted_count
    }

    pub fn ensure_connectivity(&mut self) -> bool {
        if self.check_connectivity() {
            return true;
        }

        self.graph_sealed = None;
        self.build_state = BuildState::NotBuilt;
        false
    }

    pub fn periodic_maintenance(&mut self) {
        {
            let mut count = self.operation_count.write().unwrap();
            *count = count.wrapping_add(1);
            if *count < 100 {
                return;
            }
            *count = 0;
        }

        let total_count = {
            let graph = self.graph.read().unwrap();
            graph.len()
        };
        let deleted_count = {
            let deleted = self.deleted.read().unwrap();
            deleted.len()
        };

        let deleted_ratio = if total_count > 0 {
            deleted_count as f64 / total_count as f64
        } else {
            0.0
        };

        const REBUILD_THRESHOLD: f64 = 0.20;
        if deleted_ratio > REBUILD_THRESHOLD || !self.check_connectivity() {
            self.graph_sealed = None;
            self.build_state = BuildState::NotBuilt;
        }
    }
}

impl VectorIndex for HnswIndex {
    fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<ScoredVectorId> {
        let graph = self.get_readable_graph();
        let searcher = HnswSearcher::new(&graph, &self.params, self.metric);
        let results = searcher.search(query, k, self.params.ef_search);
        let node_to_id = self.node_to_id.read().unwrap();

        let deleted = self.deleted.read().unwrap();
        results
            .into_iter()
            .filter_map(|r| {
                if deleted.contains(r.id.as_u64() as u32) {
                    return None;
                }
                let internal = r.id.as_u64() as usize;
                let original = node_to_id.get(&internal)?;
                Some(ScoredVectorId::new(*original, r.score))
            })
            .take(k)
            .collect()
    }

    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterFn>,
    ) -> Vec<ScoredVectorId> {
        let graph = self.get_readable_graph();
        let deleted = self.deleted.read().unwrap();
        let payloads = self.payloads.read().unwrap();
        let node_to_id = self.node_to_id.read().unwrap();

        let base_ef = self.params.ef_search;
        let max_ef = (base_ef * 8).max(1024);

        let mut current_ef = base_ef;
        let mut results = Vec::new();

        while results.len() < k && current_ef <= max_ef {
            let searcher = HnswSearcher::new(&graph, &self.params, self.metric);
            let raw_results = searcher.search(query, k, current_ef);

            let filtered: Vec<ScoredVectorId> = raw_results
                .into_iter()
                .filter(|r| {
                    if deleted.contains(r.id.as_u64() as u32) {
                        return false;
                    }

                    if let Some(f) = filter {
                        let node_id = r.id.as_u64() as usize;
                        let original_id = match node_to_id.get(&node_id) {
                            Some(&id) => id,
                            None => return false,
                        };
                        if node_id >= payloads.len() {
                            return false;
                        }
                        match &payloads[node_id] {
                            Some(p) => f(original_id, p),
                            None => false,
                        }
                    } else {
                        true
                    }
                })
                .collect();

            results = filtered;
            if results.len() < k {
                current_ef *= 2;
            }
        }

        results
    }

    fn search_with_params(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Vec<ScoredVectorId> {
        if let SearchParams::Hnsw(p) = params {
            let graph = self.get_readable_graph();
            let searcher = HnswSearcher::new(&graph, &self.params, self.metric);
            let results = searcher.search(query, k, p.ef);
            let deleted = self.deleted.read().unwrap();
            results
                .into_iter()
                .filter(|r| !deleted.contains(r.id.as_u64() as u32))
                .collect()
        } else {
            self.search(query, k)
        }
    }

    fn search_with_params_and_filter(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        filter: Option<&FilterFn>,
    ) -> Vec<ScoredVectorId> {
        if let SearchParams::Hnsw(p) = params {
            let graph = self.get_readable_graph();
            let deleted = self.deleted.read().unwrap();
            let payloads = self.payloads.read().unwrap();
            let node_to_id = self.node_to_id.read().unwrap();

            let base_ef = p.ef;
            let max_ef = (base_ef * 8).max(1024);

            let mut current_ef = base_ef;
            let mut results = Vec::new();

            while results.len() < k && current_ef <= max_ef {
                let searcher = HnswSearcher::new(&graph, &self.params, self.metric);
                let raw_results = searcher.search(query, k, current_ef);

                let filtered: Vec<ScoredVectorId> = raw_results
                    .into_iter()
                    .filter(|r| {
                        if deleted.contains(r.id.as_u64() as u32) {
                            return false;
                        }

                        if let Some(f) = filter {
                            let node_id = r.id.as_u64() as usize;
                            let original_id = match node_to_id.get(&node_id) {
                                Some(&id) => id,
                                None => return false,
                            };
                            if node_id >= payloads.len() {
                                return false;
                            }
                            match &payloads[node_id] {
                                Some(p) => f(original_id, p),
                                None => false,
                            }
                        } else {
                            true
                        }
                    })
                    .collect();

                results = filtered;
                if results.len() < k {
                    current_ef *= 2;
                }
            }

            results
        } else {
            self.search_with_filter(query, k, filter)
        }
    }

    fn insert(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError> {
        self.insert_internal(id, vector, payload)
    }

    fn delete(
        &mut self,
        id: VectorId,
    ) -> Result<(), IndexError> {
        let mut deleted = self.deleted.write().unwrap();
        deleted.insert(id.as_u64() as u32);
        Ok(())
    }

    fn update(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError> {
        {
            let mut deleted = self.deleted.write().unwrap();
            deleted.insert(id.as_u64() as u32);
        }

        self.insert_internal(id, vector, payload)?;

        let total_count = {
            let graph = self.graph.read().unwrap();
            graph.len()
        };
        let deleted_count = {
            let deleted = self.deleted.read().unwrap();
            deleted.len()
        };

        let deleted_ratio = if total_count > 0 {
            deleted_count as f64 / total_count as f64
        } else {
            0.0
        };

        const REBUILD_THRESHOLD: f64 = 0.20;
        if deleted_ratio > REBUILD_THRESHOLD {
            self.graph_sealed = None;
            self.build_state = BuildState::NotBuilt;
        }

        Ok(())
    }

    fn len(&self) -> usize {
        let graph = self.graph.read().unwrap();
        graph.len() - self.deleted.read().unwrap().len()
    }

    fn dimension(&self) -> usize {
        self.graph.read().unwrap().dimension()
    }

    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn size_bytes(&self) -> usize {
        let graph = self.graph.read().unwrap();
        graph.size_bytes()
    }

    fn quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    fn quantized_storage(&self) -> Option<&QuantizedStorage> {
        self.quantized_storage.as_ref()
    }

    fn search_quantized(
        &self,
        query: &[f32],
        k: usize,
        rescore_factor: usize,
    ) -> Vec<ScoredVectorId> {
        let storage = match &self.quantized_storage {
            Some(s) => s,
            None => return self.search(query, k),
        };

        let graph = self.get_readable_graph();
        let distance_fn = self.distance_fn;

        let base_ef = (self.params.ef_search * rescore_factor / 10).max(100);
        let searcher = HnswSearcher::new(&graph, &self.params, self.metric);

        // Phase 1: Quantized graph traversal using approximate distances
        let candidates = searcher.search_quantized(query, k * rescore_factor, base_ef, storage);

        let deleted = self.deleted.read().unwrap();
        let node_to_id = self.node_to_id.read().unwrap();

        // Phase 2: Rescore top candidates with exact f32 distances
        let mut scored: Vec<ScoredVectorId> = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if deleted.contains(candidate.id.as_u64() as u32) {
                continue;
            }

            let node_id = candidate.id.as_u64() as usize;
            if let Some(vector) = graph.get_vector(node_id) {
                let score = distance_fn(query, vector);
                let original_id = node_to_id
                    .get(&node_id)
                    .copied()
                    .unwrap_or(candidate.id);
                scored.push(ScoredVectorId::new(original_id, score));
            }
        }

        scored.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored
    }

    fn train_quantization(
        &mut self,
        config: &QuantizationConfig,
    ) -> Result<(), IndexError> {
        let graph = self.graph.read().unwrap();
        let node_count = graph.len();

        if node_count == 0 {
            return Err(IndexError::Internal(
                "cannot train quantization on empty index".into(),
            ));
        }

        let dim = graph.dimension();
        if config.dimension() != Some(dim) {
            return Err(IndexError::Quantization(format!(
                "dimension mismatch: index={}, config={:?}",
                dim,
                config.dimension()
            )));
        }

        let vectors: Vec<Vec<f32>> = (0..node_count)
            .filter_map(|i| graph.get_vector(i).map(|v| v.to_vec()))
            .collect();

        drop(graph);

        let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let storage = encode_vectors(config, &vector_refs);
        self.quantized_storage = Some(storage);
        self.quantization_config = Some(config.clone());

        Ok(())
    }
}

/// Wrapper to adapt Hamming distance for f32 vectors used by HNSW
/// Converts f32 to binary: positive values -> 1, non-positive -> 0
fn hamming_distance_wrapper(
    a: &[f32],
    b: &[f32],
) -> f32 {
    kideta_core::distance::hamming_distance_f32(a, b) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_hnsw_index() {
        let params = HnswParams::new(16, 200, 50);
        let index = HnswIndex::new(128, params, DistanceMetric::L2);
        assert_eq!(index.len(), 0);
        assert_eq!(index.dimension(), 128);
    }

    #[test]
    fn test_insert_and_search() {
        let params = HnswParams::new(16, 200, 50);
        let mut index = HnswIndex::new(4, params, DistanceMetric::L2);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 1.0, 0.0];

        index
            .insert(VectorId::new(1), &v1, Payload::default())
            .unwrap();
        index
            .insert(VectorId::new(2), &v2, Payload::default())
            .unwrap();
        index
            .insert(VectorId::new(3), &v3, Payload::default())
            .unwrap();

        let query = vec![0.9, 0.1, 0.1, 0.1];
        let results = index.search(&query, 2);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_delete() {
        let params = HnswParams::new(16, 200, 50);
        let mut index = HnswIndex::new(4, params, DistanceMetric::L2);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];

        index
            .insert(VectorId::new(1), &v1, Payload::default())
            .unwrap();
        index
            .insert(VectorId::new(2), &v2, Payload::default())
            .unwrap();

        assert_eq!(index.len(), 2);

        index.delete(VectorId::new(1)).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_stats() {
        let params = HnswParams::new(16, 200, 50);
        let mut index = HnswIndex::new(4, params, DistanceMetric::L2);

        for i in 0..10 {
            let vector = vec![
                (i as f32) * 0.1,
                (i as f32) * 0.2,
                (i as f32) * 0.3,
                (i as f32) * 0.4,
            ];
            index
                .insert(VectorId::new(i as u64), &vector, Payload::default())
                .unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.num_elements, 10);
        assert!(stats.num_layers >= 1);
    }

    #[test]
    fn test_search_with_filter() {
        let params = HnswParams::new(16, 200, 50);
        let mut index = HnswIndex::new(4, params, DistanceMetric::L2);

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

        let query = vec![0.1, 0.1, 0.1, 0.1];
        let results_filtered = index.search_with_filter(&query, 10, Some(&filter));

        assert!(
            !results_filtered.is_empty(),
            "Should find at least one matching vector"
        );
        assert_eq!(
            results_filtered.len(),
            2,
            "Should find exactly 2 vectors with category=1"
        );
    }

    #[test]
    fn test_search_with_filter_empty_result() {
        let params = HnswParams::new(16, 200, 50);
        let mut index = HnswIndex::new(4, params, DistanceMetric::L2);

        let mut payload1 = Payload::new();
        payload1.insert("category", kideta_core::payload::PayloadValue::Int(1));

        index
            .insert(VectorId::new(1), &[0.0, 0.0, 0.0, 0.0], payload1)
            .unwrap();

        let filter = |_id: VectorId, payload: &Payload| {
            payload
                .get("category")
                .and_then(|v| v.as_int())
                .map(|v| v == 999)
                .unwrap_or(false)
        };

        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.search_with_filter(&query, 10, Some(&filter));

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_search_without_filter_returns_some() {
        let params = HnswParams::new(16, 200, 50);
        let mut index = HnswIndex::new(4, params, DistanceMetric::L2);

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

        let query = vec![0.1, 0.1, 0.1, 0.1];
        let results = index.search(&query, 10);

        assert!(!results.is_empty(), "Should find at least some vectors");
    }

    #[test]
    fn test_search_with_filter_all_pass() {
        let params = HnswParams::new(16, 200, 50);
        let mut index = HnswIndex::new(4, params, DistanceMetric::L2);

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

        let filter = |_id: VectorId, _payload: &Payload| true;

        let query = vec![0.1, 0.1, 0.1, 0.1];
        let results = index.search_with_filter(&query, 10, Some(&filter));

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_quantized_returns_results() {
        use crate::quantization::sq8::Sq8Stats;

        let params = HnswParams::new(16, 200, 50);
        let mut index = HnswIndex::new(16, params, DistanceMetric::L2);

        // Insert vectors
        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(100);
        for i in 0u32..100 {
            let vec: Vec<f32> = (0..16)
                .map(|j| {
                    let seed = ((i * 31 + j * 17) % 1000) as f32;
                    (seed / 1000.0 - 0.5) * 10.0
                })
                .collect();
            all_vectors.push(vec.clone());
            index
                .insert(VectorId::new(i as u64), &vec, Payload::default())
                .unwrap();
        }

        // Train SQ8 quantization
        let vector_refs: Vec<&[f32]> = all_vectors.iter().map(|v| v.as_slice()).collect();
        let stats = Sq8Stats::compute(&vector_refs);
        let config = QuantizationConfig::Sq8(stats.into_config());
        index.train_quantization(&config).unwrap();
        assert!(index.quantized_storage().is_some());

        // Search with quantized traversal + exact rescore
        let query: Vec<f32> = (0..16)
            .map(|j| ((j * 37) % 1000) as f32 / 1000.0 * 2.0)
            .collect();
        let quantized_results = index.search_quantized(&query, 10, 10);

        // Verify results are returned
        assert_eq!(quantized_results.len(), 10);

        // Verify all results have valid IDs and scores
        for result in &quantized_results {
            assert!(result.score.is_finite());
        }

        // Verify quantized search returns different (but overlapping) results vs exact search
        let exact_results = index.search(&query, 10);
        let exact_ids: std::collections::HashSet<_> = exact_results.iter().map(|r| r.id).collect();
        let quantized_ids: std::collections::HashSet<_> =
            quantized_results.iter().map(|r| r.id).collect();
        let overlap = exact_ids.intersection(&quantized_ids).count();
        assert!(
            overlap >= 3,
            "Expected at least 3 overlapping IDs, got {}",
            overlap
        );
    }
}
