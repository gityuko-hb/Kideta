//! HNSW search algorithms.
//!
//! # Search Strategy
//!
//! HNSW search consists of two phases:
//! 1. **Hierarchical traversal**: Start from entry point at highest layer,
//!    greedily traverse to find closest node at each level
//! 2. **Layer 0 beam search**: At layer 0, perform beam search with ef width
//!
//! # Early Termination
//!
//! During search, we maintain a result set (max-heap) and a candidates set
//! (min-heap). Search terminates when the best candidate's distance exceeds
//! the worst result's distance (since all remaining candidates are further).

use kideta_core::distance::prefetch;
use kideta_core::metric::DistanceMetric;
use kideta_core::types::VectorId;
use kideta_core::utils::bitset::FixedBitset;
use kideta_core::utils::heap::{BoundedMaxHeap, MinHeap};

use crate::hnsw::graph::HnswGraph;
use crate::hnsw::params::HnswParams;
use crate::quantization::QuantizedStorage;
use crate::traits::ScoredVectorId;

#[allow(dead_code)]
pub struct HnswSearcher<'a> {
    graph: &'a HnswGraph,
    params: &'a HnswParams,
    distance_fn: fn(&[f32], &[f32]) -> f32,
}

impl<'a> HnswSearcher<'a> {
    pub fn new(
        graph: &'a HnswGraph,
        params: &'a HnswParams,
        metric: DistanceMetric,
    ) -> Self {
        let distance_fn = match metric {
            DistanceMetric::Cosine => kideta_core::distance::cosine_f32,
            DistanceMetric::L2 => kideta_core::distance::l2_f32,
            DistanceMetric::DotProduct => kideta_core::distance::dot_distance_f32,
            DistanceMetric::Hamming => {
                // Hamming operates on binary vectors - use wrapper
                hamming_distance_wrapper
            },
        };
        Self {
            graph,
            params,
            distance_fn,
        }
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Vec<ScoredVectorId> {
        if self.graph.is_empty() {
            return Vec::new();
        }

        let entry_point = match self.graph.get_entry_point() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let mut current_node = entry_point.node_id;
        let level = entry_point.level;

        let mut lvl = level;
        while lvl > 0 {
            current_node = self.greedy_traverse_layer(query, current_node, lvl, 1);
            lvl -= 1;
        }

        self.search_layer_0(query, k, ef, Some(current_node))
    }

    pub fn greedy_traverse_layer(
        &self,
        query: &[f32],
        start_node: usize,
        level: usize,
        _ef: usize,
    ) -> usize {
        let n = self.graph.len();
        if n == 0 || start_node >= n {
            return start_node;
        }
        let mut candidates = MinHeap::with_capacity(1);
        let mut visited = FixedBitset::new(n);
        visited.set(start_node);

        let query_vector = query;
        let Some(start_vector) = self.graph.get_vector(start_node) else {
            return start_node;
        };
        let current_dist = (self.distance_fn)(query_vector, start_vector);
        candidates.push(ScoredVectorId::new(
            VectorId::new(start_node as u64),
            current_dist,
        ));

        let mut best_node = start_node;
        let mut best_dist = current_dist;

        while let Some(candidate) = candidates.pop() {
            if candidate.score > best_dist {
                break;
            }

            let candidate_idx = candidate.id.as_u64() as usize;
            if candidate_idx >= n {
                continue;
            }

            let neighbors = self
                .graph
                .get_neighbors_unchecked(candidate_idx, level);

            // Cache for fast prefetch (avoid double get_vector)
            let graph_ptr = self.graph.vectors_data_ptr();
            let graph_dim = self.graph.dimension();

            for (j, &neighbor_id) in neighbors.iter().enumerate() {
                let neighbor_idx = neighbor_id as usize;
                if neighbor_idx >= n {
                    continue;
                }
                if visited.get(neighbor_idx) {
                    continue;
                }
                visited.set(neighbor_idx);

                // Prefetch 2 neighbors ahead (direct ptr arithmetic, no get_vector)
                let pf_j = j + 2;
                if pf_j < neighbors.len() {
                    let pf_idx = neighbors[pf_j] as usize;
                    if pf_idx < n {
                        if let Some(ptr) = graph_ptr {
                            unsafe {
                                prefetch(ptr.add(pf_idx * graph_dim));
                            }
                        }
                    }
                }

                let Some(neighbor_vector) = self.graph.get_vector(neighbor_idx) else {
                    continue;
                };

                prefetch(neighbor_vector.as_ptr());

                let dist = (self.distance_fn)(query_vector, neighbor_vector);

                if dist < best_dist {
                    best_dist = dist;
                    best_node = neighbor_idx;
                }

                candidates.push(ScoredVectorId::new(
                    VectorId::new(neighbor_idx as u64),
                    dist,
                ));
            }
        }

        best_node
    }

    pub fn search_layer(
        &self,
        query: &[f32],
        k: usize,
        level: usize,
        start_node: Option<usize>,
    ) -> Vec<ScoredVectorId> {
        let n = self.graph.len();
        if n == 0 || level >= self.graph.num_layers() {
            return Vec::new();
        }

        let ef_multiplier = if level == 0 {
            8
        } else {
            2
        };
        let ef = (k * ef_multiplier).max(1);

        let mut candidates = MinHeap::with_capacity(ef);
        let mut results = BoundedMaxHeap::new(ef);
        let mut visited = FixedBitset::new(n);

        let entry_point = match start_node {
            Some(node_id) => {
                if node_id >= n {
                    return Vec::new();
                }
                crate::hnsw::graph::EntryPoint { node_id, level }
            },
            None => match self.graph.get_entry_point() {
                Some(ep) => ep,
                None => return Vec::new(),
            },
        };

        let ep_vector = match self.graph.get_vector(entry_point.node_id) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let ep_dist = (self.distance_fn)(query, ep_vector);

        let ep_vid = VectorId::new(entry_point.node_id as u64);
        if n > entry_point.node_id {
            candidates.push(ScoredVectorId::new(ep_vid, ep_dist));
            visited.set(entry_point.node_id);
            results.push(ScoredVectorId::new(ep_vid, ep_dist));
        }

        while let Some(candidate) = candidates.pop() {
            let worst_result = match results.peek_worst() {
                Some(w) => w.score,
                None => f32::INFINITY,
            };

            if candidate.score > worst_result {
                break;
            }

            let candidate_idx = candidate.id.as_u64() as usize;
            if candidate_idx >= n {
                continue;
            }

            let neighbors = self
                .graph
                .get_neighbors_unchecked(candidate_idx, level);

            for &neighbor_id in neighbors {
                let neighbor_idx = neighbor_id as usize;
                if neighbor_idx >= n {
                    continue;
                }
                if visited.get(neighbor_idx) {
                    continue;
                }
                visited.set(neighbor_idx);

                let Some(neighbor_vector) = self.graph.get_vector(neighbor_idx) else {
                    continue;
                };

                prefetch(neighbor_vector.as_ptr());

                let dist = (self.distance_fn)(query, neighbor_vector);
                let n_vid = VectorId::new(neighbor_idx as u64);

                results.push(ScoredVectorId::new(n_vid, dist));
                candidates.push(ScoredVectorId::new(n_vid, dist));
            }
        }

        results.into_sorted_asc()
    }

    pub fn search_layer_0(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        start_node: Option<usize>,
    ) -> Vec<ScoredVectorId> {
        let n = self.graph.len();
        if n == 0 {
            return Vec::new();
        }

        let beam = ef.max(k);
        let mut candidates = MinHeap::with_capacity(beam);
        let mut results = BoundedMaxHeap::new(beam);
        let mut visited = FixedBitset::new(n);

        let entry_point = match start_node {
            Some(node_id) => {
                if node_id >= n {
                    return Vec::new();
                }
                crate::hnsw::graph::EntryPoint { node_id, level: 0 }
            },
            None => match self.graph.get_entry_point() {
                Some(ep) => ep,
                None => return Vec::new(),
            },
        };

        let ep_vector = match self.graph.get_vector(entry_point.node_id) {
            Some(v) => v,
            None => return Vec::new(),
        };
        let ep_dist = (self.distance_fn)(query, ep_vector);

        let ep_vid = VectorId::new(entry_point.node_id as u64);
        candidates.push(ScoredVectorId::new(ep_vid, ep_dist));
        visited.set(entry_point.node_id);
        results.push(ScoredVectorId::new(ep_vid, ep_dist));

        while let Some(candidate) = candidates.pop() {
            let worst_result = match results.peek_worst() {
                Some(w) => w.score,
                None => f32::INFINITY,
            };

            if candidate.score > worst_result {
                break;
            }

            let neighbors = self
                .graph
                .get_neighbors_unchecked(candidate.id.as_u64() as usize, 0);

            // Cache for fast prefetch (avoid double get_vector)
            let graph_ptr = self.graph.vectors_data_ptr();
            let graph_dim = self.graph.dimension();

            for (j, &neighbor_id) in neighbors.iter().enumerate() {
                let neighbor_idx = neighbor_id as usize;
                if visited.get(neighbor_idx) {
                    continue;
                }
                visited.set(neighbor_idx);

                // Prefetch 2 neighbors ahead (direct ptr arithmetic, no get_vector)
                let pf_j = j + 2;
                if pf_j < neighbors.len() {
                    let pf_idx = neighbors[pf_j] as usize;
                    if pf_idx < n
                        && let Some(ptr) = graph_ptr
                    {
                        unsafe {
                            prefetch(ptr.add(pf_idx * graph_dim));
                        }
                    }
                }

                let Some(neighbor_vector) = self.graph.get_vector(neighbor_idx) else {
                    continue;
                };

                prefetch(neighbor_vector.as_ptr());

                let dist = (self.distance_fn)(query, neighbor_vector);
                let n_vid = VectorId::new(neighbor_idx as u64);

                results.push(ScoredVectorId::new(n_vid, dist));
                candidates.push(ScoredVectorId::new(n_vid, dist));
            }
        }

        results.into_sorted_asc()
    }

    /// Quantized search using approximate distances during graph traversal.
    ///
    /// Uses `QuantizedStorage::approx_distance()` instead of exact f32 distances
    /// during both hierarchical traversal and layer 0 beam search. This avoids
    /// fetching full vectors during navigation, reducing memory bandwidth.
    ///
    /// Returns candidates for rescore — the caller should rescore with exact distances.
    pub fn search_quantized(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        storage: &QuantizedStorage,
    ) -> Vec<ScoredVectorId> {
        if self.graph.is_empty() {
            return Vec::new();
        }

        let entry_point = match self.graph.get_entry_point() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let mut current_node = entry_point.node_id;
        let level = entry_point.level;

        let mut lvl = level;
        while lvl > 0 {
            current_node =
                self.greedy_traverse_layer_quantized(query, current_node, lvl, 1, storage);
            lvl -= 1;
        }

        self.search_layer_0_quantized(query, k, ef, storage)
    }

    /// Greedy traversal using approximate distances.
    fn greedy_traverse_layer_quantized(
        &self,
        query: &[f32],
        start_node: usize,
        level: usize,
        _ef: usize,
        storage: &QuantizedStorage,
    ) -> usize {
        let n = self.graph.len();
        if n == 0 || start_node >= n {
            return start_node;
        }
        let mut candidates = MinHeap::with_capacity(1);
        let mut visited = FixedBitset::new(n);
        visited.set(start_node);

        let start_dist = storage
            .approx_distance(query, start_node)
            .unwrap_or(f32::MAX);
        candidates.push(ScoredVectorId::new(
            VectorId::new(start_node as u64),
            start_dist,
        ));

        let mut best_node = start_node;
        let mut best_dist = start_dist;

        while let Some(candidate) = candidates.pop() {
            if candidate.score > best_dist {
                break;
            }

            let candidate_idx = candidate.id.as_u64() as usize;
            if candidate_idx >= n {
                continue;
            }

            let neighbors = self
                .graph
                .get_neighbors_unchecked(candidate_idx, level);

            for &neighbor_id in neighbors {
                let neighbor_idx = neighbor_id as usize;
                if neighbor_idx >= n {
                    continue;
                }
                if visited.get(neighbor_idx) {
                    continue;
                }
                visited.set(neighbor_idx);

                let dist = storage
                    .approx_distance(query, neighbor_idx)
                    .unwrap_or(f32::MAX);

                if dist < best_dist {
                    best_dist = dist;
                    best_node = neighbor_idx;
                }

                candidates.push(ScoredVectorId::new(
                    VectorId::new(neighbor_idx as u64),
                    dist,
                ));
            }
        }

        best_node
    }

    /// Layer 0 beam search using approximate distances.
    fn search_layer_0_quantized(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        storage: &QuantizedStorage,
    ) -> Vec<ScoredVectorId> {
        let n = self.graph.len();
        if n == 0 {
            return Vec::new();
        }

        let mut candidates = MinHeap::with_capacity(ef);
        let mut results = BoundedMaxHeap::new(k);
        let mut visited = FixedBitset::new(n);

        let entry_point = match self.graph.get_entry_point() {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let ep_dist = storage
            .approx_distance(query, entry_point.node_id)
            .unwrap_or(f32::MAX);

        let ep_vid = VectorId::new(entry_point.node_id as u64);
        candidates.push(ScoredVectorId::new(ep_vid, ep_dist));
        visited.set(entry_point.node_id);
        results.push(ScoredVectorId::new(ep_vid, ep_dist));

        while let Some(candidate) = candidates.pop() {
            let worst_result = match results.peek_worst() {
                Some(w) => w.score,
                None => f32::INFINITY,
            };

            if candidate.score > worst_result {
                break;
            }

            let neighbors = self
                .graph
                .get_neighbors_unchecked(candidate.id.as_u64() as usize, 0);

            for &neighbor_id in neighbors {
                let neighbor_idx = neighbor_id as usize;
                if visited.get(neighbor_idx) {
                    continue;
                }
                visited.set(neighbor_idx);

                let dist = storage
                    .approx_distance(query, neighbor_idx)
                    .unwrap_or(f32::MAX);
                let n_vid = VectorId::new(neighbor_idx as u64);

                results.push(ScoredVectorId::new(n_vid, dist));
                candidates.push(ScoredVectorId::new(n_vid, dist));
            }
        }

        results.into_sorted_asc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> (HnswGraph, HnswParams) {
        let params = HnswParams::new(16, 200, 50);
        let mut graph = HnswGraph::new(4, 16, 100);

        for i in 0..10 {
            let vector = vec![
                (i as f32) * 0.1,
                (i as f32) * 0.2,
                (i as f32) * 0.3,
                (i as f32) * 0.4,
            ];
            graph.add_node(&vector);
            if i > 0 {
                graph.add_edge(0, i, 0);
            }
        }

        graph.update_entry_point(0, 0);

        (graph, params)
    }

    #[test]
    fn test_search_empty() {
        let graph = HnswGraph::new(4, 16, 10);
        let params = HnswParams::new(16, 200, 50);
        let searcher = HnswSearcher::new(&graph, &params, DistanceMetric::L2);

        let results = searcher.search(&[0.0, 0.0, 0.0, 0.0], 3, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_layer_0() {
        let (graph, params) = create_test_graph();
        let searcher = HnswSearcher::new(&graph, &params, DistanceMetric::L2);

        let query = [0.05, 0.1, 0.15, 0.2];
        let results = searcher.search_layer_0(&query, 3, 10, None);

        assert!(!results.is_empty());
    }
}

/// Wrapper to adapt Hamming distance for f32 vectors
fn hamming_distance_wrapper(
    a: &[f32],
    b: &[f32],
) -> f32 {
    kideta_core::distance::hamming_distance_f32(a, b) as f32
}
