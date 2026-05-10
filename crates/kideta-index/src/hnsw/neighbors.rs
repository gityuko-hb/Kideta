//! Neighbor selection algorithms for HNSW.
//!
//! # Neighbor Selection
//!
//! When inserting a new node, we need to select which existing nodes will
//! be its neighbors. HNSW supports two strategies:
//!
//! 1. **Simple selection**: Select the M nearest neighbors by distance.
//!    Fast but can create "chain" effects where all neighbors lie on a line.
//!
//! 2. **Heuristic selection** (Malkov-style): Adds randomization to avoid
//!    chain effects. When candidate A is considered, if A is closer to B than
//!    A is to the new node, we skip A (B already has a path to the new node
//!    through A).

use kideta_core::types::VectorId;
use kideta_core::utils::bitset::FixedBitset;
use kideta_core::utils::heap::MinHeap;
use kideta_core::utils::rng::Xorshift64;

use crate::hnsw::graph::HnswGraph;
use crate::traits::ScoredVectorId;

pub struct NeighborSelector<'a> {
    graph: &'a HnswGraph,
    distance_fn: fn(&[f32], &[f32]) -> f32,
}

impl<'a> NeighborSelector<'a> {
    pub fn new(
        graph: &'a HnswGraph,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> Self {
        Self { graph, distance_fn }
    }

    pub fn select_simple<M: Into<usize>>(
        &self,
        query: &[f32],
        candidates: &[usize],
        m: M,
    ) -> Vec<usize> {
        let m = m.into();
        let mut heap = MinHeap::with_capacity(candidates.len());

        for &candidate_id in candidates {
            let Some(vector) = self.graph.get_vector(candidate_id) else {
                continue;
            };
            let dist = (self.distance_fn)(query, vector);
            heap.push(ScoredVectorId::new(
                VectorId::new(candidate_id as u64),
                dist,
            ));
        }

        let mut results = Vec::with_capacity(m);
        for _ in 0..m {
            if let Some(scored) = heap.pop() {
                results.push(scored.id.as_u64() as usize);
            }
        }
        results
    }

    #[allow(dead_code)]
    pub fn select_heuristic<M: Into<usize>>(
        &self,
        query: &[f32],
        candidates: &[usize],
        m: M,
        _rng: &mut Xorshift64,
    ) -> Vec<usize> {
        let m = m.into();
        let mut heap = MinHeap::with_capacity(candidates.len());

        for &candidate_id in candidates {
            let Some(vector) = self.graph.get_vector(candidate_id) else {
                continue;
            };
            let dist = (self.distance_fn)(query, vector);
            heap.push(ScoredVectorId::new(
                VectorId::new(candidate_id as u64),
                dist,
            ));
        }

        let mut results = Vec::with_capacity(m);
        let mut used = FixedBitset::new(self.graph.len());

        while results.len() < m {
            let Some(best) = heap.pop() else {
                break;
            };
            let best_id = best.id.as_u64() as usize;
            let best_dist = best.score;

            let mut dominated = false;
            for &result_id in &results {
                if used.get(result_id) {
                    continue;
                }
                let Some(result_vector) = self.graph.get_vector(result_id) else {
                    continue;
                };
                let Some(best_vector) = self.graph.get_vector(best_id) else {
                    continue;
                };
                let dist_to_result = (self.distance_fn)(best_vector, result_vector);

                if dist_to_result < best_dist {
                    dominated = true;
                    break;
                }
            }

            if !dominated {
                results.push(best_id);
                used.set(best_id);
            }
        }

        results
    }
}

#[allow(dead_code)]
pub fn select_neighbors_simple<M: Into<usize>>(
    graph: &HnswGraph,
    distance_fn: fn(&[f32], &[f32]) -> f32,
    query: &[f32],
    candidates: &[usize],
    m: M,
) -> Vec<usize> {
    let selector = NeighborSelector::new(graph, distance_fn);
    selector.select_simple(query, candidates, m)
}

#[allow(dead_code)]
pub fn select_neighbors_heuristic<M: Into<usize>>(
    graph: &HnswGraph,
    distance_fn: fn(&[f32], &[f32]) -> f32,
    query: &[f32],
    candidates: &[usize],
    m: M,
    rng: &mut Xorshift64,
) -> Vec<usize> {
    let selector = NeighborSelector::new(graph, distance_fn);
    selector.select_heuristic(query, candidates, m, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> HnswGraph {
        let mut graph = HnswGraph::new(4, 16, 100);
        for i in 0..10 {
            let vector = vec![
                (i as f32) * 0.1,
                (i as f32) * 0.2,
                (i as f32) * 0.3,
                (i as f32) * 0.4,
            ];
            graph.add_node(&vector);
        }
        graph
    }

    #[test]
    fn test_simple_selection() {
        let graph = create_test_graph();
        let candidates = vec![0, 1, 2, 3, 4];
        let query = [0.25, 0.5, 0.75, 1.0];

        let neighbors = select_neighbors_simple(
            &graph,
            kideta_core::distance::l2_f32,
            &query,
            &candidates,
            3usize,
        );

        assert_eq!(neighbors.len(), 3);
    }

    #[test]
    fn test_heuristic_selection() {
        let graph = create_test_graph();
        let candidates = vec![0, 1, 2, 3, 4];
        let query = [0.25, 0.5, 0.75, 1.0];
        let mut rng = Xorshift64::new(42);

        let neighbors = select_neighbors_heuristic(
            &graph,
            kideta_core::distance::l2_f32,
            &query,
            &candidates,
            3usize,
            &mut rng,
        );

        assert_eq!(neighbors.len(), 3);
    }
}
