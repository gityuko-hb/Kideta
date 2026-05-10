//! HNSW graph data structure.
//!
//! # Graph Structure
//!
//! The graph consists of multiple layers where:
//! - Layer 0 contains all nodes with their full neighbor lists
//! - Higher layers contain progressively fewer nodes with sparse connections
//! - Each node at layer L appears in layer L+1 with probability p^L
//!
//! # Storage Layout
//!
//! - `layers[l]` — adjacency list for layer L (`Vec` of `InlineSmallVec`)
//! - `vectors` — vector storage via [`VectorSource`] enum
//! - `entry_point` — `(node_id, level)` of the highest-level node
//!
//! # Vector Storage Abstraction
//!
//! The [`VectorSource`] enum acts as a bridge between in-memory and mmap'd
//! vector storage:
//!
//! - [`VectorSource::InMemory`] wraps a [`VecVectorStore`] — used during
//!   construction and by [`InMemoryCollection`]. Supports push operations
//!   for incremental index building.
//! - [`VectorSource::Mmap`] wraps a [`Box<dyn VectorStore>`] — used by
//!   [`PersistentCollection`] after sealing. Provides zero-copy reads
//!   from memory-mapped files.
//!
//! The graph seamlessly switches between modes via [`set_vector_store`](HnswGraph::set_vector_store).
//! All search code uses the same [`get_vector`](HnswGraph::get_vector) method regardless of
//! which storage backend is active.
//!
//! [`VecVectorStore`]: kideta_core::vector_store::VecVectorStore
//! [`VectorStore`]: kideta_core::vector_store::VectorStore
//! [`InMemoryCollection`]: kideta_storage::collection::InMemoryCollection
//! [`PersistentCollection`]: kideta_storage::persistent_collection::PersistentCollection

use kideta_core::vector_store::{VecVectorStore, VectorStore};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone)]
pub struct InlineSmallVec {
    pub data: Vec<u32>,
}

impl InlineSmallVec {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push(
        &mut self,
        value: u32,
    ) {
        self.data.push(value);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(
        &self,
        i: usize,
    ) -> Option<u32> {
        self.data.get(i).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = &u32> {
        self.data.iter()
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        self.data.clear();
    }

    pub fn contains(
        &self,
        value: u32,
    ) -> bool {
        self.data.contains(&value)
    }
}

impl Default for InlineSmallVec {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub neighbors: Vec<InlineSmallVec>,
}

impl Layer {
    pub fn new(node_count: usize) -> Self {
        Self {
            neighbors: vec![InlineSmallVec::new(); node_count],
        }
    }

    pub fn grow(
        &mut self,
        additional: usize,
    ) {
        self.neighbors
            .resize_with(self.neighbors.len() + additional, InlineSmallVec::new);
    }

    pub fn add_edge(
        &mut self,
        from: usize,
        to: usize,
        max_m: usize,
    ) {
        debug_assert!(from < self.neighbors.len());
        let nbrs = &mut self.neighbors[from];
        if nbrs.len() < max_m * 2 && !nbrs.contains(to as u32) {
            nbrs.push(to as u32);
        }
    }

    pub fn add_neighbor(
        &mut self,
        node_idx: usize,
        neighbor_id: u32,
    ) {
        if node_idx >= self.neighbors.len() {
            self.grow(node_idx + 1 - self.neighbors.len());
        }
        self.neighbors[node_idx].push(neighbor_id);
    }

    pub fn active_nodes(&self) -> usize {
        self.neighbors
            .iter()
            .filter(|n| !n.is_empty())
            .count()
    }

    pub fn node_at(
        &self,
        index: usize,
    ) -> Option<usize> {
        if index < self.neighbors.len() && !self.neighbors[index].is_empty() {
            Some(index)
        } else {
            None
        }
    }

    pub fn iter_active_nodes(&self) -> impl Iterator<Item = usize> + '_ {
        self.neighbors
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.is_empty())
            .map(|(i, _)| i)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntryPoint {
    pub node_id: usize,
    pub level: usize,
}

impl EntryPoint {
    fn new(
        node_id: usize,
        level: usize,
    ) -> Self {
        Self { node_id, level }
    }
}

pub(crate) enum VectorSource {
    InMemory(VecVectorStore),
    Mmap(Arc<dyn VectorStore>),
}

impl VectorSource {
    fn len(&self) -> usize {
        match self {
            VectorSource::InMemory(s) => s.len(),
            VectorSource::Mmap(s) => s.len(),
        }
    }

    fn get_vector(
        &self,
        i: usize,
    ) -> Option<&[f32]> {
        match self {
            VectorSource::InMemory(s) => s.get_vector(i),
            VectorSource::Mmap(s) => s.get_vector(i),
        }
    }

    fn push(
        &mut self,
        v: &[f32],
    ) {
        match self {
            VectorSource::InMemory(s) => s.push(v),
            VectorSource::Mmap(_) => panic!("cannot push to mmap-backed store"),
        }
    }

    fn as_f32_vec(&self) -> Option<&Vec<f32>> {
        match self {
            VectorSource::InMemory(s) => Some(&s.data),
            VectorSource::Mmap(_) => None,
        }
    }

    fn data_ptr(&self) -> Option<*const f32> {
        match self {
            VectorSource::InMemory(s) => s.data_ptr(),
            VectorSource::Mmap(s) => s.data_ptr(),
        }
    }
}

impl std::fmt::Debug for VectorSource {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            VectorSource::InMemory(s) => f.debug_tuple("InMemory").field(&s.len()).finish(),
            VectorSource::Mmap(s) => f.debug_tuple("Mmap").field(&s.len()).finish(),
        }
    }
}

#[derive(Debug)]
pub struct HnswGraph {
    layers: Vec<Layer>,
    vectors: VectorSource,
    entry_point: RwLock<Option<EntryPoint>>,
    dimension: usize,
    max_m: usize,
}

impl Clone for HnswGraph {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            vectors: self.vectors.clone(),
            entry_point: RwLock::new(self.entry_point.read().unwrap().clone()),
            dimension: self.dimension,
            max_m: self.max_m,
        }
    }
}

impl Clone for VectorSource {
    fn clone(&self) -> Self {
        match self {
            VectorSource::InMemory(s) => VectorSource::InMemory(s.clone()),
            VectorSource::Mmap(s) => VectorSource::Mmap(Arc::clone(s)),
        }
    }
}

impl HnswGraph {
    pub fn new(
        dimension: usize,
        max_m: usize,
        expected_count: usize,
    ) -> Self {
        let layers = vec![Layer::new(expected_count)];
        Self {
            layers,
            vectors: VectorSource::InMemory(VecVectorStore::new(dimension, expected_count)),
            entry_point: RwLock::new(None),
            dimension,
            max_m,
        }
    }

    pub fn set_vectors(
        &mut self,
        vectors: Vec<f32>,
    ) {
        let mut store = VecVectorStore::new(self.dimension, vectors.len() / self.dimension);
        store.data = vectors;
        self.vectors = VectorSource::InMemory(store);
    }

    /// Replaces the vector storage with a mmap-backed store.
    ///
    /// After calling this, the graph reads vectors from the provided
    /// [`VectorStore`] (usually an `MmapVectorStorage`). The old in-memory
    /// `VecVectorStore` is dropped, freeing RAM.
    ///
    /// This is called by [`PersistentCollection`] after the index is built
    /// and vectors are flushed to disk.
    ///
    /// # Panics
    ///
    /// Panics if the graph still uses the built-in `VecVectorStore` for
    /// incremental insertion — this method should only be called after
    /// the index is fully constructed and sealed.
    pub fn set_vector_store(
        &mut self,
        store: Box<dyn VectorStore>,
    ) {
        self.vectors = VectorSource::Mmap(Arc::from(store));
    }

    pub fn with_vector_store(
        store: Box<dyn VectorStore>,
        dimension: usize,
    ) -> Self {
        Self {
            layers: Vec::new(),
            vectors: VectorSource::Mmap(Arc::from(store)),
            entry_point: RwLock::new(None),
            dimension,
            max_m: 0,
        }
    }

    pub fn add_layer(&mut self) {
        self.layers.push(Layer::new(self.len()));
    }

    pub fn get_layer(
        &self,
        layer_idx: usize,
    ) -> Option<&Layer> {
        self.layers.get(layer_idx)
    }

    pub fn get_layer_mut(
        &mut self,
        layer_idx: usize,
    ) -> Option<&mut Layer> {
        self.layers.get_mut(layer_idx)
    }

    pub fn add_neighbor(
        &mut self,
        layer_idx: usize,
        node_idx: usize,
        neighbor_id: u32,
    ) {
        while self.layers.len() <= layer_idx {
            self.add_layer();
        }
        self.layers[layer_idx].add_neighbor(node_idx, neighbor_id);
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.len() == 0
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn max_m(&self) -> usize {
        self.max_m
    }

    pub fn max_level(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    pub fn get_vector(
        &self,
        node_id: usize,
    ) -> Option<&[f32]> {
        self.vectors.get_vector(node_id)
    }

    #[allow(dead_code)]
    pub fn get_neighbors(
        &self,
        node_id: usize,
        level: usize,
    ) -> Option<Vec<u32>> {
        if level >= self.layers.len() || node_id >= self.layers[level].neighbors.len() {
            return None;
        }
        let layer = &self.layers[level];
        Some(layer.neighbors[node_id].iter().copied().collect())
    }

    pub fn get_neighbors_unchecked(
        &self,
        node_id: usize,
        level: usize,
    ) -> &[u32] {
        &self.layers[level].neighbors[node_id].data
    }

    pub fn add_node(
        &mut self,
        vector: &[f32],
    ) -> usize {
        let node_id = self.len();
        self.vectors.push(vector);

        for layer in &mut self.layers {
            layer.neighbors.push(InlineSmallVec::new());
        }
        node_id
    }

    pub fn add_edge(
        &mut self,
        from: usize,
        to: usize,
        level: usize,
    ) {
        while level >= self.layers.len() {
            self.layers.push(Layer::new(self.len()));
        }
        self.layers[level].add_edge(from, to, self.max_m);
    }

    pub fn update_entry_point(
        &self,
        node_id: usize,
        level: usize,
    ) {
        let mut ep = self.entry_point.write().unwrap();
        match ep.as_ref() {
            Some(current) if current.level >= level => {},
            _ => {
                *ep = Some(EntryPoint::new(node_id, level));
            },
        }
    }

    pub fn get_entry_point(&self) -> Option<EntryPoint> {
        self.entry_point.read().unwrap().clone()
    }

    pub fn neighbor_count(
        &self,
        node_id: usize,
        level: usize,
    ) -> usize {
        if level >= self.layers.len() {
            return 0;
        }
        self.layers[level].neighbors[node_id].len()
    }

    pub fn avg_connections(
        &self,
        level: usize,
    ) -> f64 {
        if level >= self.layers.len() {
            return 0.0;
        }
        let layer = &self.layers[level];
        let total: usize = layer.neighbors.iter().map(|n| n.len()).sum();
        let nodes = layer.neighbors.len();
        if nodes == 0 {
            0.0
        } else {
            total as f64 / nodes as f64
        }
    }

    pub fn size_bytes(&self) -> usize {
        let vectors_bytes = match &self.vectors {
            VectorSource::InMemory(s) => s.data.capacity() * std::mem::size_of::<f32>(),
            VectorSource::Mmap(_) => 0,
        };
        let layers_bytes = self.layers.iter().fold(0, |acc, layer| {
            acc + layer
                .neighbors
                .iter()
                .map(|n| n.data.capacity() * std::mem::size_of::<u32>())
                .sum::<usize>()
        }) + self.layers.iter().fold(0, |acc, layer| {
            acc + layer.neighbors.capacity() * std::mem::size_of::<InlineSmallVec>()
        });
        vectors_bytes + layers_bytes + std::mem::size_of_val(&*self.entry_point.read().unwrap())
    }

    pub fn stats(&self) -> HnswStats {
        HnswStats {
            num_elements: self.len(),
            num_layers: self.num_layers(),
            avg_connections_l0: self.avg_connections(0),
            size_bytes: self.size_bytes(),
        }
    }

    pub fn vectors_ref(&self) -> Option<&Vec<f32>> {
        self.vectors.as_f32_vec()
    }

    /// Returns a raw pointer to the start of the flat f32 vector array,
    /// if the current storage backend provides one.
    ///
    /// Used by the searcher for pointer-arithmetic-based prefetching in
    /// the hot search path. Returns `None` for storage backends that don't
    /// expose a contiguous f32 array.
    pub fn vectors_data_ptr(&self) -> Option<*const f32> {
        self.vectors.data_ptr()
    }

    pub fn layers_ref(&self) -> &Vec<Layer> {
        &self.layers
    }
}

#[derive(Debug, Clone)]
pub struct HnswStats {
    pub num_elements: usize,
    pub num_layers: usize,
    pub avg_connections_l0: f64,
    pub size_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_small_vec() {
        let mut vec = InlineSmallVec::new();
        assert!(vec.is_empty());
        vec.push(1);
        vec.push(2);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.get(0), Some(1));
        assert_eq!(vec.get(1), Some(2));
    }

    #[test]
    fn test_graph_add_node() {
        let mut graph = HnswGraph::new(4, 16, 10);
        graph.add_node(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(graph.len(), 1);
        assert_eq!(graph.get_vector(0), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    #[test]
    fn test_graph_add_edge() {
        let mut graph = HnswGraph::new(4, 16, 10);
        graph.add_node(&[1.0, 2.0, 3.0, 4.0]);
        graph.add_node(&[5.0, 6.0, 7.0, 8.0]);
        graph.add_edge(0, 1, 0);
        assert_eq!(graph.neighbor_count(0, 0), 1);
    }

    #[test]
    fn test_entry_point() {
        let graph = HnswGraph::new(4, 16, 10);
        assert!(graph.get_entry_point().is_none());
    }
}
