//! HNSW Index — Hierarchical Navigable Small World graph.
//!
//! HNSW is a state-of-the-art approximate nearest neighbor (ANN) search algorithm
//! that builds a multi-layer graph with exponentially decaying probability of
//! connections at higher layers. This provides O(log n) search complexity while
//! achieving high recall.
//!
//! # Architecture
//!
//! - **Layers**: Each node appears at level L with probability p^L where p = 1/ln(M)
//! - **Search**: Greedy best-first traversal from top layer down to layer 0
//! - **Insert**: Layer-by-layer insertion with local neighbor refinement
//!
//! # Implementation
//!
//! This module is organized into:
//! - `params.rs` — HNSW configuration parameters
//! - `graph.rs` — Graph data structure and entry point management
//! - `searcher.rs` — `search_layer` and full `search` algorithms
//! - `neighbors.rs` — Simple and heuristic neighbor selection
//! - `builder.rs` — Sequential and parallel index construction
//! - `serial.rs` — Binary serialization format

pub mod builder;
pub mod graph;
pub mod neighbors;
pub mod params;
pub mod searcher;
pub mod serial;

pub use builder::{BuildState, HnswIndex};
pub use graph::HnswGraph;
pub use params::HnswParams;
