//! Search parameters for different index types.
//!
//! This module defines the runtime search parameters used when querying
//! approximate nearest neighbor indexes. Each index type has its own
//! parameter struct that controls the search behavior.

#[derive(Debug, Clone)]
pub struct FlatSearchParams {
    pub k: usize,
}

impl FlatSearchParams {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    pub fn from_k(k: usize) -> Self {
        Self { k }
    }

    pub fn increase(&self) -> Self {
        Self { k: self.k * 2 }
    }
}
