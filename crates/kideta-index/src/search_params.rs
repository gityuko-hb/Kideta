//! Search parameters for different index types.
//!
//! This module defines the runtime search parameters used when querying
//! approximate nearest neighbor indexes. Each index type has its own
//! parameter struct that controls the search behavior.

use kideta_core::enums::IndexType;

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

#[derive(Debug, Clone)]
pub struct HnswSearchParams {
    pub ef: usize,
}

impl HnswSearchParams {
    pub fn new(ef: usize) -> Self {
        Self { ef }
    }

    pub fn from_k(k: usize) -> Self {
        Self { ef: k }
    }

    pub fn increase(&self) -> Self {
        Self { ef: self.ef * 2 }
    }
}

#[derive(Debug, Clone)]
pub enum SearchParams {
    Flat(FlatSearchParams),
    Hnsw(HnswSearchParams),
}

impl SearchParams {
    pub fn from_index_type_and_k(
        index_type: IndexType,
        k: usize,
    ) -> Self {
        match index_type {
            IndexType::Flat => SearchParams::Flat(FlatSearchParams::from_k(k)),
            IndexType::Hnsw => SearchParams::Hnsw(HnswSearchParams::from_k(k)),
            _ => panic!("Unsupported index type for search params"),
        }
    }

    pub fn index_type(&self) -> IndexType {
        match self {
            SearchParams::Flat(_) => IndexType::Flat,
            SearchParams::Hnsw(_) => IndexType::Hnsw,
        }
    }

    pub fn is_adaptive_for(
        &self,
        other: &SearchParams,
    ) -> bool {
        matches!(
            (self, other),
            (SearchParams::Flat(_), SearchParams::Flat(_))
                | (SearchParams::Hnsw(_), SearchParams::Hnsw(_))
        )
    }
}

impl From<FlatSearchParams> for SearchParams {
    fn from(params: FlatSearchParams) -> Self {
        SearchParams::Flat(params)
    }
}

impl From<HnswSearchParams> for SearchParams {
    fn from(params: HnswSearchParams) -> Self {
        SearchParams::Hnsw(params)
    }
}

impl Default for FlatSearchParams {
    fn default() -> Self {
        Self { k: 10 }
    }
}

impl Default for HnswSearchParams {
    fn default() -> Self {
        Self { ef: 100 }
    }
}
