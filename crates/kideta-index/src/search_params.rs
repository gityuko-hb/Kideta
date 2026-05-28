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
    Ivf(IvfSearchParams),
    IvfPq(IvfPqSearchParams),
}

impl SearchParams {
    pub fn from_index_type_and_k(
        index_type: IndexType,
        k: usize,
    ) -> Self {
        match index_type {
            IndexType::Flat => SearchParams::Flat(FlatSearchParams::from_k(k)),
            IndexType::Hnsw => SearchParams::Hnsw(HnswSearchParams::from_k(k)),
            IndexType::Ivf => SearchParams::Ivf(IvfSearchParams::from_k(k)),
            IndexType::IvfPQ => SearchParams::IvfPq(IvfPqSearchParams::from_k(k)),
            _ => panic!("Unsupported index type for search params"),
        }
    }

    pub fn index_type(&self) -> IndexType {
        match self {
            SearchParams::Flat(_) => IndexType::Flat,
            SearchParams::Hnsw(_) => IndexType::Hnsw,
            SearchParams::Ivf(_) => IndexType::Ivf,
            SearchParams::IvfPq(_) => IndexType::IvfPQ,
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
                | (SearchParams::Ivf(_), SearchParams::Ivf(_))
                | (SearchParams::IvfPq(_), SearchParams::IvfPq(_))
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

#[derive(Debug, Clone)]
pub struct IvfSearchParams {
    pub nprobe: usize,
}

impl IvfSearchParams {
    pub fn new(nprobe: usize) -> Self {
        Self { nprobe }
    }

    pub fn from_k(k: usize) -> Self {
        let nprobe = k.clamp(1, 100);
        Self { nprobe }
    }

    pub fn increase(&self) -> Self {
        Self {
            nprobe: self.nprobe * 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IvfPqSearchParams {
    pub nprobe: usize,
    pub rescore_factor: usize,
}

impl IvfPqSearchParams {
    pub fn new(
        nprobe: usize,
        rescore_factor: usize,
    ) -> Self {
        Self {
            nprobe,
            rescore_factor,
        }
    }

    pub fn from_k(k: usize) -> Self {
        let nprobe = k.clamp(1, 100);
        Self {
            nprobe,
            rescore_factor: 4,
        }
    }

    pub fn increase(&self) -> Self {
        Self {
            nprobe: self.nprobe * 2,
            rescore_factor: self.rescore_factor * 2,
        }
    }
}
