use crate::vector_storage::engine::MmapVectorStorage;
use kideta_core::vector_store::VectorStore;
use std::path::Path;

pub struct MultiSegmentVectorStore {
    segment_stores: Vec<MmapVectorStorage>,
    offsets: Vec<usize>,
    dim: u32,
}

impl MultiSegmentVectorStore {
    /// Create from a sorted list of segment directory paths.
    /// Each path should contain a "vectors.bin" file.
    /// Returns None if any segment file can't be opened.
    pub fn open(
        segment_dirs: &[&Path],
        dim: u32,
    ) -> Option<Self> {
        if segment_dirs.is_empty() {
            return None;
        }

        let mut stores = Vec::with_capacity(segment_dirs.len());
        let mut offsets = Vec::with_capacity(segment_dirs.len() + 1);
        offsets.push(0);

        for dir in segment_dirs {
            let path = dir.join("vectors.bin");
            let mvs = MmapVectorStorage::open_readonly(&path).ok()?;
            let count = mvs.len();
            offsets.push(*offsets.last().unwrap() + count);
            stores.push(mvs);
        }

        Some(Self {
            segment_stores: stores,
            offsets,
            dim,
        })
    }
}

impl VectorStore for MultiSegmentVectorStore {
    fn len(&self) -> usize {
        *self.offsets.last().unwrap_or(&0)
    }

    fn dimension(&self) -> usize {
        self.dim as usize
    }

    fn get_vector(
        &self,
        i: usize,
    ) -> Option<&[f32]> {
        if i >= self.len() {
            return None;
        }

        let seg_idx = match self.offsets.binary_search(&i) {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        };

        if seg_idx >= self.segment_stores.len() {
            return None;
        }

        let local = i - self.offsets[seg_idx];
        self.segment_stores[seg_idx].get_vector(local)
    }
}
