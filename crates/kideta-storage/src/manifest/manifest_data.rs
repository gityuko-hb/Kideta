use crate::manifest::segment_ref::SegmentRef;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectionStats {
    pub total_vectors: u64,
    pub total_deleted: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MappingCheckpoint {
    pub wal_lsn: u64,
    pub mapping_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u64,
    pub segments: Vec<SegmentRef>,
    pub stats: CollectionStats,
    pub wal_lsn: u64,
    #[serde(default)]
    pub mapping_checkpoint: MappingCheckpoint,
    pub created_at: DateTime<Utc>,
}

impl Manifest {
    pub fn new(
        version: u64,
        wal_lsn: u64,
    ) -> Self {
        Self {
            version,
            segments: Vec::new(),
            stats: CollectionStats::default(),
            wal_lsn,
            mapping_checkpoint: MappingCheckpoint::default(),
            created_at: Utc::now(),
        }
    }

    pub fn add_segment(
        &mut self,
        segment: SegmentRef,
    ) {
        self.stats.total_vectors += segment.vector_count;
        self.stats.total_deleted += segment.deleted_count;
        self.stats.total_bytes += segment.file_size_bytes;
        self.segments.push(segment);
    }

    pub fn remove_segment(
        &mut self,
        segment_id: u64,
    ) {
        if let Some(pos) = self
            .segments
            .iter()
            .position(|s| s.id == segment_id)
        {
            let removed = self.segments.remove(pos);
            self.stats.total_vectors = self
                .stats
                .total_vectors
                .saturating_sub(removed.vector_count);
            self.stats.total_deleted = self
                .stats
                .total_deleted
                .saturating_sub(removed.deleted_count);
            self.stats.total_bytes = self
                .stats
                .total_bytes
                .saturating_sub(removed.file_size_bytes);
        }
    }

    pub fn remove_segments(
        &mut self,
        segment_ids: &[u64],
    ) {
        for id in segment_ids {
            self.remove_segment(*id);
        }
    }

    #[allow(dead_code)]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    pub fn update_segment(
        &mut self,
        segment_id: u64,
        f: impl FnOnce(&mut SegmentRef),
    ) {
        if let Some(segment) = self
            .segments
            .iter_mut()
            .find(|s| s.id == segment_id)
        {
            f(segment);
        }
    }

    pub fn replace_segments(
        &mut self,
        remove_ids: &[u64],
        add_segment: SegmentRef,
    ) {
        self.remove_segments(remove_ids);
        self.add_segment(add_segment);
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn manifest_new_is_empty() {
        let m = Manifest::new(1, 0);
        assert_eq!(m.version, 1);
        assert!(m.is_empty());
        assert_eq!(m.stats.total_vectors, 0);
    }

    #[test]
    fn add_segment_updates_stats() {
        let mut m = Manifest::new(1, 0);
        let seg = SegmentRef::new(1, PathBuf::from("/tmp/seg1"), 100, 5, false, 1024);
        m.add_segment(seg);
        assert_eq!(m.segment_count(), 1);
        assert_eq!(m.stats.total_vectors, 100);
        assert_eq!(m.stats.total_deleted, 5);
        assert_eq!(m.stats.total_bytes, 1024);
    }

    #[test]
    fn remove_segment_updates_stats() {
        let mut m = Manifest::new(1, 0);
        let seg = SegmentRef::new(1, PathBuf::from("/tmp/seg1"), 100, 5, false, 1024);
        m.add_segment(seg);
        m.remove_segment(1);
        assert!(m.is_empty());
        assert_eq!(m.stats.total_vectors, 0);
        assert_eq!(m.stats.total_deleted, 0);
        assert_eq!(m.stats.total_bytes, 0);
    }

    #[test]
    fn remove_multiple_segments() {
        let mut m = Manifest::new(1, 0);
        m.add_segment(SegmentRef::new(
            1,
            PathBuf::from("/tmp/seg1"),
            100,
            5,
            false,
            1024,
        ));
        m.add_segment(SegmentRef::new(
            2,
            PathBuf::from("/tmp/seg2"),
            200,
            10,
            false,
            2048,
        ));
        m.remove_segments(&[1, 2]);
        assert!(m.is_empty());
    }

    #[test]
    fn update_segment_changes_index_ready() {
        let mut m = Manifest::new(1, 0);
        let seg = SegmentRef::new(1, PathBuf::from("/tmp/seg1"), 100, 5, false, 1024);
        m.add_segment(seg);
        assert!(!m.segments[0].index_ready);
        m.update_segment(1, |s| s.index_ready = true);
        assert!(m.segments[0].index_ready);
    }

    #[test]
    fn update_segment_nonexistent_id_is_noop() {
        let mut m = Manifest::new(1, 0);
        let seg = SegmentRef::new(1, PathBuf::from("/tmp/seg1"), 100, 5, false, 1024);
        m.add_segment(seg);
        m.update_segment(999, |s| s.index_ready = true);
        assert_eq!(m.segment_count(), 1);
        assert!(!m.segments[0].index_ready);
    }
}
