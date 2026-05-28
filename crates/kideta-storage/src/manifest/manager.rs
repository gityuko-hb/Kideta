use crate::manifest::commit::{
    atomic_commit, current_manifest_path, delete_manifest_version, list_manifest_versions, read_manifest,
    read_manifest_version,
};
use crate::manifest::manifest_data::{Manifest, MappingCheckpoint};
use crate::manifest::segment_ref::SegmentRef;

use std::fs;
use std::path::PathBuf;
use std::sync::RwLock;

const DEFAULT_MAX_VERSIONS: usize = 10;

pub struct ManifestManager {
    dir: PathBuf,
    current: RwLock<Manifest>,
    max_versions: usize,
}

impl ManifestManager {
    pub fn open(dir: &PathBuf) -> std::io::Result<Self> {
        fs::create_dir_all(dir)?;

        let current = if current_manifest_path(dir).exists() {
            read_manifest(dir).unwrap_or_else(|_| Manifest::new(0, 0))
        } else {
            Manifest::new(0, 0)
        };

        Ok(Self {
            dir: dir.clone(),
            current: RwLock::new(current),
            max_versions: DEFAULT_MAX_VERSIONS,
        })
    }

    pub fn commit(&self, mut manifest: Manifest) -> std::io::Result<Manifest> {
        let new_version = manifest.version + 1;
        manifest.version = new_version;
        atomic_commit(&self.dir, &manifest)?;
        *self.current.write().unwrap() = manifest.clone();
        self.cleanup_old_versions()?;
        Ok(manifest)
    }

    pub fn commit_with_lsn(&self, wal_lsn: u64) -> std::io::Result<Manifest> {
        let mut current = self.current.read().unwrap().clone();
        current.wal_lsn = wal_lsn;
        self.commit(current)
    }

    pub fn commit_mapping_checkpoint(&self, wal_lsn: u64, mapping_file: String) -> std::io::Result<Manifest> {
        let mut current = self.current.read().unwrap().clone();
        current.wal_lsn = wal_lsn;
        current.mapping_checkpoint = MappingCheckpoint { wal_lsn, mapping_file };
        self.commit(current)
    }

    pub fn get_current(&self) -> Manifest {
        self.current.read().unwrap().clone()
    }

    pub fn get_version(&self, version: u64) -> std::io::Result<Manifest> {
        read_manifest_version(&self.dir, version)
    }

    pub fn add_segment(&self, segment: SegmentRef, wal_lsn: u64) -> std::io::Result<Manifest> {
        let mut current = self.get_current();
        current.add_segment(segment);
        current.wal_lsn = wal_lsn;
        self.commit(current)
    }

    pub fn update_segment(
        &self,
        segment_id: u64,
        f: impl FnOnce(&mut SegmentRef),
        wal_lsn: u64,
    ) -> std::io::Result<Manifest> {
        let mut current = self.get_current();
        current.wal_lsn = wal_lsn;
        current.update_segment(segment_id, f);
        self.commit(current)
    }

    pub fn replace_segments(
        &self,
        remove_ids: &[u64],
        add_segment: SegmentRef,
        wal_lsn: u64,
    ) -> std::io::Result<Manifest> {
        let mut current = self.get_current();
        current.replace_segments(remove_ids, add_segment);
        current.wal_lsn = wal_lsn;
        self.commit(current)
    }

    pub fn remove_segments(&self, segment_ids: &[u64], wal_lsn: u64) -> std::io::Result<Manifest> {
        let mut current = self.get_current();
        current.remove_segments(segment_ids);
        current.wal_lsn = wal_lsn;
        self.commit(current)
    }

    pub fn list_versions(&self) -> std::io::Result<Vec<u64>> {
        list_manifest_versions(&self.dir)
    }

    pub fn wal_lsn(&self) -> u64 {
        self.current.read().unwrap().wal_lsn
    }

    fn cleanup_old_versions(&self) -> std::io::Result<()> {
        let versions = list_manifest_versions(&self.dir)?;
        if versions.len() <= self.max_versions {
            return Ok(());
        }
        let to_delete = &versions[..versions.len() - self.max_versions];
        for &v in to_delete {
            delete_manifest_version(&self.dir, v)?;
        }
        Ok(())
    }

    pub fn set_max_versions(&mut self, max: usize) {
        self.max_versions = max;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn open_creates_dir_if_missing() {
        let dir = tempdir().unwrap();
        let manager = ManifestManager::open(&dir.path().join("nonexistent")).unwrap();
        assert_eq!(manager.get_current().version, 0);
    }

    #[test]
    fn commit_increments_version() {
        let dir = tempdir().unwrap();
        let manager = ManifestManager::open(&dir.path().to_path_buf()).unwrap();
        let m = Manifest::new(0, 0);
        let committed = manager.commit(m).unwrap();
        assert_eq!(committed.version, 1);
    }

    #[test]
    fn commit_with_lsn_updates_wal_lsn() {
        let dir = tempdir().unwrap();
        let manager = ManifestManager::open(&dir.path().to_path_buf()).unwrap();
        let committed = manager.commit_with_lsn(999).unwrap();
        assert_eq!(committed.wal_lsn, 999);
        assert_eq!(manager.wal_lsn(), 999);
    }

    #[test]
    fn add_segment() {
        let dir = tempdir().unwrap();
        let manager = ManifestManager::open(&dir.path().to_path_buf()).unwrap();
        let seg = SegmentRef::new(1, PathBuf::from("/tmp/seg1"), 100, 5, false, 1024);
        manager.add_segment(seg, 10).unwrap();
        let current = manager.get_current();
        assert_eq!(current.segment_count(), 1);
        assert_eq!(current.stats.total_vectors, 100);
    }

    #[test]
    fn remove_segments() {
        let dir = tempdir().unwrap();
        let manager = ManifestManager::open(&dir.path().to_path_buf()).unwrap();
        let seg = SegmentRef::new(1, PathBuf::from("/tmp/seg1"), 100, 5, false, 1024);
        manager.add_segment(seg, 10).unwrap();
        manager.remove_segments(&[1], 20).unwrap();
        assert!(manager.get_current().is_empty());
    }

    #[test]
    fn update_segment() {
        let dir = tempdir().unwrap();
        let manager = ManifestManager::open(&dir.path().to_path_buf()).unwrap();
        let seg = SegmentRef::new(1, PathBuf::from("/tmp/seg1"), 100, 5, false, 1024);
        manager.add_segment(seg, 10).unwrap();
        manager.update_segment(1, |s| s.index_ready = true, 10).unwrap();
        let current = manager.get_current();
        assert!(current.segments[0].index_ready);
    }

    #[test]
    fn replace_segments_commits_remove_and_add_once() -> std::io::Result<()> {
        let dir = tempdir()?;
        let manager = ManifestManager::open(&dir.path().to_path_buf())?;

        let first = SegmentRef::new(1, "segment_1".into(), 10, 0, true, 100);
        let second = SegmentRef::new(2, "segment_2".into(), 20, 0, true, 200);
        manager.add_segment(first, 0)?;
        manager.add_segment(second, 0)?;
        let before = manager.get_current().version;

        let merged = SegmentRef::new(3, "segment_3".into(), 30, 0, true, 300);
        let after = manager.replace_segments(&[1, 2], merged, 99)?;

        assert_eq!(after.version, before + 1);
        assert_eq!(after.wal_lsn, 99);
        assert_eq!(after.segments.len(), 1);
        assert_eq!(after.segments[0].id, 3);
        assert_eq!(after.stats.total_vectors, 30);

        Ok(())
    }

    #[test]
    fn cleanup_preserves_latest_versions() {
        let dir = tempdir().unwrap();
        let mut manager = ManifestManager::open(&dir.path().to_path_buf()).unwrap();
        manager.set_max_versions(3);
        for i in 0..5 {
            let m = Manifest::new(i, 0);
            manager.commit(m).unwrap();
        }
        let versions = manager.list_versions().unwrap();
        assert_eq!(versions, vec![3, 4, 5]);
    }
}
