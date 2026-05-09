//! KidetaStore — custom block-based key-value store.
//!
//! ## Architecture
//!
//! - **Data layer**: 4MB page files (`data_N.dat`) of 128-byte fixed blocks
//! - **Tracker**: mmap'd array of `BlockPtr` indexed by entry ID
//! - **Bitmask**: 1 bit per block, tracks used/free state
//! - **Gaps**: per-64-block summary metadata for O(1) amortized allocation
//! - **Lazy buffer**: defers tracker + bitmask mutations until `flush()`
//!
//! ## Write flow
//!
//! ```text
//! put(id, value)
//!   1. alloc blocks via gaps.find_free_blocks()
//!   2. write value into page file(s)
//!   3. buffer.push_put(id, BlockPtr{...})
//! ```
//!
//! ## Flush flow
//!
//! ```text
//! flush()
//!   1. buffer.apply(tracker, mask, gaps)
//!   2. page_manager.flush_all()
//!   3. tracker.flush()
//!   4. buffer.clear()
//! ```

use crate::store::block::{BLOCK_SIZE, BlockPtr};
use crate::store::gaps::GapsLayer;
use crate::store::lazy::{LazyBuffer, PendingUpdate};
use crate::store::mask::BitmaskLayer;
use crate::store::page_file::PageFileManager;
use crate::store::tracker::{TRACKER_FILENAME, TrackerArray};
use std::fs;
use std::path::Path;

pub const DEFAULT_PAGE_SIZE: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct StoreConfig {
    pub page_size: usize,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
        }
    }
}

pub struct KidetaStore {
    data_dir: std::path::PathBuf,
    tracker: TrackerArray,
    mask: BitmaskLayer,
    gaps: GapsLayer,
    page_manager: PageFileManager,
    buffer: LazyBuffer,
    total_blocks: u32,
    total_pages: u32,
    page_size: usize,
}

pub enum StoreError {
    InsufficientSpace,
    InvalidId,
    Io(std::io::Error),
    AlreadyExists,
}

impl std::fmt::Display for StoreError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            StoreError::InsufficientSpace => write!(f, "insufficient space in store"),
            StoreError::InvalidId => write!(f, "invalid entry id"),
            StoreError::Io(e) => write!(f, "I/O error: {}", e),
            StoreError::AlreadyExists => write!(f, "entry already exists"),
        }
    }
}

impl std::fmt::Debug for StoreError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            StoreError::InsufficientSpace => write!(f, "StoreError::InsufficientSpace"),
            StoreError::InvalidId => write!(f, "StoreError::InvalidId"),
            StoreError::Io(e) => write!(f, "StoreError::Io({})", e),
            StoreError::AlreadyExists => write!(f, "StoreError::AlreadyExists"),
        }
    }
}

impl std::error::Error for StoreError {}

impl PartialEq for StoreError {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        match (self, other) {
            (StoreError::InsufficientSpace, StoreError::InsufficientSpace) => true,
            (StoreError::InvalidId, StoreError::InvalidId) => true,
            (StoreError::AlreadyExists, StoreError::AlreadyExists) => true,
            (StoreError::Io(a), StoreError::Io(b)) => a.kind() == b.kind(),
            _ => false,
        }
    }
}

impl Eq for StoreError {}

impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        StoreError::Io(e)
    }
}

// impl From<crate::wal::error::WalError> for StoreError {
//     fn from(e: crate::wal::error::WalError) -> Self {
//         StoreError::Io(std::io::Error::other(e.to_string()))
//     }
// }

impl From<kideta_core::mmap::error::MmapError> for StoreError {
    fn from(e: kideta_core::mmap::error::MmapError) -> Self {
        StoreError::Io(std::io::Error::other(e.to_string()))
    }
}

impl From<kideta_core::error::KidetaError> for StoreError {
    fn from(e: kideta_core::error::KidetaError) -> Self {
        StoreError::Io(std::io::Error::other(e.to_string()))
    }
}

pub type Result<T> = std::result::Result<T, StoreError>;

impl KidetaStore {
    pub fn open(
        path: &Path,
        capacity: u32,
    ) -> Result<Self> {
        Self::open_with_config(path, capacity, StoreConfig::default())
    }

    pub fn open_with_config(
        path: &Path,
        capacity: u32,
        config: StoreConfig,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(StoreError::InsufficientSpace);
        }

        let page_size = config.page_size;
        let blocks_per_page = page_size / BLOCK_SIZE;
        let total_blocks = capacity;
        let total_pages = (capacity as usize).div_ceil(blocks_per_page);

        let data_dir = path.to_path_buf();
        fs::create_dir_all(&data_dir)?;

        let tracker_path = data_dir.join(TRACKER_FILENAME);
        let tracker = TrackerArray::open(&tracker_path, capacity)?;

        let mask = BitmaskLayer::new(capacity as usize);
        let mut gaps = GapsLayer::new(capacity as usize);

        gaps.rebuild_from_mask(&mask);

        let page_manager = PageFileManager::new_with_page_size(data_dir.clone(), page_size);

        let mut store = Self {
            data_dir,
            tracker,
            mask,
            gaps,
            page_manager,
            buffer: LazyBuffer::new(),
            total_blocks,
            total_pages: total_pages as u32,
            page_size,
        };

        store.gaps.rebuild_from_mask(&store.mask);

        Ok(store)
    }

    fn ensure_capacity(
        &mut self,
        id: u32,
        n_blocks: u32,
    ) -> Result<()> {
        let blocks_per_page = self.page_size / BLOCK_SIZE;
        let num_pages = self.page_manager.num_pages() as u32;
        let needed_entries = id
            .saturating_add(1)
            .max((num_pages + 1) * blocks_per_page as u32 + n_blocks);

        if needed_entries > self.total_blocks {
            self.tracker.grow(needed_entries)?;
            self.total_blocks = self.tracker.capacity();
            let total_blocks_usize = self.total_blocks as usize;
            if total_blocks_usize > self.mask.n_blocks() {
                self.mask.grow(total_blocks_usize);
                self.gaps.grow(total_blocks_usize, &self.mask);
            }
        }
        Ok(())
    }

    pub fn put(
        &mut self,
        id: u32,
        value: &[u8],
    ) -> Result<()> {
        if value.is_empty() {
            return Err(StoreError::InvalidId);
        }

        let n_blocks = value.len().div_ceil(BLOCK_SIZE) as u32;

        if id >= self.total_blocks {
            self.ensure_capacity(id, n_blocks)?;
        }

        let (page_id, byte_offset) = self.allocate_blocks(n_blocks)?;

        self.write_value(page_id, byte_offset, value)?;

        let ptr = BlockPtr {
            page_id,
            byte_offset,
            byte_len: value.len() as u16,
            _padding: [0; 2],
        };

        self.buffer.push_put(id, ptr);

        Ok(())
    }

    fn allocate_blocks(
        &mut self,
        n_blocks: u32,
    ) -> Result<(u32, u32)> {
        let blocks_per_page = self.page_size / BLOCK_SIZE;
        let (start_block, page_id, byte_offset) =
            if let Some((sb, _)) = self.gaps.find_free_blocks(n_blocks, &self.mask) {
                let pid = sb / blocks_per_page as u32;
                let bo = sb % blocks_per_page as u32;
                let byte_offset = bo * BLOCK_SIZE as u32;
                (sb, pid, byte_offset)
            } else {
                let new_page_id = self.page_manager.num_pages() as u32;
                self.page_manager.open_page(new_page_id)?;

                let sb = new_page_id * blocks_per_page as u32;
                if sb + n_blocks > self.total_blocks {
                    return Err(StoreError::InsufficientSpace);
                }

                (sb, new_page_id, 0)
            };

        self.mask.mark_used(start_block, n_blocks);
        self.gaps
            .update_after_alloc(start_block, n_blocks, &self.mask);

        Ok((page_id, byte_offset))
    }

    fn write_value(
        &mut self,
        page_id: u32,
        byte_offset: u32,
        value: &[u8],
    ) -> Result<()> {
        let start_byte = byte_offset as usize;
        let page = self.page_manager.open_page(page_id)?;
        page.write_at(start_byte, value);
        page.flush()?;
        Ok(())
    }

    pub fn get(
        &self,
        id: u32,
    ) -> Result<Option<Vec<u8>>> {
        if id >= self.total_blocks {
            return Err(StoreError::InvalidId);
        }

        match self.live_ptr(id) {
            Some(ptr) if !ptr.is_null() => self.get_raw(ptr),
            _ => Ok(None),
        }
    }

    fn live_ptr(
        &self,
        id: u32,
    ) -> Option<BlockPtr> {
        for update in self.buffer.updates().iter().rev() {
            match update {
                PendingUpdate::Put { id: uid, ptr } if *uid == id => {
                    return Some(*ptr);
                },
                PendingUpdate::Delete { id: uid, .. } if *uid == id => {
                    return None;
                },
                _ => {},
            }
        }

        let ptr = self.tracker.get(id);
        if !ptr.is_null() && ptr.byte_len > 0 {
            return Some(ptr);
        }

        None
    }

    fn get_raw(
        &self,
        ptr: BlockPtr,
    ) -> Result<Option<Vec<u8>>> {
        let page_id = ptr.page_id;
        let byte_offset = ptr.byte_offset;
        let byte_len = ptr.byte_len;
        let start_byte = byte_offset as usize;
        let end_byte = start_byte + byte_len as usize;
        let page = self
            .page_manager
            .open_page(page_id)
            .map_err(StoreError::Io)?;
        let slice = page.slice();
        Ok(Some(slice[start_byte..end_byte].to_vec()))
    }

    pub fn delete(
        &mut self,
        id: u32,
    ) -> Result<()> {
        if id >= self.total_blocks {
            return Err(StoreError::InvalidId);
        }

        let old_ptr = self.tracker.get(id);
        if old_ptr.is_null() {
            return Ok(());
        }

        self.buffer.push_delete(id, old_ptr);
        Ok(())
    }

    pub fn update(
        &mut self,
        id: u32,
        value: &[u8],
    ) -> Result<()> {
        if id >= self.total_blocks {
            return Err(StoreError::InvalidId);
        }

        let old_ptr = self.tracker.get(id);
        let n_blocks = value.len().div_ceil(BLOCK_SIZE) as u32;

        let (page_id, byte_offset) = self.allocate_blocks(n_blocks)?;
        self.write_value(page_id, byte_offset, value)?;

        let ptr = BlockPtr {
            page_id,
            byte_offset,
            byte_len: value.len() as u16,
            _padding: [0; 2],
        };

        if !old_ptr.is_null() {
            self.buffer.push_delete(id, old_ptr);
        }

        self.buffer.push_put(id, ptr);

        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.buffer
            .apply(&mut self.tracker, &mut self.mask, &mut self.gaps);

        self.page_manager
            .flush_all()
            .map_err(StoreError::Io)?;

        self.tracker.flush().map_err(StoreError::Io)?;

        self.buffer.clear();

        Ok(())
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter {
            store: self,
            next_id: 0,
        }
    }

    pub fn len(&self) -> u32 {
        self.total_blocks
    }

    pub fn is_empty(&self) -> bool {
        self.total_blocks == 0
    }

    pub fn capacity(&self) -> u32 {
        self.total_blocks
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }
}

pub struct Iter<'a> {
    store: &'a KidetaStore,
    next_id: u32,
}

impl Iterator for Iter<'_> {
    type Item = (u32, Vec<u8>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.next_id < self.store.total_blocks {
            let id = self.next_id;
            self.next_id += 1;

            let ptr = match self.store.live_ptr(id) {
                Some(p) if !p.is_null() => p,
                _ => continue,
            };

            if let Ok(Some(data)) = self.store.get_raw(ptr) {
                return Some((id, data));
            }
        }
        None
    }
}

impl std::fmt::Debug for KidetaStore {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("KidetaStore")
            .field("data_dir", &self.data_dir)
            .field("total_blocks", &self.total_blocks)
            .field("total_pages", &self.total_pages)
            .field("buffered_updates", &self.buffer.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_store() -> (tempfile::TempDir, KidetaStore) {
        let dir = tempdir().unwrap();
        let store = KidetaStore::open(dir.path(), 10000).unwrap();
        (dir, store)
    }

    #[test]
    fn put_and_get() {
        let (_dir, mut store) = test_store();

        store.put(0, b"hello world").unwrap();
        let result = store.get(0).unwrap();
        assert_eq!(result, Some(b"hello world".to_vec()));
    }

    #[test]
    fn put_multiple_ids() {
        let (_dir, mut store) = test_store();

        store.put(1, b"one").unwrap();
        store.put(2, b"two").unwrap();
        store.put(3, b"three").unwrap();

        assert_eq!(store.get(1).unwrap(), Some(b"one".to_vec()));
        assert_eq!(store.get(2).unwrap(), Some(b"two".to_vec()));
        assert_eq!(store.get(3).unwrap(), Some(b"three".to_vec()));
    }

    #[test]
    fn get_nonexistent() {
        let (_dir, store) = test_store();
        assert_eq!(store.get(999).unwrap(), None);
    }

    #[test]
    fn delete() {
        let (_dir, mut store) = test_store();

        store.put(5, b"delete me").unwrap();
        assert_eq!(store.get(5).unwrap(), Some(b"delete me".to_vec()));

        store.delete(5).unwrap();
        assert_eq!(store.get(5).unwrap(), None);
    }

    #[test]
    fn delete_nonexistent() {
        let (_dir, mut store) = test_store();
        store.delete(100).unwrap();
    }

    #[test]
    fn update() {
        let (_dir, mut store) = test_store();

        store.put(10, b"old value").unwrap();
        assert_eq!(store.get(10).unwrap(), Some(b"old value".to_vec()));

        store.update(10, b"new value").unwrap();
        assert_eq!(store.get(10).unwrap(), Some(b"new value".to_vec()));
    }

    #[test]
    fn iter() {
        let (_dir, mut store) = test_store();

        store.put(0, b"a").unwrap();
        store.put(1, b"b").unwrap();
        store.put(2, b"c").unwrap();

        let entries: Vec<_> = store.iter().collect();
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn flush_then_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().to_path_buf();

        {
            let mut store = KidetaStore::open(&path, 10000).unwrap();
            store.put(0, b"persistent").unwrap();
            store.flush().unwrap();
        }

        {
            let store = KidetaStore::open(&path, 10000).unwrap();
            assert_eq!(store.get(0).unwrap(), Some(b"persistent".to_vec()));
        }
    }

    #[test]
    fn dynamic_capacity_growth() {
        let (_dir, mut store) = test_store();
        // With ensure_capacity, IDs beyond initial capacity grow the store
        store.put(99999, b"test").unwrap();
        assert_eq!(store.get(99999).unwrap(), Some(b"test".to_vec()));
    }

    #[test]
    fn zero_value_rejected() {
        let (_dir, mut store) = test_store();
        assert_eq!(store.put(0, b"").unwrap_err(), StoreError::InvalidId);
    }

    #[test]
    fn large_value() {
        let (_dir, mut store) = test_store();
        let value = vec![0xAB; 1000];
        store.put(0, &value).unwrap();
        assert_eq!(store.get(0).unwrap(), Some(value));
    }

    #[test]
    fn many_ids_dont_collide() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = KidetaStore::open(dir.path(), 5000).unwrap();
        for i in 0..100u32 {
            let value = format!("value_{}", i).into_bytes();
            store.put(i, &value).unwrap();
        }
        assert_eq!(store.get(0).unwrap(), Some(b"value_0".to_vec()));
        assert_eq!(store.get(7).unwrap(), Some(b"value_7".to_vec()));
        assert_eq!(store.get(50).unwrap(), Some(b"value_50".to_vec()));
    }

    #[test]
    fn minimal_5000_cap() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = KidetaStore::open(dir.path(), 5000).unwrap();
        store.put(0, b"zero").unwrap();
        store.put(7, b"seven").unwrap();
        assert_eq!(store.get(0).unwrap(), Some(b"zero".to_vec()));
        assert_eq!(store.get(7).unwrap(), Some(b"seven".to_vec()));
    }

    #[test]
    fn three_entries_sequential() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = KidetaStore::open(dir.path(), 5000).unwrap();
        store.put(0, b"aaa").unwrap();
        store.put(1, b"bbb").unwrap();
        store.put(2, b"ccc").unwrap();
        assert_eq!(store.get(0).unwrap(), Some(b"aaa".to_vec()));
        assert_eq!(store.get(1).unwrap(), Some(b"bbb".to_vec()));
        assert_eq!(store.get(2).unwrap(), Some(b"ccc".to_vec()));
    }

    #[test]
    fn five_hundred_entries_sequential() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = KidetaStore::open(dir.path(), 5000).unwrap();
        for i in 0..500u32 {
            let value = format!("val{}", i);
            store.put(i, value.as_bytes()).unwrap();
        }
        for i in 0..500u32 {
            let expected = format!("val{}", i);
            assert_eq!(
                store.get(i).unwrap(),
                Some(expected.into_bytes()),
                "id={}",
                i
            );
        }
    }

    #[test]
    fn store_dynamic_grow_on_large_id() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = KidetaStore::open(dir.path(), 100).unwrap();

        store.put(50, b"within capacity").unwrap();
        assert_eq!(store.get(50).unwrap(), Some(b"within capacity".to_vec()));

        // id beyond initial 100 → ensure_capacity grows tracker
        store.put(500, b"beyond initial").unwrap();
        assert_eq!(store.get(500).unwrap(), Some(b"beyond initial".to_vec()));

        // Suy ra capacity >= 512 từ việc get(500) thành công
        assert_eq!(store.get(50).unwrap(), Some(b"within capacity".to_vec()));
    }

    #[test]
    fn store_dynamic_grow_sequential_ids() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = KidetaStore::open(dir.path(), 10).unwrap();

        for i in 0u32..100 {
            store
                .put(i, format!("val_{}", i).as_bytes())
                .unwrap();
        }

        for i in 0u32..100 {
            let expected = format!("val_{}", i).into_bytes();
            assert_eq!(store.get(i).unwrap(), Some(expected));
        }
    }

    #[test]
    fn store_dynamic_grow_preserves_unaffected_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = KidetaStore::open(dir.path(), 10).unwrap();

        store.put(5, b"before grow").unwrap();
        store.put(7, b"also before").unwrap();

        // Triggers growth
        store.put(100, b"after grow").unwrap();

        assert_eq!(store.get(5).unwrap(), Some(b"before grow".to_vec()));
        assert_eq!(store.get(7).unwrap(), Some(b"also before".to_vec()));
    }
}
