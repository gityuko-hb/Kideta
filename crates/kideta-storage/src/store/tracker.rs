//! Tracker array — mmap'd array of `BlockPtr` indexed by entry ID.
//!
//! The tracker provides O(1) random access to any entry's block pointer.
//! It is persisted to `tracker.dat` and memory-mapped so reads are zero-copy.

use crate::store::block::BlockPtr;
use kideta_core::mmap::unix::ftruncate;
use kideta_core::mmap::{MmapMut, MmapOptions};
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub const TRACKER_FILENAME: &str = "tracker.dat";

#[derive(Debug)]
pub struct TrackerArray {
    mmap: MmapMut,
    file: File,
    capacity: u32,
}

impl TrackerArray {
    pub fn open(
        path: &Path,
        capacity: u32,
    ) -> std::io::Result<Self> {
        let (file, is_new) = Self::open_file(path, capacity)?;
        let mmap = unsafe {
            MmapOptions::new(capacity as usize * std::mem::size_of::<BlockPtr>())
                .mmap_file_mut(&file)?
        };
        let mut tracker = Self {
            mmap,
            file,
            capacity,
        };
        if is_new {
            tracker.zero_all();
        }
        Ok(tracker)
    }

    pub fn grow(
        &mut self,
        min_capacity: u32,
    ) -> std::io::Result<()> {
        if min_capacity <= self.capacity {
            return Ok(());
        }
        let new_cap = min_capacity.next_power_of_two();
        let old_bytes = self.capacity as usize * std::mem::size_of::<BlockPtr>();
        let new_bytes = new_cap as usize * std::mem::size_of::<BlockPtr>();

        ftruncate(self.file.as_raw_fd(), new_bytes)?;
        self.mmap.remap(new_bytes)?;

        let slice = unsafe { self.mmap.as_mut_slice() };
        slice[old_bytes..new_bytes].fill(0);

        self.capacity = new_cap;
        Ok(())
    }

    fn zero_all(&mut self) {
        let slice = unsafe { self.mmap.as_mut_slice() };
        for byte in slice.iter_mut() {
            *byte = 0;
        }
    }

    fn open_file(
        path: &Path,
        capacity: u32,
    ) -> std::io::Result<(File, bool)> {
        let is_new = !path.exists();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        let bytes = capacity as usize * std::mem::size_of::<BlockPtr>();
        let metadata = file.metadata()?;
        let needs_truncate = metadata.len() != bytes as u64;
        if needs_truncate {
            ftruncate(file.as_raw_fd(), bytes)?;
        }
        Ok((file, is_new))
    }

    #[inline]
    pub fn get(
        &self,
        id: u32,
    ) -> BlockPtr {
        debug_assert!(id < self.capacity);
        let index = id as usize;
        let offset = index * std::mem::size_of::<BlockPtr>();
        // SAFETY: offset and size are within bounds of mmap
        let slice = unsafe { self.mmap.as_slice() };
        let ptr_bytes: [u8; 12] = slice[offset..offset + 12].try_into().unwrap();
        // SAFETY: BlockPtr is repr(C) with size 12
        unsafe { std::mem::transmute(ptr_bytes) }
    }

    #[inline]
    pub fn set(
        &mut self,
        id: u32,
        ptr: BlockPtr,
    ) {
        debug_assert!(id < self.capacity);
        let index = id as usize;
        let offset = index * std::mem::size_of::<BlockPtr>();
        // SAFETY: BlockPtr is repr(C) with size 12
        let ptr_bytes: [u8; 12] = unsafe { std::mem::transmute(ptr) };
        // SAFETY: offset and size are within bounds of mmap
        unsafe { self.mmap.as_mut_slice()[offset..offset + 12].copy_from_slice(&ptr_bytes) };
    }

    #[inline]
    pub fn clear(
        &mut self,
        id: u32,
    ) {
        self.set(id, BlockPtr::null());
    }

    pub fn flush(&self) -> std::io::Result<()> {
        self.mmap.flush().map_err(std::io::Error::other)
    }

    #[inline]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    #[inline]
    pub fn len(&self) -> u32 {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.capacity == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn tracker_set_and_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(TRACKER_FILENAME);
        let mut tracker = TrackerArray::open(&path, 100).unwrap();

        let ptr = BlockPtr {
            page_id: 5,
            byte_offset: 42 * 128,
            byte_len: 200,
            _padding: [0; 2],
        };
        tracker.set(42, ptr);
        assert_eq!(tracker.get(42), ptr);
    }

    #[test]
    fn tracker_clear() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(TRACKER_FILENAME);
        let mut tracker = TrackerArray::open(&path, 100).unwrap();

        let ptr = BlockPtr {
            page_id: 5,
            byte_offset: 42 * 128,
            byte_len: 200,
            _padding: [0; 2],
        };
        tracker.set(10, ptr);
        assert_eq!(tracker.get(10), ptr);
        tracker.clear(10);
        assert!(tracker.get(10).is_null());
    }

    #[test]
    fn tracker_flush_does_not_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(TRACKER_FILENAME);
        let mut tracker = TrackerArray::open(&path, 100).unwrap();
        tracker.set(1, BlockPtr::null());
        tracker.flush().unwrap();
    }

    #[test]
    fn tracker_capacity() {
        let dir = tempdir().unwrap();
        let path = dir.path().join(TRACKER_FILENAME);
        let tracker = TrackerArray::open(&path, 256).unwrap();
        assert_eq!(tracker.capacity(), 256);
    }
}
