//! Writable memory mapping with exclusive access.

#![cfg(unix)]

use super::super::error::Result;
use super::Mmap;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::os::unix::io::RawFd;

pub struct MmapMut {
    pub(crate) inner: Mmap,
}

// SAFETY: MmapMut is safe to send between threads because:
// 1. The underlying mmap'd memory is managed by the kernel
// 2. MmapMut provides exclusive access for writes
// 3. The fd is a process-wide resource that can be shared
unsafe impl Send for MmapMut {}
unsafe impl Sync for MmapMut {}

impl MmapMut {
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the mapping as a mutable byte slice.
    ///
    /// # Safety
    /// The caller must ensure that the file is not truncated while the slice
    /// is alive. Although `MmapMut` represents exclusive access within this
    /// process, external processes can still modify or truncate the underlying
    /// file.
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: We have exclusive access to `self` and the mapping is valid
        // for `self.len()` bytes.
        // SAFETY: This is safe because the function is already unsafe and caller
        // guarantees exclusive access to `self`.
        unsafe { self.inner.as_mut_slice() }
    }

    pub fn flush_sync(&mut self) -> Result<()> {
        self.inner.flush()
    }
    pub fn flush_async(&mut self) -> Result<()> {
        self.inner.flush_async()
    }
    pub fn flush_range(
        &mut self,
        offset: usize,
        len: usize,
    ) -> Result<()> {
        self.inner.flush_range(offset, len)
    }
    pub fn advise(
        &self,
        advice: super::MadvFlags,
    ) -> Result<()> {
        self.inner.advise(advice)
    }
    #[inline]
    pub fn advise_sequential(&self) -> Result<()> {
        self.inner.advise_sequential()
    }
    #[inline]
    pub fn advise_random(&self) -> Result<()> {
        self.inner.advise_random()
    }
    #[inline]
    pub fn advise_willneed(&self) -> Result<()> {
        self.inner.advise_willneed()
    }
    #[inline]
    pub fn advise_dontneed(&self) -> Result<()> {
        self.inner.advise_dontneed()
    }
    #[inline]
    pub fn advise_populate_read(&self) -> Result<()> {
        self.inner.advise_populate_read()
    }
    pub fn remap(
        &mut self,
        new_len: usize,
    ) -> Result<()> {
        self.inner.remap(new_len)
    }
    pub fn lock(&self) -> Result<()> {
        self.inner.lock()
    }
    pub fn unlock(&self) -> Result<()> {
        self.inner.unlock()
    }
    #[inline]
    pub fn fd(&self) -> Option<RawFd> {
        self.inner.fd()
    }
}
impl fmt::Debug for MmapMut {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        f.debug_struct("MmapMut")
            .field("addr", &self.inner.as_ptr())
            .field("len", &self.inner.len())
            .finish()
    }
}

impl Deref for MmapMut {
    type Target = Mmap;
    fn deref(&self) -> &Mmap {
        &self.inner
    }
}

impl DerefMut for MmapMut {
    fn deref_mut(&mut self) -> &mut Mmap {
        &mut self.inner
    }
}

impl Drop for MmapMut {
    fn drop(&mut self) {}
}

#[cfg(test)]
mod tests {
    use crate::mmap::{Mmap, MmapMut, MmapOptions, Result};
    use std::fs::OpenOptions;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn mmap_mut_write_and_sync() {
        let path = {
            let mut file = NamedTempFile::new().unwrap();
            file.write_all(&[0u8; 4096]).unwrap();
            file.as_file().sync_all().unwrap();
            let (_, path) = file.keep().unwrap();
            path
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut mapping = unsafe {
            MmapOptions::new(4096)
                .mmap_file_mut(&file)
                .unwrap()
        };
        unsafe {
            mapping.as_mut_slice()[0..4].copy_from_slice(&123i32.to_le_bytes());
        }
        mapping.flush_sync().unwrap();
        drop(mapping);
        drop(file);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mapping = unsafe { MmapOptions::new(4096).mmap_file(&file).unwrap() };
        let slice = unsafe { mapping.as_slice() };
        assert_eq!(&slice[0..4], 123i32.to_le_bytes());
    }

    #[test]
    fn mmap_mut_deref() {
        let path = {
            let mut file = NamedTempFile::new().unwrap();
            file.write_all(&[0u8; 4096]).unwrap();
            let (_, path) = file.keep().unwrap();
            path
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut mapping = unsafe {
            MmapOptions::new(4096)
                .mmap_file_mut(&file)
                .unwrap()
        };

        let mmap_ref: &Mmap = &*mapping;
        assert_eq!(mmap_ref.len(), 4096);

        unsafe {
            mapping.as_mut_slice()[0] = 0xFF;
            assert_eq!(mapping.as_mut_slice()[0], 0xFF);
        }
    }

    #[test]
    fn debug_format() {
        let path = {
            let mut file = NamedTempFile::new().unwrap();
            file.write_all(&[0u8; 4096]).unwrap();
            let (_, path) = file.keep().unwrap();
            path
        };
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mapping = unsafe {
            MmapOptions::new(4096)
                .mmap_file_mut(&file)
                .unwrap()
        };
        let debug_str = format!("{:?}", mapping);
        assert!(debug_str.contains("MmapMut"));
        assert!(debug_str.contains("4096"));
    }
}
