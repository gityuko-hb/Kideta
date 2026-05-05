//! Raw memory-mapped file interface — direct syscall wrappers via `libc`.
//!
//! This module provides a safe Rust API over the POSIX `mmap`/`munmap` syscalls,
//! with **zero external dependencies** beyond `libc`. It is the foundation for
//! all disk-backed storage in Kideta: [`MmapVectorStorage`](super::MmapVectorStorage),
//! [`KidetaStore`](super::KidetaStore), and the [`WAL`](super::wal).
//!
//! # Types
//!
//! - [`Mmap`] — read-only or read-write mapping (use `MmapOptions` to configure)
//! - [`MmapMut`] — writable mapping with exclusive access; use this when you need to
//!   modify data in-place. Implements `DerefMut` for ergonomic slice access.
//!
//! # Why not `memmap2`?
//!
//! The `memmap2` crate wraps these syscalls with a safe API, but introduces an
//! extra dependency for a relatively thin abstraction. We implement the same pattern
//! directly here so that:
//!
//! - We control the `SAFETY` invariants precisely
//! - We can add storage-specific operations (e.g. `advise_readahead`)
//! - We keep the dependency surface minimal

pub mod file;
pub mod mmap_mut;
pub mod options;

use super::error::{MmapError, Result};

pub use file::ftruncate;
pub use mmap_mut::MmapMut;
pub use options::MmapOptions;

use std::ffi::c_void;
use std::fmt;
use std::os::unix::io::RawFd;
use std::ptr::NonNull;

pub struct Mmap {
    ptr: NonNull<c_void>,
    len: usize,
    #[allow(dead_code)]
    fd: Option<RawFd>,
}

// SAFETY: Mmap is safe to send between threads because:
// 1. The underlying mmap'd memory is managed by the kernel
// 2. Multiple threads can safely read from the same mapping
// 3. The fd is a process-wide resource that can be shared
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Returns the base address of the mapping.
    #[inline]
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }
    /// Returns the mutable base address of the mapping.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }
    /// Returns the mapping as a byte slice.
    ///
    /// # Safety
    /// The caller must ensure that the mapping is not modified or truncated
    /// externally while the slice is alive. Memory-mapped files can be changed
    /// by other processes, which can lead to undefined behavior.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        // SAFETY: The pointer is valid for `self.len` bytes as long as the
        // mapping exists and is not truncated externally.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.len) }
    }
    /// Returns the mapping as a mutable byte slice.
    ///
    /// # Safety
    /// The caller must ensure that no other references (including from other
    /// processes) exist to the same memory range, and that the file is not
    /// truncated while the slice is alive.
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: The pointer is valid for `self.len` bytes and we have
        // exclusive access via `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.len) }
    }
    pub fn flush(&self) -> Result<()> {
        self.flush_range(0, self.len)
    }
    pub fn flush_async(&self) -> Result<()> {
        self.flush_range_async(0, self.len)
    }
    pub fn flush_range(
        &self,
        offset: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        let base_ptr = self.ptr.as_ptr();
        let end_offset = offset + len;
        let page_start = (offset / page_size) * page_size;
        let aligned_addr = unsafe { base_ptr.byte_offset(page_start as isize) };
        let aligned_len = end_offset - page_start;
        let ret = unsafe { libc::msync(aligned_addr, aligned_len, libc::MS_SYNC) };
        if ret == 0 {
            Ok(())
        } else {
            Err(MmapError::Sync {
                kind: "sync",
                code: std::io::Error::last_os_error().raw_os_error(),
            })
        }
    }
    pub fn flush_range_async(
        &self,
        offset: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };
        let base_ptr = self.ptr.as_ptr();
        let end_offset = offset + len;
        let page_start = (offset / page_size) * page_size;
        let aligned_addr = unsafe { base_ptr.byte_offset(page_start as isize) };
        let aligned_len = end_offset - page_start;
        let ret = unsafe { libc::msync(aligned_addr, aligned_len, libc::MS_ASYNC) };
        if ret == 0 {
            Ok(())
        } else {
            Err(MmapError::Sync {
                kind: "async",
                code: std::io::Error::last_os_error().raw_os_error(),
            })
        }
    }
    pub fn advise(
        &self,
        advice: MadvFlags,
    ) -> Result<()> {
        // SAFETY: `self.ptr` and `self.len` describe a valid mapping.
        let ret = unsafe { libc::madvise(self.ptr.as_ptr(), self.len, advice.bits()) };
        if ret == 0 {
            Ok(())
        } else {
            Err(MmapError::Advis {
                kind: advice.name(),
                code: std::io::Error::last_os_error().raw_os_error(),
            })
        }
    }
    #[inline]
    pub fn advise_sequential(&self) -> Result<()> {
        self.advise(MadvFlags::SEQUENTIAL)
    }
    #[inline]
    pub fn advise_random(&self) -> Result<()> {
        self.advise(MadvFlags::RANDOM)
    }
    #[inline]
    pub fn advise_willneed(&self) -> Result<()> {
        self.advise(MadvFlags::WILLNEED)
    }
    #[inline]
    pub fn advise_dontneed(&self) -> Result<()> {
        self.advise(MadvFlags::DONTNEED)
    }
    #[cfg(any(target_os = "linux", doc))]
    pub fn advise_populate_read(&self) -> Result<()> {
        // SAFETY: `self.ptr` and `self.len` describe a valid mapping.
        let ret = unsafe {
            libc::madvise(
                self.ptr.as_ptr(),
                self.len,
                libc::MADV_POPULATE_READ as libc::c_int,
            )
        };
        if ret == 0 {
            Ok(())
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::EINVAL) {
                self.advise_willneed()
            } else {
                Err(MmapError::Advis {
                    kind: "POPULATE_READ",
                    code: err.raw_os_error(),
                })
            }
        }
    }
    #[cfg(not(any(target_os = "linux", doc)))]
    pub fn advise_populate_read(&self) -> Result<()> {
        self.advise_willneed()
    }
    pub fn lock(&self) -> Result<()> {
        // SAFETY: `self.ptr` and `self.len` describe a valid mapping.
        let ret = unsafe { libc::mlock(self.ptr.as_ptr(), self.len) };
        if ret == 0 {
            Ok(())
        } else {
            Err(MmapError::Lock)
        }
    }
    pub fn unlock(&self) -> Result<()> {
        // SAFETY: `self.ptr` and `self.len` describe a valid mapping.
        let ret = unsafe { libc::munlock(self.ptr.as_ptr(), self.len) };
        if ret == 0 {
            Ok(())
        } else {
            Err(MmapError::Unlock)
        }
    }
    #[inline]
    pub fn fd(&self) -> Option<RawFd> {
        self.fd
    }
}

impl fmt::Debug for Mmap {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        f.debug_struct("Mmap")
            .field("addr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` and `self.len` were obtained from a successful `mmap` call.
        let ret = unsafe { libc::munmap(self.ptr.as_ptr(), self.len) };
        if ret != 0 {
            tracing::error!(
                "munmap failed: addr={:?}, len={}, err={}",
                self.ptr,
                self.len,
                std::io::Error::last_os_error()
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MremapFlags(u32);

impl MremapFlags {
    pub const MAYMOVE: MremapFlags = MremapFlags(libc::MREMAP_MAYMOVE as u32);
    pub const FIXED: MremapFlags = MremapFlags(libc::MREMAP_FIXED as u32);

    #[inline]
    fn bits(&self) -> libc::c_int {
        self.0 as libc::c_int
    }
}

impl std::ops::BitOr<MremapFlags> for MremapFlags {
    type Output = Self;
    fn bitor(
        self,
        rhs: MremapFlags,
    ) -> Self {
        MremapFlags(self.0 | rhs.0)
    }
}

/// Remap an existing memory mapping.
///
/// # Safety
/// The caller must ensure that `addr` and `old_len` describe a valid existing
/// mapping.
pub fn mremap(
    addr: NonNull<c_void>,
    old_len: usize,
    new_len: usize,
    flags: MremapFlags,
) -> Result<NonNull<c_void>> {
    // SAFETY: The mapping is valid per the requirements above.
    let new_addr = unsafe { libc::mremap(addr.as_ptr(), old_len, new_len, flags.bits()) };
    if new_addr == libc::MAP_FAILED {
        Err(MmapError::Mremap {
            code: std::io::Error::last_os_error().raw_os_error(),
        })
    } else {
        // SAFETY: `mremap` succeeded and returned a non-null address.
        Ok(unsafe { NonNull::new_unchecked(new_addr) })
    }
}

impl Mmap {
    pub fn remap(
        &mut self,
        new_len: usize,
    ) -> Result<()> {
        if new_len == self.len {
            return Ok(());
        }
        let new_ptr = mremap(self.ptr, self.len, new_len, MremapFlags::MAYMOVE)?;
        self.ptr = new_ptr;
        self.len = new_len;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MadvFlags(u32);

impl MadvFlags {
    pub const NORMAL: MadvFlags = MadvFlags(0);
    pub const SEQUENTIAL: MadvFlags = MadvFlags(libc::MADV_SEQUENTIAL as u32);
    pub const RANDOM: MadvFlags = MadvFlags(libc::MADV_RANDOM as u32);
    pub const WILLNEED: MadvFlags = MadvFlags(libc::MADV_WILLNEED as u32);
    pub const DONTNEED: MadvFlags = MadvFlags(libc::MADV_DONTNEED as u32);
    pub const REMOVE: MadvFlags = MadvFlags(libc::MADV_REMOVE as u32);

    #[cfg(target_os = "linux")]
    pub const POPULATE_READ: MadvFlags = MadvFlags(libc::MADV_POPULATE_READ as u32);

    #[cfg(not(target_os = "linux"))]
    pub const POPULATE_READ: MadvFlags = MadvFlags(0);

    fn name(&self) -> &'static str {
        const S: u32 = libc::MADV_SEQUENTIAL as u32;
        const R: u32 = libc::MADV_RANDOM as u32;
        const W: u32 = libc::MADV_WILLNEED as u32;
        const D: u32 = libc::MADV_DONTNEED as u32;
        const RM: u32 = libc::MADV_REMOVE as u32;
        #[cfg(target_os = "linux")]
        const P: u32 = libc::MADV_POPULATE_READ as u32;
        match self.0 {
            0 => "NORMAL",
            S => "SEQUENTIAL",
            R => "RANDOM",
            W => "WILLNEED",
            D => "DONTNEED",
            RM => "REMOVE",
            #[cfg(target_os = "linux")]
            P => "POPULATE_READ",
            _ => "UNKNOWN",
        }
    }
    #[inline]
    fn bits(&self) -> libc::c_int {
        self.0 as libc::c_int
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn test_file() -> (std::fs::File, std::path::PathBuf) {
        let mut f = NamedTempFile::new().unwrap();
        let data: [u8; 4096] = [0xAB; 4096];
        f.write_all(&data).unwrap();
        f.as_file().flush().unwrap();
        f.keep().unwrap()
    }

    fn open_test_file(path: &std::path::Path) -> std::fs::File {
        OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap()
    }

    #[test]
    fn mmap_file_read() {
        let (file, _path) = test_file();
        let mapping = unsafe { MmapOptions::new(4096).mmap_file(&file).unwrap() };
        assert_eq!(mapping.len(), 4096);
        let slice = unsafe { mapping.as_slice() };
        assert!(slice.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn mmap_file_write_and_flush() {
        let (_file, path) = test_file();
        let file = open_test_file(&path);
        let mut mapping = unsafe { MmapOptions::new(4096).mmap_file(&file).unwrap() };
        let slice = unsafe { mapping.as_mut_slice() };
        slice[0] = 0xCD;
        slice[1] = 0xEF;
        mapping.flush().unwrap();
        drop(mapping);
        drop(file);

        let file = open_test_file(&path);
        let mapping = unsafe { MmapOptions::new(4096).mmap_file(&file).unwrap() };
        let slice = unsafe { mapping.as_slice() };
        assert_eq!(slice[0], 0xCD);
        assert_eq!(slice[1], 0xEF);
    }

    #[test]
    fn mmap_anonymous() {
        let mut mapping = unsafe { MmapOptions::new(65536).mmap_anonymous().unwrap() };
        assert_eq!(mapping.len(), 65536);
        let slice = unsafe { mapping.as_mut_slice() };
        slice.fill(0x42);
        assert!(slice.iter().all(|&b| b == 0x42));
    }

    #[test]
    fn advise_willneed_does_not_error() {
        let mapping = unsafe { MmapOptions::new(65536).mmap_anonymous().unwrap() };
        mapping.advise_willneed().unwrap();
        mapping.advise_dontneed().unwrap();
        mapping.advise_sequential().unwrap();
        mapping.advise_random().unwrap();
    }

    #[test]
    fn advise_populate_read_does_not_error() {
        let mapping = unsafe { MmapOptions::new(65536).mmap_anonymous().unwrap() };
        mapping.advise_populate_read().unwrap();
    }

    #[test]
    fn mremap_expands_anonymous_mapping() {
        let mut mapping = unsafe {
            MmapOptions::new(4096)
                .mmap_anonymous_mut()
                .unwrap()
        };
        let slice = unsafe { mapping.as_mut_slice() };
        slice.fill(0x42);
        assert_eq!(mapping.len(), 4096);

        mapping.remap(8192).unwrap();
        assert_eq!(mapping.len(), 8192);

        let slice = unsafe { mapping.as_mut_slice() };
        assert!(slice[0..4096].iter().all(|&b| b == 0x42));
    }

    #[test]
    fn debug_format() {
        let mapping = unsafe { MmapOptions::new(4096).mmap_anonymous().unwrap() };
        let debug_str = format!("{:?}", mapping);
        assert!(debug_str.contains("Mmap"));
        assert!(debug_str.contains("4096"));
    }
}
