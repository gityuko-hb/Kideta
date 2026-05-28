//! Native Windows mmap implementation using Win32 API.
//!
//! Uses CreateFileMappingW and MapViewOfFile for memory-mapped file access.

#![cfg(windows)]

use std::ffi::c_void;
use std::fs::File;
use std::os::windows::io::AsRawHandle;
use std::ptr::NonNull;

use crate::mmap::error::{MmapError, Result as MmapResult};

fn map_error(reason: &str) -> MmapError {
    MmapError::Map {
        reason: reason.to_string(),
        code: std::io::Error::last_os_error().raw_os_error(),
    }
}

mod bindings {
    use std::ffi::c_void;
    use std::os::windows::io::RawHandle;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn CreateFileMappingW(
            hFile: RawHandle,
            lpFileMappingAttributes: *const c_void,
            flProtect: u32,
            dwMaximumSizeHigh: u32,
            dwMaximumSizeLow: u32,
            lpName: *const u16,
        ) -> RawHandle;

        pub fn MapViewOfFile(
            hFileMappingObject: RawHandle,
            dwDesiredAccess: u32,
            dwFileOffsetHigh: u32,
            dwFileOffsetLow: u32,
            dwNumberOfBytesToMap: usize,
        ) -> *mut c_void;

        pub fn UnmapViewOfFile(lpBaseAddress: *const c_void) -> i32;
        pub fn FlushViewOfFile(lpBaseAddress: *const c_void, dwNumberOfBytesToFlush: usize) -> i32;
        pub fn CloseHandle(hObject: RawHandle) -> i32;

        pub fn VirtualAlloc(
            lpAddress: *const c_void,
            dwSize: usize,
            flAllocationType: u32,
            flProtect: u32,
        ) -> *mut c_void;

        pub fn VirtualFree(lpAddress: *const c_void, dwSize: usize, dwFreeType: u32) -> i32;
    }
}

const FILE_MAP_READ: u32 = 4;
const FILE_MAP_WRITE: u32 = 2;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_READONLY: u32 = 0x02;

const MEM_COMMIT: u32 = 0x00001000;
const MEM_RESERVE: u32 = 0x00002000;
const MEM_RELEASE: u32 = 0x00008000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MadvFlags(u32);

impl MadvFlags {
    pub const NORMAL: MadvFlags = MadvFlags(0);
    pub const SEQUENTIAL: MadvFlags = MadvFlags(0);
    pub const RANDOM: MadvFlags = MadvFlags(0);
    pub const WILLNEED: MadvFlags = MadvFlags(0);
    pub const DONTNEED: MadvFlags = MadvFlags(0);
    pub const POPULATE_READ: MadvFlags = MadvFlags(0);
}

pub struct Mmap {
    ptr: NonNull<c_void>,
    len: usize,
    #[allow(dead_code)]
    mapping_handle: std::os::windows::io::RawHandle,
}

unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub unsafe fn map_file(file: &File, offset: u64) -> MmapResult<Self> {
        let handle = file.as_raw_handle();
        let file_size = file.metadata()?.len();

        let mapping = unsafe {
            bindings::CreateFileMappingW(
                handle,
                std::ptr::null(),
                PAGE_READONLY,
                (file_size >> 32) as u32,
                file_size as u32,
                std::ptr::null(),
            )
        };

        if mapping.is_null() {
            return Err(map_error("CreateFileMappingW failed"));
        }

        let ptr = unsafe { bindings::MapViewOfFile(mapping, FILE_MAP_READ, (offset >> 32) as u32, offset as u32, 0) };

        if ptr.is_null() {
            unsafe { bindings::CloseHandle(mapping) };
            return Err(map_error("MapViewOfFile failed"));
        }

        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            len: file_size as usize,
            mapping_handle: mapping,
        })
    }

    pub unsafe fn map_anonymous(len: usize) -> MmapResult<Self> {
        if len == 0 {
            return Err(MmapError::ZeroSize);
        }
        let ptr = unsafe { bindings::VirtualAlloc(std::ptr::null(), len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE) };
        if ptr.is_null() {
            return Err(map_error("VirtualAlloc failed"));
        }
        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            len,
            mapping_handle: std::ptr::null_mut(),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_ptr(&self) -> *const c_void {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.ptr.as_ptr()
    }

    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.len) }
    }

    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.len) }
    }

    pub fn flush(&self) -> MmapResult<()> {
        unsafe {
            if bindings::FlushViewOfFile(self.ptr.as_ptr(), self.len) != 0 {
                Ok(())
            } else {
                Err(map_error("FlushViewOfFile failed"))
            }
        }
    }

    pub fn flush_range(&self, _offset: usize, _len: usize) -> MmapResult<()> {
        self.flush()
    }

    pub fn flush_async(&self) -> MmapResult<()> {
        self.flush()
    }

    pub fn flush_range_async(&self, _offset: usize, _len: usize) -> MmapResult<()> {
        self.flush()
    }

    pub fn advise(&self, _advice: MadvFlags) -> MmapResult<()> {
        Ok(())
    }

    #[inline]
    pub fn advise_sequential(&self) -> MmapResult<()> {
        Ok(())
    }

    #[inline]
    pub fn advise_random(&self) -> MmapResult<()> {
        Ok(())
    }

    #[inline]
    pub fn advise_willneed(&self) -> MmapResult<()> {
        Ok(())
    }

    #[inline]
    pub fn advise_dontneed(&self) -> MmapResult<()> {
        Ok(())
    }

    #[inline]
    pub fn advise_populate_read(&self) -> MmapResult<()> {
        Ok(())
    }

    pub fn lock(&self) -> MmapResult<()> {
        Ok(())
    }

    pub fn unlock(&self) -> MmapResult<()> {
        Ok(())
    }

    pub fn remap(&mut self, _new_len: usize) -> MmapResult<()> {
        Err(MmapError::Map {
            reason: "remap not supported on Windows; recreate mapping".to_string(),
            code: None,
        })
    }
}

impl std::fmt::Debug for Mmap {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Mmap")
            .field("addr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            if self.mapping_handle.is_null() {
                bindings::VirtualFree(self.ptr.as_ptr(), 0, MEM_RELEASE);
            } else {
                bindings::UnmapViewOfFile(self.ptr.as_ptr());
                bindings::CloseHandle(self.mapping_handle);
            }
        }
    }
}

pub struct MmapMut {
    pub(crate) inner: Mmap,
}

unsafe impl Send for MmapMut {}
unsafe impl Sync for MmapMut {}

impl MmapMut {
    pub unsafe fn map_file_mut(file: &File, len: usize, offset: u64) -> MmapResult<Self> {
        let handle = file.as_raw_handle();
        let file_size = file.metadata()?.len();
        let map_size = len.max(file_size as usize);

        let mapping = unsafe {
            bindings::CreateFileMappingW(
                handle,
                std::ptr::null(),
                PAGE_READWRITE,
                (map_size as u64 >> 32) as u32,
                map_size as u32,
                std::ptr::null(),
            )
        };

        if mapping.is_null() {
            return Err(map_error("CreateFileMappingW failed"));
        }

        let ptr = unsafe { bindings::MapViewOfFile(mapping, FILE_MAP_WRITE, (offset >> 32) as u32, offset as u32, 0) };

        if ptr.is_null() {
            unsafe { bindings::CloseHandle(mapping) };
            return Err(map_error("MapViewOfFile failed"));
        }

        let inner = Mmap {
            ptr: NonNull::new(ptr).unwrap(),
            len,
            mapping_handle: mapping,
        };

        Ok(Self { inner })
    }

    pub unsafe fn map_anonymous_mut(len: usize) -> MmapResult<Self> {
        if len == 0 {
            return Err(MmapError::ZeroSize);
        }
        let mmap = unsafe { Mmap::map_anonymous(len) }?;
        Ok(Self { inner: mmap })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const c_void {
        self.inner.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.inner.as_mut_ptr()
    }

    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        self.inner.as_slice()
    }

    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner.as_mut_slice()
    }

    pub fn flush(&self) -> MmapResult<()> {
        self.inner.flush()
    }

    pub fn flush_sync(&self) -> MmapResult<()> {
        self.inner.flush()
    }

    pub fn flush_async(&self) -> MmapResult<()> {
        self.inner.flush_async()
    }

    pub fn flush_range(&self, offset: usize, len: usize) -> MmapResult<()> {
        self.inner.flush_range(offset, len)
    }

    pub fn advise(&self, advice: MadvFlags) -> MmapResult<()> {
        self.inner.advise(advice)
    }

    #[inline]
    pub fn advise_sequential(&self) -> MmapResult<()> {
        self.inner.advise_sequential()
    }

    #[inline]
    pub fn advise_random(&self) -> MmapResult<()> {
        self.inner.advise_random()
    }

    #[inline]
    pub fn advise_willneed(&self) -> MmapResult<()> {
        self.inner.advise_willneed()
    }

    #[inline]
    pub fn advise_dontneed(&self) -> MmapResult<()> {
        self.inner.advise_dontneed()
    }

    #[inline]
    pub fn advise_populate_read(&self) -> MmapResult<()> {
        self.inner.advise_populate_read()
    }

    pub fn lock(&self) -> MmapResult<()> {
        self.inner.lock()
    }

    pub fn unlock(&self) -> MmapResult<()> {
        self.inner.unlock()
    }

    pub fn remap(&mut self, new_len: usize) -> MmapResult<()> {
        self.inner.remap(new_len)
    }
}

impl std::fmt::Debug for MmapMut {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("MmapMut")
            .field("addr", &self.inner.ptr)
            .field("len", &self.inner.len)
            .finish()
    }
}

impl Drop for MmapMut {
    fn drop(&mut self) {
        // Mmap's Drop handles the cleanup
    }
}

impl std::ops::Deref for MmapMut {
    type Target = Mmap;
    fn deref(&self) -> &Mmap {
        &self.inner
    }
}

impl std::ops::DerefMut for MmapMut {
    fn deref_mut(&mut self) -> &mut Mmap {
        &mut self.inner
    }
}

pub struct MmapOptions {
    len: usize,
    offset: u64,
}

impl MmapOptions {
    pub fn new(len: usize) -> Self {
        Self { len, offset: 0 }
    }

    pub fn offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    pub unsafe fn mmap_file(&self, file: &File) -> MmapResult<Mmap> {
        unsafe { Mmap::map_file(file, self.offset) }
    }

    pub unsafe fn mmap_file_mut(&self, file: &File) -> MmapResult<MmapMut> {
        unsafe { MmapMut::map_file_mut(file, self.len, self.offset) }
    }

    pub unsafe fn mmap_anonymous(&self) -> MmapResult<Mmap> {
        if self.len == 0 {
            return Err(MmapError::ZeroSize);
        }
        unsafe { Mmap::map_anonymous(self.len) }
    }

    pub unsafe fn mmap_anonymous_mut(&self) -> MmapResult<MmapMut> {
        if self.len == 0 {
            return Err(MmapError::ZeroSize);
        }
        unsafe { MmapMut::map_anonymous_mut(self.len) }
    }
}
