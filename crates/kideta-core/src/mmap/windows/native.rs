//! Native Windows mmap implementation using Win32 API.
//!
//! Uses CreateFileMappingW and MapViewOfFile for memory-mapped file access.

#![cfg(windows)]

use std::ffi::c_void;
use std::fs::File;
use std::os::windows::io::AsRawHandle;
use std::ptr::NonNull;

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
        pub fn FlushViewOfFile(
            lpBaseAddress: *const c_void,
            dwNumberOfBytesToFlush: usize,
        ) -> i32;
        pub fn CloseHandle(hObject: RawHandle) -> i32;
    }
}

const FILE_MAP_READ: u32 = 4;
const FILE_MAP_WRITE: u32 = 2;
const PAGE_READWRITE: u32 = 0x04;
const PAGE_READONLY: u32 = 0x02;

pub struct Mmap {
    ptr: NonNull<c_void>,
    len: usize,
    mapping_handle: std::os::windows::io::RawHandle,
}

unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    /// Maps a file into memory using Windows `CreateFileMappingW` and `MapViewOfFile`.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - The file is not modified or truncated while the mapping exists.
    /// - The file is not unmapped by other code holding a reference to it.
    /// - Multiple mutable mappings to the same file are not created simultaneously.
    pub unsafe fn map_file(file: &File) -> std::io::Result<Self> {
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
            return Err(std::io::Error::last_os_error());
        }

        let ptr = unsafe { bindings::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) };

        if ptr.is_null() {
            unsafe { bindings::CloseHandle(mapping) };
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            len: file_size as usize,
            mapping_handle: mapping,
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

    /// Returns a read-only slice over the mapped memory.
    ///
    /// # Safety
    ///
    /// Caller must ensure the mapped memory is not modified while holding the slice.
    /// This can happen if another mutable mapping to the same file exists elsewhere.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.len) }
    }

    /// Returns a mutable slice over the mapped memory.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - No other slices (mutable or immutable) to this mapping exist.
    /// - No other threads are accessing the same memory region.
    /// - Changes are not visible to other processes until `flush()` is called.
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.len) }
    }

    pub fn flush(&self) -> std::io::Result<()> {
        unsafe {
            if bindings::FlushViewOfFile(self.ptr.as_ptr(), self.len) != 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            bindings::UnmapViewOfFile(self.ptr.as_ptr());
            bindings::CloseHandle(self.mapping_handle);
        }
    }
}

pub struct MmapMut {
    ptr: NonNull<c_void>,
    len: usize,
    mapping_handle: std::os::windows::io::RawHandle,
}

unsafe impl Send for MmapMut {}
unsafe impl Sync for MmapMut {}

impl MmapMut {
    /// Maps a file into memory with read-write access using Windows `CreateFileMappingW` and `MapViewOfFile`.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - The file is not modified or truncated while the mapping exists.
    /// - The file is not unmapped by other code holding a reference to it.
    /// - No other mappings (mutable or immutable) to the same file exist simultaneously.
    /// - Only one `MmapMut` for a given file exists at a time.
    pub unsafe fn map_file_mut(
        file: &File,
        len: usize,
    ) -> std::io::Result<Self> {
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
            return Err(std::io::Error::last_os_error());
        }

        let ptr = unsafe { bindings::MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, 0) };

        if ptr.is_null() {
            unsafe { bindings::CloseHandle(mapping) };
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            len,
            mapping_handle: mapping,
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

    /// Returns a read-only slice over the mapped memory.
    ///
    /// # Safety
    ///
    /// Caller must ensure the mapped memory is not modified while holding the slice.
    /// This can happen if another mutable mapping to the same file exists elsewhere.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr() as *const u8, self.len) }
    }

    /// Returns a mutable slice over the mapped memory.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - No other slices (mutable or immutable) to this mapping exist.
    /// - No other threads are accessing the same memory region.
    /// - Changes are not visible to other processes until `flush()` is called.
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut u8, self.len) }
    }

    pub fn flush(&self) -> std::io::Result<()> {
        unsafe {
            if bindings::FlushViewOfFile(self.ptr.as_ptr(), self.len) != 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        }
    }
}

impl Drop for MmapMut {
    fn drop(&mut self) {
        unsafe {
            bindings::UnmapViewOfFile(self.ptr.as_ptr());
            bindings::CloseHandle(self.mapping_handle);
        }
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

    pub fn offset(
        mut self,
        offset: u64,
    ) -> Self {
        self.offset = offset;
        self
    }

    /// Creates a read-only memory mapping using the options configured on this builder.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - The file is not modified or truncated while the mapping exists.
    /// - No mutable mappings to the same file exist.
    /// - The file remains open and valid for the lifetime of the mapping.
    pub unsafe fn mmap_file(
        &self,
        file: &File,
    ) -> std::io::Result<Mmap> {
        unsafe { Mmap::map_file(file) }
    }

    /// Creates a read-write memory mapping using the options configured on this builder.
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - The file is not modified or truncated while the mapping exists.
    /// - No other mappings to the same file exist.
    /// - Only one mutable mapping for the file is active at a time.
    /// - The file remains open and valid for the lifetime of the mapping.
    pub unsafe fn mmap_file_mut(
        &self,
        file: &File,
    ) -> std::io::Result<MmapMut> {
        unsafe { MmapMut::map_file_mut(file, self.len) }
    }
}
