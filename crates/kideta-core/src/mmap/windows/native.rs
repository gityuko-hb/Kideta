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

pub struct Mmap {
    ptr: NonNull<c_void>,
    len: usize,
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
    ptr: NonNull<c_void>,
    len: usize,
    mapping_handle: std::os::windows::io::RawHandle,
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

        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            len,
            mapping_handle: mapping,
        })
    }

    pub unsafe fn map_anonymous_mut(len: usize) -> MmapResult<Self> {
        if len == 0 {
            return Err(MmapError::ZeroSize);
        }
        let mmap = unsafe { Mmap::map_anonymous(len) }?;
        let ptr = mmap.ptr;
        let mapping_handle = mmap.mapping_handle;
        std::mem::forget(mmap);
        Ok(Self {
            ptr,
            len,
            mapping_handle,
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
}

impl Drop for MmapMut {
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
