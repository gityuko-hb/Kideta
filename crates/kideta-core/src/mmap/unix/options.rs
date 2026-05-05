//! Builder for [`Mmap`](super::Mmap) and [`MmapMut`](super::MmapMut) objects.

#![cfg(unix)]

use super::super::error::{MmapError, Result};
use super::Mmap;
use crate::mmap::unix::mmap_mut::MmapMut;
use std::fs::File;
use std::os::fd::RawFd;
use std::os::unix::io::AsRawFd;
use std::ptr::NonNull;

pub struct MmapOptions {
    len: usize,
    offset: u64,
    prot: ProtFlags,
    flags: MapFlags,
    addr: Option<NonNull<u8>>,
}

impl MmapOptions {
    pub fn new(len: usize) -> Self {
        Self {
            len,
            offset: 0,
            prot: ProtFlags::READ | ProtFlags::WRITE,
            flags: MapFlags::empty(),
            addr: None,
        }
    }

    pub fn offset(
        mut self,
        offset: u64,
    ) -> Self {
        self.offset = offset;
        self
    }

    pub fn prot(
        mut self,
        prot: ProtFlags,
    ) -> Self {
        self.prot = prot;
        self
    }

    pub fn flags(
        mut self,
        flags: MapFlags,
    ) -> Self {
        self.flags = flags;
        self
    }

    pub fn address_hint(
        mut self,
        addr: Option<NonNull<u8>>,
    ) -> Self {
        self.addr = addr;
        self
    }

    pub unsafe fn mmap_file(
        self,
        file: &File,
    ) -> Result<Mmap> {
        self.validate()?;
        let fd = file.as_raw_fd();
        unsafe { self.do_mmap(fd) }
    }

    pub unsafe fn mmap_anonymous(self) -> Result<Mmap> {
        self.validate()?;
        if self.len == 0 {
            return Err(MmapError::ZeroSize);
        }
        unsafe { self.do_mmap(-1) }
    }

    pub unsafe fn mmap_file_mut(
        self,
        file: &File,
    ) -> Result<MmapMut> {
        self.validate()?;
        let fd = file.as_raw_fd();
        unsafe { self.do_mmap_mut(fd) }
    }

    pub unsafe fn mmap_anonymous_mut(self) -> Result<MmapMut> {
        self.validate()?;
        if self.len == 0 {
            return Err(MmapError::ZeroSize);
        }
        unsafe { self.do_mmap_mut(-1) }
    }

    fn validate(&self) -> Result<()> {
        if self.offset > 0 {
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 };
            if self.offset % page_size != 0 {
                return Err(MmapError::Map {
                    reason: format!(
                        "offset ({}) must be a multiple of page size ({})",
                        self.offset, page_size
                    ),
                    code: None,
                });
            }
        }
        Ok(())
    }

    unsafe fn do_mmap(
        self,
        fd: RawFd,
    ) -> Result<Mmap> {
        let addr = self
            .addr
            .map(|p| p.as_ptr() as *mut libc::c_void)
            .unwrap_or(std::ptr::null_mut());

        let map_flags = if fd < 0 {
            libc::MAP_ANONYMOUS | libc::MAP_PRIVATE
        } else {
            self.flags.bits()
                | libc::MAP_SHARED
                | (if self.flags.is_empty() {
                    0
                } else {
                    0
                })
        };
        let prot = self.prot.bits();

        let ptr = unsafe {
            libc::mmap(
                addr,
                self.len,
                prot,
                map_flags,
                fd,
                self.offset as libc::off_t,
            )
        };

        if ptr == libc::MAP_FAILED {
            let code = std::io::Error::last_os_error().raw_os_error();
            return Err(MmapError::Map {
                reason: std::io::Error::last_os_error().to_string(),
                code,
            });
        }

        Ok(Mmap {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len: self.len,
            fd: if fd >= 0 {
                Some(fd)
            } else {
                None
            },
        })
    }

    unsafe fn do_mmap_mut(
        self,
        fd: RawFd,
    ) -> Result<MmapMut> {
        let mmap = unsafe { self.do_mmap(fd)? };
        Ok(MmapMut { inner: mmap })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProtFlags(u32);

impl ProtFlags {
    pub const READ: ProtFlags = ProtFlags(libc::PROT_READ as u32);
    pub const WRITE: ProtFlags = ProtFlags(libc::PROT_WRITE as u32);
    pub const EXEC: ProtFlags = ProtFlags(libc::PROT_EXEC as u32);
    pub const NONE: ProtFlags = ProtFlags(libc::PROT_NONE as u32);

    #[inline]
    pub fn bits(&self) -> libc::c_int {
        self.0 as libc::c_int
    }
}

impl std::ops::BitOr<ProtFlags> for ProtFlags {
    type Output = Self;
    fn bitor(
        self,
        rhs: ProtFlags,
    ) -> Self {
        ProtFlags(self.0 | rhs.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MapFlags(u32);

impl MapFlags {
    pub const SHARED: MapFlags = MapFlags(libc::MAP_SHARED as u32);
    pub const PRIVATE: MapFlags = MapFlags(libc::MAP_PRIVATE as u32);
    pub const ANONYMOUS: MapFlags = MapFlags(libc::MAP_ANONYMOUS as u32);
    pub const FIXED: MapFlags = MapFlags(libc::MAP_FIXED as u32);
    pub const HUGETLB: MapFlags = MapFlags(libc::MAP_HUGETLB as u32);
    pub const NORESERVE: MapFlags = MapFlags(libc::MAP_NORESERVE as u32);
    pub const LOCKED: MapFlags = MapFlags(libc::MAP_LOCKED as u32);

    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn bits(&self) -> libc::c_int {
        self.0 as libc::c_int
    }
}

impl std::ops::BitOr<MapFlags> for MapFlags {
    type Output = Self;
    fn bitor(
        self,
        rhs: MapFlags,
    ) -> Self {
        MapFlags(self.0 | rhs.0)
    }
}

impl Default for MmapOptions {
    fn default() -> Self {
        Self::new(0)
    }
}
