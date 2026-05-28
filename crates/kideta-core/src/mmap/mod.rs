//! Memory-mapped file interface — cross-platform.
//!
//! This module provides a safe Rust API over platform-specific syscalls:
//! - Unix: POSIX `mmap`/`munmap` (via `libc`)
//! - Windows: Native `CreateFileMappingW` + `MapViewOfFile`
//!
//! # Architecture
//!
//! This module was originally part of `kideta-storage` and was moved to
//! `kideta-core` to break a circular dependency:
//!
//! ```text
//! kideta-core (mmap primitives + VectorStore trait)
//!     ↕                          ↕
//! kideta-index (consumes mmap, VectorStore)
//! kideta-storage (uses mmap, implements VectorStore via MmapVectorStorage)
//! ```
//!
//! The lower-level types [`Mmap`] and [`MmapMut`] provide the raw syscall wrappers.
//! See also:
//! - [`kideta_core::vector_store::VectorStore`](crate::vector_store::VectorStore)
//! - `kideta_storage::vector_storage::MmapVectorStorage` (impl of VectorStore)
//!
//! # Usage
//!
//! ```rust,no_run
//! use kideta_core::mmap::{Mmap, MmapOptions};
//! use std::fs::OpenOptions;
//!
//! let file = OpenOptions::new()
//!     .read(true)
//!     .write(true)
//!     .create(true)
//!     .open("/tmp/test.mmap").unwrap();
//! let mmap = unsafe { MmapOptions::new(4096).mmap_file(&file).unwrap() };
//! let slice = unsafe { mmap.as_slice() };
//! assert_eq!(slice.len(), 4096);
//! ```

#![cfg(any(unix, windows))]

pub mod error;

#[cfg(unix)]
pub mod unix;
#[cfg(unix)]
pub use unix::{Mmap, MmapMut, MmapOptions};

#[cfg(windows)]
pub mod windows;
#[cfg(windows)]
pub use windows::{Mmap, MmapMut, MmapOptions};

#[cfg(windows)]
pub use windows::ftruncate;

pub fn ftruncate_file(
    file: &std::fs::File,
    len: usize,
) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        crate::mmap::unix::ftruncate(file.as_raw_fd(), len)
    }
    #[cfg(windows)]
    {
        crate::mmap::windows::ftruncate(file, len)
    }
}

// Re-export common types
pub use error::{MmapError, Result};
