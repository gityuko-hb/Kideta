//! Windows memory-mapped file interface.
//!
//! Uses CreateFileMappingW and MapViewOfFile for native memory-mapped file access.

mod file;
mod native;

pub use file::ftruncate;
pub use native::{MadvFlags, Mmap, MmapMut, MmapOptions};
