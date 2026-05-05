//! Windows memory-mapped file interface.
//!
//! Uses CreateFileMappingW and MapViewOfFile for native memory-mapped file access.
mod native;

pub use native::{Mmap, MmapMut, MmapOptions};
