#![cfg(any(unix, windows))]

pub mod error;

#[cfg(unix)]
pub mod unix;
#[cfg(unix)]
// pub use unix::{Mmap, MmapMut, MmapOptions};

#[cfg(windows)]
pub mod windows;
#[cfg(windows)]
// pub use windows::{Mmap, MmapMut, MmapOptions};

// Re-export common types
pub use error::{MmapError, Result};
