#[cfg(unix)]
pub mod composite;
pub mod dtype;
#[cfg(unix)]
pub mod engine;
pub mod header;

#[cfg(unix)]
pub use engine::{MmapVectorStorage, MmapVectorStorageError};
