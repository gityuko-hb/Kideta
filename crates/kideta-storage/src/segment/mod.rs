//! Segment management — immutable data segments with delta-log deletes.
//!
//! ## File Layout
//!
//! Each segment lives in its own directory:
//! ```text
//! segment_dir/
//!   vectors.bin     — MmapVectorStorage
//!   payload.store   — KidetaStore (payload key-value)
//!   deleted.bitmap  — RoaringBitmap of deleted vector IDs
//!   delta.log       — append-only delete log (replayed on load)
//!   meta.json       — SegmentMeta (JSON, includes BloomFilter)
//! ```
//!
//! ## Lifecycle
//!
//! ```text
//! Open → Growing → Flushing → Sealed → Indexed → Compacted
//! ```

#[cfg(unix)]
pub mod deletion_bloom;
#[cfg(unix)]
pub mod delta;
#[cfg(unix)]
pub mod manager;
#[cfg(unix)]
pub mod meta;
#[cfg(unix)]
#[allow(clippy::module_inception)]
pub mod segment;
#[cfg(unix)]
pub mod state;

#[cfg(unix)]
pub use deletion_bloom::DeletionBloomFilter;
#[cfg(unix)]
pub use delta::{DeltaLogReader, DeltaLogWriter};
#[cfg(unix)]
pub use manager::{BackgroundIndexer, SegmentManager, SegmentManagerConfig, WriteVectorResult};
#[cfg(unix)]
pub use meta::SegmentMeta;
#[cfg(unix)]
pub use segment::Segment;
#[cfg(unix)]
pub use state::SegmentState;
