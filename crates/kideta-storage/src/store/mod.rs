#![cfg(unix)]

//! KidetaStore — custom block-based key-value engine.
//!
//! ## Layers (bottom → up)
//!
//! 1. **[`block`](block)]** — `BlockPtr` packed struct (12 bytes) + constants
//! 2. **[`tracker`](tracker)]** — mmap'd array of `BlockPtr` indexed by entry ID
//! 3. **[`page_file`](page_file)]** — `PageFileManager` for 4MB `data_N.dat` page files
//! 4. **[`mask`](mask)]** — `BitmaskLayer` (1 bit/block, Vec<u64>)
//! 5. **[`gaps`](gaps)]** — `GapsLayer` (GapInfo per 64-block region)
//! 6. **[`lazy`](lazy)]** — `LazyBuffer` for deferred metadata updates
//! 7. **[`engine`](engine)]** — `KidetaStore` tying everything together

pub mod block;
pub mod engine;
pub mod gaps;
pub mod lazy;
pub mod mask;
pub mod page_file;
pub mod tracker;

pub use block::{BLOCK_SIZE, BLOCKS_PER_PAGE, BlockPtr, PAGE_SIZE};
pub use engine::{KidetaStore, Result, StoreError};
pub use gaps::{GapInfo, GapsLayer};
pub use lazy::{LazyBuffer, PendingUpdate};
pub use mask::BitmaskLayer;
pub use page_file::PageFileManager;
pub use tracker::TrackerArray;
