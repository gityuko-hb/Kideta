//! Compaction engine — tiered compaction with scoring, scheduling, and merging.
//!
//! ## Design
//!
//! Compaction runs in a background thread pool (configurable parallelism) triggered
//! either by a cron interval (default: every 5 minutes) or on-demand via an mpsc
//! channel.
//!
//! ## Tiered Layout
//!
//! ```text
//! L0: < 10 MB   — fresh segments, high churn
//! L1: < 100 MB  — medium segments
//! L2: >= 100 MB — large stable segments
//! ```
//!
//! Compaction policy: compact L0→L1 frequently, L1→L2 less frequently.

pub mod merger;
pub mod picker;
pub mod scheduler;
pub mod scorer;
pub mod tiered;
pub mod trigger;
