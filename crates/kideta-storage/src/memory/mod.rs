//! Memory usage tracking for tiered storage.
//!
//! Tracks memory consumption per tier (Hot/Warm/Cold) using atomic counters.
//! Used by the OOM handler to determine when eviction is needed.
//!
//! # Usage
//!
//! ```
//! use kideta_storage::memory::{MemoryTracker, MemoryTier, global_tracker};
//!
//! let tracker = global_tracker();
//! tracker.track_alloc(MemoryTier::Hot, 1024);
//! assert_eq!(tracker.get_bytes(MemoryTier::Hot), 1024);
//! tracker.track_dealloc(MemoryTier::Hot, 1024);
//! assert_eq!(tracker.get_bytes(MemoryTier::Hot), 0);
//! ```

use std::hash::Hash;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    Hot,
    Warm,
    Cold,
    Archive,
}

impl MemoryTier {
    pub fn name(&self) -> &'static str {
        match self {
            MemoryTier::Hot => "Hot",
            MemoryTier::Warm => "Warm",
            MemoryTier::Cold => "Cold",
            MemoryTier::Archive => "Archive",
        }
    }
}

impl std::fmt::Display for MemoryTier {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

pub struct MemoryTracker {
    hot_bytes: AtomicUsize,
    warm_bytes: AtomicUsize,
    cold_bytes: AtomicUsize,
    archive_bytes: AtomicUsize,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            hot_bytes: AtomicUsize::new(0),
            warm_bytes: AtomicUsize::new(0),
            cold_bytes: AtomicUsize::new(0),
            archive_bytes: AtomicUsize::new(0),
        }
    }

    pub fn track_alloc(
        &self,
        tier: MemoryTier,
        bytes: usize,
    ) {
        match tier {
            MemoryTier::Hot => self.hot_bytes.fetch_add(bytes, Ordering::Relaxed),
            MemoryTier::Warm => self
                .warm_bytes
                .fetch_add(bytes, Ordering::Relaxed),
            MemoryTier::Cold => self
                .cold_bytes
                .fetch_add(bytes, Ordering::Relaxed),
            MemoryTier::Archive => self
                .archive_bytes
                .fetch_add(bytes, Ordering::Relaxed),
        };
    }

    pub fn track_dealloc(
        &self,
        tier: MemoryTier,
        bytes: usize,
    ) {
        match tier {
            MemoryTier::Hot => self.hot_bytes.fetch_sub(bytes, Ordering::Relaxed),
            MemoryTier::Warm => self
                .warm_bytes
                .fetch_sub(bytes, Ordering::Relaxed),
            MemoryTier::Cold => self
                .cold_bytes
                .fetch_sub(bytes, Ordering::Relaxed),
            MemoryTier::Archive => self
                .archive_bytes
                .fetch_sub(bytes, Ordering::Relaxed),
        };
    }

    pub fn get_bytes(
        &self,
        tier: MemoryTier,
    ) -> usize {
        match tier {
            MemoryTier::Hot => self.hot_bytes.load(Ordering::Relaxed),
            MemoryTier::Warm => self.warm_bytes.load(Ordering::Relaxed),
            MemoryTier::Cold => self.cold_bytes.load(Ordering::Relaxed),
            MemoryTier::Archive => self.archive_bytes.load(Ordering::Relaxed),
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.hot_bytes.load(Ordering::Relaxed)
            + self.warm_bytes.load(Ordering::Relaxed)
            + self.cold_bytes.load(Ordering::Relaxed)
            + self.archive_bytes.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn reset(&self) {
        self.hot_bytes.store(0, Ordering::Relaxed);
        self.warm_bytes.store(0, Ordering::Relaxed);
        self.cold_bytes.store(0, Ordering::Relaxed);
        self.archive_bytes.store(0, Ordering::Relaxed);
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MemoryTracker {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("MemoryTracker")
            .field("hot_bytes", &self.hot_bytes.load(Ordering::Relaxed))
            .field("warm_bytes", &self.warm_bytes.load(Ordering::Relaxed))
            .field("cold_bytes", &self.cold_bytes.load(Ordering::Relaxed))
            .field("archive_bytes", &self.archive_bytes.load(Ordering::Relaxed))
            .field("total_bytes", &self.total_bytes())
            .finish()
    }
}

static GLOBAL_TRACKER: LazyLock<MemoryTracker> = LazyLock::new(MemoryTracker::new);

pub fn global_tracker() -> &'static MemoryTracker {
    &GLOBAL_TRACKER
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn track_alloc_and_dealloc() {
        let tracker = MemoryTracker::new();
        tracker.track_alloc(MemoryTier::Hot, 1024);
        assert_eq!(tracker.get_bytes(MemoryTier::Hot), 1024);
        tracker.track_alloc(MemoryTier::Warm, 2048);
        assert_eq!(tracker.get_bytes(MemoryTier::Warm), 2048);
        tracker.track_dealloc(MemoryTier::Hot, 512);
        assert_eq!(tracker.get_bytes(MemoryTier::Hot), 512);
    }

    #[test]
    fn total_bytes() {
        let tracker = MemoryTracker::new();
        assert_eq!(tracker.total_bytes(), 0);
        tracker.track_alloc(MemoryTier::Hot, 100);
        tracker.track_alloc(MemoryTier::Warm, 200);
        tracker.track_alloc(MemoryTier::Cold, 300);
        assert_eq!(tracker.total_bytes(), 600);
    }

    #[test]
    fn concurrent_access() {
        let tracker = Arc::new(MemoryTracker::new());
        let mut handles = Vec::new();

        for _ in 0..4 {
            let t = tracker.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    t.track_alloc(MemoryTier::Hot, i);
                    t.track_dealloc(MemoryTier::Hot, i);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn global_tracker_access() {
        let tracker = global_tracker();
        tracker.reset();
        tracker.track_alloc(MemoryTier::Hot, 12345);
        assert_eq!(tracker.get_bytes(MemoryTier::Hot), 12345);
        tracker.track_dealloc(MemoryTier::Hot, 12345);
        assert_eq!(tracker.get_bytes(MemoryTier::Hot), 0);
    }
}
