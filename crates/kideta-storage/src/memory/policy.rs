//! Eviction policies and OOM handling for tiered storage.
//!
//! When memory pressure exceeds thresholds, the `EvictionManager` evicts
//! segments from warm to cold tiers based on the configured policy (LRU or LFU).
//!
//! # LFU Counter Persistence
//!
//! Access frequency counters are persisted to disk alongside segments as
//! `LFU_counts.db` to survive restarts. On crash, counts may be slightly stale
//! but that's acceptable since LFU is an approximation.

#![cfg(unix)]

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::{MemoryTier, MemoryTracker, global_tracker};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    Lru,
    Lfu,
}

impl std::fmt::Display for EvictionPolicy {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            EvictionPolicy::Lru => write!(f, "LRU"),
            EvictionPolicy::Lfu => write!(f, "LFU"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvictionConfig {
    pub hot_threshold_bytes: usize,
    pub warm_threshold_bytes: usize,
    pub cold_threshold_bytes: usize,
    pub archive_threshold_bytes: usize,
    pub policy: EvictionPolicy,
}

impl EvictionConfig {
    pub fn new(
        hot_threshold_bytes: usize,
        warm_threshold_bytes: usize,
        cold_threshold_bytes: usize,
        archive_threshold_bytes: usize,
        policy: EvictionPolicy,
    ) -> Self {
        Self {
            hot_threshold_bytes,
            warm_threshold_bytes,
            cold_threshold_bytes,
            archive_threshold_bytes,
            policy,
        }
    }

    pub fn threshold_bytes(
        &self,
        tier: MemoryTier,
    ) -> usize {
        match tier {
            MemoryTier::Hot => self.hot_threshold_bytes,
            MemoryTier::Warm => self.warm_threshold_bytes,
            MemoryTier::Cold => self.cold_threshold_bytes,
            MemoryTier::Archive => self.archive_threshold_bytes,
        }
    }
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            hot_threshold_bytes: 1024 * 1024 * 1024,
            warm_threshold_bytes: 4 * 1024 * 1024 * 1024,
            cold_threshold_bytes: 16 * 1024 * 1024 * 1024,
            archive_threshold_bytes: usize::MAX,
            policy: EvictionPolicy::Lru,
        }
    }
}

#[derive(Debug, Clone)]
pub enum EvictionNeeded {
    Hot(usize),
    Warm(usize),
    Cold(usize),
    Archive(usize),
}

impl EvictionNeeded {
    pub fn tier(&self) -> MemoryTier {
        match self {
            EvictionNeeded::Hot(_) => MemoryTier::Hot,
            EvictionNeeded::Warm(_) => MemoryTier::Warm,
            EvictionNeeded::Cold(_) => MemoryTier::Cold,
            EvictionNeeded::Archive(_) => MemoryTier::Archive,
        }
    }

    pub fn needed_bytes(&self) -> usize {
        match self {
            EvictionNeeded::Hot(n) => *n,
            EvictionNeeded::Warm(n) => *n,
            EvictionNeeded::Cold(n) => *n,
            EvictionNeeded::Archive(n) => *n,
        }
    }
}

impl std::fmt::Display for EvictionNeeded {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            EvictionNeeded::Hot(n) => write!(f, "EvictionNeeded::Hot({} bytes over limit)", n),
            EvictionNeeded::Warm(n) => write!(f, "EvictionNeeded::Warm({} bytes over limit)", n),
            EvictionNeeded::Cold(n) => write!(f, "EvictionNeeded::Cold({} bytes over limit)", n),
            EvictionNeeded::Archive(n) => {
                write!(f, "EvictionNeeded::Archive({} bytes over limit)", n)
            },
        }
    }
}

struct LfuCounters {
    counts: RwLock<HashMap<u64, AtomicU64>>,
    path: Option<std::path::PathBuf>,
}

impl LfuCounters {
    fn new(path: Option<PathBuf>) -> Self {
        let counts = if let Some(ref p) = path {
            Self::load_from_disk(p)
        } else {
            HashMap::new()
        };
        Self {
            counts: RwLock::new(counts),
            path: path.map(|p| p.to_path_buf()),
        }
    }

    fn load_from_disk(path: &Path) -> HashMap<u64, AtomicU64> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return HashMap::new(),
        };
        let mut reader = BufReader::new(file);
        let mut map = HashMap::new();
        let mut buf = [0u8; 16];
        while reader.read_exact(&mut buf).is_ok() {
            let segment_id = u64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ]);
            let count = u64::from_le_bytes([
                buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15],
            ]);
            map.insert(segment_id, AtomicU64::new(count));
        }
        map
    }

    fn flush_to_disk(&self) -> std::io::Result<()> {
        let path = match &self.path {
            Some(p) => p,
            None => return Ok(()),
        };
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);
        let counts = self.counts.read().unwrap();
        for (segment_id, count) in counts.iter() {
            let mut buf = [0u8; 16];
            buf[0..8].copy_from_slice(&segment_id.to_le_bytes());
            buf[8..16].copy_from_slice(&count.load(Ordering::Relaxed).to_le_bytes());
            writer.write_all(&buf)?;
        }
        writer.flush()
    }

    fn increment(
        &self,
        segment_id: u64,
    ) {
        let mut counts = self.counts.write().unwrap();
        let counter = counts
            .entry(segment_id)
            .or_insert_with(|| AtomicU64::new(0));
        counter.fetch_add(1, Ordering::Relaxed);
    }

    fn get_count(
        &self,
        segment_id: &u64,
    ) -> u64 {
        let counts = self.counts.read().unwrap();
        counts
            .get(segment_id)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    fn get_least_frequently_used(&self) -> Option<u64> {
        let counts = self.counts.read().unwrap();
        counts
            .iter()
            .min_by_key(|(_, v)| v.load(Ordering::Relaxed))
            .map(|(k, _)| *k)
    }

    fn remove(
        &self,
        segment_id: &u64,
    ) {
        let mut counts = self.counts.write().unwrap();
        counts.remove(segment_id);
    }
}

struct LruOrder {
    order: RwLock<Vec<u64>>,
    path: Option<std::path::PathBuf>,
}

impl LruOrder {
    fn new(path: Option<PathBuf>) -> Self {
        let order = if let Some(ref p) = path {
            Self::load_from_disk(p)
        } else {
            Vec::new()
        };
        Self {
            order: RwLock::new(order),
            path: path.map(|p| p.to_path_buf()),
        }
    }

    fn load_from_disk(path: &Path) -> Vec<u64> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let mut reader = BufReader::new(file);
        let mut result = Vec::new();
        let mut buf = [0u8; 8];
        while reader.read_exact(&mut buf).is_ok() {
            let segment_id = u64::from_le_bytes([
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
            ]);
            result.push(segment_id);
        }
        result
    }

    fn flush_to_disk(&self) -> std::io::Result<()> {
        let path = match &self.path {
            Some(p) => p,
            None => return Ok(()),
        };
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);
        let order = self.order.read().unwrap();
        for segment_id in order.iter() {
            writer.write_all(&segment_id.to_le_bytes())?;
        }
        writer.flush()
    }

    fn touch(
        &self,
        segment_id: u64,
    ) {
        let mut order = self.order.write().unwrap();
        order.retain(|&id| id != segment_id);
        order.push(segment_id);
    }

    fn get_least_recently_used(&self) -> Option<u64> {
        let order = self.order.read().unwrap();
        order.first().copied()
    }

    fn remove(
        &self,
        segment_id: &u64,
    ) {
        let mut order = self.order.write().unwrap();
        order.retain(|&id| id != *segment_id);
    }
}

pub struct EvictionManager {
    config: EvictionConfig,
    tracker: &'static MemoryTracker,
    lfu_counters: LfuCounters,
    lru_order: LruOrder,
    segment_sizes: RwLock<HashMap<u64, usize>>,
}

impl EvictionManager {
    pub fn new(config: EvictionConfig) -> Self {
        Self::with_persistence(config, None, None)
    }

    pub fn with_persistence(
        config: EvictionConfig,
        lfu_path: Option<PathBuf>,
        lru_path: Option<PathBuf>,
    ) -> Self {
        Self {
            config,
            tracker: global_tracker(),
            lfu_counters: LfuCounters::new(lfu_path),
            lru_order: LruOrder::new(lru_path),
            segment_sizes: RwLock::new(HashMap::new()),
        }
    }

    pub fn try_alloc(
        &self,
        tier: MemoryTier,
        bytes: usize,
    ) -> Result<(), EvictionNeeded> {
        self.tracker.track_alloc(tier, bytes);
        let current = self.tracker.get_bytes(tier);
        let threshold = self.config.threshold_bytes(tier);

        if current > threshold {
            let overage = current.saturating_sub(threshold);
            self.tracker.track_dealloc(tier, bytes);
            return Err(match tier {
                MemoryTier::Hot => EvictionNeeded::Hot(overage),
                MemoryTier::Warm => EvictionNeeded::Warm(overage),
                MemoryTier::Cold => EvictionNeeded::Cold(overage),
                MemoryTier::Archive => EvictionNeeded::Archive(overage),
            });
        }
        Ok(())
    }

    pub fn evict_until_below_threshold<F>(
        &self,
        tier: MemoryTier,
        mut evict_callback: F,
    ) -> usize
    where
        F: FnMut(u64, usize) -> bool,
    {
        let threshold = self.config.threshold_bytes(tier);
        let mut evicted_bytes = 0;

        while self.tracker.get_bytes(tier) > threshold {
            let segment_id = match self.config.policy {
                EvictionPolicy::Lfu => self.lfu_counters.get_least_frequently_used(),
                EvictionPolicy::Lru => self.lru_order.get_least_recently_used(),
            };

            let Some(id) = segment_id else { break };

            let size = self
                .segment_sizes
                .read()
                .unwrap()
                .get(&id)
                .copied()
                .unwrap_or(0);

            if evict_callback(id, size) {
                self.lfu_counters.remove(&id);
                self.lru_order.remove(&id);
                self.segment_sizes.write().unwrap().remove(&id);
                self.tracker.track_dealloc(tier, size);
                evicted_bytes += size;
            } else {
                break;
            }
        }

        let _ = self.lfu_counters.flush_to_disk();
        let _ = self.lru_order.flush_to_disk();

        evicted_bytes
    }

    pub fn threshold_bytes(
        &self,
        tier: MemoryTier,
    ) -> usize {
        self.config.threshold_bytes(tier)
    }

    pub fn record_access(
        &self,
        segment_id: u64,
    ) {
        if self.config.policy == EvictionPolicy::Lfu {
            self.lfu_counters.increment(segment_id);
        }
        self.lru_order.touch(segment_id);
    }

    pub fn register_segment(
        &self,
        segment_id: u64,
        size_bytes: usize,
        tier: MemoryTier,
    ) {
        self.tracker.track_alloc(tier, size_bytes);
        self.record_access(segment_id);
        self.segment_sizes
            .write()
            .unwrap()
            .insert(segment_id, size_bytes);
    }

    pub fn unregister_segment(
        &self,
        segment_id: u64,
        size_bytes: usize,
        tier: MemoryTier,
    ) {
        self.tracker.track_dealloc(tier, size_bytes);
        self.segment_sizes
            .write()
            .unwrap()
            .remove(&segment_id);
    }
}

impl Drop for EvictionManager {
    fn drop(&mut self) {
        let _ = self.lfu_counters.flush_to_disk();
        let _ = self.lru_order.flush_to_disk();
    }
}

impl std::fmt::Debug for EvictionManager {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("EvictionManager")
            .field("config", &self.config)
            .field("tracker", self.tracker)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eviction_config_default() {
        let config = EvictionConfig::default();
        assert_eq!(config.threshold_bytes(MemoryTier::Hot), 1024 * 1024 * 1024);
        assert_eq!(
            config.threshold_bytes(MemoryTier::Warm),
            4 * 1024 * 1024 * 1024
        );
        assert_eq!(
            config.threshold_bytes(MemoryTier::Cold),
            16 * 1024 * 1024 * 1024
        );
        assert_eq!(config.threshold_bytes(MemoryTier::Archive), usize::MAX);
    }

    #[test]
    fn eviction_config_custom() {
        let config = EvictionConfig::new(100, 200, 300, 400, EvictionPolicy::Lfu);
        assert_eq!(config.threshold_bytes(MemoryTier::Hot), 100);
        assert_eq!(config.threshold_bytes(MemoryTier::Warm), 200);
        assert_eq!(config.threshold_bytes(MemoryTier::Cold), 300);
        assert_eq!(config.threshold_bytes(MemoryTier::Archive), 400);
    }

    #[test]
    fn try_alloc_within_threshold() {
        let config = EvictionConfig::new(
            1024 * 1024,
            1024 * 1024,
            usize::MAX,
            usize::MAX,
            EvictionPolicy::Lru,
        );
        let manager = EvictionManager::new(config);

        let result = manager.try_alloc(MemoryTier::Hot, 512 * 1024);
        assert!(result.is_ok());
        assert_eq!(manager.tracker.get_bytes(MemoryTier::Hot), 512 * 1024);
    }

    #[test]
    fn try_alloc_exceeds_threshold_returns_error() {
        let config = EvictionConfig::new(
            1024,
            1024 * 1024,
            usize::MAX,
            usize::MAX,
            EvictionPolicy::Lru,
        );
        let manager = EvictionManager::new(config);

        let result = manager.try_alloc(MemoryTier::Hot, 2048);
        assert!(result.is_err());
        match result.unwrap_err() {
            EvictionNeeded::Hot(overage) => {
                assert!(overage > 0);
            },
            _ => panic!("Expected Hot eviction"),
        }
    }

    #[test]
    fn evict_until_below_threshold() {
        let config = EvictionConfig::new(100, 1000, usize::MAX, usize::MAX, EvictionPolicy::Lru);
        let manager = EvictionManager::new(config);

        manager.register_segment(1, 500, MemoryTier::Hot);
        manager.register_segment(2, 500, MemoryTier::Hot);

        let mut evicted_count = 0;
        let evicted = manager.evict_until_below_threshold(MemoryTier::Hot, |_id, _size| {
            evicted_count += 1;
            true
        });

        assert_eq!(evicted_count, 2);
        assert!(manager.tracker.get_bytes(MemoryTier::Hot) <= 100);
    }

    #[test]
    fn lfu_policy_tracks_accesses() {
        let manager = EvictionManager::new(EvictionConfig {
            hot_threshold_bytes: 1000,
            warm_threshold_bytes: 1000,
            cold_threshold_bytes: usize::MAX,
            archive_threshold_bytes: usize::MAX,
            policy: EvictionPolicy::Lfu,
        });

        let segment: u64 = 42;
        manager.register_segment(segment, 1500, MemoryTier::Hot);
        manager.record_access(segment);
        manager.record_access(segment);
        manager.record_access(segment);

        let evicted = manager.evict_until_below_threshold(MemoryTier::Hot, |id, _size| {
            assert_eq!(id, segment);
            true
        });

        assert_eq!(evicted, 1500);
    }
}
