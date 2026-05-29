use crate::compaction::picker::CandidatePicker;
use crate::compaction::scorer::CompactionScorer;
use crate::compaction::{CompactionResult, compact_pair};
use crate::segment::SegmentManager;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

pub struct CompactionMetrics {
    pub total_merges: AtomicU64,
    pub total_vectors_merged: AtomicU64,
    pub total_bytes_freed: AtomicU64,
    pub last_merge_time_ms: AtomicU64,
}

impl CompactionMetrics {
    pub fn new() -> Self {
        Self {
            total_merges: AtomicU64::new(0),
            total_vectors_merged: AtomicU64::new(0),
            total_bytes_freed: AtomicU64::new(0),
            last_merge_time_ms: AtomicU64::new(0),
        }
    }

    pub fn record_merge(
        &self,
        result: &CompactionResult,
        elapsed_ms: u64,
    ) {
        self.total_merges.fetch_add(1, Ordering::Relaxed);
        self.total_vectors_merged
            .fetch_add(result.vectors_merged, Ordering::Relaxed);
        self.total_bytes_freed
            .fetch_add(result.bytes_freed, Ordering::Relaxed);
        self.last_merge_time_ms
            .store(elapsed_ms, Ordering::Relaxed);
    }

    pub fn total_merges(&self) -> u64 {
        self.total_merges.load(Ordering::Relaxed)
    }

    pub fn total_vectors_merged(&self) -> u64 {
        self.total_vectors_merged.load(Ordering::Relaxed)
    }

    pub fn total_bytes_freed(&self) -> u64 {
        self.total_bytes_freed.load(Ordering::Relaxed)
    }
}

impl Default for CompactionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CompactionScheduler {
    segment_manager: Option<Arc<SegmentManager>>,
    picker: CandidatePicker,
    metrics: CompactionMetrics,
    next_segment_id: AtomicU64,
    running: RwLock<bool>,
}

impl CompactionScheduler {
    pub fn new(
        segment_manager: Arc<SegmentManager>,
        max_candidates: usize,
    ) -> Self {
        Self {
            segment_manager: Some(segment_manager),
            picker: CandidatePicker::new(CompactionScorer::default(), max_candidates),
            metrics: CompactionMetrics::new(),
            next_segment_id: AtomicU64::new(0),
            running: RwLock::new(false),
        }
    }

    pub fn run_once(
        &self,
        output_dir: &Path,
    ) -> Vec<crate::store::Result<CompactionResult>> {
        let Some(ref manager) = self.segment_manager else {
            return Vec::new();
        };
        let sealed = manager.sealed_segments();
        let candidates = self.picker.pick(&sealed);

        let mut results = Vec::new();
        let mut i = 0;

        while i + 1 < candidates.len() {
            let seg_a = &candidates[i];
            let seg_b = &candidates[i + 1];

            let seg_id = self
                .next_segment_id
                .fetch_add(1, Ordering::SeqCst);
            let start = std::time::Instant::now();

            let result = compact_pair(seg_a, seg_b, output_dir, seg_id);

            let elapsed = start.elapsed().as_millis() as u64;
            if let Ok(ref result) = result {
                self.metrics.record_merge(result, elapsed);
            }

            results.push(result);
            i += 2;
        }

        results
    }

    pub fn metrics(&self) -> &CompactionMetrics {
        &self.metrics
    }

    pub fn is_running(&self) -> bool {
        *self.running.read().unwrap()
    }

    pub fn set_running(
        &self,
        running: bool,
    ) {
        *self.running.write().unwrap() = running;
    }

    pub fn next_segment_id(&self) -> u64 {
        self.next_segment_id
            .fetch_add(1, Ordering::SeqCst)
    }
}

impl Default for CompactionScheduler {
    fn default() -> Self {
        Self {
            segment_manager: None,
            picker: CandidatePicker::new(CompactionScorer::default(), 10),
            metrics: CompactionMetrics::new(),
            next_segment_id: AtomicU64::new(0),
            running: RwLock::new(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_initial_state() {
        let m = CompactionMetrics::new();
        assert_eq!(m.total_merges(), 0);
        assert_eq!(m.total_vectors_merged(), 0);
        assert_eq!(m.total_bytes_freed(), 0);
    }

    #[test]
    fn scheduler_default_no_panic() {
        let scheduler = CompactionScheduler::default();
        assert!(!scheduler.is_running());
    }
}
