use crate::compaction::scorer::CompactionScorer;
use crate::segment::Segment;
use std::sync::Arc;

pub struct CandidatePicker {
    scorer: CompactionScorer,
    max_candidates: usize,
}

impl CandidatePicker {
    pub fn new(scorer: CompactionScorer, max_candidates: usize) -> Self {
        Self { scorer, max_candidates }
    }

    pub fn pick(&self, segments: &[Arc<Segment>]) -> Vec<Arc<Segment>> {
        let mut scored: Vec<_> = segments
            .iter()
            .map(|s: &Arc<Segment>| (s.clone(), self.scorer.score_segment(s)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(self.max_candidates).map(|(s, _)| s).collect()
    }

    pub fn pick_pairs(&self, segments: &[Arc<Segment>]) -> Vec<(Arc<Segment>, Arc<Segment>)> {
        let candidates = self.pick(segments);

        let mut pairs = Vec::new();
        for window in candidates.windows(2) {
            if let [a, b] = window {
                pairs.push((a.clone(), b.clone()));
            }
        }

        pairs
    }
}

impl Default for CandidatePicker {
    fn default() -> Self {
        Self::new(CompactionScorer::default(), 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_respects_max() {
        let picker = CandidatePicker::default();
        let segments: Vec<Arc<Segment>> = vec![];
        let picked = picker.pick(&segments);
        assert!(picked.is_empty());
    }

    #[test]
    fn pick_pairs_empty() {
        let picker = CandidatePicker::default();
        let segments: Vec<Arc<Segment>> = vec![];
        let pairs = picker.pick_pairs(&segments);
        assert!(pairs.is_empty());
    }
}
