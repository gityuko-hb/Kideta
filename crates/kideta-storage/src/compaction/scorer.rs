use crate::segment::Segment;
use std::sync::Arc;

pub struct CompactionScorer {
    deleted_weight: f64,
    size_weight: f64,
}

impl CompactionScorer {
    pub fn new(
        deleted_weight: f64,
        size_weight: f64,
    ) -> Self {
        Self {
            deleted_weight,
            size_weight,
        }
    }

    pub fn score(
        &self,
        vector_count: u64,
        deleted_count: u64,
    ) -> f64 {
        if vector_count == 0 {
            return 0.0;
        }
        let deleted_ratio = deleted_count as f64 / vector_count as f64;
        let size_penalty = (vector_count as f64).log10() / 10.0;
        self.deleted_weight * deleted_ratio + self.size_weight * size_penalty
    }

    pub fn score_segment(
        &self,
        segment: &Arc<Segment>,
    ) -> f64 {
        let vc = segment.vector_count();
        let dc = segment.deleted_count();
        self.score(vc, dc)
    }
}

impl Default for CompactionScorer {
    fn default() -> Self {
        Self::new(0.6, 0.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_empty_segment() {
        let scorer = CompactionScorer::default();
        let s = scorer.score(0, 0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn score_high_deleted_ratio() {
        let scorer = CompactionScorer::new(0.6, 0.4);
        let s = scorer.score(100, 50);
        assert!(s > 0.3);
    }

    #[test]
    fn score_low_deleted_ratio() {
        let scorer = CompactionScorer::new(0.6, 0.4);
        let s = scorer.score(1000, 10);
        assert!(s < scorer.score(100, 50));
    }
}
