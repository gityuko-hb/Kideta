use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentState {
    Open,
    Growing,
    Flushing,
    Sealed,
    Indexed,
    Compacted,
}

impl SegmentState {
    pub fn transition_to(
        &self,
        next: SegmentState,
    ) -> Result<(), SegmentStateError> {
        let valid = matches!(
            (self, &next),
            (SegmentState::Open, SegmentState::Growing)
                | (SegmentState::Growing, SegmentState::Flushing)
                | (SegmentState::Flushing, SegmentState::Sealed)
                | (SegmentState::Sealed, SegmentState::Indexed)
                | (SegmentState::Indexed, SegmentState::Compacted)
        );

        if valid {
            Ok(())
        } else {
            Err(SegmentStateError {
                from: *self,
                to: next,
            })
        }
    }

    pub fn is_writable(&self) -> bool {
        matches!(self, SegmentState::Open | SegmentState::Growing)
    }

    pub fn is_sealed(&self) -> bool {
        matches!(
            self,
            SegmentState::Sealed | SegmentState::Indexed | SegmentState::Compacted
        )
    }

    pub fn is_indexed(&self) -> bool {
        matches!(self, SegmentState::Indexed | SegmentState::Compacted)
    }
}

impl fmt::Display for SegmentState {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            SegmentState::Open => write!(f, "Open"),
            SegmentState::Growing => write!(f, "Growing"),
            SegmentState::Flushing => write!(f, "Flushing"),
            SegmentState::Sealed => write!(f, "Sealed"),
            SegmentState::Indexed => write!(f, "Indexed"),
            SegmentState::Compacted => write!(f, "Compacted"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SegmentStateError {
    pub from: SegmentState,
    pub to: SegmentState,
}

impl fmt::Display for SegmentStateError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        write!(
            f,
            "invalid segment state transition from {} to {}",
            self.from, self.to
        )
    }
}

impl std::error::Error for SegmentStateError {}
