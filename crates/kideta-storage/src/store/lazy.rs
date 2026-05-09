//! Lazy update buffer — stages tracker and bitmask mutations in memory before flush.
//!
//! KidetaStore uses a write-ordering model where data is written to page files
//! immediately, but tracker + bitmask updates are deferred.  This batches random
//! metadata I/O into a single fsync on flush.

use crate::store::block::BlockPtr;

#[derive(Debug, Clone)]
pub enum PendingUpdate {
    Put { id: u32, ptr: BlockPtr },
    Delete { id: u32, old_ptr: BlockPtr },
}

pub struct LazyBuffer {
    updates: Vec<PendingUpdate>,
}

impl LazyBuffer {
    pub fn new() -> Self {
        Self {
            updates: Vec::new(),
        }
    }

    pub fn push_put(
        &mut self,
        id: u32,
        ptr: BlockPtr,
    ) {
        self.updates.push(PendingUpdate::Put { id, ptr });
    }

    pub fn push_delete(
        &mut self,
        id: u32,
        old_ptr: BlockPtr,
    ) {
        self.updates
            .push(PendingUpdate::Delete { id, old_ptr });
    }

    pub fn apply(
        &self,
        tracker: &mut crate::store::tracker::TrackerArray,
        mask: &mut crate::store::mask::BitmaskLayer,
        gaps: &mut crate::store::gaps::GapsLayer,
    ) {
        for update in &self.updates {
            match update {
                PendingUpdate::Put { id, ptr } => {
                    tracker.set(*id, *ptr);
                    let start_block = ptr.byte_offset / crate::store::block::BLOCK_SIZE as u32;
                    mask.mark_used(start_block, ptr.n_blocks());
                    gaps.update_after_alloc(start_block, ptr.n_blocks(), mask);
                },
                PendingUpdate::Delete { id, old_ptr } => {
                    if !old_ptr.is_null() {
                        tracker.clear(*id);
                        let start_block =
                            old_ptr.byte_offset / crate::store::block::BLOCK_SIZE as u32;
                        mask.mark_free(start_block, old_ptr.n_blocks());
                        gaps.update_after_free(start_block, old_ptr.n_blocks(), mask);
                    }
                },
            }
        }
    }

    pub fn clear(&mut self) {
        self.updates.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    pub fn len(&self) -> usize {
        self.updates.len()
    }

    pub fn updates(&self) -> &[PendingUpdate] {
        &self.updates
    }
}

impl Default for LazyBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LazyBuffer {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("LazyBuffer")
            .field("pending_updates", &self.updates.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_buffer_push_put() {
        let mut buffer = LazyBuffer::new();
        let ptr = BlockPtr {
            page_id: 1,
            byte_offset: 5 * 128,
            byte_len: 200,
            _padding: [0; 2],
        };
        buffer.push_put(42, ptr);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn lazy_buffer_push_delete() {
        let mut buffer = LazyBuffer::new();
        let ptr = BlockPtr {
            page_id: 1,
            byte_offset: 5 * 128,
            byte_len: 200,
            _padding: [0; 2],
        };
        buffer.push_delete(42, ptr);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn lazy_buffer_clear() {
        let mut buffer = LazyBuffer::new();
        buffer.push_put(1, BlockPtr::null());
        buffer.push_delete(2, BlockPtr::null());
        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn lazy_buffer_debug() {
        let mut buffer = LazyBuffer::new();
        buffer.push_put(1, BlockPtr::null());
        let debug_str = format!("{:?}", buffer);
        assert!(debug_str.contains("LazyBuffer"));
    }
}
