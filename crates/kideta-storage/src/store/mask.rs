//! Bitmask layer — tracks used/free state of every block with 1 bit per block.
//!
//! The bitmask is stored as a `Vec<u64>` where each bit represents one block.
//! Bit = 1 means "used", bit = 0 means "free".
//!
//! The mask is kept entirely in memory and reconstructed from the GapsLayer
//! on startup (since gaps can derive the mask state).

pub struct BitmaskLayer {
    words: Vec<u64>,
    n_blocks: usize,
}

impl BitmaskLayer {
    pub fn new(n_blocks: usize) -> Self {
        let n_words = n_blocks.div_ceil(64);
        Self {
            words: vec![0u64; n_words],
            n_blocks,
        }
    }

    pub fn with_capacity(n_blocks: usize) -> Self {
        Self::new(n_blocks)
    }

    /// Grow bitmask to cover at least `new_n_blocks` blocks.
    /// New bits are initialized to 0 (free).
    pub fn grow(
        &mut self,
        new_n_blocks: usize,
    ) {
        if new_n_blocks <= self.n_blocks {
            return;
        }
        let new_words = new_n_blocks.div_ceil(64);
        self.words.resize(new_words, 0u64);
        self.n_blocks = new_n_blocks;
    }

    #[inline]
    pub fn is_used(
        &self,
        block_id: u32,
    ) -> bool {
        debug_assert!((block_id as usize) < self.n_blocks);
        let word = block_id / 64;
        let bit = block_id % 64;
        (self.words[word as usize] >> bit) & 1 == 1
    }

    pub fn mark_used(
        &mut self,
        start_block: u32,
        n_blocks: u32,
    ) {
        let mut block = start_block;
        let end = start_block + n_blocks;
        while block < end {
            let word = block / 64;
            let bit = block % 64;
            let remaining_in_word = 64 - bit;
            let count = (end - block).min(remaining_in_word);

            let mask = if count == 64 {
                u64::MAX
            } else {
                ((1u64 << count) - 1) << bit
            };

            self.words[word as usize] |= mask;
            block += count;
        }
    }

    pub fn mark_free(
        &mut self,
        start_block: u32,
        n_blocks: u32,
    ) {
        let mut block = start_block;
        let end = start_block + n_blocks;
        while block < end {
            let word = block / 64;
            let bit = block % 64;
            let remaining_in_word = 64 - bit;
            let count = (end - block).min(remaining_in_word);

            let mask = if count == 64 {
                u64::MAX
            } else {
                ((1u64 << count) - 1) << bit
            };

            self.words[word as usize] &= !mask;
            block += count;
        }
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    pub fn n_words(&self) -> usize {
        self.words.len()
    }

    pub fn used_blocks(&self) -> usize {
        self.words
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }

    pub fn free_blocks(&self) -> usize {
        self.n_blocks - self.used_blocks()
    }

    pub fn get_word(
        &self,
        word_idx: usize,
    ) -> u64 {
        self.words[word_idx]
    }

    pub fn set_word(
        &mut self,
        word_idx: usize,
        value: u64,
    ) {
        self.words[word_idx] = value;
    }

    pub fn is_full(&self) -> bool {
        self.words.iter().all(|&w| w == u64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_all_free() {
        let mask = BitmaskLayer::new(1000);
        for i in 0..1000 {
            assert!(!mask.is_used(i));
        }
    }

    #[test]
    fn mark_used_single_block() {
        let mut mask = BitmaskLayer::new(200);
        mask.mark_used(50, 1);
        assert!(mask.is_used(50));
        assert!(!mask.is_used(49));
        assert!(!mask.is_used(51));
    }

    #[test]
    fn mark_used_across_word_boundary() {
        let mut mask = BitmaskLayer::new(200);
        mask.mark_used(60, 10);

        for i in 60..70 {
            assert!(mask.is_used(i));
        }
        assert!(!mask.is_used(59));
        assert!(!mask.is_used(70));
    }

    #[test]
    fn mark_free() {
        let mut mask = BitmaskLayer::new(200);
        mask.mark_used(50, 20);
        assert!(mask.is_used(55));

        mask.mark_free(55, 1);
        assert!(!mask.is_used(55));
        assert!(mask.is_used(54));
        assert!(mask.is_used(56));
    }

    #[test]
    fn used_and_free_blocks() {
        let mut mask = BitmaskLayer::new(128);
        assert_eq!(mask.used_blocks(), 0);
        assert_eq!(mask.free_blocks(), 128);

        mask.mark_used(0, 64);
        assert_eq!(mask.used_blocks(), 64);
        assert_eq!(mask.free_blocks(), 64);

        mask.mark_used(64, 64);
        assert_eq!(mask.used_blocks(), 128);
        assert!(mask.is_full());
    }

    #[test]
    fn mark_used_full_range() {
        let mut mask = BitmaskLayer::new(256);
        mask.mark_used(0, 256);
        assert!(mask.is_full());
    }

    #[test]
    fn mark_free_all() {
        let mut mask = BitmaskLayer::new(256);
        mask.mark_used(0, 256);
        mask.mark_free(0, 256);
        assert_eq!(mask.used_blocks(), 0);
        assert_eq!(mask.free_blocks(), 256);
    }
}
