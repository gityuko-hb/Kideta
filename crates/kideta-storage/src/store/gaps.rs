//! Gaps layer — summary metadata per 64-block region for O(1) amortized allocation.
//!
//! Each region of 64 consecutive blocks has a `GapInfo` that tracks:
//! - `max_gap`: size of the largest contiguous free run in the region
//! - `leading_free`: number of free blocks from the start of the region
//! - `trailing_free`: number of free blocks from the end of the region
//!
//! This allows `find_free_blocks` to skip regions that cannot possibly satisfy
//! an allocation request in O(1) time per region.

use crate::store::mask::BitmaskLayer;

const REGION_BITS: u32 = 64;

#[derive(Debug, Clone, Copy, Default)]
pub struct GapInfo {
    pub max_gap: u8,
    pub leading_free: u8,
    pub trailing_free: u8,
}

impl GapInfo {
    pub fn from_mask(bits: u64) -> Self {
        if bits == 0 {
            return Self {
                max_gap: 64,
                leading_free: 64,
                trailing_free: 64,
            };
        }
        if bits == u64::MAX {
            return Self {
                max_gap: 0,
                leading_free: 0,
                trailing_free: 0,
            };
        }

        let leading = bits.trailing_ones() as u8;
        let trailing = bits.leading_zeros() as u8;

        let mut max_gap: u8 = 0;
        let mut current_gap: u8 = 0;
        let mut in_gap = false;
        let mut pos: u8 = 0;

        while pos < 64 {
            let bit = (bits >> pos) & 1;
            if bit == 0 {
                if !in_gap {
                    in_gap = true;
                    current_gap = 1;
                } else {
                    current_gap += 1;
                }
                max_gap = max_gap.max(current_gap);
            } else {
                in_gap = false;
            }
            pos += 1;
        }

        Self {
            max_gap,
            leading_free: leading,
            trailing_free: trailing,
        }
    }

    pub fn can_fit(
        &self,
        n_blocks: u32,
    ) -> bool {
        self.max_gap as u32 >= n_blocks
    }
}

#[derive(Debug)]
pub struct GapsLayer {
    infos: Vec<GapInfo>,
    n_blocks: usize,
}

impl GapsLayer {
    pub fn new(n_blocks: usize) -> Self {
        let n_regions = n_blocks.div_ceil(64);
        Self {
            infos: vec![GapInfo::default(); n_regions],
            n_blocks,
        }
    }

    pub fn rebuild_from_mask(
        &mut self,
        mask: &BitmaskLayer,
    ) {
        for (i, info) in self.infos.iter_mut().enumerate() {
            let word = mask.get_word(i);
            *info = GapInfo::from_mask(word);
        }
    }

    pub fn find_free_blocks(
        &self,
        n: u32,
        mask: &BitmaskLayer,
    ) -> Option<(u32, u32)> {
        for (region_idx, info) in self.infos.iter().enumerate() {
            if !info.can_fit(n) {
                continue;
            }

            let region_start = region_idx as u32 * REGION_BITS;
            let region_end = (region_start + REGION_BITS).min(self.n_blocks as u32);

            if let Some(found) = self.scan_region_for_free(region_start, region_end, n, mask) {
                return Some(found);
            }
        }
        None
    }

    fn scan_region_for_free(
        &self,
        region_start: u32,
        region_end: u32,
        n: u32,
        mask: &BitmaskLayer,
    ) -> Option<(u32, u32)> {
        let mut block = region_start;
        while block + n <= region_end {
            let mut all_free = true;
            for i in 0..n {
                if mask.is_used(block + i) {
                    all_free = false;
                    break;
                }
            }
            if all_free {
                return Some((block, n));
            }
            block += 1;
        }
        None
    }

    pub fn update_after_alloc(
        &mut self,
        start_block: u32,
        n_blocks: u32,
        mask: &BitmaskLayer,
    ) {
        let region_start = (start_block / REGION_BITS) as usize;
        let region_end = ((start_block + n_blocks - 1) / REGION_BITS) as usize;

        for region_idx in region_start..=region_end {
            if region_idx < self.infos.len() {
                self.infos[region_idx] = GapInfo::from_mask(mask.get_word(region_idx));
            }
        }
    }

    pub fn update_after_free(
        &mut self,
        start_block: u32,
        n_blocks: u32,
        mask: &BitmaskLayer,
    ) {
        self.update_after_alloc(start_block, n_blocks, mask);
    }

    /// Grow gaps layer to cover at least `new_n_blocks` blocks.
    /// New GapInfo entries are default-initialized (all free).
    /// Then rebuild from current mask to ensure accuracy.
    pub fn grow(
        &mut self,
        new_n_blocks: usize,
        mask: &BitmaskLayer,
    ) {
        if new_n_blocks <= self.n_blocks {
            return;
        }
        let new_regions = new_n_blocks.div_ceil(64);
        self.infos.resize(new_regions, GapInfo::default());
        self.n_blocks = new_n_blocks;
        self.rebuild_from_mask(mask);
    }

    #[inline]
    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    #[inline]
    pub fn n_regions(&self) -> usize {
        self.infos.len()
    }

    #[inline]
    pub fn get_info(
        &self,
        region_idx: usize,
    ) -> GapInfo {
        self.infos[region_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mask_with_free(
        n_blocks: usize,
        free_start: u32,
        free_len: u32,
    ) -> BitmaskLayer {
        let mut mask = BitmaskLayer::new(n_blocks);
        for i in 0..n_blocks as u32 {
            if i >= free_start && i < free_start + free_len {
                continue;
            }
            mask.mark_used(i, 1);
        }
        mask
    }

    #[test]
    fn gap_info_all_free() {
        let info = GapInfo::from_mask(0);
        assert_eq!(info.max_gap, 64);
        assert_eq!(info.leading_free, 64);
        assert_eq!(info.trailing_free, 64);
    }

    #[test]
    fn gap_info_all_used() {
        let info = GapInfo::from_mask(u64::MAX);
        assert_eq!(info.max_gap, 0);
        assert_eq!(info.leading_free, 0);
        assert_eq!(info.trailing_free, 0);
    }

    #[test]
    fn gap_info_leading_free() {
        let bits: u64 = 0b0000111111111111u64;
        let info = GapInfo::from_mask(bits);
        assert_eq!(info.leading_free, 12);
        assert_eq!(info.trailing_free, 52);
    }

    #[test]
    fn gap_info_trailing_free() {
        let bits: u64 = 0b1111111111111111000000000000000000000000000000000000000000000000u64;
        let info = GapInfo::from_mask(bits);
        assert_eq!(info.trailing_free, 0);
        assert_eq!(info.leading_free, 0);
    }

    #[test]
    fn gap_info_max_gap_middle() {
        let bits = 0xFFFF_0000_FFFFFFFFu64;
        let info = GapInfo::from_mask(bits);
        assert!(info.max_gap >= 16);
    }

    #[test]
    fn find_free_blocks_single_region() {
        let n_blocks = 128;
        let mask = make_mask_with_free(n_blocks, 64, 32);
        let mut gaps = GapsLayer::new(n_blocks);
        gaps.rebuild_from_mask(&mask);

        let result = gaps.find_free_blocks(16, &mask);
        assert!(result.is_some());
        let (start, len) = result.unwrap();
        assert_eq!(len, 16);
        assert!(start >= 64 && start < 96);
    }

    #[test]
    fn find_free_blocks_not_found() {
        let n_blocks = 128;
        let mask = make_mask_with_free(n_blocks, 0, 64);
        let mut gaps = GapsLayer::new(n_blocks);
        gaps.rebuild_from_mask(&mask);

        let result = gaps.find_free_blocks(65, &mask);
        assert!(result.is_none());
    }

    #[test]
    fn find_free_blocks_across_regions() {
        let n_blocks = 256;
        let mask = make_mask_with_free(n_blocks, 60, 100);
        let mut gaps = GapsLayer::new(n_blocks);
        gaps.rebuild_from_mask(&mask);

        let result = gaps.find_free_blocks(20, &mask);
        assert!(result.is_some());
    }

    #[test]
    fn update_after_alloc() {
        let n_blocks = 128;
        let mut mask = BitmaskLayer::new(n_blocks);
        let mut gaps = GapsLayer::new(n_blocks);
        gaps.rebuild_from_mask(&mask);

        mask.mark_used(0, 10);
        gaps.update_after_alloc(0, 10, &mask);

        let info = gaps.get_info(0);
        assert!(info.can_fit(10));
        assert!(info.can_fit(54));
    }

    #[test]
    fn rebuild_from_mask() {
        let n_blocks = 256;
        let mask = make_mask_with_free(n_blocks, 4, 200);
        let mut gaps = GapsLayer::new(n_blocks);
        gaps.rebuild_from_mask(&mask);

        let info = gaps.get_info(0);
        assert!(info.max_gap >= 50 || info.leading_free >= 50 || info.trailing_free >= 50);
    }
}
