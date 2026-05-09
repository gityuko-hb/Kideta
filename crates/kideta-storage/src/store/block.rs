//! Block pointer — the fundamental address type in KidetaStore.
//!
//! A `BlockPtr` encodes the location of a value stored in the data grid using
//! exactly 12 bytes: a 4-byte page ID, a 4-byte byte offset within that page,
//! and a 2-byte byte length.  The remaining 2 bytes are padding to reach 12 bytes
//! total so the struct has a known ABI when mmap'd as part of the tracker array.

use std::fmt;

pub const BLOCK_SIZE: usize = 128;
pub const PAGE_SIZE: usize = 64 * 1024 * 1024;
pub const BLOCKS_PER_PAGE: usize = PAGE_SIZE / BLOCK_SIZE;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C, packed)]
pub struct BlockPtr {
    pub page_id: u32,
    pub byte_offset: u32,
    pub byte_len: u16,
    pub _padding: [u8; 2],
}

const _: () = assert!(std::mem::size_of::<BlockPtr>() == 12);

impl BlockPtr {
    #[inline]
    pub const fn null() -> Self {
        Self {
            page_id: u32::MAX,
            byte_offset: 0,
            byte_len: 0,
            _padding: [0; 2],
        }
    }

    #[inline]
    pub const fn is_null(&self) -> bool {
        self.page_id == u32::MAX
    }

    #[inline]
    pub const fn data_len(&self) -> usize {
        self.byte_len as usize
    }

    #[inline]
    pub const fn n_blocks(&self) -> u32 {
        (self.byte_len as u32).div_ceil(BLOCK_SIZE as u32)
    }
}

unsafe impl Send for BlockPtr {}
unsafe impl Sync for BlockPtr {}

impl fmt::Display for BlockPtr {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        if self.is_null() {
            write!(f, "null")
        } else {
            let page_id = self.page_id;
            let byte_offset = self.byte_offset;
            let byte_len = self.byte_len;
            write!(
                f,
                "BlockPtr(page={}, byte_offset={}, len={})",
                page_id, byte_offset, byte_len
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_ptr_has_max_page_id() {
        let ptr = BlockPtr::null();
        assert!(ptr.is_null());
        let page_id = ptr.page_id;
        assert_eq!(page_id, u32::MAX);
    }

    #[test]
    fn null_ptr_data_len_is_zero() {
        let ptr = BlockPtr::null();
        assert_eq!(ptr.data_len(), 0);
    }

    #[test]
    fn n_blocks_rounds_up() {
        let ptr_1 = BlockPtr {
            page_id: 0,
            byte_offset: 0,
            byte_len: 1,
            _padding: [0; 2],
        };
        assert_eq!(ptr_1.n_blocks(), 1);

        let ptr_128 = BlockPtr {
            page_id: 0,
            byte_offset: 0,
            byte_len: 128,
            _padding: [0; 2],
        };
        assert_eq!(ptr_128.n_blocks(), 1);

        let ptr_129 = BlockPtr {
            page_id: 0,
            byte_offset: 0,
            byte_len: 129,
            _padding: [0; 2],
        };
        assert_eq!(ptr_129.n_blocks(), 2);
    }

    #[test]
    fn block_ptr_is_packed_12_bytes() {
        assert_eq!(std::mem::size_of::<BlockPtr>(), 12);
    }

    #[test]
    fn display_null() {
        let ptr = BlockPtr::null();
        assert_eq!(format!("{}", ptr), "null");
    }

    #[test]
    fn display_non_null() {
        let ptr = BlockPtr {
            page_id: 5,
            byte_offset: 42 * 128,
            byte_len: 200,
            _padding: [0; 2],
        };
        assert_eq!(
            format!("{}", ptr),
            "BlockPtr(page=5, byte_offset=5376, len=200)"
        );
    }
}
