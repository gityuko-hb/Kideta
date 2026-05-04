//! CRC32C (Castagnoli) — hardware-accelerated on x86_64 SSE4.2, software
//! fallback on all other targets.
//!
//! CRC32C is a 32-bit cyclic redundancy check polynomial that provides
//! strong error detection capabilities. It's widely used in storage systems
//! (iSCSI, SATA, ZFS) and is the de facto standard for detecting torn writes
//! in write-ahead logs.
//!
//! ## Performance
//!
//! - **x86_64 + SSE4.2**: Hardware CRC instruction — extremely fast
//! - **Other platforms**: Software table-based implementation
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::crc32c::{crc32c, crc32c_combine};
//!
//! // Compute CRC of data
//! let data = b"hello, world";
//! let hash = crc32c(0, data);
//! println!("CRC32C of 'hello, world': {:08x}", hash);
//!
//! // Verify data integrity
//! let data2 = b"hello, world";
//! assert_eq!(crc32c(0, data2), hash);
//!
//! // Different data = different hash
//! let data3 = b"hello, worlD";
//! assert_ne!(crc32c(0, data3), hash);
//! ```
//!
//! ## Incremental CRC (for large data)
//!
//! ```
//! use kideta_core::utils::crc32c::crc32c;
//!
//! // Compute CRC incrementally
//! let chunk1 = b"part1_";
//! let chunk2 = b"part2_";
//! let chunk3 = b"part3";
//!
//! let partial = crc32c(0, chunk1);
//! let combined = crc32c(partial, chunk2);
//! let final_hash = crc32c(combined, chunk3);
//!
//! // Should equal CRC of complete data
//! let full = crc32c(0, b"part1_part2_part3");
//! assert_eq!(final_hash, full);
//! ```
//!
//! ## Use Case: WAL Record Checksums
//!
//! Each WAL record includes a CRC32C to detect torn writes (partial writes
//! that get interrupted by power loss):
//!
//! ```
//! use kideta_core::utils::crc32c::crc32c;
//!
//! // Simulate WAL record
//! let record = b"WAL_RECORD: vector_id=42, data=[1.0, 2.0, 3.0]";
//!
//! // Compute checksum for integrity check on read
//! let stored_checksum = crc32c(0, record);
//!
//! // Later: verify the record wasn't corrupted
//! let record_check = crc32c(0, record);
//! if record_check == stored_checksum {
//!     println!("Record integrity OK");
//! } else {
//!     println!("Record corrupted! Possible torn write.");
//! }
//! ```

// ── Software lookup table ─────────────────────────────────────────────────────

/// CRC32C polynomial (reflected): 0x82F63B78
const POLY: u32 = 0x82F63B78;

/// Pre-computed 256-entry CRC32C table.
const TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        t[i as usize] = crc;
        i += 1;
    }
    t
};

/// Software CRC32C over `data`, starting from `crc` (use 0 for a fresh hash).
#[inline]
pub fn crc32c_sw(
    mut crc: u32,
    data: &[u8],
) -> u32 {
    crc = !crc;
    for &b in data {
        crc = (crc >> 8) ^ TABLE[((crc ^ b as u32) & 0xFF) as usize];
    }
    !crc
}

// ── Hardware path (x86_64 SSE4.2) ────────────────────────────────────────────

/// Hardware CRC32C over `data` using SSE4.2 instructions.
///
/// # Safety
/// This function is unsafe because it requires the `sse4.2` target feature.
/// The caller must ensure that the CPU supports SSE4.2 before calling this
/// function.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_hw(
    crc: u32,
    data: &[u8],
) -> u32 {
    use std::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};
    let mut crc64 = (!crc) as u64;
    let mut p = data.as_ptr();
    let mut len = data.len();

    // Process 8 bytes at a time.
    while len >= 8 {
        // SAFETY: p points into `data`, and we checked len >= 8.
        let word = unsafe { (p as *const u64).read_unaligned() };
        crc64 = _mm_crc32_u64(crc64, word);
        // SAFETY: p is still within `data` bounds.
        p = unsafe { p.add(8) };
        len -= 8;
    }

    let mut crc32 = crc64 as u32;
    // Process remaining bytes.
    // SAFETY: p and len correctly describe the remaining slice.
    let tail = unsafe { std::slice::from_raw_parts(p, len) };
    for &b in tail {
        crc32 = _mm_crc32_u8(crc32, b);
    }
    !crc32
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Compute CRC32C of `data` with initial value `crc` (pass 0 for a fresh hash).
///
/// Uses hardware instruction on SSE4.2 x86_64, software table otherwise.
#[inline]
pub fn crc32c(
    crc: u32,
    data: &[u8],
) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse4.2") {
        return unsafe { crc32c_hw(crc, data) };
    }
    crc32c_sw(crc, data)
}

/// Finalise a CRC32C hash begun with multiple `crc32c` calls.
///
/// This is the same as calling `crc32c(0, data)` on all data concatenated.
#[inline]
pub fn crc32c_combine(
    crc: u32,
    data: &[u8],
) -> u32 {
    crc32c(crc, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference vectors from the iSCSI / RFC 3720 test suite.
    #[test]
    fn zeros_32_bytes() {
        let data = [0u8; 32];
        assert_eq!(crc32c(0, &data), 0x8A9136AA);
    }

    #[test]
    fn ones_32_bytes() {
        let data = [0xFFu8; 32];
        assert_eq!(crc32c(0, &data), 0x62A8AB43);
    }

    #[test]
    fn incrementing_32_bytes() {
        let data: [u8; 32] = core::array::from_fn(|i| i as u8);
        assert_eq!(crc32c(0, &data), 0x46DD794E);
    }

    #[test]
    fn empty_input() {
        assert_eq!(crc32c(0, b""), 0x00000000);
    }

    #[test]
    fn dispatcher_matches_sw() {
        let data = b"hello, kideta WAL record";
        assert_eq!(crc32c(0, data), crc32c_sw(0, data));
    }

    #[test]
    fn incremental_same_as_full() {
        let data = b"split this data into two chunks for testing";
        let full = crc32c(0, data);
        let mid = 20;
        let partial = crc32c(0, &data[..mid]);
        let combined = crc32c(partial, &data[mid..]);
        assert_eq!(full, combined);
    }
}
