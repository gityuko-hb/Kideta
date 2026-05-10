//! Variable-length integer encoding — LEB128 (unsigned).
//!
//! Variable-length encoding (also known as LEB128 or varint) stores integers
//! using a variable number of bytes. This is efficient for small numbers
//! (which are common) while still supporting full 64-bit values.
//!
//! ## Encoding Scheme
//!
//! Each byte stores 7 bits of data, with the MSB as a continuation flag:
//! - If MSB = 0: this is the last byte
//! - If MSB = 1: more bytes follow
//!
//! ## Efficiency
//!
//! | Value Range | Bytes Needed |
//! |-------------|--------------|
//! | 0 - 127     | 1            |
//! | 128 - 16383 | 2            |
//! | 16384 - 2M  | 3            |
//! | ...         | ...          |
//! | u64::MAX    | 10           |
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::varint::{encode_u64, decode_u64, encoded_len_u64};
//!
//! // Encode
//! let mut buf = [0u8; 10];
//! let written = encode_u64(300, &mut buf);
//! println!("300 encodes to {} bytes", written);
//!
//! // Decode
//! let (value, read) = decode_u64(&buf[..written]).unwrap();
//! assert_eq!(value, 300);
//!
//! // Check encoded length without encoding
//! assert_eq!(encoded_len_u64(300), 2);
//! assert_eq!(encoded_len_u64(127), 1);  //边界值
//! assert_eq!(encoded_len_u64(128), 2);  //边界值
//! ```
//!
//! ## Zigzag Encoding for Signed Integers
//!
//! For signed integers, use zigzag encoding to handle negative numbers efficiently:
//!
//! ```
//! use kideta_core::utils::varint::{encode_i64, decode_i64};
//!
//! // Encode signed integers (zigzag maps -1 to 1, 1 to 2, etc.)
//! let mut buf = [0u8; 10];
//!
//! // Encode various values
//! let cases = [0i64, 1, -1, 64, -64, i64::MAX, i64::MIN];
//! for val in cases {
//!     let written = encode_i64(val, &mut buf);
//!     let (decoded, _) = decode_i64(&buf[..written]).unwrap();
//!     assert_eq!(decoded, val, "roundtrip failed for {}", val);
//! }
//! ```
//!
//! ## Use Case: WAL Record Headers
//!
//! Variable-length encoding is essential for WAL to minimize record sizes:
//!
//! ```
//! use kideta_core::utils::varint::{encode_u64, decode_u64, MAX_VARINT_LEN64};
//!
//! // Simulate a WAL record: [type][lsn][vector_id][dimension]
//! let mut record = Vec::new();
//!
//! // Encode each field
//! let type_code: u64 = 1;  // INSERT operation
//! let lsn: u64 = 1000000;  // log sequence number
//! let vector_id: u64 = 42;
//! let dimension: u64 = 768;
//!
//! let mut buf = [0u8; MAX_VARINT_LEN64];
//! record.push(encode_u64(type_code, &mut buf) as u8);
//! record.push(encode_u64(lsn, &mut buf) as u8);
//! record.push(encode_u64(vector_id, &mut buf) as u8);
//! record.push(encode_u64(dimension, &mut buf) as u8);
//!
//! // Total size is much smaller than fixed 8 bytes each
//! println!("WAL record size: {} bytes", record.len());
//! ```

/// Maximum bytes required to encode any `u64` value (ceil(64/7) = 10).
pub const MAX_VARINT_LEN64: usize = 10;

/// Maximum bytes required to encode any `u32` value (ceil(32/7) = 5).
pub const MAX_VARINT_LEN32: usize = 5;

// ── Encode ────────────────────────────────────────────────────────────────────

/// Encode `value` into `buf` using unsigned LEB128.
///
/// Returns the number of bytes written.
///
/// # Panics
/// Panics if `buf.len() < MAX_VARINT_LEN64`.
#[inline]
pub fn encode_u64(
    mut value: u64,
    buf: &mut [u8],
) -> usize {
    debug_assert!(buf.len() >= MAX_VARINT_LEN64, "buffer too small for varint");
    let mut i = 0;
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf[i] = byte;
            i += 1;
            break;
        }
        buf[i] = byte | 0x80;
        i += 1;
    }
    i
}

/// Encode `value` into `buf` using unsigned LEB128 (convenience for u32).
#[inline]
pub fn encode_u32(
    value: u32,
    buf: &mut [u8],
) -> usize {
    encode_u64(value as u64, buf)
}

/// Encode `value` as a zigzag-encoded signed varint into `buf`.
///
/// Zigzag maps `0 → 0, -1 → 1, 1 → 2, -2 → 3, ...`, so small negative
/// numbers take few bytes.
#[inline]
pub fn encode_i64(
    value: i64,
    buf: &mut [u8],
) -> usize {
    let zig = ((value << 1) ^ (value >> 63)) as u64;
    encode_u64(zig, buf)
}

// ── Decode ────────────────────────────────────────────────────────────────────

/// Decode an unsigned LEB128 varint from `buf`.
///
/// Returns `(value, bytes_consumed)`, or `None` if the buffer is truncated or
/// the encoded value overflows 64 bits.
#[inline]
pub fn decode_u64(buf: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0u32;
    for (i, &byte) in buf.iter().enumerate().take(MAX_VARINT_LEN64) {
        let low7 = (byte & 0x7F) as u64;
        if shift < 64 {
            value |= low7 << shift;
        } else if low7 != 0 {
            return None; // overflow
        }
        shift += 7;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None // truncated
}

/// Decode a zigzag-encoded signed varint.
#[inline]
pub fn decode_i64(buf: &[u8]) -> Option<(i64, usize)> {
    let (zig, n) = decode_u64(buf)?;
    let value = ((zig >> 1) as i64) ^ -((zig & 1) as i64);
    Some((value, n))
}

/// Decode a `u32` varint (values > u32::MAX return `None`).
#[inline]
pub fn decode_u32(buf: &[u8]) -> Option<(u32, usize)> {
    let (v, n) = decode_u64(buf)?;
    if v > u32::MAX as u64 {
        return None;
    }
    Some((v as u32, n))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// How many bytes does `value` encode to?
#[inline]
pub fn encoded_len_u64(mut value: u64) -> usize {
    if value == 0 {
        return 1;
    }
    let mut n = 0;
    while value > 0 {
        value >>= 7;
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_u64(v: u64) {
        let mut buf = [0u8; MAX_VARINT_LEN64];
        let written = encode_u64(v, &mut buf);
        let (decoded, read) = decode_u64(&buf[..written]).unwrap();
        assert_eq!(decoded, v);
        assert_eq!(read, written);
    }

    #[test]
    fn u64_zero() {
        roundtrip_u64(0);
    }
    #[test]
    fn u64_one() {
        roundtrip_u64(1);
    }
    #[test]
    fn u64_127() {
        roundtrip_u64(127);
    } // exactly 1 byte
    #[test]
    fn u64_128() {
        roundtrip_u64(128);
    } // 2 bytes
    #[test]
    fn u64_max() {
        roundtrip_u64(u64::MAX);
    }
    #[test]
    fn u64_large() {
        roundtrip_u64(1_000_000_000_000);
    }

    #[test]
    fn single_byte_values() {
        for v in 0u64..128 {
            let mut buf = [0u8; MAX_VARINT_LEN64];
            let n = encode_u64(v, &mut buf);
            assert_eq!(n, 1, "v={v} should encode to 1 byte");
            let (decoded, read) = decode_u64(&buf[..n]).unwrap();
            assert_eq!(decoded, v);
            assert_eq!(read, 1);
        }
    }

    #[test]
    fn u64_length_matches_encoded_len() {
        for v in [0u64, 1, 127, 128, 300, 16384, u32::MAX as u64, u64::MAX] {
            let mut buf = [0u8; MAX_VARINT_LEN64];
            let written = encode_u64(v, &mut buf);
            assert_eq!(written, encoded_len_u64(v));
        }
    }

    #[test]
    fn i64_zigzag_roundtrip() {
        for v in [0i64, 1, -1, 64, -64, i64::MAX, i64::MIN] {
            let mut buf = [0u8; MAX_VARINT_LEN64];
            let n = encode_i64(v, &mut buf);
            let (decoded, read) = decode_i64(&buf[..n]).unwrap();
            assert_eq!(decoded, v);
            assert_eq!(read, n);
        }
    }

    #[test]
    fn truncated_returns_none() {
        let mut buf = [0u8; MAX_VARINT_LEN64];
        encode_u64(300, &mut buf); // 2 bytes, buf[0] has high bit set
        assert!(decode_u64(&buf[..1]).is_none());
    }
}
