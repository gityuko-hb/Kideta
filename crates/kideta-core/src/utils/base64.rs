//! Base64url encode / decode (RFC 4648 §5) — no padding, URL-safe alphabet.
//!
//! This module provides Base64url encoding, which is a URL-safe variant of
//! standard Base64. It uses `-` and `_` instead of `+` and `/`, and omits
//! padding characters (`=`). This makes it safe for use in URLs, query
//! parameters, and HTTP headers.
//!
//! ## Why Base64url?
//!
//! Standard Base64 uses `+` and `/` which need URL encoding, and `=` padding
//! which can cause issues in some contexts. Base64url avoids all these problems.
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::base64::{encode, decode};
//!
//! // Encode binary data to a URL-safe string
//! let data = b"Hello, World!";
//! let encoded = encode(data);
//! println!("Encoded: {}", encoded); // "SGVsbG8sIFdvcmxkIQ"
//!
//! // Decode back to original bytes
//! let decoded = decode(&encoded).unwrap();
//! assert_eq!(decoded, data);
//!
//! // Round trip with binary data
//! let binary = vec![0xFB, 0xFF, 0x00, 0x1F];
//! let enc = encode(&binary);
//! assert!(!enc.contains('+')); // URL-safe
//! assert!(!enc.contains('/')); // URL-safe
//! assert!(!enc.contains('=')); // No padding
//! let dec = decode(&enc).unwrap();
//! assert_eq!(dec, binary);
//! ```
//!
//! ## Error Handling
//!
//! Decoding returns `None` on invalid input:
//!
//! ```
//! use kideta_core::utils::base64::decode;
//!
//! // Valid input decodes successfully
//! assert!(decode("SGVsbG8").is_some());
//!
//! // Invalid characters return None
//! assert!(decode("SGVsbG8!").is_none()); // Contains !
//! assert!(decode("SGVsbG8=").is_none()); // Contains padding
//! ```
//!
//! ## Use Case: Scroll API Cursor Tokens
//!
//! Encode vector IDs and offsets into opaque cursor tokens:
//!
//! ```
//! use kideta_core::utils::base64::{encode, decode};
//!
//! struct ScrollCursor {
//!     vector_id: u64,
//!     offset: u64,
//! }
//!
//! fn encode_cursor(id: u64, offset: u64) -> String {
//!     let bytes: Vec<u8> = id.to_le_bytes().into_iter()
//!         .chain(offset.to_le_bytes().into_iter())
//!         .collect();
//!     encode(&bytes)
//! }
//!
//! fn decode_cursor(token: &str) -> Option<ScrollCursor> {
//!     let bytes = decode(token)?;
//!     if bytes.len() != 16 { return None; }
//!     let id = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
//!     let offset = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
//!     Some(ScrollCursor { vector_id: id, offset })
//! }
//!
//! let cursor = encode_cursor(42, 100);
//! let parsed = decode_cursor(&cursor).unwrap();
//! assert_eq!(parsed.vector_id, 42);
//! assert_eq!(parsed.offset, 100);
//! ```

const ENCODE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Lookup table: ASCII → 6-bit value.  0xFF = invalid character.
const DECODE: [u8; 256] = {
    let mut t = [0xFFu8; 256];
    let mut i = 0u8;
    // A-Z → 0..25
    while i < 26 {
        t[(b'A' + i) as usize] = i;
        i += 1;
    }
    i = 0;
    // a-z → 26..51
    while i < 26 {
        t[(b'a' + i) as usize] = 26 + i;
        i += 1;
    }
    i = 0;
    // 0-9 → 52..61
    while i < 10 {
        t[(b'0' + i) as usize] = 52 + i;
        i += 1;
    }
    t[b'-' as usize] = 62;
    t[b'_' as usize] = 63;
    t
};

/// Encode `src` as a base64url string (no `=` padding).
pub fn encode(src: &[u8]) -> String {
    let out_len = src.len().div_ceil(3) * 4;
    // Actual output is ≤ out_len (no padding chars needed when trimmed).
    let actual_len = (src.len() * 4).div_ceil(3);
    let mut out = Vec::with_capacity(actual_len);

    for chunk in src.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 {
            chunk[1] as u32
        } else {
            0
        };
        let b2 = if chunk.len() > 2 {
            chunk[2] as u32
        } else {
            0
        };
        let combined = (b0 << 16) | (b1 << 8) | b2;

        out.push(ENCODE[((combined >> 18) & 0x3F) as usize]);
        out.push(ENCODE[((combined >> 12) & 0x3F) as usize]);
        if chunk.len() > 1 {
            out.push(ENCODE[((combined >> 6) & 0x3F) as usize]);
        }
        if chunk.len() > 2 {
            out.push(ENCODE[(combined & 0x3F) as usize]);
        }
    }

    // Suppress the unused-variable warning for out_len in release builds.
    let _ = out_len;
    // SAFETY: all bytes come from the ENCODE table which is ASCII.
    unsafe { String::from_utf8_unchecked(out) }
}

/// Decode a base64url string into bytes.
///
/// Returns `None` on any invalid character.
pub fn decode(src: &str) -> Option<Vec<u8>> {
    let src = src.as_bytes();
    let mut out = Vec::with_capacity(src.len() * 3 / 4 + 2);
    let mut i = 0;

    while i < src.len() {
        let remaining = src.len() - i;
        if remaining == 1 {
            return None; // single char is always invalid
        }

        let c0 = DECODE[src[i] as usize];
        let c1 = DECODE[src[i + 1] as usize];
        if c0 == 0xFF || c1 == 0xFF {
            return None;
        }
        out.push((c0 << 2) | (c1 >> 4));

        if remaining >= 3 {
            let c2 = DECODE[src[i + 2] as usize];
            if c2 == 0xFF {
                return None;
            }
            out.push(((c1 & 0x0F) << 4) | (c2 >> 2));
        }

        if remaining >= 4 {
            let c3 = DECODE[src[i + 3] as usize];
            if c3 == 0xFF {
                return None;
            }
            out.push(((DECODE[src[i + 2] as usize] & 0x03) << 6) | c3);
        }

        i += 4;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let encoded = encode(data);
        let decoded = decode(&encoded).expect("decode failed");
        assert_eq!(decoded, data, "roundtrip failed for {data:?}");
    }

    #[test]
    fn empty() {
        assert_eq!(encode(b""), "");
        assert_eq!(decode(""), Some(vec![]));
    }

    #[test]
    fn one_byte() {
        roundtrip(b"A");
    }

    #[test]
    fn two_bytes() {
        roundtrip(b"AB");
    }

    #[test]
    fn three_bytes() {
        roundtrip(b"ABC");
    }

    #[test]
    fn four_bytes() {
        roundtrip(b"ABCD");
    }

    #[test]
    fn known_vector() {
        // "Man" → "TWFu" in standard base64 (same for url-safe, no padding)
        assert_eq!(encode(b"Man"), "TWFu");
        assert_eq!(decode("TWFu"), Some(b"Man".to_vec()));
    }

    #[test]
    fn url_safe_chars() {
        // Encode binary that would produce '+' and '/' in standard base64.
        // 0b11111011 = 0xFB → last 6 bits map to '/', url-safe maps to '_'.
        let encoded = encode(&[0xFB, 0xFF]);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('='));
        roundtrip(&[0xFB, 0xFF]);
    }

    #[test]
    fn binary_roundtrip() {
        let data: Vec<u8> = (0u8..=255).collect();
        roundtrip(&data);
    }

    #[test]
    fn invalid_char_returns_none() {
        assert_eq!(decode("AB=="), None); // '=' is not valid
        assert_eq!(decode("A!"), None);
    }
}
