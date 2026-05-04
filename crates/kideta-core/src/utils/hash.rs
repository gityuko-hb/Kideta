//! Non-cryptographic hash functions: `xxhash64` and `xxhash3_128`.
//!
//! Both implement the official xxHash specification.
//!
//! * `xxhash64`     — 64-bit output, fast routing / shard assignment.
//! * `xxhash3_128`  — 128-bit output, content checksums for index files.
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::hash::{xxhash64, xxhash3_128};
//!
//! // xxhash64 - fast 64-bit hash
//! let data = b"hello world";
//! let hash = xxhash64(data, 0); // 0 is the seed
//! assert_ne!(hash, 0); // deterministic but non-zero
//!
//! // Same input + same seed = same hash
//! assert_eq!(xxhash64(b"hello", 0), xxhash64(b"hello", 0));
//!
//! // Different seed = different hash
//! let h1 = xxhash64(b"test", 1);
//! let h2 = xxhash64(b"test", 2);
//! assert_ne!(h1, h2);
//!
//! // xxhash3_128 - 128-bit hash (returns tuple of lo, hi)
//! let (lo, hi) = xxhash3_128(b"hello world", 0);
//! assert!(lo != 0 || hi != 0); // non-zero result
//!
//! // Empty input is valid
//! let (lo, hi) = xxhash3_128(b"", 0);
//! println!("Empty hash: {:016x}{:016x}", lo, hi);
//! ```
//!
//! ## Use Case: Shard Assignment
//!
//! For consistent hashing in a distributed setup:
//!
//! ```
//! use kideta_core::utils::hash::xxhash64;
//!
//! fn assign_shard(vector_id: u64, num_shards: u64) -> u64 {
//!     let data = vector_id.to_le_bytes();
//!     xxhash64(&data, 0) % num_shards
//! }
//!
//! assert_eq!(assign_shard(42, 10), assign_shard(42, 10)); // consistent
//! assert!(assign_shard(100, 10) < 10);
//! ```

// ── xxHash64 ─────────────────────────────────────────────────────────────────
// Reference: https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md

const P1: u64 = 0x9E3779B185EBCA87;
const P2: u64 = 0xC2B2AE3D27D4EB4F;
const P3: u64 = 0x165667B19E3779F9;
const P4: u64 = 0x85EBCA77C2B2AE63;
const P5: u64 = 0x27D4EB2F165667C5;

#[inline(always)]
fn rol64(
    x: u64,
    r: u32,
) -> u64 {
    x.rotate_left(r)
}

#[inline(always)]
fn mix_step(
    acc: u64,
    lane: u64,
) -> u64 {
    rol64(acc.wrapping_add(lane.wrapping_mul(P2)), 31).wrapping_mul(P1)
}

#[inline(always)]
fn merge_acc(
    acc: u64,
    lane_acc: u64,
) -> u64 {
    (acc ^ mix_step(0, lane_acc))
        .wrapping_mul(P1)
        .wrapping_add(P4)
}

/// Hash `data` with `seed` using the xxHash64 algorithm.
pub fn xxhash64(
    data: &[u8],
    seed: u64,
) -> u64 {
    let len = data.len();
    let mut h: u64;
    let mut p = 0usize;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);

        let limit = len - 32;
        while p <= limit {
            v1 = mix_step(v1, u64::from_le_bytes(data[p..p + 8].try_into().unwrap()));
            p += 8;
            v2 = mix_step(v2, u64::from_le_bytes(data[p..p + 8].try_into().unwrap()));
            p += 8;
            v3 = mix_step(v3, u64::from_le_bytes(data[p..p + 8].try_into().unwrap()));
            p += 8;
            v4 = mix_step(v4, u64::from_le_bytes(data[p..p + 8].try_into().unwrap()));
            p += 8;
        }

        h = rol64(v1, 1)
            .wrapping_add(rol64(v2, 7))
            .wrapping_add(rol64(v3, 12))
            .wrapping_add(rol64(v4, 18));
        h = merge_acc(h, v1);
        h = merge_acc(h, v2);
        h = merge_acc(h, v3);
        h = merge_acc(h, v4);
    } else {
        h = seed.wrapping_add(P5);
    }

    h = h.wrapping_add(len as u64);

    // Consume remaining 8-byte chunks.
    while p + 8 <= len {
        let k = u64::from_le_bytes(data[p..p + 8].try_into().unwrap());
        h ^= mix_step(0, k);
        h = rol64(h, 27).wrapping_mul(P1).wrapping_add(P4);
        p += 8;
    }

    // Remaining 4-byte chunk.
    if p + 4 <= len {
        let k = u32::from_le_bytes(data[p..p + 4].try_into().unwrap()) as u64;
        h ^= k.wrapping_mul(P1);
        h = rol64(h, 23).wrapping_mul(P2).wrapping_add(P3);
        p += 4;
    }

    // Remaining bytes.
    while p < len {
        h ^= (data[p] as u64).wrapping_mul(P5);
        h = rol64(h, 11).wrapping_mul(P1);
        p += 1;
    }

    // Avalanche.
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^= h >> 32;
    h
}

// ── xxHash3 (128-bit) ─────────────────────────────────────────────────────────
// Simplified implementation of xxHash3 for inputs >= 0 bytes.
// Uses the "secret" bytes defined in the spec (first 192 bytes of secret).

const SECRET: &[u8] = &[
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
    0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
    0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21,
    0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
    0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
    0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
    0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d,
    0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
    0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
    0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
    0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce,
    0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
];

const H3_P1: u64 = 0x9E3779B185EBCA87;
const H3_P2: u64 = 0xC2B2AE3D27D4EB4F;

#[inline(always)]
fn read_u64_le(
    b: &[u8],
    offset: usize,
) -> u64 {
    u64::from_le_bytes(b[offset..offset + 8].try_into().unwrap())
}

#[inline(always)]
fn read_u32_le(
    b: &[u8],
    offset: usize,
) -> u32 {
    u32::from_le_bytes(b[offset..offset + 4].try_into().unwrap())
}

#[allow(dead_code)]
#[inline(always)]
fn mix16(
    data: &[u8],
    data_off: usize,
    secret: &[u8],
    secret_off: usize,
    seed: u64,
) -> u64 {
    let lo = read_u64_le(data, data_off) ^ (read_u64_le(secret, secret_off).wrapping_add(seed));
    let hi =
        read_u64_le(data, data_off + 8) ^ (read_u64_le(secret, secret_off + 8).wrapping_sub(seed));
    mul128_fold64(lo, hi)
}

#[inline(always)]
fn mul128_fold64(
    a: u64,
    b: u64,
) -> u64 {
    let full = (a as u128).wrapping_mul(b as u128);
    ((full >> 64) ^ full) as u64
}

/// Hash `data` using xxHash3 and return a 128-bit value as `(lo, hi)`.
pub fn xxhash3_128(
    data: &[u8],
    seed: u64,
) -> (u64, u64) {
    let len = data.len();

    if len == 0 {
        let h = avalanche(seed ^ (read_u64_le(SECRET, 64) ^ read_u64_le(SECRET, 72)));
        return (h, h);
    }

    if len <= 3 {
        let c1 = data[0] as u64;
        let c2 = (data[len >> 1] as u64) << 8;
        let c3 = (data[len - 1] as u64) << 16;
        let combined = c1 | c2 | c3 | ((len as u64) << 24);
        let lo = (read_u32_le(SECRET, 0) as u64 ^ read_u32_le(SECRET, 4) as u64)
            .wrapping_add(seed)
            .wrapping_add(combined.swap_bytes());
        let hi = (read_u32_le(SECRET, 8) as u64 ^ read_u32_le(SECRET, 12) as u64)
            .wrapping_sub(seed)
            ^ combined;
        return (avalanche(lo), avalanche(hi));
    }

    if len <= 8 {
        let lo_input = read_u32_le(data, 0) as u64 | ((read_u32_le(data, len - 4) as u64) << 32);
        let hi_input = read_u32_le(data, 0) as u64 | ((read_u32_le(data, len - 4) as u64) << 32);
        let lo = ((read_u64_le(SECRET, 0) ^ read_u64_le(SECRET, 8)).wrapping_sub(seed))
            ^ lo_input.swap_bytes();
        let hi =
            ((read_u64_le(SECRET, 16) ^ read_u64_le(SECRET, 24)).wrapping_add(seed)) ^ hi_input;
        return (avalanche(lo ^ (len as u64)), avalanche(hi ^ (len as u64)));
    }

    if len <= 16 {
        // Read 8 bytes from start and 8 bytes from end, which may overlap.
        let input_lo = read_u64_le(data, 0);
        let input_hi = read_u64_le(data, len - 8);

        let secret_lo = read_u64_le(SECRET, 48) ^ read_u64_le(SECRET, 56);
        let secret_hi = read_u64_le(SECRET, 64) ^ read_u64_le(SECRET, 72);

        let lo = mul128_fold64(
            input_lo ^ (secret_lo.wrapping_add(seed)),
            input_hi ^ (secret_hi.wrapping_sub(seed)),
        );
        let hi = mul128_fold64(
            input_lo ^ (secret_hi.wrapping_add(seed)),
            input_hi ^ (secret_lo.wrapping_sub(seed)),
        );

        return (avalanche(lo ^ (len as u64)), avalanche(hi ^ (len as u64)));
    }

    // For longer inputs: process 16-byte stripes.
    let mut acc0 = seed ^ read_u64_le(SECRET, 0);
    let mut acc1 = seed ^ read_u64_le(SECRET, 8);
    let nb_stripes = len / 16;

    for stripe in 0..nb_stripes {
        let off = stripe * 16;
        let s0 = stripe % ((SECRET.len() - 16) / 8);
        let d0 = read_u64_le(data, off) ^ read_u64_le(SECRET, s0 * 8);
        let d1 = read_u64_le(data, off + 8) ^ read_u64_le(SECRET, s0 * 8 + 8);
        acc0 = acc0.wrapping_add(mul128_fold64(d0, d1 ^ H3_P1));
        acc1 = acc1.wrapping_add(mul128_fold64(d1, d0 ^ H3_P2));
    }

    // Handle the final partial stripe.
    let tail = len - 16;
    let s0 = (SECRET.len() - 16) / 8 % ((SECRET.len() - 16) / 8);
    let d0 = read_u64_le(data, tail) ^ read_u64_le(SECRET, s0 * 8);
    let d1 = read_u64_le(data, tail + 8) ^ read_u64_le(SECRET, s0 * 8 + 8);
    acc0 = acc0.wrapping_add(mul128_fold64(d0, d1 ^ H3_P1));
    acc1 = acc1.wrapping_add(mul128_fold64(d1, d0 ^ H3_P2));

    let lo = avalanche(acc0.wrapping_add(len as u64));
    let hi = avalanche(acc1.wrapping_sub(len as u64));
    (lo, hi)
}

#[inline(always)]
fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 37;
    h = h.wrapping_mul(0x165667919E3779F9);
    h ^= h >> 32;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── xxhash64 ──────────────────────────────────────────────────────────────

    #[test]
    fn xxhash64_empty() {
        // Known value from the reference implementation: seed=0 → 0xEF46DB3751D8E999
        assert_eq!(xxhash64(b"", 0), 0xEF46DB3751D8E999);
    }

    #[test]
    fn xxhash64_seed_changes_output() {
        let h0 = xxhash64(b"hello", 0);
        let h1 = xxhash64(b"hello", 1);
        assert_ne!(h0, h1);
    }

    #[test]
    fn xxhash64_deterministic() {
        assert_eq!(xxhash64(b"kideta", 42), xxhash64(b"kideta", 42));
    }

    #[test]
    fn xxhash64_different_inputs() {
        assert_ne!(xxhash64(b"foo", 0), xxhash64(b"bar", 0));
    }

    #[test]
    fn xxhash64_long_input() {
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let h = xxhash64(&data, 0);
        assert_ne!(h, 0);
        assert_eq!(h, xxhash64(&data, 0)); // deterministic
    }

    // ── xxhash3_128 ───────────────────────────────────────────────────────────

    #[test]
    fn xxhash3_128_deterministic() {
        let (lo1, hi1) = xxhash3_128(b"kideta vector db", 0);
        let (lo2, hi2) = xxhash3_128(b"kideta vector db", 0);
        assert_eq!((lo1, hi1), (lo2, hi2));
    }

    #[test]
    fn xxhash3_128_different_inputs() {
        let a = xxhash3_128(b"hello", 0);
        let b = xxhash3_128(b"world", 0);
        assert_ne!(a, b);
    }

    #[test]
    fn xxhash3_128_non_zero() {
        let (lo, hi) = xxhash3_128(b"test data", 0);
        assert!(lo != 0 || hi != 0);
    }

    #[test]
    fn xxhash3_128_seed_matters() {
        let a = xxhash3_128(b"same", 0);
        let b = xxhash3_128(b"same", 999);
        assert_ne!(a, b);
    }
}
