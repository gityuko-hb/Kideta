//! SHA-256 — block-based implementation (FIPS 180-4).
//!
//! A pure-Rust implementation of SHA-256 (Secure Hash Algorithm 2) with a
//! 256-bit digest. Used for HMAC-SHA256 JWT signing and cryptographic purposes.
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::sha256::sha256;
//!
//! // Compute SHA-256 hash
//! let data = b"hello world";
//! let hash = sha256(data);
//!
//! // Hash is 32 bytes (256 bits)
//! assert_eq!(hash.len(), 32);
//!
//! // Print as hex
//! let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
//! println!("SHA-256: {}", hex);
//! ```
//!
//! ## Known Test Vectors
//!
//! ```
//! use kideta_core::utils::sha256::sha256;
//!
//! fn to_hex(bytes: &[u8]) -> String {
//!     bytes.iter().map(|b| format!("{:02x}", b)).collect()
//! }
//!
//! // Empty string (NIST test vector)
//! let hash = sha256(b"");
//! assert_eq!(to_hex(&hash), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
//!
//! // "abc" (NIST test vector)
//! let hash = sha256(b"abc");
//! assert_eq!(to_hex(&hash), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
//! ```

// ── Round constants ───────────────────────────────────────────────────────────

#[rustfmt::skip]
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// ── Initial hash values ───────────────────────────────────────────────────────

const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

// ── Core functions ────────────────────────────────────────────────────────────

#[inline(always)]
fn ch(
    e: u32,
    f: u32,
    g: u32,
) -> u32 {
    (e & f) ^ (!e & g)
}
#[inline(always)]
fn maj(
    a: u32,
    b: u32,
    c: u32,
) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}
#[inline(always)]
fn sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
#[inline(always)]
fn sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}
#[inline(always)]
fn gamma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
#[inline(always)]
fn gamma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

fn compress(
    state: &mut [u32; 8],
    block: &[u8; 64],
) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
    }
    for i in 16..64 {
        w[i] = gamma1(w[i - 2])
            .wrapping_add(w[i - 7])
            .wrapping_add(gamma0(w[i - 15]))
            .wrapping_add(w[i - 16]);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let t1 = h
            .wrapping_add(sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let t2 = sigma0(a).wrapping_add(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// SHA-256 digest of `data`.  Returns 32 bytes (256 bits).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = H0;
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut block = [0u8; 64];

    // Process full 64-byte blocks.
    let full_blocks = data.len() / 64;
    for i in 0..full_blocks {
        block.copy_from_slice(&data[i * 64..(i + 1) * 64]);
        compress(&mut state, &block);
    }

    // Padding: copy remainder, append 0x80.
    let remainder = data.len() % 64;
    block = [0u8; 64];
    block[..remainder].copy_from_slice(&data[full_blocks * 64..]);
    block[remainder] = 0x80;

    if remainder >= 56 {
        // Not enough room for the length; need an extra block.
        compress(&mut state, &block);
        block = [0u8; 64];
    }

    // Append bit length as big-endian u64 in the final 8 bytes.
    block[56..64].copy_from_slice(&bit_len.to_be_bytes());
    compress(&mut state, &block);

    let mut out = [0u8; 32];
    for (i, &h) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&h.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn sha256_empty() {
        // NIST known answer.
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc() {
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_long() {
        // "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_deterministic() {
        assert_eq!(sha256(b"kideta"), sha256(b"kideta"));
    }

    #[test]
    fn sha256_different_inputs() {
        assert_ne!(sha256(b"foo"), sha256(b"bar"));
    }

    #[test]
    fn sha256_64_byte_boundary() {
        // Test that padding across a 64-byte block boundary works.
        let data = [0x61u8; 64]; // exactly one block
        let h = sha256(&data);
        assert_ne!(h, [0u8; 32]);
        assert_eq!(h, sha256(&data)); // deterministic
    }
}
