//! `RoaringBitmap` — compressed bitset for 32-bit unsigned integers.
//!
//! A Roaring Bitmap is a compressed bitset optimized for storing sorted
//! integers. It provides excellent compression and fast operations.
//!
//! ## How It Works
//!
//! Integers are split into 65,536-element chunks indexed by the top 16 bits.
//! Each chunk uses one of three containers automatically:
//!
//! * **Array** (< 4096 elements): sorted `Vec<u16>` — best for sparse sets
//! * **Bitset** (≥ 4096 elements): flat 8 KiB bitvector — best for dense sets
//! * **Run** (RLE): run-length encoded — best for consecutive ranges
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::roaring::RoaringBitmap;
//!
//! // Create and populate
//! let mut rb = RoaringBitmap::new();
//!
//! // Insert values
//! assert!(rb.insert(42));  // new insertion
//! assert!(!rb.insert(42)); // duplicate - returns false
//!
//! // Check membership
//! assert!(rb.contains(42));
//! assert!(!rb.contains(43));
//!
//! // Bulk operations
//! for i in 100..200 {
//!     rb.insert(i);
//! }
//! assert_eq!(rb.len(), 100 + 1); // 42 + 100 values
//!
//! // Remove
//! rb.remove(42);
//! assert!(!rb.contains(42));
//! ```
//!
//! ## Set Operations
//!
//! ```
//! use kideta_core::utils::roaring::RoaringBitmap;
//!
//! let mut a = RoaringBitmap::new();
//! let mut b = RoaringBitmap::new();
//!
//! a.insert(1);
//! a.insert(2);
//! a.insert(3);
//!
//! b.insert(2);
//! b.insert(3);
//! b.insert(4);
//!
//! // Union: all elements in either set
//! let u: Vec<u32> = a.union(&b).iter().collect();
//! assert_eq!(u, vec![1, 2, 3, 4]);
//!
//! // Intersection: elements in both sets
//! let i: Vec<u32> = a.intersection(&b).iter().collect();
//! assert_eq!(i, vec![2, 3]);
//!
//! // Difference: elements in a but not in b
//! let d: Vec<u32> = a.difference(&b).iter().collect();
//! assert_eq!(d, vec![1]);
//! ```
//!
//! ## Serialization
//!
//! ```
//! use kideta_core::utils::roaring::RoaringBitmap;
//!
//! let mut rb = RoaringBitmap::new();
//! for i in [5, 10, 15, 20, 100, 200] {
//!     rb.insert(i);
//! }
//!
//! // Serialize to bytes
//! let bytes = rb.serialize();
//!
//! // Deserialize back
//! let rb2 = RoaringBitmap::deserialize(&bytes).unwrap();
//!
//! assert_eq!(rb.len(), rb2.len());
//! let v1: Vec<u32> = rb.iter().collect();
//! let v2: Vec<u32> = rb2.iter().collect();
//! assert_eq!(v1, v2);
//! ```
//!
//! ## Use Case: Tracking Deleted Vector IDs
//!
//! Use a RoaringBitmap to track deleted vectors in a segment:
//!
//! ```
//! use kideta_core::utils::roaring::RoaringBitmap;
//!
//! let mut deleted = RoaringBitmap::new();
//!
//! // Mark some vectors as deleted
//! deleted.insert(5);
//! deleted.insert(10);
//! deleted.insert(15);
//!
//! // Check before read
//! fn can_read_vector(id: u32, deleted: &RoaringBitmap) -> bool {
//!     !deleted.contains(id)
//! }
//!
//! assert!(can_read_vector(1, &deleted));  // not deleted
//! assert!(!can_read_vector(5, &deleted)); // deleted
//! ```

// ── Array container ───────────────────────────────────────────────────────────

const ARRAY_THRESHOLD: usize = 4096;
const BITSET_WORDS: usize = 1024; // 1024 × 64 = 65536 bits

#[derive(Clone, Debug)]
enum Container {
    Array(Vec<u16>),
    Bitset(Box<[u64; BITSET_WORDS]>),
    Run(Vec<(u16, u16)>), // (start, length), inclusive: covers start..=start+length
}

#[allow(dead_code)]
impl Container {
    // ── construction ──────────────────────────────────────────────────────────

    fn new_array() -> Self {
        Container::Array(Vec::new())
    }
    fn new_bitset() -> Self {
        Container::Bitset(Box::new([0u64; BITSET_WORDS]))
    }

    // ── cardinality ───────────────────────────────────────────────────────────

    fn len(&self) -> usize {
        match self {
            Container::Array(v) => v.len(),
            Container::Bitset(b) => b.iter().map(|w| w.count_ones() as usize).sum(),
            Container::Run(runs) => runs.iter().map(|&(_, l)| l as usize + 1).sum(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ── membership ────────────────────────────────────────────────────────────

    fn contains(
        &self,
        lo: u16,
    ) -> bool {
        match self {
            Container::Array(v) => v.binary_search(&lo).is_ok(),
            Container::Bitset(b) => {
                let (word, bit) = (lo as usize / 64, lo as usize % 64);
                b[word] >> bit & 1 == 1
            },
            Container::Run(runs) => {
                for &(s, l) in runs {
                    if lo < s {
                        return false;
                    }
                    if lo <= s.saturating_add(l) {
                        return true;
                    }
                }
                false
            },
        }
    }

    // ── insert ────────────────────────────────────────────────────────────────

    fn insert(
        &mut self,
        lo: u16,
    ) -> bool {
        match self {
            Container::Array(v) => match v.binary_search(&lo) {
                Ok(_) => false,
                Err(i) => {
                    v.insert(i, lo);
                    if v.len() >= ARRAY_THRESHOLD {
                        *self = self.to_bitset();
                    }
                    true
                },
            },
            Container::Bitset(b) => {
                let (word, bit) = (lo as usize / 64, lo as usize % 64);
                let already = b[word] >> bit & 1 == 1;
                b[word] |= 1u64 << bit;
                !already
            },
            Container::Run(_) => {
                *self = self.to_array();
                self.insert(lo)
            },
        }
    }

    // ── remove ────────────────────────────────────────────────────────────────

    fn remove(
        &mut self,
        lo: u16,
    ) -> bool {
        match self {
            Container::Array(v) => match v.binary_search(&lo) {
                Ok(i) => {
                    v.remove(i);
                    true
                },
                Err(_) => false,
            },
            Container::Bitset(b) => {
                let (word, bit) = (lo as usize / 64, lo as usize % 64);
                let had = b[word] >> bit & 1 == 1;
                b[word] &= !(1u64 << bit);
                if had && self.len() < ARRAY_THRESHOLD / 2 {
                    *self = self.to_array();
                }
                had
            },
            Container::Run(_) => {
                *self = self.to_array();
                self.remove(lo)
            },
        }
    }

    // ── conversion helpers ────────────────────────────────────────────────────

    fn to_array(&self) -> Container {
        let mut v = Vec::new();
        self.iter_lo(|x| v.push(x));
        Container::Array(v)
    }

    fn to_bitset(&self) -> Container {
        let mut b = Box::new([0u64; BITSET_WORDS]);
        self.iter_lo(|x| {
            b[x as usize / 64] |= 1u64 << (x as usize % 64);
        });
        Container::Bitset(b)
    }

    fn to_run(&self) -> Container {
        let mut vals: Vec<u16> = Vec::new();
        self.iter_lo(|x| vals.push(x));
        let mut runs: Vec<(u16, u16)> = Vec::new();
        let mut i = 0;
        while i < vals.len() {
            let start = vals[i];
            let mut length = 0u16;
            while i + 1 < vals.len() && vals[i + 1] == vals[i] + 1 {
                length += 1;
                i += 1;
            }
            runs.push((start, length));
            i += 1;
        }
        Container::Run(runs)
    }

    // ── iteration ─────────────────────────────────────────────────────────────

    fn iter_lo(
        &self,
        mut f: impl FnMut(u16),
    ) {
        match self {
            Container::Array(v) => v.iter().for_each(|&x| f(x)),
            Container::Bitset(b) => {
                for (wi, &w) in b.iter().enumerate() {
                    let mut word = w;
                    while word != 0 {
                        let bit = word.trailing_zeros() as usize;
                        f((wi * 64 + bit) as u16);
                        word &= word - 1;
                    }
                }
            },
            Container::Run(runs) => {
                for &(s, l) in runs {
                    for x in s..=s.saturating_add(l) {
                        f(x);
                    }
                }
            },
        }
    }

    // ── set operations ────────────────────────────────────────────────────────

    fn union(
        &self,
        other: &Container,
    ) -> Container {
        let mut result = self.to_bitset();
        other.iter_lo(|x| {
            result.insert(x);
        });
        if result.len() < ARRAY_THRESHOLD {
            result.to_array()
        } else {
            result
        }
    }

    fn intersection(
        &self,
        other: &Container,
    ) -> Container {
        let mut v = Vec::new();
        if let Container::Array(a) = self {
            for &x in a {
                if other.contains(x) {
                    v.push(x);
                }
            }
        } else {
            self.iter_lo(|x| {
                if other.contains(x) {
                    v.push(x);
                }
            });
        }
        Container::Array(v)
    }

    fn difference(
        &self,
        other: &Container,
    ) -> Container {
        let mut v = Vec::new();
        self.iter_lo(|x| {
            if !other.contains(x) {
                v.push(x);
            }
        });
        Container::Array(v)
    }

    // ── serialization (Roaring standard format) ───────────────────────────────

    fn serialize(
        &self,
        out: &mut Vec<u8>,
    ) {
        // Container type: 0 = array, 1 = bitset, 2 = run
        match self {
            Container::Array(v) => {
                out.push(0);
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for &x in v {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            },
            Container::Bitset(b) => {
                out.push(1);
                for &w in b.iter() {
                    out.extend_from_slice(&w.to_le_bytes());
                }
            },
            Container::Run(runs) => {
                out.push(2);
                out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
                for &(s, l) in runs {
                    out.extend_from_slice(&s.to_le_bytes());
                    out.extend_from_slice(&l.to_le_bytes());
                }
            },
        }
    }

    fn deserialize(data: &[u8]) -> Option<(Container, usize)> {
        if data.is_empty() {
            return None;
        }
        let kind = data[0];
        let mut off = 1;

        match kind {
            0 => {
                // Array
                if off + 4 > data.len() {
                    return None;
                }
                let count = u32::from_le_bytes(data[off..off + 4].try_into().ok()?) as usize;
                off += 4;
                if off + count * 2 > data.len() {
                    return None;
                }
                let v: Vec<u16> = (0..count)
                    .map(|i| {
                        u16::from_le_bytes(
                            data[off + i * 2..off + i * 2 + 2]
                                .try_into()
                                .unwrap(),
                        )
                    })
                    .collect();
                off += count * 2;
                Some((Container::Array(v), off))
            },
            1 => {
                // Bitset
                if off + BITSET_WORDS * 8 > data.len() {
                    return None;
                }
                let mut b = Box::new([0u64; BITSET_WORDS]);
                for i in 0..BITSET_WORDS {
                    b[i] = u64::from_le_bytes(
                        data[off + i * 8..off + i * 8 + 8]
                            .try_into()
                            .ok()?,
                    );
                }
                off += BITSET_WORDS * 8;
                Some((Container::Bitset(b), off))
            },
            2 => {
                // Run
                if off + 4 > data.len() {
                    return None;
                }
                let count = u32::from_le_bytes(data[off..off + 4].try_into().ok()?) as usize;
                off += 4;
                if off + count * 4 > data.len() {
                    return None;
                }
                let runs: Vec<(u16, u16)> = (0..count)
                    .map(|i| {
                        let s = u16::from_le_bytes(
                            data[off + i * 4..off + i * 4 + 2]
                                .try_into()
                                .unwrap(),
                        );
                        let l = u16::from_le_bytes(
                            data[off + i * 4 + 2..off + i * 4 + 4]
                                .try_into()
                                .unwrap(),
                        );
                        (s, l)
                    })
                    .collect();
                off += count * 4;
                Some((Container::Run(runs), off))
            },
            _ => None,
        }
    }
}

// ── RoaringBitmap ─────────────────────────────────────────────────────────────

/// A compressed bitset for `u32` values, using the Roaring Bitmap format.
#[derive(Clone, Debug, Default)]
pub struct RoaringBitmap {
    /// Sorted list of (high-16-bits key, container) pairs.
    chunks: Vec<(u16, Container)>,
}

impl RoaringBitmap {
    pub fn new() -> Self {
        Self::default()
    }

    // ── chunk management ──────────────────────────────────────────────────────

    fn chunk_idx(
        &self,
        key: u16,
    ) -> Result<usize, usize> {
        self.chunks
            .binary_search_by_key(&key, |&(k, _)| k)
    }

    fn get_or_create_chunk(
        &mut self,
        key: u16,
    ) -> &mut Container {
        match self.chunk_idx(key) {
            Ok(i) => &mut self.chunks[i].1,
            Err(i) => {
                self.chunks
                    .insert(i, (key, Container::new_array()));
                &mut self.chunks[i].1
            },
        }
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Total number of values stored.
    pub fn len(&self) -> usize {
        self.chunks.iter().map(|(_, c)| c.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty() || self.len() == 0
    }

    /// Insert `value`.  Returns `true` if it was not already present.
    pub fn insert(
        &mut self,
        value: u32,
    ) -> bool {
        let key = (value >> 16) as u16;
        let lo = value as u16;
        self.get_or_create_chunk(key).insert(lo)
    }

    /// Return `true` if `value` is in the set.
    pub fn contains(
        &self,
        value: u32,
    ) -> bool {
        let key = (value >> 16) as u16;
        let lo = value as u16;
        match self.chunk_idx(key) {
            Ok(i) => self.chunks[i].1.contains(lo),
            Err(_) => false,
        }
    }

    /// Remove `value`.  Returns `true` if it was present.
    pub fn remove(
        &mut self,
        value: u32,
    ) -> bool {
        let key = (value >> 16) as u16;
        let lo = value as u16;
        match self.chunk_idx(key) {
            Ok(i) => {
                let removed = self.chunks[i].1.remove(lo);
                if self.chunks[i].1.is_empty() {
                    self.chunks.remove(i);
                }
                removed
            },
            Err(_) => false,
        }
    }

    /// Convert all containers to run-length encoded form.
    pub fn run_optimize(&mut self) {
        for (_, c) in &mut self.chunks {
            *c = c.to_run();
        }
    }

    /// Iterate over all values in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.chunks.iter().flat_map(|(key, c)| {
            let base = (*key as u32) << 16;
            let mut vals = Vec::new();
            c.iter_lo(|lo| vals.push(base | lo as u32));
            vals.into_iter()
        })
    }

    // ── set operations (3.15.4) ───────────────────────────────────────────────

    pub fn union(
        &self,
        other: &RoaringBitmap,
    ) -> RoaringBitmap {
        let mut result = self.clone();
        for &v in other.iter().collect::<Vec<_>>().iter() {
            result.insert(v);
        }
        result
    }

    pub fn intersection(
        &self,
        other: &RoaringBitmap,
    ) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for &v in self.iter().collect::<Vec<_>>().iter() {
            if other.contains(v) {
                result.insert(v);
            }
        }
        result
    }

    pub fn difference(
        &self,
        other: &RoaringBitmap,
    ) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        for &v in self.iter().collect::<Vec<_>>().iter() {
            if !other.contains(v) {
                result.insert(v);
            }
        }
        result
    }

    // ── serialization (3.15.5) ────────────────────────────────────────────────

    /// Serialize in a compact binary format.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // Header: number of chunks (u32 LE)
        out.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());
        for (key, container) in &self.chunks {
            out.extend_from_slice(&key.to_le_bytes());
            container.serialize(&mut out);
        }
        out
    }

    /// Deserialize from the binary format produced by `serialize`.
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }
        let n_chunks = u32::from_le_bytes(data[0..4].try_into().ok()?) as usize;
        let mut off = 4;
        let mut chunks = Vec::with_capacity(n_chunks);
        for _ in 0..n_chunks {
            if off + 2 > data.len() {
                return None;
            }
            let key = u16::from_le_bytes(data[off..off + 2].try_into().ok()?);
            off += 2;
            let (container, consumed) = Container::deserialize(&data[off..])?;
            off += consumed;
            chunks.push((key, container));
        }
        Some(Self { chunks })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_contains() {
        let mut rb = RoaringBitmap::new();
        assert!(rb.insert(42));
        assert!(!rb.insert(42)); // duplicate
        assert!(rb.contains(42));
        assert!(!rb.contains(43));
    }

    #[test]
    fn remove() {
        let mut rb = RoaringBitmap::new();
        rb.insert(10);
        rb.insert(20);
        assert!(rb.remove(10));
        assert!(!rb.remove(10)); // already gone
        assert_eq!(rb.len(), 1);
    }

    #[test]
    fn large_set_uses_bitset() {
        let mut rb = RoaringBitmap::new();
        for i in 0u32..5000 {
            rb.insert(i);
        }
        assert_eq!(rb.len(), 5000);
        for i in 0u32..5000 {
            assert!(rb.contains(i));
        }
        assert!(!rb.contains(5000));
    }

    #[test]
    fn cross_chunk_values() {
        let mut rb = RoaringBitmap::new();
        rb.insert(0); // chunk 0
        rb.insert(65535); // chunk 0
        rb.insert(65536); // chunk 1
        rb.insert(131071); // chunk 1
        rb.insert(u32::MAX); // chunk 65535
        assert_eq!(rb.len(), 5);
        for &v in &[0u32, 65535, 65536, 131071, u32::MAX] {
            assert!(rb.contains(v));
        }
    }

    #[test]
    fn iter_ascending() {
        let mut rb = RoaringBitmap::new();
        let vals = [5u32, 1, 3, 2, 4];
        for &v in &vals {
            rb.insert(v);
        }
        let collected: Vec<u32> = rb.iter().collect();
        assert_eq!(collected, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn union_op() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();
        a.insert(1);
        a.insert(2);
        b.insert(2);
        b.insert(3);
        let u = a.union(&b);
        let vals: Vec<u32> = u.iter().collect();
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    fn intersection_op() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();
        a.insert(1);
        a.insert(2);
        a.insert(3);
        b.insert(2);
        b.insert(3);
        b.insert(4);
        let i = a.intersection(&b);
        let vals: Vec<u32> = i.iter().collect();
        assert_eq!(vals, vec![2, 3]);
    }

    #[test]
    fn difference_op() {
        let mut a = RoaringBitmap::new();
        let mut b = RoaringBitmap::new();
        a.insert(1);
        a.insert(2);
        a.insert(3);
        b.insert(2);
        let d = a.difference(&b);
        let vals: Vec<u32> = d.iter().collect();
        assert_eq!(vals, vec![1, 3]);
    }

    #[test]
    fn run_optimize_and_contains() {
        let mut rb = RoaringBitmap::new();
        for i in 10u32..20 {
            rb.insert(i);
        }
        rb.run_optimize();
        for i in 10u32..20 {
            assert!(rb.contains(i));
        }
        assert!(!rb.contains(9));
        assert!(!rb.contains(20));
    }

    #[test]
    fn serialize_deserialize() {
        let mut rb = RoaringBitmap::new();
        for i in [0u32, 1, 100, 65536, 131000, u32::MAX] {
            rb.insert(i);
        }
        let bytes = rb.serialize();
        let rb2 = RoaringBitmap::deserialize(&bytes).unwrap();
        assert_eq!(rb.len(), rb2.len());
        let v1: Vec<u32> = rb.iter().collect();
        let v2: Vec<u32> = rb2.iter().collect();
        assert_eq!(v1, v2);
    }
}
