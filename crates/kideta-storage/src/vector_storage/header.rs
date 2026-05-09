//! Binary header for `MmapVectorStorage` files (64 bytes, packed).
//!
//! Layout:
//! ```text
//! Byte 0-3:   Magic "KVST" (4 bytes)
//! Byte 4-5:   Version (u16, LE)
//! Byte 6-9:   Dimension (u32, LE)
//! Byte 10-17: Vector count (u64, LE)
//! Byte 18:    dtype (u8, see `VectorDtype`)
//! Byte 19-55: Reserved (37 bytes)
//! Byte 56-63: xxhash3 checksum of bytes 0-55 (u64, LE)
//! ```

use crate::vector_storage::dtype::VectorDtype;
use kideta_core::utils::hash::xxhash3_128;

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct VectorStorageHeader {
    magic: [u8; 4],
    version: u16,
    dim: u32,
    count: u64,
    dtype: u8,
    reserved: [u8; 37],
    checksum: u64,
}

const EXPECTED_MAGIC: &[u8; 4] = b"KVST";
const CURRENT_VERSION: u16 = 1;
const HEADER_SIZE: usize = 64;

impl VectorStorageHeader {
    pub const SIZE: usize = HEADER_SIZE;

    pub fn new(
        dim: u32,
        dtype: VectorDtype,
    ) -> Self {
        let mut header = Self {
            magic: *EXPECTED_MAGIC,
            version: CURRENT_VERSION,
            dim,
            count: 0,
            dtype: dtype as u8,
            reserved: [0u8; 37],
            checksum: 0,
        };
        header.checksum = header.compute_checksum();
        header
    }

    pub fn dimension(&self) -> u32 {
        self.dim
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn dtype(&self) -> VectorDtype {
        VectorDtype::from_u8(self.dtype)
    }

    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn magic(&self) -> &[u8; 4] {
        &self.magic
    }

    pub fn is_valid(&self) -> bool {
        self.magic == *EXPECTED_MAGIC && self.checksum == self.compute_checksum()
    }

    pub fn set_count(
        &mut self,
        count: u64,
    ) {
        self.count = count;
        self.checksum = self.compute_checksum();
    }

    pub fn increment_count(
        &mut self,
        n: u64,
    ) {
        self.count += n;
        self.checksum = self.compute_checksum();
    }

    pub fn byte_size(
        &self,
        num_vectors: u64,
    ) -> usize {
        self.dtype().byte_size(self.dim) * num_vectors as usize
    }

    fn compute_checksum(&self) -> u64 {
        let bytes = self.as_bytes();
        let (lo, _hi) = xxhash3_128(&bytes[..56], 0);
        lo
    }

    pub fn as_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&self.magic);
        bytes[4..6].copy_from_slice(&self.version.to_le_bytes());
        bytes[6..10].copy_from_slice(&self.dim.to_le_bytes());
        bytes[10..18].copy_from_slice(&self.count.to_le_bytes());
        bytes[18] = self.dtype;
        bytes[19..56].copy_from_slice(&self.reserved);
        bytes[56..64].copy_from_slice(&self.checksum.to_le_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Option<Self> {
        let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if &magic != EXPECTED_MAGIC {
            return None;
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        let dim = u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]);
        let count = u64::from_le_bytes([
            bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17],
        ]);
        let dtype = bytes[18];
        let reserved = {
            let mut arr = [0u8; 37];
            arr.copy_from_slice(&bytes[19..56]);
            arr
        };
        let checksum = u64::from_le_bytes([
            bytes[56], bytes[57], bytes[58], bytes[59], bytes[60], bytes[61], bytes[62], bytes[63],
        ]);
        let header = Self {
            magic,
            version,
            dim,
            count,
            dtype,
            reserved,
            checksum,
        };
        if header.checksum != header.compute_checksum() {
            return None;
        }
        Some(header)
    }
}

impl std::fmt::Display for VectorStorageHeader {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let dim = self.dim;
        let count = self.count;
        let version = self.version;
        write!(
            f,
            "VectorStorageHeader {{ dim: {}, count: {}, dtype: {:?}, version: {} }}",
            dim,
            count,
            self.dtype(),
            version
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_is_64_bytes() {
        assert_eq!(std::mem::size_of::<VectorStorageHeader>(), 64);
    }

    #[test]
    fn header_packed() {
        assert_eq!(std::mem::align_of::<VectorStorageHeader>(), 1);
    }

    #[test]
    fn new_header_is_valid() {
        let header = VectorStorageHeader::new(128, VectorDtype::F32);
        assert!(header.is_valid());
        assert_eq!(header.dimension(), 128);
        assert_eq!(header.count(), 0);
        assert_eq!(header.dtype(), VectorDtype::F32);
        assert_eq!(header.version(), 1);
    }

    #[test]
    fn header_roundtrip() {
        let header = VectorStorageHeader::new(256, VectorDtype::F32);
        let bytes = header.as_bytes();
        let restored = VectorStorageHeader::from_bytes(&bytes).unwrap();
        assert!(restored.is_valid());
        assert_eq!(restored.dimension(), header.dimension());
        assert_eq!(restored.count(), header.count());
        assert_eq!(restored.dtype(), header.dtype());
    }

    #[test]
    fn increment_count_updates_checksum() {
        let mut header = VectorStorageHeader::new(128, VectorDtype::F32);
        assert_eq!(header.count(), 0);
        header.increment_count(10);
        assert_eq!(header.count(), 10);
        assert!(header.is_valid());
    }

    #[test]
    fn invalid_magic_returns_none() {
        let mut bytes = [0u8; 64];
        bytes[0] = b'X';
        bytes[1] = b'X';
        bytes[2] = b'X';
        bytes[3] = b'X';
        let result = VectorStorageHeader::from_bytes(&bytes);
        assert!(result.is_none());
    }

    #[test]
    fn corrupted_checksum_returns_none() {
        let header = VectorStorageHeader::new(128, VectorDtype::F32);
        let mut bytes = header.as_bytes();
        bytes[56] ^= 0xFF;
        let result = VectorStorageHeader::from_bytes(&bytes);
        assert!(result.is_none());
    }

    #[test]
    fn display_format() {
        let header = VectorStorageHeader::new(128, VectorDtype::F32);
        let s = format!("{}", header);
        assert!(s.contains("128"));
        assert!(s.contains("F32"));
    }
}
