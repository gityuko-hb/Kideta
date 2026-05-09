//! PQ Codebook — binary serialization format for PqConfig.
//!
//! The codebook is serialized as a compact binary format for persistence.
//! It can be stored in segment metadata and reloaded at startup without retraining.
//!
//! # Binary Format
//!
//! ```ignore
//! [magic: 4 bytes] = b'PQCB'
//! [version: u8]    = 1
//! [dimension: u32]
//! [num_subspaces: u32]
//! [subspace_dim: u32]
//! [bytes_per_subvec: u32]
//! [num_centroids: u32]
//! [codebook: M * K * subspace_dim * 4 bytes f32]
//! [checksum: u32]  CRC32C of everything above
//! ```

use crate::quantization::config::PqConfig;

const MAGIC: [u8; 4] = *b"PQCB";
const VERSION: u8 = 1;

#[derive(Debug, thiserror::Error)]
pub enum CodebookError {
    #[error("invalid magic: expected {expected:?}, got {got:?}")]
    InvalidMagic { expected: [u8; 4], got: [u8; 4] },
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u8),
    #[error("invalid dimension: {0}")]
    InvalidDimension(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl PqConfig {
    /// Serialize the codebook to binary format.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + 1 + 4 * 4 + self.codebook.len() * 4 + 4);

        buf.extend_from_slice(&MAGIC);
        buf.push(VERSION);
        buf.extend_from_slice(&(self.dimension as u32).to_le_bytes());
        buf.extend_from_slice(&(self.num_subspaces as u32).to_le_bytes());
        buf.extend_from_slice(&(self.subspace_dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.bytes_per_subvec as u32).to_le_bytes());
        buf.extend_from_slice(&(self.num_centroids() as u32).to_le_bytes());

        for val in &self.codebook {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        let checksum = Self::crc32c(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());

        buf
    }

    /// Deserialize the codebook from binary format.
    pub fn deserialize(data: &[u8]) -> Result<Self, CodebookError> {
        if data.len() < 4 + 1 + 4 * 5 + 4 {
            return Err(CodebookError::InvalidDimension(format!(
                "data too short: {} bytes",
                data.len()
            )));
        }

        let magic: [u8; 4] = data[0..4].try_into().unwrap();
        if magic != MAGIC {
            return Err(CodebookError::InvalidMagic {
                expected: MAGIC,
                got: magic,
            });
        }

        let version = data[4];
        if version != VERSION {
            return Err(CodebookError::UnsupportedVersion(version));
        }

        let mut offset = 5;
        let dimension = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let num_subspaces =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let subspace_dim =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let bytes_per_subvec =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let _num_centroids =
            u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;

        let expected_codebook_len = num_subspaces * 256 * subspace_dim;
        let expected_total = 5 + 4 * 5 + expected_codebook_len * 4 + 4;

        if data.len() < expected_total {
            return Err(CodebookError::InvalidDimension(format!(
                "data too short: expected {} bytes, got {}",
                expected_total,
                data.len()
            )));
        }

        let stored_checksum = u32::from_le_bytes(
            data[data.len() - 4..data.len()]
                .try_into()
                .unwrap(),
        );
        let computed_checksum = Self::crc32c(&data[..data.len() - 4]);

        if stored_checksum != computed_checksum {
            return Err(CodebookError::InvalidDimension(format!(
                "checksum mismatch: expected {:08x}, got {:08x}",
                computed_checksum, stored_checksum
            )));
        }

        let mut codebook = Vec::with_capacity(expected_codebook_len);
        for i in 0..expected_codebook_len {
            let val = f32::from_le_bytes(
                data[offset + i * 4..offset + (i + 1) * 4]
                    .try_into()
                    .unwrap(),
            );
            codebook.push(val);
        }

        Ok(Self {
            dimension,
            num_subspaces,
            subspace_dim,
            bytes_per_subvec,
            codebook,
        })
    }

    fn crc32c(data: &[u8]) -> u32 {
        kideta_core::utils::crc32c::crc32c(0, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantization::pq::PqTrainer;

    fn make_test_config() -> PqConfig {
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..64)
                    .map(|j| ((i * 64 + j) as f32) * 0.01)
                    .collect()
            })
            .collect();
        let refs: Vec<_> = vectors.iter().map(|v| v.as_slice()).collect();
        let trainer = PqTrainer::new(8, 256, 5);
        trainer.train(&refs)
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let original = make_test_config();
        let serialized = original.serialize();
        let deserialized = PqConfig::deserialize(&serialized).unwrap();

        assert_eq!(original.dimension, deserialized.dimension);
        assert_eq!(original.num_subspaces, deserialized.num_subspaces);
        assert_eq!(original.subspace_dim, deserialized.subspace_dim);
        assert_eq!(original.bytes_per_subvec, deserialized.bytes_per_subvec);
        assert_eq!(original.codebook, deserialized.codebook);
    }

    #[test]
    fn test_deserialize_invalid_magic() {
        let num_subspaces: u32 = 2;
        let subspace_dim: u32 = 8;
        let bytes_per_subvec: u32 = 8;
        let num_centroids: u32 = 256;
        let codebook_len = num_subspaces as usize * 256 * subspace_dim as usize;
        let header_size = 5 + 4 * 5;
        let total_size = header_size + codebook_len * 4 + 4;

        let mut data = vec![0u8; total_size];
        data[4] = 1;
        data[5..9].copy_from_slice(&64u32.to_le_bytes());
        data[9..13].copy_from_slice(&num_subspaces.to_le_bytes());
        data[13..17].copy_from_slice(&subspace_dim.to_le_bytes());
        data[17..21].copy_from_slice(&bytes_per_subvec.to_le_bytes());
        data[21..25].copy_from_slice(&num_centroids.to_le_bytes());

        let result = PqConfig::deserialize(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_wrong_version() {
        let num_subspaces: u32 = 2;
        let subspace_dim: u32 = 8;
        let bytes_per_subvec: u32 = 8;
        let num_centroids: u32 = 256;
        let codebook_len = num_subspaces as usize * 256 * subspace_dim as usize;
        let header_size = 5 + 4 * 5;
        let total_size = header_size + codebook_len * 4 + 4;

        let mut data = vec![0u8; total_size];
        data[0..4].copy_from_slice(b"PQCB");
        data[4] = 99;
        data[5..9].copy_from_slice(&64u32.to_le_bytes());
        data[9..13].copy_from_slice(&num_subspaces.to_le_bytes());
        data[13..17].copy_from_slice(&subspace_dim.to_le_bytes());
        data[17..21].copy_from_slice(&bytes_per_subvec.to_le_bytes());
        data[21..25].copy_from_slice(&num_centroids.to_le_bytes());

        let result = PqConfig::deserialize(&data);
        assert!(matches!(result, Err(CodebookError::UnsupportedVersion(99))));
    }

    #[test]
    fn test_deserialize_truncated() {
        let data = vec![0u8; 10];
        let result = PqConfig::deserialize(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialized_contains_correct_header() {
        let config = make_test_config();
        let serialized = config.serialize();

        assert_eq!(&serialized[0..4], b"PQCB");
        assert_eq!(serialized[4], 1);
        let dim = u32::from_le_bytes(serialized[5..9].try_into().unwrap()) as usize;
        assert_eq!(dim, 64);
    }

    #[test]
    fn test_codebook_crc32_check() {
        let config = make_test_config();
        let mut serialized = config.serialize();

        let len = serialized.len();
        serialized[len - 5] ^= 0xFF;

        let result = PqConfig::deserialize(&serialized);
        assert!(result.is_err());
    }
}
