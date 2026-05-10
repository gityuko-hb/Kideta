use kideta_core::enums::{IndexType, QuantizationType};
use kideta_index::quantization::QuantizationConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::SystemTime;

use kideta_core::utils::bloom::BloomFilter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub id: u64,
    pub path: PathBuf,
    pub state: String,
    pub vector_count: u64,
    pub deleted_count: u64,
    pub index_type: IndexType,
    pub index_ready: bool,
    pub created_at: SystemTime,
    pub file_size_bytes: u64,
    #[serde(with = "serde_base64")]
    pub bloom_filter: Vec<u8>,
    pub bloom_hash_count: usize,
    pub bloom_bit_count: usize,
    #[serde(default)]
    pub quantization_type: QuantizationType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization_config: Option<QuantizationConfig>,
}

mod serde_base64 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(
        bytes: &[u8],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let encoded = kideta_core::utils::base64::encode(bytes);
        serializer.serialize_str(&encoded)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        kideta_core::utils::base64::decode(&s)
            .ok_or_else(|| serde::de::Error::custom("invalid base64"))
    }
}

impl SegmentMeta {
    pub fn new(
        id: u64,
        path: PathBuf,
        vector_count: u64,
        index_type: IndexType,
    ) -> Self {
        let bloom = BloomFilter::build_from_ids(&[], 0.01);
        let bloom_bytes = serialize_bloom(&bloom);

        Self {
            id,
            path,
            state: "Open".to_string(),
            vector_count,
            deleted_count: 0,
            index_type,
            index_ready: false,
            created_at: SystemTime::now(),
            file_size_bytes: 0,
            bloom_filter: bloom_bytes,
            bloom_hash_count: bloom.num_hashes() as usize,
            bloom_bit_count: bloom.num_bits(),
            quantization_type: QuantizationType::default(),
            quantization_config: None,
        }
    }

    pub fn with_bloom(
        mut self,
        ids: &[u32],
    ) -> Self {
        let bloom = BloomFilter::build_from_ids(ids, 0.01);
        self.bloom_filter = serialize_bloom(&bloom);
        self.bloom_hash_count = bloom.num_hashes() as usize;
        self.bloom_bit_count = bloom.num_bits();
        self
    }

    pub fn may_exist(
        &self,
        id: u32,
    ) -> bool {
        let bloom = deserialize_bloom(
            &self.bloom_filter,
            self.bloom_hash_count,
            self.bloom_bit_count,
        );
        bloom.contains_u32(id)
    }

    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }

    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn save(
        &self,
        dir: &std::path::Path,
    ) -> std::io::Result<()> {
        let json = self
            .to_json()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(dir.join("meta.json"), json)
    }

    pub fn load(dir: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(dir.join("meta.json"))?;
        Self::from_json(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

fn serialize_bloom(bloom: &BloomFilter) -> Vec<u8> {
    let mut bytes = Vec::new();
    for (i, chunk) in bloom.words().chunks(8).enumerate() {
        if chunk.len() == 8 {
            bytes.extend_from_slice(&u64::to_le_bytes(chunk[0]));
        } else {
            let mut val = 0u64;
            for (j, &b) in chunk.iter().enumerate() {
                val |= b << (j * 8);
            }
            bytes.extend_from_slice(&u64::to_le_bytes(val));
        }
        if i * 8 >= bloom.num_bits() {
            break;
        }
    }
    bytes
}

fn deserialize_bloom(
    bytes: &[u8],
    num_hashes: usize,
    num_bits: usize,
) -> BloomFilter {
    let mut bit_array = Vec::new();
    for chunk in bytes.chunks(8) {
        if chunk.len() == 8 {
            bit_array.push(u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
    }
    BloomFilter::from_raw(bit_array, num_bits, num_hashes as u32)
}
