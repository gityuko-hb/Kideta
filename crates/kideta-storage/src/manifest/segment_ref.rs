use kideta_core::enums::QuantizationType;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRef {
    pub id: u64,
    pub path: PathBuf,
    pub vector_count: u64,
    pub deleted_count: u64,
    pub index_ready: bool,
    pub created_at: SystemTime,
    pub file_size_bytes: u64,
    #[serde(default)]
    pub quantization_type: QuantizationType,
}

impl SegmentRef {
    pub fn new(
        id: u64,
        path: PathBuf,
        vector_count: u64,
        deleted_count: u64,
        index_ready: bool,
        file_size_bytes: u64,
    ) -> Self {
        Self {
            id,
            path,
            vector_count,
            deleted_count,
            index_ready,
            created_at: SystemTime::now(),
            file_size_bytes,
            quantization_type: QuantizationType::default(),
        }
    }
}
