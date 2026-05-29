use crate::segment::Segment;
use crate::vector_storage::dtype::VectorDtype;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::store::KidetaStore;

fn sync_dir(dir: &Path) -> std::io::Result<()> {
    let file = std::fs::File::open(dir)?;
    file.sync_all()?;
    Ok(())
}

pub struct CompactionResult {
    pub new_segment_id: u64,
    pub new_segment_path: PathBuf,
    pub old_segment_ids: Vec<u64>,
    pub vectors_merged: u64,
    pub bytes_freed: u64,
    pub new_size_bytes: u64,
}

pub fn compact_pair(
    seg_a: &Arc<Segment>,
    seg_b: &Arc<Segment>,
    output_dir: &Path,
    next_segment_id: u64,
) -> crate::store::Result<CompactionResult> {
    let start_size = seg_a.meta().file_size_bytes + seg_b.meta().file_size_bytes;

    let final_dir = output_dir.join(format!("segment_{:016x}", next_segment_id));
    let tmp_dir = output_dir.join(format!("segment_{:016x}.tmp", next_segment_id));

    // Clean up stale .tmp from previous crash
    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir).map_err(crate::store::StoreError::Io)?;
    }
    fs::create_dir_all(&tmp_dir).map_err(crate::store::StoreError::Io)?;

    let dim = seg_a.dimension() as u32;

    // Open raw output storage (always F32, source of truth after compaction)
    let raw_path = tmp_dir.join("vectors.bin");
    let mut raw_output = crate::vector_storage::MmapVectorStorage::open(&raw_path, dim, VectorDtype::F32)
        .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;

    // Open quantized output storage (F32 during build, may be re-quantized later)
    let quantized_path = tmp_dir.join("vectors.quantized");
    let mut quantized_output = crate::vector_storage::MmapVectorStorage::open(&quantized_path, dim, VectorDtype::F32)
        .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;

    let payload_dir = tmp_dir.join("payload.store");
    let mut merged_payload = KidetaStore::open(&payload_dir, 10000)?;

    let old_segment_ids = vec![seg_a.id, seg_b.id];

    let mut merged_count = 0u64;
    let _total_a = seg_a.vector_count();
    let _total_b = seg_b.vector_count();

    for seg in [seg_a.clone(), seg_b.clone()].iter() {
        let total = seg.vector_count();
        for local_id in 0..total {
            let id = local_id as u32;
            if seg.tombstones().contains(id) {
                continue;
            }

            // Read F32 vector — prefer raw storage when available
            let f32_vec = if seg.has_raw_storage() && seg.raw_count() > 0 {
                match seg.read_vector_f32(id) {
                    Ok(v) => v,
                    Err(_) => {
                        let bytes = seg
                            .read_vector(id)
                            .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;
                        seg.dtype().cast_to_f32(bytes.as_ref(), dim)
                    }
                }
            } else {
                let bytes = seg
                    .read_vector(id)
                    .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;
                seg.dtype().cast_to_f32(bytes.as_ref(), dim)
            };

            if let Some(payload) = seg.read_payload(local_id as u32) {
                let _ = merged_payload.put(merged_count as u32, &payload);
            }

            raw_output
                .append_vector(&f32_vec)
                .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;
            quantized_output
                .append_vector(&f32_vec)
                .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;
            merged_count += 1;
        }
    }

    raw_output
        .flush()
        .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;
    quantized_output
        .flush()
        .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;

    merged_payload.flush()?;

    // Sync the tmp directory before atomic rename
    sync_dir(&tmp_dir).map_err(crate::store::StoreError::Io)?;

    // Atomic rename: .tmp → final
    fs::rename(&tmp_dir, &final_dir).map_err(crate::store::StoreError::Io)?;

    // Sync parent directory to persist rename
    sync_dir(output_dir).map_err(crate::store::StoreError::Io)?;

    let new_size = raw_output.len() as u64 * 4 * dim as u64;

    let payload_size = merged_payload.len() as u64 * 256;

    Ok(CompactionResult {
        new_segment_id: next_segment_id,
        new_segment_path: final_dir,
        old_segment_ids,
        vectors_merged: merged_count,
        bytes_freed: start_size.saturating_sub(new_size + payload_size),
        new_size_bytes: new_size + payload_size + 4096,
    })
}

pub fn compact_batch(
    pairs: Vec<(Arc<Segment>, Arc<Segment>)>,
    output_dir: &Path,
    next_segment_id: u64,
    parallelism: usize,
) -> Vec<crate::store::Result<CompactionResult>> {
    let mut results = Vec::with_capacity(pairs.len());
    let mut current_id = next_segment_id;

    for chunk in pairs.chunks(parallelism.max(1)) {
        let chunk_results: Vec<_> = chunk
            .iter()
            .map(|(a, b)| {
                let id = current_id;
                current_id += 1;
                compact_pair(a, b, output_dir, id)
            })
            .collect();

        results.extend(chunk_results);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_result_basic() {
        let result = CompactionResult {
            new_segment_id: 1,
            new_segment_path: PathBuf::from("/tmp/seg1"),
            old_segment_ids: vec![],
            vectors_merged: 100,
            bytes_freed: 500,
            new_size_bytes: 1000,
        };
        assert_eq!(result.vectors_merged, 100);
    }

    #[test]
    fn test_compact_pair_with_payloads() {
        let dir = tempfile::tempdir().unwrap();
        let dim = 4;

        let seg_a_dir = dir.path().join("seg_a");
        let seg_a =
            Segment::open_with_dual_storage(1, &seg_a_dir, dim, kideta_core::enums::IndexType::Flat, 256).unwrap();
        seg_a
            .append_vector(&[1.0, 2.0, 3.0, 4.0], Some(b"pay_a1".to_vec()))
            .unwrap();
        seg_a
            .append_vector(&[5.0, 6.0, 7.0, 8.0], Some(b"pay_a2".to_vec()))
            .unwrap();
        seg_a.delete(1).unwrap();
        seg_a.seal().unwrap();

        let seg_b_dir = dir.path().join("seg_b");
        let seg_b =
            Segment::open_with_dual_storage(2, &seg_b_dir, dim, kideta_core::enums::IndexType::Flat, 256).unwrap();
        seg_b
            .append_vector(&[9.0, 10.0, 11.0, 12.0], Some(b"pay_b1".to_vec()))
            .unwrap();
        seg_b
            .append_vector(&[13.0, 14.0, 15.0, 16.0], Some(b"pay_b2".to_vec()))
            .unwrap();
        seg_b.seal().unwrap();

        let result = compact_pair(&Arc::new(seg_a), &Arc::new(seg_b), &dir.path().join("merged"), 3).unwrap();

        assert_eq!(result.vectors_merged, 3);
        assert_eq!(result.old_segment_ids.len(), 2);

        let merged = Segment::open_with_dual_storage(
            3,
            &result.new_segment_path,
            dim,
            kideta_core::enums::IndexType::Flat,
            256,
        )
        .unwrap();
        assert_eq!(merged.vector_count(), 3);

        let pay = merged.read_payload(0).unwrap();
        assert_eq!(String::from_utf8(pay).unwrap(), "pay_a1");

        let pay = merged.read_payload(1).unwrap();
        assert_eq!(String::from_utf8(pay).unwrap(), "pay_b1");

        let pay = merged.read_payload(2).unwrap();
        assert_eq!(String::from_utf8(pay).unwrap(), "pay_b2");
    }
}
