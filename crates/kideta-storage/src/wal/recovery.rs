use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::error::{Result, WalError};
use super::format::{Record, RecordType, wal_file_name};
use super::reader::WalReader;

#[derive(Debug, Clone, Default)]
pub struct WalState {
    pub last_lsn: u64,
    pub last_checkpoint_lsn: u64,
    pub last_checkpoint_file_id: u64,
    pub collections: HashMap<String, CollectionState>,
    pub vectors: HashMap<u64, VectorState>,
}

#[derive(Debug, Clone)]
pub struct CollectionState {
    pub name: String,
    pub created_at_lsn: u64,
    pub dropped: bool,
    pub dropped_at_lsn: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct VectorState {
    pub id: u64,
    pub data: Vec<u8>,
    pub updated_at_lsn: u64,
    pub deleted: bool,
}

pub trait ReplayHandler: Send + Sync {
    fn apply_insert(
        &self,
        lsn: u64,
        id: u64,
        vector: &[u8],
    ) -> Result<()>;
    fn apply_delete(
        &self,
        lsn: u64,
        id: u64,
    ) -> Result<()>;
    fn apply_upsert(
        &self,
        lsn: u64,
        id: u32,
        vector: &[u8],
    ) -> Result<()>;
    fn apply_create_collection(
        &self,
        lsn: u64,
        name: &str,
    ) -> Result<()>;
    fn apply_drop_collection(
        &self,
        lsn: u64,
        name: &str,
    ) -> Result<()>;
    fn apply_segment_sealed(
        &self,
        lsn: u64,
        segment_id: u64,
    ) -> Result<()>;
    fn apply_compaction_done(
        &self,
        lsn: u64,
        old_segments: &[u64],
        new_segment: u64,
    ) -> Result<()>;
    fn apply_checkpoint(
        &self,
        lsn: u64,
    ) -> Result<()>;
}

pub struct WalRecovery {
    dir: PathBuf,
    state: WalState,
    truncation_enabled: bool,
}

impl WalRecovery {
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
            state: WalState::default(),
            truncation_enabled: true,
        }
    }

    pub fn recover<H: ReplayHandler>(
        &mut self,
        handler: &H,
    ) -> Result<WalState> {
        let wal_files = self.list_wal_files()?;
        if wal_files.is_empty() {
            tracing::info!("no WAL files found, starting fresh");
            return Ok(WalState::default());
        }

        tracing::info!("found {} WAL file(s) to recover", wal_files.len());

        let mut prev_lsn: u64 = 0;
        for file_id in &wal_files {
            let path = self.dir.join(wal_file_name(*file_id));
            let mut reader = WalReader::open(&path)?;

            loop {
                match reader.read_record() {
                    Ok(Some(record)) => {
                        if record.lsn <= prev_lsn && prev_lsn != 0 {
                            tracing::warn!(
                                "LSN {} at or below previous LSN {} — skipping (possible duplicate)",
                                record.lsn,
                                prev_lsn
                            );
                            continue;
                        }

                        if record.lsn < prev_lsn {
                            return Err(WalError::LsnNotMonotonic {
                                prev: prev_lsn,
                                curr: record.lsn,
                            });
                        }
                        prev_lsn = record.lsn;

                        self.replay_record(handler, &record)?;

                        if record.record_type == RecordType::Checkpoint {
                            self.state.last_checkpoint_lsn = record.lsn;
                            self.state.last_checkpoint_file_id = *file_id;
                            tracing::info!("checkpoint reached at LSN {}", record.lsn);
                            break;
                        }
                    },
                    Ok(None) => break,
                    Err(WalError::UnexpectedEof { .. }) => {
                        tracing::warn!("WAL file {} ended unexpectedly, moving to next", file_id);
                        break;
                    },
                    Err(WalError::BadCrc { lsn, .. }) => {
                        tracing::warn!("torn write / corrupt record at LSN {lsn}, truncating WAL");
                        break;
                    },
                    Err(e) => {
                        tracing::warn!(
                            "WAL record error at file {file_id}: {e}, skipping remaining"
                        );
                        break;
                    },
                }
            }

            self.state.last_lsn = prev_lsn;
        }

        tracing::info!(
            "recovery complete: last_lsn={}, checkpoint_lsn={}, collections={}, vectors={}",
            self.state.last_lsn,
            self.state.last_checkpoint_lsn,
            self.state.collections.len(),
            self.state.vectors.len()
        );

        Ok(self.state.clone())
    }

    fn replay_record<H: ReplayHandler>(
        &mut self,
        handler: &H,
        record: &Record,
    ) -> Result<()> {
        match record.record_type {
            RecordType::Insert => {
                if record.payload.len() < 8 {
                    return Err(WalError::CorruptRecord {
                        lsn: record.lsn,
                        reason: "Insert payload too short",
                    });
                }
                let id = u64::from_le_bytes(record.payload[..8].try_into().unwrap());
                let vector = &record.payload[8..];
                handler.apply_insert(record.lsn, id, vector)?;
                self.state.vectors.insert(
                    id,
                    VectorState {
                        id,
                        data: vector.to_vec(),
                        updated_at_lsn: record.lsn,
                        deleted: false,
                    },
                );
            },
            RecordType::Delete => {
                if record.payload.len() < 8 {
                    return Err(WalError::CorruptRecord {
                        lsn: record.lsn,
                        reason: "Delete payload too short",
                    });
                }
                let id = u64::from_le_bytes(record.payload[..8].try_into().unwrap());
                handler.apply_delete(record.lsn, id)?;
                if let Some(v) = self.state.vectors.get_mut(&id) {
                    v.deleted = true;
                    v.updated_at_lsn = record.lsn;
                }
            },
            RecordType::Upsert => {
                if record.payload.len() < 4 {
                    return Err(WalError::CorruptRecord {
                        lsn: record.lsn,
                        reason: "Upsert payload too short",
                    });
                }
                let id = u32::from_le_bytes(record.payload[..4].try_into().unwrap());
                let vector = &record.payload[4..];
                handler.apply_upsert(record.lsn, id, vector)?;
                self.state.vectors.insert(
                    id as u64,
                    VectorState {
                        id: id as u64,
                        data: vector.to_vec(),
                        updated_at_lsn: record.lsn,
                        deleted: false,
                    },
                );
            },
            RecordType::CreateCollection => {
                let name = String::from_utf8_lossy(&record.payload).to_string();
                handler.apply_create_collection(record.lsn, &name)?;
                self.state.collections.insert(
                    name.clone(),
                    CollectionState {
                        name,
                        created_at_lsn: record.lsn,
                        dropped: false,
                        dropped_at_lsn: None,
                    },
                );
            },
            RecordType::DropCollection => {
                let name = String::from_utf8_lossy(&record.payload).to_string();
                handler.apply_drop_collection(record.lsn, &name)?;
                if let Some(c) = self.state.collections.get_mut(&name) {
                    c.dropped = true;
                    c.dropped_at_lsn = Some(record.lsn);
                }
            },
            RecordType::SegmentSealed => {
                if record.payload.len() < 8 {
                    return Err(WalError::CorruptRecord {
                        lsn: record.lsn,
                        reason: "SegmentSealed payload too short",
                    });
                }
                let segment_id = u64::from_le_bytes(record.payload[..8].try_into().unwrap());
                handler.apply_segment_sealed(record.lsn, segment_id)?;
            },
            RecordType::CompactionDone => {
                let (old_segments, new_segment) = Self::decode_compaction_payload(&record.payload)?;
                handler.apply_compaction_done(record.lsn, &old_segments, new_segment)?;
            },
            RecordType::Checkpoint => {
                handler.apply_checkpoint(record.lsn)?;
            },
        }
        Ok(())
    }

    fn decode_compaction_payload(payload: &[u8]) -> Result<(Vec<u64>, u64)> {
        if payload.len() < 4 {
            return Err(WalError::CorruptRecord {
                lsn: 0,
                reason: "CompactionDone payload too short",
            });
        }
        let count = u32::from_le_bytes(payload[..4].try_into().unwrap()) as usize;
        let expected_len = 4 + count * 8 + 8;
        if payload.len() < expected_len {
            return Err(WalError::CorruptRecord {
                lsn: 0,
                reason: "CompactionDone payload length mismatch",
            });
        }
        let mut old_segments = Vec::with_capacity(count);
        let mut off = 4;
        for _ in 0..count {
            let s = u64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
            old_segments.push(s);
            off += 8;
        }
        let new_segment = u64::from_le_bytes(payload[off..off + 8].try_into().unwrap());
        Ok((old_segments, new_segment))
    }

    pub fn truncate_pre_checkpoint(&mut self) -> Result<u64> {
        if !self.truncation_enabled {
            return Ok(0);
        }

        let checkpoint_file_id = self.state.last_checkpoint_file_id;
        if checkpoint_file_id == 0 {
            return Ok(0);
        }

        let wal_files = self.list_wal_files()?;
        let mut deleted = 0u64;

        for file_id in &wal_files {
            if *file_id < checkpoint_file_id {
                let path = self.dir.join(wal_file_name(*file_id));
                if path.exists() {
                    fs::remove_file(&path).map_err(|_| WalError::io("remove_wal_file"))?;
                    tracing::info!("truncated WAL file {}", path.display());
                    deleted += 1;
                }
            }
        }

        Ok(deleted)
    }

    pub fn get_state(&self) -> &WalState {
        &self.state
    }

    pub fn set_truncation(
        &mut self,
        enabled: bool,
    ) {
        self.truncation_enabled = enabled;
    }

    fn list_wal_files(&self) -> Result<Vec<u64>> {
        let mut ids = Vec::new();
        let entries = fs::read_dir(&self.dir).map_err(|_| WalError::io("read_dir"))?;
        for entry in entries {
            let entry = entry.map_err(|_| WalError::io("read_dir_entry"))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("wal_")
                && name.ends_with(".kid")
                && let Ok(id) = name[4..name.len() - 4].parse::<u64>()
            {
                ids.push(id);
            }
        }
        ids.sort();
        Ok(ids)
    }
}

pub struct NoOpHandler;

impl ReplayHandler for NoOpHandler {
    fn apply_insert(
        &self,
        _lsn: u64,
        _id: u64,
        _vector: &[u8],
    ) -> Result<()> {
        Ok(())
    }
    fn apply_delete(
        &self,
        _lsn: u64,
        _id: u64,
    ) -> Result<()> {
        Ok(())
    }
    fn apply_upsert(
        &self,
        _lsn: u64,
        _id: u32,
        _vector: &[u8],
    ) -> Result<()> {
        Ok(())
    }
    fn apply_create_collection(
        &self,
        _lsn: u64,
        _name: &str,
    ) -> Result<()> {
        Ok(())
    }
    fn apply_drop_collection(
        &self,
        _lsn: u64,
        _name: &str,
    ) -> Result<()> {
        Ok(())
    }
    fn apply_segment_sealed(
        &self,
        _lsn: u64,
        _segment_id: u64,
    ) -> Result<()> {
        Ok(())
    }
    fn apply_compaction_done(
        &self,
        _lsn: u64,
        _old_segments: &[u64],
        _new_segment: u64,
    ) -> Result<()> {
        Ok(())
    }
    fn apply_checkpoint(
        &self,
        _lsn: u64,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::DEFAULT_MAX_WAL_FILE_SIZE;
    use super::super::writer::WalWriter;
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn recovery_replays_all_records() {
        let dir = tempdir().unwrap();
        let mut writer = WalWriter::open(dir.path(), DEFAULT_MAX_WAL_FILE_SIZE).unwrap();
        writer.insert(1, b"vec1").unwrap();
        writer.insert(2, b"vec2").unwrap();
        writer.checkpoint().unwrap();
        writer.sync().unwrap();

        let mut recovery = WalRecovery::new(dir.path());
        let state = recovery.recover(&NoOpHandler).unwrap();
        assert_eq!(state.last_lsn, 2);
        assert_eq!(state.last_checkpoint_lsn, 2);
        assert_eq!(state.vectors.len(), 2);
    }

    #[test]
    fn recovery_truncates_old_files() {
        let dir = tempdir().unwrap();
        let mut writer = WalWriter::open(dir.path(), DEFAULT_MAX_WAL_FILE_SIZE).unwrap();
        writer.insert(1, b"vec").unwrap();
        writer.checkpoint().unwrap();
        writer.sync().unwrap();

        let mut recovery = WalRecovery::new(dir.path());
        recovery.recover(&NoOpHandler).unwrap();
        let truncated = recovery.truncate_pre_checkpoint().unwrap();
        assert_eq!(
            truncated, 0,
            "file 0 contains checkpoint — should not be truncated"
        );
        assert!(dir.path().join(wal_file_name(0)).exists());
    }
}
