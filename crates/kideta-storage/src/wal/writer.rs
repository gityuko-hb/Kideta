use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use super::error::{Result, WalError};
#[cfg(test)]
use super::format::DEFAULT_MAX_WAL_FILE_SIZE;
use super::format::{RecordEncoder, RecordType, WAL_HEADER_SIZE, WalHeader, wal_file_name};

pub struct WalWriter {
    dir: PathBuf,
    file: BufWriter<File>,
    current_file_id: u64,
    current_file_path: PathBuf,
    max_file_size: u64,
    current_offset: u64,
    lsn_counter: u64,
    pending: Vec<Vec<u8>>,
    sync_policy: SyncPolicy,
    file_id_counter: u64,
    scratch: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncPolicy {
    /// fsync after every record
    Fsync,
    /// batch fsync every `interval` ms
    Batch { interval_ms: u64 },
    /// fsync only on explicit flush
    #[default]
    NoSync,
}

impl WalWriter {
    pub fn open(
        dir: &Path,
        max_file_size: u64,
    ) -> Result<Self> {
        std::fs::create_dir_all(dir).map_err(|_| WalError::io("create_dir"))?;

        let existing_files = Self::list_wal_files(dir)?;
        let (file_id, needs_header) = if let Some(max_id) = existing_files.last() {
            (*max_id, false)
        } else {
            (0, true)
        };

        let file_path = dir.join(wal_file_name(file_id));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(false)
            .open(&file_path)
            .map_err(|_| WalError::io("open_wal"))?;

        let metadata = file
            .metadata()
            .map_err(|_| WalError::io("metadata"))?;
        let current_offset = metadata.len();

        let mut writer = Self {
            dir: dir.to_path_buf(),
            file: BufWriter::new(file),
            current_file_id: file_id,
            current_file_path: file_path,
            max_file_size,
            current_offset,
            lsn_counter: 0,
            pending: Vec::new(),
            sync_policy: SyncPolicy::NoSync,
            file_id_counter: file_id,
            scratch: Vec::with_capacity(4096),
        };

        if needs_header {
            writer.write_header()?;
        }

        Ok(writer)
    }

    pub fn append(
        &mut self,
        record_type: RecordType,
        payload: &[u8],
    ) -> Result<u64> {
        let lsn = self.lsn_counter;
        self.lsn_counter += 1;

        let record = super::format::Record {
            lsn,
            record_type,
            payload: payload.to_vec(),
        };

        self.scratch.clear();
        RecordEncoder::encode(&record, &mut self.scratch);
        let encoded_len = self.scratch.len();

        let mut replacement = Vec::with_capacity(self.scratch.capacity());
        std::mem::swap(&mut self.scratch, &mut replacement);
        let encoded = replacement;

        if self.current_offset + encoded_len as u64 > self.max_file_size {
            self.rotate()?;
        }

        self.file
            .write_all(&encoded)
            .map_err(|_| WalError::io("write"))?;
        self.current_offset += encoded_len as u64;
        self.pending.push(encoded);

        match self.sync_policy {
            SyncPolicy::Fsync => self.sync()?,
            SyncPolicy::Batch { .. } => {},
            SyncPolicy::NoSync => {},
        }

        Ok(lsn)
    }

    pub fn insert(
        &mut self,
        global_id: u64,
        vector: &[u8],
    ) -> Result<u64> {
        let mut payload = Vec::with_capacity(8 + vector.len());
        payload.extend_from_slice(&global_id.to_le_bytes());
        payload.extend_from_slice(vector);
        self.append(RecordType::Insert, &payload)
    }

    pub fn delete(
        &mut self,
        global_id: u64,
    ) -> Result<u64> {
        let payload = global_id.to_le_bytes();
        self.append(RecordType::Delete, &payload)
    }

    pub fn upsert(
        &mut self,
        id: u32,
        vector: &[u8],
    ) -> Result<u64> {
        let mut payload = Vec::with_capacity(4 + vector.len());
        payload.extend_from_slice(&id.to_le_bytes());
        payload.extend_from_slice(vector);
        self.append(RecordType::Upsert, &payload)
    }

    pub fn create_collection(
        &mut self,
        name: &str,
    ) -> Result<u64> {
        self.append(RecordType::CreateCollection, name.as_bytes())
    }

    pub fn drop_collection(
        &mut self,
        name: &str,
    ) -> Result<u64> {
        self.append(RecordType::DropCollection, name.as_bytes())
    }

    pub fn segment_sealed(
        &mut self,
        segment_id: u64,
    ) -> Result<u64> {
        self.append(RecordType::SegmentSealed, &segment_id.to_le_bytes())
    }

    pub fn compaction_done(
        &mut self,
        old_segments: &[u64],
        new_segment: u64,
    ) -> Result<u64> {
        let mut payload = Vec::with_capacity(8 * (old_segments.len() + 1) + 4);
        payload.extend_from_slice(&(old_segments.len() as u32).to_le_bytes());
        for &s in old_segments {
            payload.extend_from_slice(&s.to_le_bytes());
        }
        payload.extend_from_slice(&new_segment.to_le_bytes());
        self.append(RecordType::CompactionDone, &payload)
    }

    pub fn checkpoint(&mut self) -> Result<u64> {
        self.append(RecordType::Checkpoint, &[])
    }

    pub fn sync(&mut self) -> Result<()> {
        self.file
            .flush()
            .map_err(|_| WalError::io("flush"))?;
        let file = self.file.get_mut();
        file.sync_all()
            .map_err(|_| WalError::io("sync_all"))?;
        self.pending.clear();
        Ok(())
    }

    pub fn set_sync_policy(
        &mut self,
        policy: SyncPolicy,
    ) {
        self.sync_policy = policy;
    }

    pub fn flush(&mut self) -> Result<()> {
        self.file
            .flush()
            .map_err(|_| WalError::io("flush"))?;
        Ok(())
    }

    pub fn current_lsn(&self) -> u64 {
        self.lsn_counter
    }

    pub fn current_file_id(&self) -> u64 {
        self.current_file_id
    }

    pub fn data_dir(&self) -> &Path {
        &self.dir
    }

    fn write_header(&mut self) -> Result<()> {
        let header = WalHeader::new(self.current_file_id);
        let mut buf = vec![0u8; WalHeader::SIZE];
        header.serialize_into(&mut buf);
        self.file
            .write_all(&buf)
            .map_err(|_| WalError::io("write_header"))?;
        self.current_offset = WAL_HEADER_SIZE;
        Ok(())
    }

    fn rotate(&mut self) -> Result<()> {
        self.file
            .flush()
            .map_err(|_| WalError::io("flush_before_rotate"))?;
        self.file
            .get_mut()
            .sync_all()
            .map_err(|_| WalError::io("sync_before_rotate"))?;

        self.file_id_counter += 1;
        self.current_file_id = self.file_id_counter;
        self.current_file_path = self.dir.join(wal_file_name(self.current_file_id));

        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(false)
            .open(&self.current_file_path)
            .map_err(|_| WalError::io("create_wal_file"))?;

        self.file = BufWriter::new(file);
        self.current_offset = 0;
        self.write_header()?;
        Ok(())
    }

    fn list_wal_files(dir: &Path) -> Result<Vec<u64>> {
        let mut files: Vec<u64> = Vec::new();
        let entries = std::fs::read_dir(dir).map_err(|_| WalError::io("read_dir"))?;
        for entry in entries {
            let entry = entry.map_err(|_| WalError::io("read_dir_entry"))?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("wal_")
                && name_str.ends_with(".kid")
                && let Ok(id) = name_str[4..name_str.len() - 4].parse::<u64>()
            {
                files.push(id);
            }
        }
        files.sort();
        Ok(files)
    }
}

impl Drop for WalWriter {
    fn drop(&mut self) {
        if let Err(e) = self.file.flush() {
            tracing::error!("WalWriter flush on drop failed: {e}");
        }
    }
}

pub struct WalWriterThreaded {
    inner: Arc<RwLock<WalWriter>>,
    bg_thread: Option<std::thread::JoinHandle<()>>,
}

impl WalWriterThreaded {
    pub fn open(
        dir: &Path,
        max_file_size: u64,
    ) -> Result<Self> {
        let inner = Arc::new(RwLock::new(WalWriter::open(dir, max_file_size)?));
        Ok(Self {
            inner,
            bg_thread: None,
        })
    }

    pub fn start_group_commit(
        mut self,
        interval: Duration,
    ) -> Self {
        let inner = Arc::clone(&self.inner);
        let handle = std::thread::spawn(move || {
            loop {
                std::thread::sleep(interval);
                let mut writer = match inner.write() {
                    Ok(w) => w,
                    Err(_) => break,
                };
                if writer.pending.is_empty() {
                    continue;
                }
                if let Err(e) = writer.sync() {
                    tracing::error!("group_commit sync failed: {e}");
                }
            }
        });
        self.bg_thread = Some(handle);
        self
    }

    pub fn append(
        &self,
        record_type: RecordType,
        payload: &[u8],
    ) -> Result<u64> {
        self.inner
            .write()
            .unwrap()
            .append(record_type, payload)
    }

    pub fn sync(&self) -> Result<()> {
        self.inner.write().unwrap().sync()
    }

    pub fn flush(&self) -> Result<()> {
        self.inner.write().unwrap().flush()
    }

    pub fn current_lsn(&self) -> u64 {
        self.inner.read().unwrap().current_lsn()
    }

    pub fn data_dir(&self) -> PathBuf {
        self.inner
            .read()
            .unwrap()
            .data_dir()
            .to_path_buf()
    }
}

impl Drop for WalWriterThreaded {
    fn drop(&mut self) {
        if let Some(h) = self.bg_thread.take() {
            drop(h.join());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn writer_open_creates_file() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::open(dir.path(), DEFAULT_MAX_WAL_FILE_SIZE).unwrap();
        assert!(dir.path().join(wal_file_name(0)).exists());
        assert_eq!(writer.current_lsn(), 0);
    }

    #[test]
    fn writer_append_increments_lsn() {
        let dir = tempdir().unwrap();
        let mut writer = WalWriter::open(dir.path(), DEFAULT_MAX_WAL_FILE_SIZE).unwrap();
        let lsn1 = writer.insert(1, b"vec1").unwrap();
        let lsn2 = writer.insert(2, b"vec2").unwrap();
        assert_eq!(lsn1, 0);
        assert_eq!(lsn2, 1);
    }

    #[test]
    fn writer_multiple_files() {
        let dir = tempdir().unwrap();
        let max_size = 256u64;
        let mut writer = WalWriter::open(dir.path(), max_size).unwrap();
        for i in 0..10 {
            writer
                .insert(i, b"this is a somewhat longer vector data")
                .unwrap();
        }
        let files = std::fs::read_dir(dir.path()).unwrap();
        let wal_files: Vec<_> = files
            .filter_map(|e| {
                let n = e
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .to_string();
                if n.ends_with(".kid") {
                    Some(n)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            wal_files.len() >= 2,
            "expected multiple WAL files, got {wal_files:?}"
        );
    }

    #[test]
    fn sync_policy_none_does_not_sync_on_append() {
        let dir = tempdir().unwrap();
        let mut writer = WalWriter::open(dir.path(), DEFAULT_MAX_WAL_FILE_SIZE).unwrap();
        writer.set_sync_policy(SyncPolicy::NoSync);
        writer.insert(1, b"test").unwrap();
        drop(writer);
        let path = dir.path().join(wal_file_name(0));
        let meta = std::fs::metadata(&path).unwrap();
        assert!(meta.len() > WAL_HEADER_SIZE);
    }

    #[test]
    fn checkpoint_record() {
        let dir = tempdir().unwrap();
        let mut writer = WalWriter::open(dir.path(), DEFAULT_MAX_WAL_FILE_SIZE).unwrap();
        writer.insert(1, b"vec").unwrap();
        let lsn = writer.checkpoint().unwrap();
        assert_eq!(lsn, 1);
    }
}
