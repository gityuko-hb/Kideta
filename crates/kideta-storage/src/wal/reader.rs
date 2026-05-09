use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use super::error::{Result, WalError};
use super::format::{Record, RecordDecoder, WAL_HEADER_SIZE, WalHeader};

pub struct WalReader {
    file: BufReader<File>,
    file_id: u64,
    file_path: String,
}

pub struct Iter {
    reader: WalReader,
    done: bool,
}

impl WalReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|_| WalError::io("open"))?;
        let file_id = Self::extract_file_id(path)?;
        let mut reader = BufReader::new(file);
        Self::validate_header(&mut reader)?;
        Ok(Self {
            file: reader,
            file_id,
            file_path: path.to_string_lossy().to_string(),
        })
    }

    pub fn open_at_lsn(
        dir: &Path,
        file_id: u64,
    ) -> Result<Self> {
        let name = super::format::wal_file_name(file_id);
        let path = dir.join(&name);
        Self::open(&path)
    }

    pub fn file_id(&self) -> u64 {
        self.file_id
    }

    pub fn read_record(&mut self) -> Result<Option<Record>> {
        let mut crc_buf = [0u8; 4];
        let n = self
            .file
            .read(&mut crc_buf)
            .map_err(|_| WalError::io("read_crc"))?;
        if n == 0 {
            return Ok(None);
        }
        if n < 4 {
            return Err(WalError::UnexpectedEof {
                expected: 4,
                found: n,
            });
        }

        let header_len_buf = &mut [0u8; 4];
        let n = self
            .file
            .read(header_len_buf)
            .map_err(|_| WalError::io("read_header_len"))?;
        if n < 4 {
            return Err(WalError::UnexpectedEof {
                expected: 4,
                found: n,
            });
        }
        let header_len = u32::from_le_bytes(*header_len_buf) as usize;

        if header_len > 1024 * 1024 {
            return Err(WalError::CorruptRecord {
                lsn: 0,
                reason: "header_len too large (possible torn write)",
            });
        }

        let mut header_buf = vec![0u8; header_len];
        let mut total_read = 0;
        while total_read < header_len {
            let n = self
                .file
                .read(&mut header_buf[total_read..])
                .map_err(|_| WalError::io("read_header"))?;
            if n == 0 {
                return Err(WalError::UnexpectedEof {
                    expected: header_len,
                    found: total_read,
                });
            }
            total_read += n;
        }

        let mut header_off = 0usize;
        let lsn_from_header =
            super::format::decode_varint(&header_buf, &mut header_off).unwrap_or(0);
        let payload_len_decoded =
            super::format::decode_varint(&header_buf, &mut header_off).unwrap_or(0) as usize;

        let mut payload = vec![0u8; payload_len_decoded];
        let mut total_payload = 0usize;
        while total_payload < payload_len_decoded {
            let n = self
                .file
                .read(&mut payload[total_payload..])
                .map_err(|_| WalError::io("read_payload"))?;
            if n == 0 {
                return Err(WalError::UnexpectedEof {
                    expected: payload_len_decoded,
                    found: total_payload,
                });
            }
            total_payload += n;
        }

        let mut full_record = Vec::with_capacity(4 + 4 + header_len + payload_len_decoded);
        full_record.extend_from_slice(&crc_buf);
        full_record.extend_from_slice(&u32::to_le_bytes(header_len as u32));
        full_record.extend_from_slice(&header_buf);
        full_record.extend_from_slice(&payload);

        let record = RecordDecoder::decode(&full_record, Some(lsn_from_header))?;
        Ok(Some(record))
    }

    pub fn seek_to_data(&mut self) -> Result<()> {
        self.file
            .seek(SeekFrom::Start(WAL_HEADER_SIZE))
            .map_err(|_| WalError::io("seek"))?;
        Ok(())
    }

    fn validate_header(reader: &mut BufReader<File>) -> Result<WalHeader> {
        let mut header_buf = [0u8; WalHeader::SIZE];
        reader
            .read_exact(&mut header_buf)
            .map_err(|_| WalError::io("read_header"))?;
        WalHeader::deserialize_from(&header_buf)
    }

    fn extract_file_id(path: &Path) -> Result<u64> {
        let name = path
            .file_name()
            .ok_or(WalError::CorruptRecord {
                lsn: 0,
                reason: "invalid filename",
            })?
            .to_string_lossy();
        if name.len() < 13 || &name[..4] != "wal_" || &name[name.len() - 4..] != ".kid" {
            return Err(WalError::CorruptRecord {
                lsn: 0,
                reason: "invalid WAL filename format",
            });
        }
        let id = name[4..name.len() - 4]
            .parse::<u64>()
            .map_err(|_| WalError::CorruptRecord {
                lsn: 0,
                reason: "invalid WAL file id",
            })?;
        Ok(id)
    }
}

impl Iterator for Iter {
    type Item = Result<super::format::Record>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.reader.read_record() {
            Ok(Some(rec)) => Some(Ok(rec)),
            Ok(None) => {
                self.done = true;
                None
            },
            Err(e) => {
                self.done = true;
                Some(Err(e))
            },
        }
    }
}

impl WalReader {
    pub fn iter(&mut self) -> Iter {
        let mut file = BufReader::new(File::open(std::path::Path::new(&self.file_path)).unwrap());
        file.seek(SeekFrom::Start(WAL_HEADER_SIZE))
            .expect("iter seek to data");
        Iter {
            reader: Self {
                file,
                file_id: self.file_id,
                file_path: self.file_path.clone(),
            },
            done: false,
        }
    }
}

pub struct WalFiles {
    dir: std::path::PathBuf,
    files: Vec<u64>,
    current: usize,
}

impl WalFiles {
    pub fn open(dir: &Path) -> Result<Self> {
        let files = Self::list_sorted(dir)?;
        if files.is_empty() {
            return Err(WalError::NoWalFiles);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            files,
            current: 0,
        })
    }

    fn list_sorted(dir: &Path) -> Result<Vec<u64>> {
        let mut ids = Vec::new();
        let entries = std::fs::read_dir(dir).map_err(|_| WalError::io("read_dir"))?;
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

    pub fn current_file_id(&self) -> Option<u64> {
        self.files.get(self.current).copied()
    }
}

impl Iterator for WalFiles {
    type Item = Result<WalReader>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.files.len() {
            return None;
        }
        let file_id = self.files[self.current];
        self.current += 1;
        Some(WalReader::open_at_lsn(&self.dir, file_id))
    }
}

#[cfg(test)]
mod tests {
    use crate::wal::RecordType;

    use super::super::DEFAULT_MAX_WAL_FILE_SIZE;
    use super::super::writer::WalWriter;
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn reader_reads_written_records() {
        let dir = tempdir().unwrap();
        let mut writer = WalWriter::open(dir.path(), DEFAULT_MAX_WAL_FILE_SIZE).unwrap();
        let lsn1 = writer.insert(1, b"vector_data_1").unwrap();
        let lsn2 = writer.delete(1).unwrap();
        writer.sync().unwrap();

        let mut reader = WalReader::open_at_lsn(dir.path(), 0).unwrap();
        let rec1 = reader.read_record().unwrap().unwrap();
        assert_eq!(rec1.lsn, lsn1);
        assert_eq!(rec1.record_type, RecordType::Insert);

        let rec2 = reader.read_record().unwrap().unwrap();
        assert_eq!(rec2.lsn, lsn2);
        assert_eq!(rec2.record_type, RecordType::Delete);
    }

    #[test]
    fn reader_eof_returns_none() {
        let dir = tempdir().unwrap();
        let mut writer = WalWriter::open(dir.path(), DEFAULT_MAX_WAL_FILE_SIZE).unwrap();
        writer.checkpoint().unwrap();
        writer.sync().unwrap();

        let mut reader = WalReader::open_at_lsn(dir.path(), 0).unwrap();
        reader.seek_to_data().unwrap();
        assert!(reader.read_record().unwrap().is_some());
        assert!(reader.read_record().unwrap().is_none());
        assert!(reader.read_record().unwrap().is_none());
    }

    #[test]
    fn iter_reads_all_records() {
        let dir = tempdir().unwrap();
        let mut writer = WalWriter::open(dir.path(), DEFAULT_MAX_WAL_FILE_SIZE).unwrap();
        for i in 0..5 {
            writer.insert(i, b"vector").unwrap();
        }
        writer.sync().unwrap();

        let mut reader = WalReader::open_at_lsn(dir.path(), 0).unwrap();
        reader.seek_to_data().unwrap();
        let count = reader.iter().count();
        assert_eq!(count, 5);
    }
}
