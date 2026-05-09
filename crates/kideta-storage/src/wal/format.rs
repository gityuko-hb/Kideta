use super::error::{Result, WalError};

pub const WAL_MAGIC: u32 = 0x_57_41_4C_31;
pub const WAL_VERSION: u32 = 2;
pub const WAL_HEADER_SIZE: u64 = 24;
pub const MAX_WAL_FILE_SIZE: u64 = 128 * 1024 * 1024;
pub const DEFAULT_MAX_WAL_FILE_SIZE: u64 = 64 * 1024 * 1024;

pub fn wal_file_name(id: u64) -> String {
    format!("wal_{id:09}.kid")
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordType {
    Insert = 1,
    Delete = 2,
    Upsert = 3,
    CreateCollection = 4,
    DropCollection = 5,
    SegmentSealed = 6,
    CompactionDone = 7,
    Checkpoint = 8,
}

impl RecordType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Insert),
            2 => Some(Self::Delete),
            3 => Some(Self::Upsert),
            4 => Some(Self::CreateCollection),
            5 => Some(Self::DropCollection),
            6 => Some(Self::SegmentSealed),
            7 => Some(Self::CompactionDone),
            8 => Some(Self::Checkpoint),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub lsn: u64,
    pub record_type: RecordType,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub struct WalHeader {
    pub magic: u32,
    pub version: u32,
    pub wal_id: u64,
    pub created_at: u64,
}

impl WalHeader {
    pub const SIZE: usize = 24;

    pub fn new(wal_id: u64) -> Self {
        Self {
            magic: WAL_MAGIC,
            version: WAL_VERSION,
            wal_id,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    pub fn serialize_into(
        &self,
        buf: &mut [u8],
    ) {
        debug_assert!(buf.len() >= Self::SIZE);
        let mut off = 0;
        buf[off..off + 4].copy_from_slice(&self.magic.to_le_bytes());
        off += 4;
        buf[off..off + 4].copy_from_slice(&self.version.to_le_bytes());
        off += 4;
        buf[off..off + 8].copy_from_slice(&self.wal_id.to_le_bytes());
        off += 8;
        buf[off..off + 8].copy_from_slice(&self.created_at.to_le_bytes());
    }

    pub fn deserialize_from(buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(WalError::UnexpectedEof {
                expected: Self::SIZE,
                found: buf.len(),
            });
        }
        let mut off = 0;
        let magic = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        off += 4;
        let version = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        off += 4;
        let wal_id = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        off += 8;
        let created_at = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());

        if magic != WAL_MAGIC {
            return Err(WalError::InvalidMagic {
                expected: WAL_MAGIC,
                found: magic,
            });
        }
        if version != WAL_VERSION {
            return Err(WalError::UnsupportedVersion {
                expected: WAL_VERSION,
                found: version,
            });
        }

        Ok(Self {
            magic,
            version,
            wal_id,
            created_at,
        })
    }
}

pub fn encode_varint(
    mut value: u64,
    buf: &mut Vec<u8>,
) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

pub fn encode_varint_into(
    value: u64,
    buf: &mut [u8],
) -> usize {
    let mut n = 0usize;
    let mut val = value;
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf[n] = byte;
            n += 1;
            break;
        } else {
            buf[n] = byte | 0x80;
            n += 1;
        }
    }
    n
}

pub fn decode_varint(
    buf: &[u8],
    consumed: &mut usize,
) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift = 0usize;
    loop {
        if *consumed >= buf.len() {
            return Err(WalError::UnexpectedEof {
                expected: 1,
                found: buf.len() - *consumed,
            });
        }
        let byte = buf[*consumed];
        *consumed += 1;
        result |= ((byte & 0x7F) as u64) << (shift * 7);
        if byte & 0x80 == 0 {
            break;
        }
        shift += 1;
        if shift > 10 {
            return Err(WalError::CorruptRecord {
                lsn: 0,
                reason: "varint overflow",
            });
        }
    }
    Ok(result)
}

pub fn varint_encoded_len(value: u64) -> usize {
    if value < (1 << 7) {
        1
    } else if value < (1 << 14) {
        2
    } else if value < (1 << 21) {
        3
    } else if value < (1 << 28) {
        4
    } else if value < (1 << 35) {
        5
    } else if value < (1 << 42) {
        6
    } else if value < (1 << 49) {
        7
    } else if value < (1 << 56) {
        8
    } else if value < (1 << 63) {
        9
    } else {
        10
    }
}

pub struct RecordEncoder;

impl RecordEncoder {
    pub fn encode(
        record: &Record,
        out: &mut Vec<u8>,
    ) {
        let mut header_buf = Vec::with_capacity(32);
        encode_varint(record.lsn, &mut header_buf);
        encode_varint(record.payload.len() as u64, &mut header_buf);
        header_buf.push(record.record_type as u8);

        let header_len = header_buf.len() as u32;

        let mut crc_input = Vec::with_capacity(header_buf.len() + record.payload.len());
        crc_input.extend_from_slice(&header_buf);
        crc_input.extend_from_slice(&record.payload);

        let crc = crc32c(0, &crc_input);

        out.reserve(4 + 4 + header_len as usize + record.payload.len());
        out.extend_from_slice(&crc.to_le_bytes());
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&header_buf);
        out.extend_from_slice(&record.payload);
    }
}

pub struct RecordDecoder;

impl RecordDecoder {
    pub fn decode(
        buf: &[u8],
        lsn: Option<u64>,
    ) -> Result<Record> {
        let mut off = 0;

        if buf.len() < 4 {
            return Err(WalError::UnexpectedEof {
                expected: 4,
                found: buf.len(),
            });
        }
        let stored_crc = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        off += 4;

        if buf.len() < off + 4 {
            return Err(WalError::UnexpectedEof {
                expected: off + 4,
                found: buf.len(),
            });
        }
        let header_len = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        off += 4;

        if buf.len() < off + header_len {
            return Err(WalError::UnexpectedEof {
                expected: off + header_len,
                found: buf.len(),
            });
        }

        let crc_input_end = 4 + 4 + header_len;
        let payload_len = if buf.len() > crc_input_end {
            buf.len() - crc_input_end
        } else {
            0
        };

        let mut crc_input = Vec::with_capacity(header_len + payload_len);
        crc_input.extend_from_slice(&buf[off..off + header_len]);
        if payload_len > 0 {
            crc_input.extend_from_slice(&buf[crc_input_end..]);
        }

        let computed_crc = crc32c(0, &crc_input);
        if computed_crc != stored_crc {
            return Err(WalError::BadCrc {
                expected: stored_crc,
                actual: computed_crc,
                lsn: lsn.unwrap_or(0),
            });
        }

        let header = &buf[off..off + header_len];
        off += header_len;

        let mut header_off = 0;
        let record_lsn = decode_varint(header, &mut header_off)?;
        let payload_len_decoded = decode_varint(header, &mut header_off)? as usize;
        if header_off >= header.len() {
            return Err(WalError::CorruptRecord {
                lsn: record_lsn,
                reason: "missing record type",
            });
        }
        let record_type_byte = header[header_off];
        let record_type = RecordType::from_u8(record_type_byte).ok_or(WalError::CorruptRecord {
            lsn: record_lsn,
            reason: "invalid record type",
        })?;

        let payload = if payload_len_decoded > 0 {
            buf[off..off + payload_len_decoded].to_vec()
        } else {
            Vec::new()
        };

        Ok(Record {
            lsn: record_lsn,
            record_type,
            payload,
        })
    }
}

fn crc32c(
    crc: u32,
    data: &[u8],
) -> u32 {
    kideta_core::utils::crc32c::crc32c(crc, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        for v in [
            0u64,
            1,
            127,
            128,
            255,
            256,
            65535,
            65536,
            u32::MAX as u64,
            u64::MAX,
        ] {
            let mut buf = Vec::new();
            encode_varint(v, &mut buf);
            let mut off = 0;
            let decoded = decode_varint(&buf, &mut off).unwrap();
            assert_eq!(v, decoded, "varint roundtrip failed for {v}");
        }
    }

    #[test]
    fn header_serialize_roundtrip() {
        let h = WalHeader::new(42);
        let mut buf = vec![0u8; WalHeader::SIZE];
        h.serialize_into(&mut buf);
        let h2 = WalHeader::deserialize_from(&buf).unwrap();
        assert_eq!(h.magic, h2.magic);
        assert_eq!(h.version, h2.version);
        assert_eq!(h.wal_id, h2.wal_id);
        assert_eq!(h.created_at, h2.created_at);
    }

    #[test]
    fn record_encode_decode_roundtrip() {
        let record = Record {
            lsn: 12345,
            record_type: RecordType::Insert,
            payload: b"hello, WAL!".to_vec(),
        };
        let mut encoded = Vec::new();
        RecordEncoder::encode(&record, &mut encoded);

        let decoded = RecordDecoder::decode(&encoded, Some(record.lsn)).unwrap();
        assert_eq!(decoded.lsn, record.lsn);
        assert_eq!(decoded.record_type, record.record_type);
        assert_eq!(decoded.payload, record.payload);
    }

    #[test]
    fn bad_crc_detected() {
        let record = Record {
            lsn: 1,
            record_type: RecordType::Checkpoint,
            payload: b"test".to_vec(),
        };
        let mut encoded = Vec::new();
        RecordEncoder::encode(&record, &mut encoded);

        encoded[8] ^= 0xFF;

        let result = RecordDecoder::decode(&encoded, Some(1));
        assert!(matches!(result, Err(WalError::BadCrc { .. })));
    }

    #[test]
    fn wal_file_name_format() {
        assert_eq!(wal_file_name(0), "wal_000000000.kid");
        assert_eq!(wal_file_name(99), "wal_000000099.kid");
        assert_eq!(wal_file_name(99999999), "wal_099999999.kid");
    }
}
