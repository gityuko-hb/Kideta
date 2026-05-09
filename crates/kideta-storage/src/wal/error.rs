use std::fmt;

#[derive(Debug)]
pub enum WalError {
    Io {
        kind: &'static str,
        code: Option<i32>,
    },
    CorruptRecord {
        lsn: u64,
        reason: &'static str,
    },
    BadCrc {
        expected: u32,
        actual: u32,
        lsn: u64,
    },
    TornWrite {
        lsn: u64,
    },
    UnexpectedEof {
        expected: usize,
        found: usize,
    },
    MaxSizeExceeded {
        current: u64,
        max: u64,
    },
    InvalidMagic {
        expected: u32,
        found: u32,
    },
    UnsupportedVersion {
        expected: u32,
        found: u32,
    },
    LsnNotMonotonic {
        prev: u64,
        curr: u64,
    },
    NoWalFiles,
    FileNotFound(String),
}

impl fmt::Display for WalError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::Io { kind, code } => {
                write!(f, "WAL I/O error ({kind}")?;
                if let Some(c) = code {
                    write!(f, ", code={c}")?;
                }
                write!(f, ")")
            },
            Self::CorruptRecord { lsn, reason } => {
                write!(f, "corrupt WAL record at LSN {lsn}: {reason}")
            },
            Self::BadCrc {
                expected,
                actual,
                lsn,
            } => {
                write!(
                    f,
                    "CRC mismatch at LSN {lsn}: expected {expected:#010x}, got {actual:#010x}"
                )
            },
            Self::TornWrite { lsn } => {
                write!(f, "torn write detected at LSN {lsn}")
            },
            Self::UnexpectedEof { expected, found } => {
                write!(
                    f,
                    "unexpected EOF: expected {expected} bytes, found {found}"
                )
            },
            Self::MaxSizeExceeded { current, max } => {
                write!(f, "WAL file size {current} exceeds max {max}")
            },
            Self::InvalidMagic { expected, found } => {
                write!(
                    f,
                    "invalid WAL magic: expected {expected:#010x}, found {found:#010x}"
                )
            },
            Self::UnsupportedVersion { expected, found } => {
                write!(
                    f,
                    "unsupported WAL version: expected {expected}, found {found}"
                )
            },
            Self::LsnNotMonotonic { prev, curr } => {
                write!(f, "LSN not monotonic: prev={prev}, curr={curr}")
            },
            Self::NoWalFiles => {
                write!(f, "no WAL files found")
            },
            Self::FileNotFound(path) => {
                write!(f, "WAL file not found: {path}")
            },
        }
    }
}

impl std::error::Error for WalError {}

impl WalError {
    pub fn io(kind: &'static str) -> Self {
        Self::Io {
            kind,
            code: std::io::Error::last_os_error().raw_os_error(),
        }
    }

    pub fn io_with_code(
        kind: &'static str,
        code: Option<i32>,
    ) -> Self {
        Self::Io { kind, code }
    }
}

pub type Result<T> = std::result::Result<T, WalError>;
