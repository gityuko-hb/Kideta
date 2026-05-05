//! Error types for memory mapping operations.

use std::fmt;

pub type Result<T> = std::result::Result<T, MmapError>;

#[derive(Debug)]
pub enum MmapError {
    Map {
        reason: String,
        code: Option<i32>,
    },
    Unmap {
        code: Option<i32>,
    },
    Sync {
        kind: &'static str,
        code: Option<i32>,
    },
    Advis {
        kind: &'static str,
        code: Option<i32>,
    },
    Lock,
    Unlock,
    OutOfBounds {
        offset: u64,
        len: usize,
        file_size: u64,
    },
    ZeroSize,
    Ftruncate {
        code: Option<i32>,
    },
    Mremap {
        code: Option<i32>,
    },
}

impl fmt::Display for MmapError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            MmapError::Map { reason, code } => {
                write!(f, "mmap failed: {reason}")?;
                if let Some(c) = code {
                    write!(f, " (OS error {c})")?;
                }
                Ok(())
            },
            MmapError::Unmap { code } => {
                write!(f, "munmap failed")?;
                if let Some(c) = code {
                    write!(f, " (OS error {c})")?;
                }
                Ok(())
            },
            MmapError::Sync { kind, code } => {
                write!(f, "msync ({kind}) failed")?;
                if let Some(c) = code {
                    write!(f, " (OS error {c})")?;
                }
                Ok(())
            },
            MmapError::Advis { kind, code } => {
                write!(f, "madvise ({kind}) failed")?;
                if let Some(c) = code {
                    write!(f, " (OS error {c})")?;
                }
                Ok(())
            },
            MmapError::Lock => write!(f, "mlock failed — check CAP_IPC_LOCK or RLIMIT_MEMLOCK"),
            MmapError::Unlock => write!(f, "munlock failed"),
            MmapError::OutOfBounds {
                offset,
                len,
                file_size,
            } => {
                write!(
                    f,
                    "mmap out of bounds: offset={offset}, len={len}, file_size={file_size}"
                )
            },
            MmapError::ZeroSize => write!(f, "anonymous mapping requires size > 0"),
            MmapError::Ftruncate { code } => {
                write!(f, "ftruncate failed")?;
                if let Some(c) = code {
                    write!(f, " (OS error {c})")?;
                }
                Ok(())
            },
            MmapError::Mremap { code } => {
                write!(f, "mremap failed")?;
                if let Some(c) = code {
                    write!(f, " (OS error {c})")?;
                }
                Ok(())
            },
        }
    }
}

impl std::error::Error for MmapError {}

impl From<MmapError> for std::io::Error {
    fn from(e: MmapError) -> Self {
        std::io::Error::other(e.to_string())
    }
}
