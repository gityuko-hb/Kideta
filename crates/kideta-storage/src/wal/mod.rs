#![cfg(unix)]

pub mod error;
pub mod format;
pub mod reader;
pub mod recovery;
pub mod writer;

pub use error::{Result, WalError};
pub use format::{
    DEFAULT_MAX_WAL_FILE_SIZE, MAX_WAL_FILE_SIZE, Record, RecordType, WAL_HEADER_SIZE, WAL_MAGIC,
    WAL_VERSION, WalHeader, wal_file_name,
};
pub use reader::{WalFiles, WalReader};
pub use recovery::{NoOpHandler, ReplayHandler, WalRecovery, WalState};
pub use writer::{SyncPolicy, WalWriter, WalWriterThreaded};
