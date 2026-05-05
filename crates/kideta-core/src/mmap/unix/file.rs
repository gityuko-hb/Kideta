//! File manipulation utilities for mmap.

#![cfg(unix)]

use std::os::unix::io::RawFd;

use super::super::error::{MmapError, Result};

pub fn ftruncate(
    fd: RawFd,
    len: usize,
) -> Result<()> {
    let ret = unsafe { libc::ftruncate(fd, len as libc::off_t) };
    if ret == 0 {
        Ok(())
    } else {
        Err(MmapError::Ftruncate {
            code: std::io::Error::last_os_error().raw_os_error(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::os::unix::io::AsRawFd;
    use tempfile::NamedTempFile;

    #[test]
    fn ftruncate_grows_file() {
        let mut file = NamedTempFile::new().unwrap();
        let data = [0xAB; 1024];
        file.write_all(&data).unwrap();
        let path = file.path().to_path_buf();
        file.persist(&path).unwrap();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let fd = file.as_raw_fd();
        let metadata_before = std::fs::metadata(&path).unwrap();
        assert_eq!(metadata_before.len(), 1024);

        ftruncate(fd, 4096).unwrap();

        let metadata_after = std::fs::metadata(&path).unwrap();
        assert_eq!(metadata_after.len(), 4096);
    }

    #[test]
    fn ftruncate_shrinks_file() {
        let mut file = NamedTempFile::new().unwrap();
        let data = [0xAB; 4096];
        file.write_all(&data).unwrap();
        let path = file.path().to_path_buf();
        file.persist(&path).unwrap();

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let fd = file.as_raw_fd();

        ftruncate(fd, 1024).unwrap();

        let metadata = std::fs::metadata(&path).unwrap();
        assert_eq!(metadata.len(), 1024);
    }

    #[test]
    fn ftruncate_invalid_fd_returns_error() {
        let result = ftruncate(-1, 1024);
        assert!(result.is_err());
    }
}
