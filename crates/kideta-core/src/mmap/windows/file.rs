//! File truncation for Windows — uses SetFilePointerEx + SetEndOfFile.

#![cfg(windows)]

use crate::mmap::error::{MmapError, Result};

const FILE_BEGIN: u32 = 0;

pub fn ftruncate(
    file: &std::fs::File,
    len: usize,
) -> Result<()> {
    use std::os::windows::io::AsRawHandle;
    let handle = file.as_raw_handle();

    let distance = len as i64;
    let moved =
        unsafe { kernel32::SetFilePointerEx(handle, distance, std::ptr::null_mut(), FILE_BEGIN) };
    if moved == 0 {
        let code = std::io::Error::last_os_error().raw_os_error();
        return Err(MmapError::Ftruncate { code });
    }

    let set = unsafe { kernel32::SetEndOfFile(handle) };
    if set == 0 {
        let code = std::io::Error::last_os_error().raw_os_error();
        return Err(MmapError::Ftruncate { code });
    }

    Ok(())
}

mod kernel32 {
    use std::os::windows::io::RawHandle;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn SetFilePointerEx(
            hFile: RawHandle,
            liDistanceToMove: i64,
            lpNewFilePointer: *mut i64,
            dwMoveMethod: u32,
        ) -> i32;

        pub fn SetEndOfFile(hFile: RawHandle) -> i32;
    }
}
