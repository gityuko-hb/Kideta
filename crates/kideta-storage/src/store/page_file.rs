//! Page file manager — manages data page files (`data_N.dat`).

use kideta_core::mmap::unix::ftruncate;
use kideta_core::mmap::{MmapMut, MmapOptions};
use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

pub const PAGE_FILENAME_PREFIX: &str = "data_";
pub const PAGE_FILENAME_SUFFIX: &str = ".dat";

#[derive(Debug)]
struct Page {
    mmap: Arc<RwLock<MmapMut>>,
    id: u32,
}

#[derive(Debug)]
pub struct PageFileManager {
    data_dir: PathBuf,
    pages: RwLock<Vec<Option<Page>>>,
    page_size: usize,
}

impl PageFileManager {
    pub fn new(data_dir: PathBuf) -> Self {
        Self::new_with_page_size(data_dir, 4 * 1024 * 1024)
    }

    pub fn new_with_page_size(
        data_dir: PathBuf,
        page_size: usize,
    ) -> Self {
        Self {
            data_dir,
            pages: RwLock::new(Vec::new()),
            page_size,
        }
    }

    pub fn open_page(
        &self,
        page_id: u32,
    ) -> std::io::Result<PageRef> {
        let page = self.load_or_create_page(page_id)?;
        Ok(PageRef {
            page_id,
            mmap: page.mmap,
            _manager: self as *const PageFileManager as usize,
        })
    }

    fn load_or_create_page(
        &self,
        page_id: u32,
    ) -> std::io::Result<Page> {
        {
            let pages = self.pages.read().unwrap();
            if let Some(Some(page)) = pages.get(page_id as usize) {
                return Ok(Page {
                    mmap: page.mmap.clone(),
                    id: page.id,
                });
            }
        }

        let filename = page_filename(&self.data_dir, page_id);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&filename)?;

        let file_size = file.metadata()?.len();
        let ps = self.page_size;
        if (file_size as usize) < ps {
            ftruncate(file.as_raw_fd(), ps)?;
        }

        let mmap = unsafe { MmapOptions::new(ps).mmap_file_mut(&file)? };
        let page_mmap = Arc::new(RwLock::new(mmap));

        let mut pages = self.pages.write().unwrap();
        while pages.len() <= page_id as usize {
            pages.push(None);
        }
        pages[page_id as usize] = Some(Page {
            mmap: page_mmap.clone(),
            id: page_id,
        });

        Ok(Page {
            mmap: page_mmap,
            id: page_id,
        })
    }

    pub fn num_pages(&self) -> usize {
        self.pages.read().unwrap().len()
    }

    pub fn flush_all(&self) -> std::io::Result<()> {
        let pages = self.pages.read().unwrap();
        for page in pages.iter().flatten() {
            page.mmap
                .read()
                .unwrap()
                .flush()
                .map_err(std::io::Error::other)?;
        }
        Ok(())
    }
}

pub struct PageRef {
    page_id: u32,
    mmap: Arc<RwLock<MmapMut>>,
    _manager: usize,
}

impl PageRef {
    pub fn page_id(&self) -> u32 {
        self.page_id
    }

    pub fn slice(&self) -> Vec<u8> {
        // SAFETY: We have exclusive access through the RwLock
        unsafe { self.mmap.read().unwrap().as_slice().to_vec() }
    }

    pub fn write_at(
        &self,
        offset: usize,
        data: &[u8],
    ) {
        let mut guard = self.mmap.write().unwrap();
        // SAFETY: We have exclusive access through the RwLock
        let slice = unsafe { guard.as_mut_slice() };
        slice[offset..offset + data.len()].copy_from_slice(data);
    }

    pub fn flush(&self) -> std::io::Result<()> {
        self.mmap
            .read()
            .unwrap()
            .flush()
            .map_err(std::io::Error::other)
    }
}

impl std::fmt::Debug for PageRef {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("PageRef")
            .field("page_id", &self.page_id)
            .finish()
    }
}

fn page_filename(
    dir: &Path,
    page_id: u32,
) -> std::path::PathBuf {
    dir.join(format!(
        "{}{}{}",
        PAGE_FILENAME_PREFIX, page_id, PAGE_FILENAME_SUFFIX
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn open_page_creates_file() {
        let dir = tempdir().unwrap();
        let manager = PageFileManager::new_with_page_size(dir.path().to_path_buf(), 4096);
        let page = manager.open_page(0).unwrap();

        assert_eq!(page.page_id(), 0);
        assert_eq!(page.slice().len(), 4096);

        let filename = dir.path().join("data_0.dat");
        assert!(filename.exists());
    }

    #[test]
    fn open_page_twice_same_page() {
        let dir = tempdir().unwrap();
        let manager = PageFileManager::new_with_page_size(dir.path().to_path_buf(), 4096);

        let page1 = manager.open_page(0).unwrap();
        page1.write_at(0, &[0xAB]);

        let page2 = manager.open_page(0).unwrap();
        assert_eq!(page2.slice()[0], 0xAB);
    }

    #[test]
    fn write_and_flush_page() {
        let dir = tempdir().unwrap();
        let manager = PageFileManager::new_with_page_size(dir.path().to_path_buf(), 4096);
        let page = manager.open_page(0).unwrap();

        let mut data = Vec::new();
        data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        data.extend_from_slice(&[0xFF; 120]);
        page.write_at(0, &data);
        page.flush().unwrap();

        drop(page);

        let page2 = manager.open_page(0).unwrap();
        assert_eq!(&page2.slice()[0..4], &0xDEADBEEFu32.to_le_bytes());
    }
}
