//! Columnar memory-mapped vector storage.
//!
//! ## File Layout
//!
//! ```text
//! Bytes 0-63:    Header (VectorStorageHeader, 64 bytes)
//! Bytes 64+:      Vector data (dim * bytes_per_component per vector, varies by dtype)
//! ```
//!
//! ## Supported Dtypes
//!
//! | Dtype   | Bytes/Component | Notes                              |
//! |---------|-----------------|------------------------------------|
//! | F32     | 4               | Native f32                         |
//! | F16     | 2               | half::f16                          |
//! | BF16    | 2               | half::bf16                         |
//! | I8      | 1               | SQ8 quantized                      |
//! | BINARY  | 1 bit           | Packed bits, (dim+7)/8 bytes       |
//!
//! ## Growth Strategy
//!
//! Uses a doubling strategy for O(1) amortized appends. When capacity is reached,
//! the file is extended to 2x the current capacity, and the mmap is remapped.

use crate::vector_storage::dtype::VectorDtype;
use crate::vector_storage::header::VectorStorageHeader;
use kideta_core::mmap::Mmap;
use kideta_core::mmap::unix::{MmapOptions, ftruncate};
use std::borrow::Cow;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const DEFAULT_INITIAL_CAPACITY: u64 = 64;
const VECTOR_DATA_OFFSET: usize = 64;

pub enum MmapVectorStorageError {
    Io(std::io::Error),
    InvalidHeader,
    ChecksumMismatch,
    OutOfBounds,
    ZeroDimension,
    DtypeMismatch,
    DtypeNotSupported,
}

impl std::fmt::Display for MmapVectorStorageError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            MmapVectorStorageError::Io(e) => write!(f, "I/O error: {}", e),
            MmapVectorStorageError::InvalidHeader => {
                write!(f, "invalid or corrupted vector storage header")
            },
            MmapVectorStorageError::ChecksumMismatch => write!(f, "header checksum mismatch"),
            MmapVectorStorageError::OutOfBounds => write!(f, "vector index out of bounds"),
            MmapVectorStorageError::ZeroDimension => {
                write!(f, "dimension must be greater than zero")
            },
            MmapVectorStorageError::DtypeMismatch => {
                write!(f, "input dtype does not match storage dtype")
            },
            MmapVectorStorageError::DtypeNotSupported => {
                write!(f, "stored dtype is not supported")
            },
        }
    }
}

impl std::fmt::Debug for MmapVectorStorageError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl std::error::Error for MmapVectorStorageError {}

impl From<std::io::Error> for MmapVectorStorageError {
    fn from(e: std::io::Error) -> Self {
        MmapVectorStorageError::Io(e)
    }
}

impl From<kideta_core::mmap::error::MmapError> for MmapVectorStorageError {
    fn from(e: kideta_core::mmap::error::MmapError) -> Self {
        MmapVectorStorageError::Io(std::io::Error::other(e.to_string()))
    }
}

pub type Result<T> = std::result::Result<T, MmapVectorStorageError>;

pub struct MmapVectorStorage {
    file: File,
    mmap: Mmap,
    header: VectorStorageHeader,
    dim: u32,
    dtype: VectorDtype,
    capacity: u64,
    max_capacity: u64,
    offset: AtomicU64,
}

unsafe impl Send for MmapVectorStorage {}
unsafe impl Sync for MmapVectorStorage {}

impl MmapVectorStorage {
    pub fn open(
        path: &Path,
        dim: u32,
        dtype: VectorDtype,
    ) -> Result<Self> {
        Self::open_with_max_capacity(path, dim, dtype, 0)
    }

    pub fn open_with_max_capacity(
        path: &Path,
        dim: u32,
        dtype: VectorDtype,
        max_capacity: u64,
    ) -> Result<Self> {
        if dim == 0 {
            return Err(MmapVectorStorageError::ZeroDimension);
        }

        let file_path = path.to_path_buf();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&file_path)?;

        let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);

        let (header, capacity) = if file_size == 0 {
            let capacity = DEFAULT_INITIAL_CAPACITY.min(max_capacity.max(DEFAULT_INITIAL_CAPACITY));
            let header = VectorStorageHeader::new(dim, dtype);
            (header, capacity)
        } else {
            let mut header_bytes = [0u8; 64];
            file.read_exact(&mut header_bytes)?;
            let header = VectorStorageHeader::from_bytes(&header_bytes)
                .ok_or(MmapVectorStorageError::InvalidHeader)?;
            if !header.is_valid() {
                return Err(MmapVectorStorageError::ChecksumMismatch);
            }
            let cap = Self::compute_capacity(header.dimension(), DEFAULT_INITIAL_CAPACITY);
            (header, cap)
        };

        let required_size = VECTOR_DATA_OFFSET + header.byte_size(capacity);
        if file_size < required_size as u64 {
            ftruncate(file.as_raw_fd(), required_size)?;
        }

        let mmap = unsafe {
            MmapOptions::new(required_size)
                .offset(0)
                .mmap_file(&file)?
        };

        Ok(Self {
            file,
            mmap,
            header,
            dim: header.dimension(),
            dtype: header.dtype(),
            capacity,
            max_capacity,
            offset: AtomicU64::new(header.count()),
        })
    }

    pub fn open_readonly(path: &Path) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).open(path)?;

        let mut header_bytes = [0u8; 64];
        file.read_exact(&mut header_bytes)?;
        let header = VectorStorageHeader::from_bytes(&header_bytes)
            .ok_or(MmapVectorStorageError::InvalidHeader)?;
        if !header.is_valid() {
            return Err(MmapVectorStorageError::ChecksumMismatch);
        }

        let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mmap = unsafe { MmapOptions::new(file_size as usize).mmap_file(&file)? };

        Ok(Self {
            file,
            mmap,
            header,
            dim: header.dimension(),
            dtype: header.dtype(),
            capacity: header.count(),
            max_capacity: 0,
            offset: AtomicU64::new(header.count()),
        })
    }

    #[inline]
    pub fn read_vector(
        &self,
        i: usize,
    ) -> Result<Cow<'_, [u8]>> {
        debug_assert!(
            i < self.len(),
            "vector index {} out of bounds (len={})",
            i,
            self.len()
        );

        if i >= self.len() {
            return Err(MmapVectorStorageError::OutOfBounds);
        }

        let byte_offset = VECTOR_DATA_OFFSET + i * self.dtype.byte_size(self.dim);
        let byte_len = self.dtype.byte_size(self.dim);
        let ptr = self.mmap.as_ptr() as *const u8;
        let slice =
            unsafe { std::slice::from_raw_parts(ptr.byte_offset(byte_offset as isize), byte_len) };
        Ok(Cow::Borrowed(slice))
    }

    pub fn read_vector_f32(
        &self,
        i: usize,
    ) -> Result<Vec<f32>> {
        let bytes = self.read_vector(i)?;
        Ok(self.dtype.cast_to_f32(bytes.as_ref(), self.dim))
    }

    pub fn append_vector(
        &mut self,
        vector: &[f32],
    ) -> Result<()> {
        self.append_batch(vector)
    }

    pub fn prefetch(
        &self,
        i: usize,
    ) -> Result<()> {
        self.prefetch_range(i, i + 1)
    }

    pub fn append_batch(
        &mut self,
        vectors: &[f32],
    ) -> Result<()> {
        let n_vectors = vectors.len() / self.dim as usize;
        if !vectors.len().is_multiple_of(self.dim as usize) {
            return Err(MmapVectorStorageError::OutOfBounds);
        }

        let current_offset = self.offset.load(Ordering::Relaxed);
        let new_offset = current_offset + n_vectors as u64;

        if new_offset > self.capacity {
            self.grow(new_offset)?;
        }

        let stored_bytes = self.dtype.convert_f32_to_bytes(vectors);
        let data_offset =
            VECTOR_DATA_OFFSET + current_offset as usize * self.dtype.byte_size(self.dim);
        let byte_len = stored_bytes.len();

        unsafe {
            let slice = self.mmap.as_mut_slice();
            slice[data_offset..data_offset + byte_len].copy_from_slice(&stored_bytes);
        }

        self.offset.store(new_offset, Ordering::Relaxed);
        self.header.increment_count(n_vectors as u64);

        let header_bytes = self.header.as_bytes();
        unsafe {
            let slice = self.mmap.as_mut_slice();
            slice[0..64].copy_from_slice(&header_bytes);
        }

        Ok(())
    }

    fn grow(
        &mut self,
        required: u64,
    ) -> Result<()> {
        let mut new_capacity = self.capacity;
        while new_capacity < required {
            if self.max_capacity > 0 && new_capacity * 2 > self.max_capacity {
                new_capacity = self.max_capacity;
                break;
            }
            new_capacity *= 2;
        }

        if new_capacity <= self.capacity {
            return Err(MmapVectorStorageError::OutOfBounds);
        }

        let new_size = VECTOR_DATA_OFFSET + self.dtype.byte_size(self.dim) * new_capacity as usize;
        ftruncate(self.file.as_raw_fd(), new_size)?;

        self.mmap.remap(new_size)?;

        self.capacity = new_capacity;
        Ok(())
    }

    pub fn prefetch_range(
        &self,
        start: usize,
        end: usize,
    ) -> Result<()> {
        if start >= end || end > self.len() {
            return Ok(());
        }

        let start_offset = VECTOR_DATA_OFFSET + start * self.dtype.byte_size(self.dim);
        let end_offset = VECTOR_DATA_OFFSET + end * self.dtype.byte_size(self.dim);

        let ptr = self.mmap.as_ptr();
        let addr = unsafe { ptr.byte_offset(start_offset as isize) } as *mut libc::c_void;
        let len = end_offset - start_offset;

        let ret = unsafe { libc::madvise(addr, len, libc::MADV_POPULATE_READ) };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EINVAL) {
                self.mmap.advise_willneed()?;
            }
        }

        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        self.mmap
            .flush_range(VECTOR_DATA_OFFSET, self.mmap.len() - VECTOR_DATA_OFFSET)?;
        Ok(())
    }

    pub fn header(&self) -> &VectorStorageHeader {
        &self.header
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.offset.load(Ordering::Relaxed) as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.dim as usize
    }

    #[inline]
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    #[inline]
    pub fn dtype(&self) -> VectorDtype {
        self.dtype
    }

    fn compute_capacity(
        dim: u32,
        initial: u64,
    ) -> u64 {
        let max_vectors: u64 = 1 << 20;
        initial.min(max_vectors / u64::from(dim.max(1)))
    }
}

impl std::fmt::Debug for MmapVectorStorage {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("MmapVectorStorage")
            .field("dim", &self.dim)
            .field("dtype", &self.dtype)
            .field("count", &self.len())
            .field("capacity", &self.capacity)
            .finish()
    }
}

use kideta_core::vector_store::VectorStore;

/// Bridges `MmapVectorStorage` into the abstract [`VectorStore`] trait.
///
/// This enables HNSW, Vamana, and other index types to read vectors
/// directly from the mmap'd file without copying them into RAM.
///
/// # Zero-Copy Access
///
/// For `F32` dtype, [`get_vector`](VectorStore::get_vector) returns a
/// borrowed `&[f32]` pointing directly into the mmap'd region —
/// no per-access allocation or conversion. For other dtypes, it returns
/// `None` and the caller must fall back to `read_vector_f32`.
impl VectorStore for MmapVectorStorage {
    fn len(&self) -> usize {
        self.len()
    }

    fn dimension(&self) -> usize {
        self.dimension()
    }

    fn get_vector(
        &self,
        i: usize,
    ) -> Option<&[f32]> {
        if self.dtype != VectorDtype::F32 {
            return None;
        }
        if i >= self.len() {
            return None;
        }
        let byte_offset = VECTOR_DATA_OFFSET + i * self.dim as usize * 4;
        let ptr = self.mmap.as_ptr() as *const u8;
        let f32_ptr = unsafe { ptr.add(byte_offset) as *const f32 };
        Some(unsafe { std::slice::from_raw_parts(f32_ptr, self.dim as usize) })
    }
}
