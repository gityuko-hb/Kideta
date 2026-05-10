use crate::store::KidetaStore;
use crate::vector_storage::dtype::VectorDtype;
use crate::vector_storage::{MmapVectorStorage, MmapVectorStorageError};
use kideta_core::enums::IndexType;
use std::borrow::Cow;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use super::delta::{DeltaLogReader, DeltaLogWriter};
use super::meta::SegmentMeta;
use super::state::SegmentState;

pub struct DeletedBitmap {
    bits: RwLock<Vec<u64>>,
    #[allow(dead_code)]
    max_id: u32,
}

impl DeletedBitmap {
    pub fn new(max_id: u32) -> Self {
        let num_words = ((max_id as usize) / 64) + 1;
        Self {
            bits: RwLock::new(vec![0u64; num_words]),
            max_id,
        }
    }

    pub fn open(path: &PathBuf) -> std::io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let file_size = file.metadata()?.len();
        let num_words = if file_size == 0 {
            1024
        } else {
            (file_size as usize / 8).max(1024)
        };

        let mut bits = vec![0u64; num_words];
        if file_size > 0 {
            file.read_exact(unsafe {
                std::slice::from_raw_parts_mut(bits.as_mut_ptr() as *mut u8, file_size as usize)
            })?;
        }

        Ok(Self {
            bits: RwLock::new(bits),
            max_id: (num_words * 64) as u32,
        })
    }

    pub fn set(
        &self,
        id: u32,
    ) {
        let word = id as usize / 64;
        let bit = id as usize % 64;
        let mut bits = self.bits.write().unwrap();
        if word < bits.len() {
            bits[word] |= 1u64 << bit;
        }
    }

    pub fn clear(
        &self,
        id: u32,
    ) {
        let word = id as usize / 64;
        let bit = id as usize % 64;
        let mut bits = self.bits.write().unwrap();
        if word < bits.len() {
            bits[word] &= !(1u64 << bit);
        }
    }

    pub fn get(
        &self,
        id: u32,
    ) -> bool {
        let word = id as usize / 64;
        let bit = id as usize % 64;
        let bits = self.bits.read().unwrap();
        word < bits.len() && (bits[word] & (1u64 << bit)) != 0
    }

    pub fn to_vec(&self) -> Vec<u32> {
        let bits = self.bits.read().unwrap();
        let mut ids = Vec::new();
        for (word_idx, &word) in bits.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros();
                let id = (word_idx * 64 + bit as usize) as u32;
                ids.push(id);
                w &= w - 1;
            }
        }
        ids
    }

    pub fn flush(
        &self,
        path: &PathBuf,
    ) -> std::io::Result<()> {
        let bits = self.bits.read().unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let bytes =
            unsafe { std::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len() * 8) };
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }
}

pub struct Segment {
    pub id: u64,
    dir: PathBuf,
    state: RwLock<SegmentState>,
    vector_storage: RwLock<MmapVectorStorage>,
    payload_store: RwLock<KidetaStore>,
    deleted_bitmap: DeletedBitmap,
    delta_writer: RwLock<Option<DeltaLogWriter>>,
    meta: RwLock<SegmentMeta>,
}

impl Segment {
    pub fn open(
        id: u64,
        dir: &PathBuf,
        dim: u32,
        dtype: VectorDtype,
        index_type: IndexType,
    ) -> crate::store::Result<Self> {
        Self::open_with_config(id, dir, dim, dtype, index_type, 256, 0)
    }

    pub fn open_with_payload_capacity(
        id: u64,
        dir: &PathBuf,
        dim: u32,
        dtype: VectorDtype,
        index_type: IndexType,
        payload_capacity: u32,
    ) -> crate::store::Result<Self> {
        // 0 = unlimited (vectors grow dynamically via MmapVectorStorage doubling)
        let max_vectors = 0u64;
        Self::open_with_config(
            id,
            dir,
            dim,
            dtype,
            index_type,
            payload_capacity,
            max_vectors,
        )
    }

    pub fn open_with_config(
        id: u64,
        dir: &PathBuf,
        dim: u32,
        dtype: VectorDtype,
        index_type: IndexType,
        payload_capacity: u32,
        max_vector_capacity: u64,
    ) -> crate::store::Result<Self> {
        if !dir.exists() {
            fs::create_dir_all(dir).map_err(crate::store::StoreError::Io)?;
        }

        let vector_path = dir.join("vectors.bin");
        let vector_storage = MmapVectorStorage::open_with_max_capacity(
            &vector_path,
            dim,
            dtype,
            max_vector_capacity,
        )
        .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;

        let payload_path = dir.join("payload.store");
        let actual_capacity = Self::detect_payload_capacity(&payload_path, payload_capacity);
        let payload_store = KidetaStore::open(&payload_path, actual_capacity)?;

        let deleted_path = dir.join("deleted.bitmap");
        let deleted_bitmap =
            DeletedBitmap::open(&deleted_path).map_err(crate::store::StoreError::Io)?;

        let delta_path = dir.join("delta.log");
        let delta_writer = if delta_path.exists() {
            DeltaLogReader::open(&delta_path)
                .ok()
                .and_then(|_| DeltaLogWriter::open(&delta_path).ok())
        } else {
            DeltaLogWriter::open(&delta_path).ok()
        };

        let delta_deleted: Vec<u32> = DeltaLogReader::open(&delta_path)
            .ok()
            .and_then(|mut r| r.replay().ok())
            .unwrap_or_default();
        for id in delta_deleted {
            deleted_bitmap.set(id);
        }

        let meta = if dir.join("meta.json").exists() {
            SegmentMeta::load(dir)
                .unwrap_or_else(|_| SegmentMeta::new(id, dir.clone(), 0, index_type))
        } else {
            SegmentMeta::new(id, dir.clone(), 0, index_type)
        };

        let vector_count = vector_storage.len() as u64;
        let mut meta = meta;
        meta.vector_count = vector_count;

        let state = if vector_storage.is_empty() && !dir.join("meta.json").exists() {
            SegmentState::Growing
        } else if vector_storage.is_empty() {
            SegmentState::Open
        } else {
            SegmentState::Sealed
        };

        Ok(Self {
            id,
            dir: dir.clone(),
            state: RwLock::new(state),
            vector_storage: RwLock::new(vector_storage),
            payload_store: RwLock::new(payload_store),
            deleted_bitmap,
            delta_writer: RwLock::new(delta_writer),
            meta: RwLock::new(meta),
        })
    }

    pub fn append_vector(
        &self,
        vector: &[f32],
        payload: Option<Vec<u8>>,
    ) -> crate::store::Result<u32> {
        {
            let state = self.state.read().unwrap();
            if !state.is_writable() {
                return Err(crate::store::StoreError::Io(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    format!("segment {:?} is not writable", *state),
                )));
            }
        }

        let idx = self.vector_storage.read().unwrap().len() as u32;
        self.vector_storage
            .write()
            .unwrap()
            .append_vector(vector)
            .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;

        if let Some(data) = payload
            && !data.is_empty()
            && data != b"{}"
        {
            self.payload_store
                .write()
                .unwrap()
                .put(idx, &data)?;
        }

        {
            let mut meta = self.meta.write().unwrap();
            meta.vector_count += 1;
        }

        Ok(idx)
    }

    pub fn read_vector(
        &self,
        id: u32,
    ) -> std::result::Result<Cow<'static, [u8]>, MmapVectorStorageError> {
        if self.deleted_bitmap.get(id) {
            return Err(MmapVectorStorageError::OutOfBounds);
        }
        let guard = self.vector_storage.read().unwrap();
        let result = guard.read_vector(id as usize)?;
        Ok(Cow::Owned(result.into_owned()))
    }

    pub fn read_vector_f32(
        &self,
        id: u32,
    ) -> std::result::Result<Vec<f32>, MmapVectorStorageError> {
        self.vector_storage
            .read()
            .unwrap()
            .read_vector_f32(id as usize)
    }

    pub fn read_payload(
        &self,
        local_id: u32,
    ) -> Option<Vec<u8>> {
        self.payload_store
            .read()
            .ok()
            .and_then(|store| store.get(local_id).ok().flatten())
    }

    pub fn delete(
        &self,
        id: u32,
    ) -> crate::store::Result<()> {
        {
            let state = self.state.read().unwrap();
            if state.is_sealed()
                && let Some(ref mut writer) = *self.delta_writer.write().unwrap()
            {
                writer
                    .append(id)
                    .map_err(crate::store::StoreError::Io)?;
            }
        }

        self.deleted_bitmap.set(id);
        {
            let mut meta = self.meta.write().unwrap();
            meta.deleted_count += 1;
        }
        Ok(())
    }

    pub fn may_exist(
        &self,
        id: u32,
    ) -> bool {
        let meta = self.meta.read().unwrap();
        if !meta.may_exist(id) {
            return false;
        }
        drop(meta);
        !self.deleted_bitmap.get(id)
    }

    pub fn iter_vectors(&self) -> Vec<(u32, Cow<'static, [u8]>)> {
        let storage = self.vector_storage.read().unwrap();
        let len = storage.len() as u32;
        (0..len)
            .filter_map(|id| {
                if self.deleted_bitmap.get(id) {
                    return None;
                }
                storage
                    .read_vector(id as usize)
                    .ok()
                    .map(|bytes| (id, Cow::Owned(bytes.into_owned())))
            })
            .collect()
    }

    pub fn flush(&self) -> crate::store::Result<()> {
        self.vector_storage
            .read()
            .unwrap()
            .flush()
            .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;

        {
            let mut payload = self.payload_store.write().unwrap();
            payload.flush()?;
        }

        self.deleted_bitmap
            .flush(&self.dir.join("deleted.bitmap"))
            .map_err(crate::store::StoreError::Io)?;

        if let Some(ref mut writer) = *self.delta_writer.write().unwrap() {
            writer
                .flush()
                .map_err(crate::store::StoreError::Io)?;
        }

        let mut meta = self.meta.write().unwrap();
        meta.file_size_bytes = self.estimate_size();
        meta.save(&self.dir)
            .map_err(crate::store::StoreError::Io)?;

        Ok(())
    }

    pub fn seal(&self) -> crate::store::Result<()> {
        {
            let mut state = self.state.write().unwrap();
            state
                .transition_to(SegmentState::Flushing)
                .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;
            *state = SegmentState::Flushing;
        }

        self.flush()?;

        {
            let mut state = self.state.write().unwrap();
            state
                .transition_to(SegmentState::Sealed)
                .map_err(|e| crate::store::StoreError::Io(std::io::Error::other(e.to_string())))?;
            *state = SegmentState::Sealed;
        }

        let ids: Vec<u32> = (0..self.vector_storage.read().unwrap().len() as u32).collect();
        {
            let mut meta = self.meta.write().unwrap();
            *meta = meta.clone().with_bloom(&ids);
        }

        Ok(())
    }

    pub fn set_indexed(&self) -> crate::store::Result<()> {
        {
            let mut state = self.state.write().unwrap();
            if *state != SegmentState::Sealed {
                return Err(crate::store::StoreError::Io(std::io::Error::other(
                    "can only transition Sealed -> Indexed",
                )));
            }
            *state = SegmentState::Indexed;
        }

        {
            let mut meta = self.meta.write().unwrap();
            meta.index_ready = true;
            meta.state = "Indexed".to_string();
            meta.save(&self.dir)
                .map_err(crate::store::StoreError::Io)?;
        }

        Ok(())
    }

    pub fn state(&self) -> SegmentState {
        *self.state.read().unwrap()
    }

    pub fn meta(&self) -> SegmentMeta {
        self.meta.read().unwrap().clone()
    }

    pub fn vector_count(&self) -> u64 {
        self.meta.read().unwrap().vector_count
    }

    pub fn deleted_count(&self) -> u64 {
        self.meta.read().unwrap().deleted_count
    }

    pub fn deleted_bitmap(&self) -> &DeletedBitmap {
        &self.deleted_bitmap
    }

    pub fn dimension(&self) -> usize {
        self.vector_storage.read().unwrap().dimension()
    }

    pub fn dtype(&self) -> VectorDtype {
        self.vector_storage.read().unwrap().dtype()
    }

    pub fn dir(&self) -> &PathBuf {
        &self.dir
    }

    fn detect_payload_capacity(
        payload_path: &Path,
        default_capacity: u32,
    ) -> u32 {
        let tracker_path = payload_path.join("tracker.dat");
        if tracker_path.exists()
            && let Ok(meta) = std::fs::metadata(&tracker_path)
        {
            let entry_size = std::mem::size_of::<crate::store::block::BlockPtr>();
            let cap = (meta.len() as usize / entry_size) as u32;
            if cap > 0 {
                return cap;
            }
        }
        default_capacity
    }

    fn estimate_size(&self) -> u64 {
        let vs = self.vector_storage.read().unwrap();
        let vector_size = vs.len() as u64 * vs.dtype().byte_size(vs.dimension() as u32) as u64;
        let payload_size = self.payload_store.read().unwrap().len() as u64 * 256;
        let bitmap_size = self.deleted_bitmap.to_vec().len() as u64 * 4;
        vector_size + payload_size + bitmap_size + 4096
    }
}

impl std::fmt::Debug for Segment {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("Segment")
            .field("id", &self.id)
            .field("state", &self.state())
            .field("vector_count", &self.vector_count())
            .field("deleted_count", &self.deleted_count())
            .finish()
    }
}
