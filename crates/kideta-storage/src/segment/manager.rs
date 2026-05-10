use crate::segment::Segment;
use crate::vector_storage::dtype::VectorDtype;
use kideta_core::enums::IndexType;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::Semaphore;

use super::meta::SegmentMeta;

#[derive(Debug, Clone)]
pub struct SegmentManagerConfig {
    pub max_segment_size: usize,
    pub max_age_secs: u64,
    pub index_build_batch_size: usize,
    pub index_build_parallelism: usize,
}

impl Default for SegmentManagerConfig {
    fn default() -> Self {
        Self {
            max_segment_size: 1_000_000,
            max_age_secs: 300,
            index_build_batch_size: 1000,
            index_build_parallelism: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WriteVectorResult {
    pub segment_id: u64,
    pub local_index: u32,
    pub sealed_segment: Option<Arc<Segment>>,
}

pub struct SegmentManager {
    collection_dir: PathBuf,
    dim: u32,
    dtype: VectorDtype,
    index_type: IndexType,
    config: SegmentManagerConfig,
    segments: RwLock<Vec<Arc<Segment>>>,
    growing_segment: RwLock<Option<Arc<Segment>>>,
    next_segment_id: AtomicU64,
    indexer: RwLock<Option<BackgroundIndexer>>,
}

impl SegmentManager {
    pub fn new(
        collection_dir: PathBuf,
        dim: u32,
        dtype: VectorDtype,
        index_type: IndexType,
        config: SegmentManagerConfig,
    ) -> crate::store::Result<Self> {
        if !collection_dir.exists() {
            fs::create_dir_all(&collection_dir).map_err(crate::store::StoreError::Io)?;
        }

        let next_segment_id = AtomicU64::new(0);

        let mut mgr = Self {
            collection_dir,
            dim,
            dtype,
            index_type,
            config,
            segments: RwLock::new(Vec::new()),
            growing_segment: RwLock::new(None),
            next_segment_id,
            indexer: RwLock::new(None),
        };

        mgr.load_existing_segments()?;

        Ok(mgr)
    }

    fn load_existing_segments(&mut self) -> crate::store::Result<()> {
        let mut entries = Vec::new();
        if let Ok(dir_entries) = fs::read_dir(&self.collection_dir) {
            for entry in dir_entries.flatten() {
                let path = entry.path();
                if path.is_dir()
                    && let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && name.starts_with("segment_")
                    && let Ok(meta) = SegmentMeta::load(&path)
                {
                    let seg = Segment::open(meta.id, &path, self.dim, self.dtype, self.index_type)?;
                    entries.push(Arc::new(seg));
                }
            }
        }
        let mut segments = self.segments.write().unwrap();
        *segments = entries;
        Ok(())
    }

    pub fn write_vector(
        &self,
        vector: &[f32],
        payload: Option<Vec<u8>>,
    ) -> crate::store::Result<WriteVectorResult> {
        let growing = {
            let g = self.growing_segment.read().unwrap();
            g.clone()
        };

        let seg = match growing {
            Some(s) => s.clone(),
            None => {
                let seg_id = self
                    .next_segment_id
                    .fetch_add(1, Ordering::SeqCst);
                let dir = self
                    .collection_dir
                    .join(format!("segment_{:016x}", seg_id));
                let payload_cap = 256u32;
                let seg = Segment::open_with_payload_capacity(
                    seg_id,
                    &dir,
                    self.dim,
                    self.dtype,
                    self.index_type,
                    payload_cap,
                )?;
                let seg = Arc::new(seg);
                {
                    let mut g = self.growing_segment.write().unwrap();
                    *g = Some(seg.clone());
                }
                seg
            },
        };

        let local_index = seg.append_vector(vector, payload)?;
        let segment_id = seg.id;

        let seg_len = seg.vector_count() as usize;
        let sealed_segment = if seg_len >= self.config.max_segment_size {
            self.seal_growing()?
        } else {
            None
        };

        Ok(WriteVectorResult {
            segment_id,
            local_index,
            sealed_segment,
        })
    }

    #[deprecated(
        note = "use read_vector_in_segment(segment_id, local_index) instead; flat ID is ambiguous with multiple segments"
    )]
    pub fn read_vector(
        &self,
        _id: u32,
    ) -> crate::store::Result<std::borrow::Cow<'static, [u8]>> {
        Err(crate::store::StoreError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "read_vector(id) is deprecated, use read_vector_in_segment(segment_id, local_index)",
        )))
    }

    pub fn read_vector_in_segment(
        &self,
        segment_id: u64,
        local_index: u32,
    ) -> crate::store::Result<std::borrow::Cow<'static, [u8]>> {
        if let Some(ref growing) = *self.growing_segment.read().unwrap()
            && growing.id == segment_id
        {
            if !growing.deleted_bitmap().get(local_index) {
                return growing
                    .read_vector(local_index)
                    .map(|cow| std::borrow::Cow::Owned(cow.into_owned()))
                    .map_err(|e| {
                        crate::store::StoreError::Io(std::io::Error::other(e.to_string()))
                    });
            }
            return Err(crate::store::StoreError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("vector {} not found in segment {}", local_index, segment_id),
            )));
        }

        let segments = self.segments.read().unwrap();
        for seg in segments.iter() {
            if seg.id == segment_id
                && local_index < seg.vector_count() as u32
                && !seg.deleted_bitmap().get(local_index)
            {
                return seg
                    .read_vector(local_index)
                    .map(|cow| std::borrow::Cow::Owned(cow.into_owned()))
                    .map_err(|e| {
                        crate::store::StoreError::Io(std::io::Error::other(e.to_string()))
                    });
            }
            if seg.id == segment_id {
                break;
            }
        }

        Err(crate::store::StoreError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("vector {} not found in segment {}", local_index, segment_id),
        )))
    }

    #[deprecated(
        note = "use delete_vector_in_segment(segment_id, local_index) instead; flat ID is ambiguous with multiple segments"
    )]
    pub fn delete_vector(
        &self,
        _id: u32,
    ) -> crate::store::Result<()> {
        Err(crate::store::StoreError::Io(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "delete_vector(id) is deprecated, use delete_vector_in_segment(segment_id, local_index)",
        )))
    }

    pub fn delete_vector_in_segment(
        &self,
        segment_id: u64,
        local_index: u32,
    ) -> crate::store::Result<()> {
        if let Some(ref growing) = *self.growing_segment.read().unwrap()
            && growing.id == segment_id
        {
            return growing.delete(local_index);
        }

        let segments = self.segments.read().unwrap();
        for seg in segments.iter() {
            if seg.id == segment_id && local_index < seg.vector_count() as u32 {
                return seg.delete(local_index);
            }
            if seg.id == segment_id {
                break;
            }
        }

        Ok(())
    }

    pub fn seal_growing(&self) -> crate::store::Result<Option<Arc<Segment>>> {
        let growing = {
            let mut g = self.growing_segment.write().unwrap();
            g.take()
        };

        match growing {
            Some(seg) => {
                seg.seal()?;

                {
                    let mut segments = self.segments.write().unwrap();
                    segments.push(seg.clone());
                }

                self.trigger_index_build(seg.clone())?;

                Ok(Some(seg))
            },
            None => Ok(None),
        }
    }

    fn trigger_index_build(
        &self,
        seg: Arc<Segment>,
    ) -> crate::store::Result<()> {
        if let Some(ref indexer) = *self.indexer.read().unwrap() {
            indexer.schedule(seg);
        }
        Ok(())
    }

    pub fn start_indexer(&self) {
        let semaphore = Arc::new(Semaphore::new(self.config.index_build_parallelism));
        let indexer = BackgroundIndexer::new(
            self.collection_dir.clone(),
            semaphore,
            self.config.index_build_batch_size,
        );
        *self.indexer.write().unwrap() = Some(indexer);
    }

    pub fn stop_indexer(&self) {
        *self.indexer.write().unwrap() = None;
    }

    pub fn segments(&self) -> Vec<Arc<Segment>> {
        let mut all = Vec::new();
        if let Some(ref g) = *self.growing_segment.read().unwrap() {
            all.push(g.clone());
        }
        all.extend(self.segments.read().unwrap().clone());
        all
    }

    pub fn sealed_segments(&self) -> Vec<Arc<Segment>> {
        self.segments
            .read()
            .unwrap()
            .iter()
            .filter(|s| s.state().is_sealed())
            .cloned()
            .collect()
    }

    pub fn indexed_segments(&self) -> Vec<Arc<Segment>> {
        self.segments
            .read()
            .unwrap()
            .iter()
            .filter(|s| s.state().is_indexed())
            .cloned()
            .collect()
    }

    pub fn total_vector_count(&self) -> u64 {
        let mut total = 0u64;
        for seg in self.segments() {
            total += seg.vector_count();
        }
        total
    }

    pub fn total_deleted_count(&self) -> u64 {
        let mut total = 0u64;
        for seg in self.segments() {
            total += seg.deleted_count();
        }
        total
    }

    pub fn flush_all(&self) -> crate::store::Result<()> {
        if let Some(ref g) = *self.growing_segment.read().unwrap() {
            g.flush()?;
        }
        for seg in self.segments.read().unwrap().iter() {
            seg.flush()?;
        }
        Ok(())
    }

    pub fn add_segment(
        &self,
        seg: Arc<Segment>,
    ) -> crate::store::Result<()> {
        let mut segments = self.segments.write().unwrap();
        if !segments.iter().any(|s| s.id == seg.id) {
            segments.push(seg);
        }
        Ok(())
    }

    pub fn remove_segments(
        &self,
        ids: &[u64],
    ) -> crate::store::Result<()> {
        let mut segments = self.segments.write().unwrap();
        segments.retain(|s| !ids.contains(&s.id));
        Ok(())
    }

    pub fn segment_dir(
        &self,
        segment_id: u64,
    ) -> PathBuf {
        self.collection_dir
            .join(format!("segment_{:016x}", segment_id))
    }

    pub fn max_segment_id(&self) -> Option<u64> {
        let segments = self.segments.read().unwrap();
        let sealed_max = segments.iter().map(|s| s.id).max();
        let growing_id = self
            .growing_segment
            .read()
            .unwrap()
            .as_ref()
            .map(|s| s.id);
        sealed_max.into_iter().chain(growing_id).max()
    }

    pub fn reload_segments(&self) -> crate::store::Result<()> {
        let mut segments = self.segments.write().unwrap();
        segments.clear();
        {
            let mut g = self.growing_segment.write().unwrap();
            *g = None;
        }
        let mut entries = Vec::new();
        if let Ok(dir_entries) = fs::read_dir(&self.collection_dir) {
            for entry in dir_entries.flatten() {
                let path = entry.path();
                if path.is_dir()
                    && let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && name.starts_with("segment_")
                    && let Ok(meta) = super::meta::SegmentMeta::load(&path)
                    && let Ok(seg) =
                        Segment::open(meta.id, &path, self.dim, self.dtype, self.index_type)
                {
                    entries.push(Arc::new(seg));
                }
            }
        }
        *segments = entries;
        Ok(())
    }
}

pub struct BackgroundIndexer {
    #[allow(dead_code)]
    collection_dir: PathBuf,
    #[allow(dead_code)]
    semaphore: Arc<Semaphore>,
    #[allow(dead_code)]
    batch_size: usize,
    queue: Arc<RwLock<Vec<Arc<Segment>>>>,
    handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    progress_percent: Arc<AtomicU64>,
    eta_seconds: Arc<AtomicU64>,
    vectors_indexed: Arc<AtomicU64>,
    vectors_total: Arc<AtomicU64>,
    current_segment: RwLock<Option<String>>,
    is_indexing: Arc<AtomicU64>,
}

impl BackgroundIndexer {
    pub fn new(
        collection_dir: PathBuf,
        semaphore: Arc<Semaphore>,
        batch_size: usize,
    ) -> Self {
        let queue = Arc::new(RwLock::new(Vec::new()));
        let q = queue.clone();

        let progress = Arc::new(AtomicU64::new(0));
        let eta = Arc::new(AtomicU64::new(0));
        let indexed = Arc::new(AtomicU64::new(0));
        let total = Arc::new(AtomicU64::new(0));
        let indexing = Arc::new(AtomicU64::new(0));

        let prog = progress.clone();
        let eta_s = eta.clone();
        let idx = indexed.clone();
        let _tot = total.clone();
        let ind = indexing.clone();
        let sem = semaphore.clone();

        let handle = std::thread::spawn(move || {
            loop {
                let _seg = {
                    let mut q = q.write().unwrap();
                    if q.is_empty() {
                        std::thread::sleep(Duration::from_millis(100));
                        continue;
                    }
                    ind.store(0, Ordering::Relaxed);
                    q.remove(0)
                };
                let _permit = sem.acquire();
                prog.store(100, Ordering::Relaxed);
                eta_s.store(0, Ordering::Relaxed);
                idx.fetch_add(1, Ordering::Relaxed);
            }
        });

        Self {
            collection_dir,
            semaphore,
            batch_size,
            queue,
            handle: std::sync::Mutex::new(Some(handle)),
            progress_percent: progress,
            eta_seconds: eta,
            vectors_indexed: indexed,
            vectors_total: total,
            current_segment: RwLock::new(None),
            is_indexing: indexing,
        }
    }

    pub fn schedule(
        &self,
        seg: Arc<Segment>,
    ) {
        let seg_id = seg.id;
        let mut q = self.queue.write().unwrap();
        if !q.iter().any(|s| s.id == seg_id) {
            q.push(seg);
        }
    }

    pub fn get_progress(&self) -> (u64, u64, u64, u64, Option<String>) {
        let progress = self.progress_percent.load(Ordering::Relaxed);
        let eta = self.eta_seconds.load(Ordering::Relaxed);
        let indexed = self.vectors_indexed.load(Ordering::Relaxed);
        let total = self.vectors_total.load(Ordering::Relaxed);
        let segment = self.current_segment.read().unwrap().clone();
        (progress, eta, indexed, total, segment)
    }

    pub fn is_indexing(&self) -> bool {
        self.is_indexing.load(Ordering::Relaxed) == 1
    }

    pub fn set_progress(
        &self,
        percent: u64,
        eta: u64,
        indexed: u64,
        total: u64,
    ) {
        self.progress_percent
            .store(percent, Ordering::Relaxed);
        self.eta_seconds.store(eta, Ordering::Relaxed);
        self.vectors_indexed
            .store(indexed, Ordering::Relaxed);
        self.vectors_total.store(total, Ordering::Relaxed);
    }
}

impl Drop for BackgroundIndexer {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.lock().unwrap().take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn segment_manager_write_and_read() {
        let dir = tempdir().unwrap();
        let config = SegmentManagerConfig::default();
        let mgr = SegmentManager::new(
            dir.path().to_path_buf(),
            128,
            VectorDtype::F32,
            IndexType::Hnsw,
            config,
        )
        .unwrap();

        let result = mgr.write_vector(&vec![0.1; 128], None).unwrap();
        assert_eq!(result.local_index, 0);

        let vector = mgr
            .read_vector_in_segment(result.segment_id, result.local_index)
            .unwrap();
        assert_eq!(vector.len(), 128 * 4);
    }

    #[test]
    fn segment_manager_delete() {
        let dir = tempdir().unwrap();
        let config = SegmentManagerConfig::default();
        let mgr = SegmentManager::new(
            dir.path().to_path_buf(),
            128,
            VectorDtype::F32,
            IndexType::Hnsw,
            config,
        )
        .unwrap();

        let result = mgr.write_vector(&vec![0.1; 128], None).unwrap();
        mgr.delete_vector_in_segment(result.segment_id, result.local_index)
            .unwrap();
        let delete_result = mgr.read_vector_in_segment(result.segment_id, result.local_index);
        assert!(delete_result.is_err());
    }

    #[test]
    fn segment_manager_seal() {
        let dir = tempdir().unwrap();
        let config = SegmentManagerConfig {
            max_segment_size: 10,
            ..Default::default()
        };
        let mgr = SegmentManager::new(
            dir.path().to_path_buf(),
            128,
            VectorDtype::F32,
            IndexType::Hnsw,
            config,
        )
        .unwrap();

        for i in 0..12 {
            mgr.write_vector(&vec![0.1 + i as f32; 128], None)
                .unwrap();
        }

        let sealed = mgr.sealed_segments();
        assert!(!sealed.is_empty());
    }

    #[test]
    fn test_segment_manager_add_and_remove() {
        let dir = tempdir().unwrap();
        let config = SegmentManagerConfig::default();
        let mgr = SegmentManager::new(
            dir.path().to_path_buf(),
            128,
            VectorDtype::F32,
            IndexType::Hnsw,
            config,
        )
        .unwrap();

        let seg_dir = dir.path().join("segment_00000000000000ff");
        let seg = Segment::open_with_payload_capacity(
            0xff,
            &seg_dir,
            128,
            VectorDtype::F32,
            IndexType::Hnsw,
            256,
        )
        .unwrap();
        seg.seal().unwrap();
        let seg = Arc::new(seg);

        mgr.add_segment(seg.clone()).unwrap();
        assert!(mgr.sealed_segments().iter().any(|s| s.id == 0xff));

        mgr.remove_segments(&[0xff]).unwrap();
        assert!(!mgr.sealed_segments().iter().any(|s| s.id == 0xff));
    }
}
