//! IvfIndex — Inverted File Index with k-means clustering.
//!
//! IVF partitions vectors into `num_clusters` clusters using k-means.
//! Search scans only `nprobe` nearest clusters instead of the full dataset,
//! providing approximate nearest-neighbor search with O(n/k + k*nprobe) complexity
//! vs O(n) for brute force.

use std::io::{Read, Write};

use kideta_core::distance::{Metric, get_distance_fn, prefetch};
use kideta_core::metric::DistanceMetric;
use kideta_core::payload::Payload;
use kideta_core::types::VectorId;
use kideta_core::utils::heap::{BoundedMaxHeap, MaxHeap};

use crate::ivf::params::IvfParams;
use crate::ivf::trainer::IvfTrainer;
use crate::quantization::config::PqConfig;
use crate::quantization::pq::{PqOps, PqTrainer};
use crate::quantization::{QuantizationConfig, QuantizedStorage, Sq8Stats, encode_vectors};
use crate::search_params::SearchParams;
use crate::traits::{FilterFn, IndexError, ScoredVectorId, VectorIndex};

const IVF_MAGIC: [u8; 4] = *b"IVFX";
const IVF_VERSION: u32 = 2;
const IVF_MIN_SUPPORTED_VERSION: u32 = 1;

pub struct IvfIndex {
    dimension: usize,
    vectors: Vec<f32>,
    ids: Vec<VectorId>,
    payloads: Vec<Payload>,
    centroids: Vec<f32>,
    inverted_lists: Vec<Vec<VectorId>>,
    deleted: Vec<bool>,
    params: IvfParams,
    metric: DistanceMetric,
    distance_fn: fn(&[f32], &[f32]) -> f32,
    quantization_config: Option<QuantizationConfig>,
    quantized_storage: Option<QuantizedStorage>,
    is_built: bool,
    residuals: Option<Vec<f32>>,
    pq_config: Option<PqConfig>,
    pq_codes: Option<Vec<u8>>,
}

impl IvfIndex {
    pub fn new(
        dimension: usize,
        params: IvfParams,
        metric: DistanceMetric,
    ) -> Self {
        let distance_fn = match metric {
            DistanceMetric::Cosine => get_distance_fn(Metric::Cosine),
            DistanceMetric::L2 => get_distance_fn(Metric::L2),
            DistanceMetric::DotProduct => get_distance_fn(Metric::Dot),
            DistanceMetric::Hamming => {
                // Hamming operates on binary vectors - use wrapper from kideta-core
                get_distance_fn(Metric::Hamming)
            },
        };

        Self {
            dimension,
            vectors: Vec::new(),
            ids: Vec::new(),
            payloads: Vec::new(),
            centroids: Vec::new(),
            inverted_lists: vec![],
            deleted: Vec::new(),
            params,
            metric,
            distance_fn,
            quantization_config: None,
            quantized_storage: None,
            is_built: false,
            residuals: None,
            pq_config: None,
            pq_codes: None,
        }
    }

    #[inline]
    fn vector_slice(
        &self,
        local_id: usize,
    ) -> Option<&[f32]> {
        if local_id >= self.vectors.len() / self.dimension {
            return None;
        }
        let offset = local_id * self.dimension;
        Some(&self.vectors[offset..offset + self.dimension])
    }

    #[inline]
    fn centroid_slice(
        &self,
        cluster: usize,
    ) -> Option<&[f32]> {
        let k = self.centroids.len() / self.dimension;
        if cluster >= k {
            return None;
        }
        let offset = cluster * self.dimension;
        Some(&self.centroids[offset..offset + self.dimension])
    }

    pub fn build(
        &mut self,
        ids: Vec<VectorId>,
        vectors: Vec<f32>,
        payloads: Vec<Payload>,
        quantization_config: Option<QuantizationConfig>,
    ) -> Result<(), IndexError> {
        if vectors.is_empty() {
            return Err(IndexError::EmptyIndex);
        }

        let n = vectors.len() / self.dimension;

        if ids.len() != n || payloads.len() != n {
            return Err(IndexError::Internal("ids/payloads length mismatch".into()));
        }

        let mut seen = std::collections::HashSet::with_capacity(ids.len());
        for &id in &ids {
            if !seen.insert(id) {
                return Err(IndexError::DuplicateId(id));
            }
        }

        self.vectors = vectors;
        self.ids = ids;
        self.payloads = payloads;
        self.deleted = vec![false; n];

        let k = self.params.num_clusters.min(n);
        self.inverted_lists = vec![Vec::new(); k];

        let vector_slices: Vec<&[f32]> = (0..n)
            .map(|i| {
                let offset = i * self.dimension;
                &self.vectors[offset..offset + self.dimension]
            })
            .collect();

        let trainer = IvfTrainer::new(self.params, self.dimension);
        let (centroids, assignments) = trainer
            .train(&vector_slices)
            .map_err(|e| IndexError::Internal(format!("train error: {:?}", e)))?;

        self.centroids = centroids;

        for (vid, &cluster) in assignments.iter().enumerate() {
            if cluster < k {
                self.inverted_lists[cluster].push(VectorId::new(vid as u64));
            }
        }

        if let Some(bytes_per_subvec) = self.params.pq_bytes_per_subvec {
            self.train_pq_on_residuals(&assignments, bytes_per_subvec)?;
        }

        if let Some(ref config) = quantization_config
            && !config.is_none()
        {
            self.set_quantization_config(config.clone())?;
        }

        self.is_built = true;
        Ok(())
    }

    fn train_pq_on_residuals(
        &mut self,
        assignments: &[usize],
        bytes_per_subvec: usize,
    ) -> Result<(), IndexError> {
        if self.vectors.is_empty() {
            return Err(IndexError::EmptyIndex);
        }

        let n = self.vectors.len() / self.dimension;

        let residuals = self.compute_residuals(assignments);
        self.residuals = Some(residuals);

        let residual_slices: Vec<&[f32]> = (0..n)
            .filter(|&i| !self.deleted[i])
            .map(|i| {
                let offset = i * self.dimension;
                self.residuals.as_ref().unwrap()[offset..offset + self.dimension].into()
            })
            .collect();

        let pq_trainer = PqTrainer::new(bytes_per_subvec, 256, 50);
        let pq_config = pq_trainer.train(&residual_slices);
        self.pq_config = Some(pq_config);

        let mut pq_codes = vec![0u8; n * self.pq_config.as_ref().unwrap().num_subspaces];
        for i in 0..n {
            if self.deleted[i] {
                continue;
            }
            let offset = i * self.dimension;
            let residual = &self.residuals.as_ref().unwrap()[offset..offset + self.dimension];
            let code_offset = i * self.pq_config.as_ref().unwrap().num_subspaces;
            let code_slice = &mut pq_codes
                [code_offset..code_offset + self.pq_config.as_ref().unwrap().num_subspaces];
            PqOps::encode_to_slice(self.pq_config.as_ref().unwrap(), residual, code_slice);
        }
        self.pq_codes = Some(pq_codes);

        Ok(())
    }

    fn compute_residuals(
        &self,
        assignments: &[usize],
    ) -> Vec<f32> {
        let n = self.vectors.len() / self.dimension;
        let mut residuals = vec![0.0_f32; n * self.dimension];

        for (i, &cluster) in assignments.iter().enumerate() {
            let v_offset = i * self.dimension;
            let r_offset = i * self.dimension;

            if let Some(centroid) = self.centroid_slice(cluster) {
                for j in 0..self.dimension {
                    residuals[r_offset + j] = self.vectors[v_offset + j] - centroid[j];
                }
            }
        }

        residuals
    }

    pub fn insert(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError> {
        if vector.len() != self.dimension {
            return Err(IndexError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        if self.ids.contains(&id) {
            return Err(IndexError::DuplicateId(id));
        }

        let local_id = self.vectors.len() / self.dimension;

        self.vectors.extend_from_slice(vector);
        self.ids.push(id);
        self.payloads.push(payload);
        self.deleted.push(false);

        if !self.centroids.is_empty() {
            let cluster = self.assign_to_cluster(vector);
            if cluster < self.inverted_lists.len() {
                self.inverted_lists[cluster].push(VectorId::new(local_id as u64));
            }
        }

        Ok(())
    }

    fn assign_to_cluster(
        &self,
        vector: &[f32],
    ) -> usize {
        let k = self.centroids.len() / self.dimension;
        let mut best_cluster = 0;
        let mut best_dist = f32::MAX;

        for ki in 0..k {
            if let Some(centroid) = self.centroid_slice(ki) {
                let dist = (self.distance_fn)(vector, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = ki;
                }
            }
        }

        best_cluster
    }

    pub fn delete(
        &mut self,
        id: VectorId,
    ) -> Result<(), IndexError> {
        let local_id = self
            .ids
            .iter()
            .position(|&x| x == id)
            .ok_or(IndexError::NotFound(id))?;
        self.deleted[local_id] = true;
        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<ScoredVectorId> {
        if query.len() != self.dimension || !self.is_built {
            return Vec::new();
        }

        if self.is_empty() {
            return Vec::new();
        }

        let rf = if self.pq_config.is_some() {
            4usize
        } else {
            1
        };
        let heap_k = k * rf;
        let k_clusters = self
            .params
            .num_clusters
            .min(self.inverted_lists.len());

        let centroid_dists = self.compute_all_centroid_dists(query);

        let top_clusters = self.select_top_nprobe(&centroid_dists, self.params.nprobe, k_clusters);

        let mut results = BoundedMaxHeap::new(heap_k);

        for &cluster in &top_clusters {
            if let Some(centroid) = self.centroid_slice(cluster) {
                let adc_cache = self.pq_config.as_ref().map(|cfg| {
                    let qr: Vec<f32> = query
                        .iter()
                        .zip(centroid.iter())
                        .map(|(q, c)| q - c)
                        .collect();
                    PqOps::build_adc_tables(cfg, &qr)
                });

                for (j, &vid) in self.inverted_lists[cluster].iter().enumerate() {
                    let lid = vid.as_u64() as usize;
                    if lid >= self.deleted.len() || self.deleted[lid] {
                        continue;
                    }

                    // Prefetch only for full-precision path
                    if self.pq_config.is_none() && self.quantized_storage.is_none() {
                        let pf_j = j + 4;
                        if pf_j < self.inverted_lists[cluster].len() {
                            let pf_lid = self.inverted_lists[cluster][pf_j].as_u64() as usize;
                            let pf_off = pf_lid * self.dimension;
                            prefetch(&self.vectors[pf_off] as *const f32);
                        }
                    }

                    let dist = if let (Some(tables), Some(cfg), Some(codes)) = (
                        adc_cache.as_ref(),
                        self.pq_config.as_ref(),
                        self.pq_codes.as_ref(),
                    ) {
                        let code_offset = lid * cfg.num_subspaces;
                        let code = &codes[code_offset..code_offset + cfg.num_subspaces];
                        PqOps::approx_l2_with_tables(cfg, tables, code)
                    } else if self.quantized_storage.is_some() {
                        self.approx_distance(lid, query, centroid_dists[cluster])
                    } else {
                        self.full_precision_distance(lid, query)
                    };

                    results.push(ScoredVectorId::new(self.ids[lid], dist));
                }
            }
        }

        let mut sorted = results.into_sorted_vec();
        sorted.reverse();

        if rf > 1 && sorted.len() > k {
            let ids: Vec<VectorId> = sorted.iter().take(heap_k).map(|r| r.id).collect();
            IvfIndex::rescore(self, &ids, query, rf)
                .into_iter()
                .take(k)
                .collect()
        } else {
            sorted
        }
    }

    fn full_precision_distance(
        &self,
        local_id: usize,
        query: &[f32],
    ) -> f32 {
        if let Some(vector) = self.vector_slice(local_id) {
            (self.distance_fn)(query, vector)
        } else {
            f32::MAX
        }
    }

    fn approx_distance(
        &self,
        local_id: usize,
        query: &[f32],
        _centroid_dist: f32,
    ) -> f32 {
        if let Some(storage) = &self.quantized_storage {
            storage
                .approx_distance(query, local_id)
                .unwrap_or(f32::MAX)
        } else {
            self.full_precision_distance(local_id, query)
        }
    }

    fn compute_all_centroid_dists(
        &self,
        query: &[f32],
    ) -> Vec<f32> {
        let k = self.centroids.len() / self.dimension;
        let mut dists = Vec::with_capacity(k);

        for ki in 0..k {
            if let Some(centroid) = self.centroid_slice(ki) {
                dists.push((self.distance_fn)(query, centroid));
            }
        }

        dists
    }

    fn select_top_nprobe(
        &self,
        dists: &[f32],
        nprobe: usize,
        k: usize,
    ) -> Vec<usize> {
        let n = nprobe.min(dists.len().min(k));

        let mut heap: MaxHeap<(f32, usize)> = MaxHeap::with_capacity(n);

        for (i, &dist) in dists.iter().enumerate().take(k) {
            if heap.len() < n {
                heap.push((dist, i));
            } else if let Some(&(worst_dist, _)) = heap.peek()
                && dist < worst_dist
            {
                heap.pop();
                heap.push((dist, i));
            }
        }

        let mut result: Vec<usize> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|(_, i)| i)
            .collect();
        result.sort_by(|&a, &b| {
            dists[a]
                .partial_cmp(&dists[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    fn should_fallback_filtered_to_full_scan(
        &self,
        nprobe: usize,
    ) -> bool {
        let nlist = self.inverted_lists.len().max(1);
        nprobe * 10 > nlist * 3
    }

    pub fn search_with_filter<F: Fn(VectorId, &Payload) -> bool + Send + Sync>(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&F>,
    ) -> Vec<ScoredVectorId> {
        if query.len() != self.dimension || !self.is_built {
            return Vec::new();
        }

        if self.is_empty() {
            return Vec::new();
        }

        if self.should_fallback_filtered_to_full_scan(self.params.nprobe) {
            let mut results = BoundedMaxHeap::new(k);
            for local_id in 0..self.ids.len() {
                if local_id >= self.deleted.len() || self.deleted[local_id] {
                    continue;
                }
                if let Some(f) = filter
                    && !f(self.ids[local_id], &self.payloads[local_id])
                {
                    continue;
                }
                let dist = self.full_precision_distance(local_id, query);
                results.push(ScoredVectorId::new(self.ids[local_id], dist));
            }
            let mut sorted = results.into_sorted_vec();
            sorted.reverse();
            return sorted;
        }

        let k_clusters = self
            .params
            .num_clusters
            .min(self.inverted_lists.len());
        let centroid_dists = self.compute_all_centroid_dists(query);
        let top_clusters = self.select_top_nprobe(&centroid_dists, self.params.nprobe, k_clusters);

        let mut results = BoundedMaxHeap::new(k);

        for &cluster in &top_clusters {
            if let Some(centroid) = self.centroid_slice(cluster) {
                let adc_cache = self.pq_config.as_ref().map(|cfg| {
                    let qr: Vec<f32> = query
                        .iter()
                        .zip(centroid.iter())
                        .map(|(q, c)| q - c)
                        .collect();
                    PqOps::build_adc_tables(cfg, &qr)
                });

                for (j, &vid) in self.inverted_lists[cluster].iter().enumerate() {
                    let lid = vid.as_u64() as usize;
                    if lid >= self.deleted.len() || self.deleted[lid] {
                        continue;
                    }

                    if let Some(f) = filter
                        && !f(self.ids[lid], &self.payloads[lid])
                    {
                        continue;
                    }

                    // Prefetch only for full-precision path
                    if self.pq_config.is_none() && self.quantized_storage.is_none() {
                        let pf_j = j + 4;
                        if pf_j < self.inverted_lists[cluster].len() {
                            let pf_lid = self.inverted_lists[cluster][pf_j].as_u64() as usize;
                            let pf_off = pf_lid * self.dimension;
                            prefetch(&self.vectors[pf_off] as *const f32);
                        }
                    }

                    let dist = if let (Some(tables), Some(cfg), Some(codes)) = (
                        adc_cache.as_ref(),
                        self.pq_config.as_ref(),
                        self.pq_codes.as_ref(),
                    ) {
                        let code_offset = lid * cfg.num_subspaces;
                        let code = &codes[code_offset..code_offset + cfg.num_subspaces];
                        PqOps::approx_l2_with_tables(cfg, tables, code)
                    } else if self.quantized_storage.is_some() {
                        self.approx_distance(lid, query, centroid_dists[cluster])
                    } else {
                        self.full_precision_distance(lid, query)
                    };

                    results.push(ScoredVectorId::new(self.ids[lid], dist));
                }
            }
        }

        let mut sorted = results.into_sorted_vec();
        sorted.reverse();
        sorted
    }

    fn search_with_nprobe(
        &self,
        query: &[f32],
        k: usize,
        nprobe_override: usize,
    ) -> Vec<ScoredVectorId> {
        if query.len() != self.dimension || !self.is_built {
            return Vec::new();
        }

        if self.is_empty() {
            return Vec::new();
        }

        let k_clusters = self
            .params
            .num_clusters
            .min(self.inverted_lists.len());

        let centroid_dists = self.compute_all_centroid_dists(query);

        let top_clusters = self.select_top_nprobe(&centroid_dists, nprobe_override, k_clusters);

        let mut results = BoundedMaxHeap::new(k);

        for &cluster in &top_clusters {
            if let Some(centroid) = self.centroid_slice(cluster) {
                let adc_cache = self.pq_config.as_ref().map(|cfg| {
                    let qr: Vec<f32> = query
                        .iter()
                        .zip(centroid.iter())
                        .map(|(q, c)| q - c)
                        .collect();
                    PqOps::build_adc_tables(cfg, &qr)
                });

                for (j, &vid) in self.inverted_lists[cluster].iter().enumerate() {
                    let lid = vid.as_u64() as usize;
                    if lid >= self.deleted.len() || self.deleted[lid] {
                        continue;
                    }

                    // Prefetch only for full-precision path
                    if self.pq_config.is_none() && self.quantized_storage.is_none() {
                        let pf_j = j + 4;
                        if pf_j < self.inverted_lists[cluster].len() {
                            let pf_lid = self.inverted_lists[cluster][pf_j].as_u64() as usize;
                            let pf_off = pf_lid * self.dimension;
                            prefetch(&self.vectors[pf_off] as *const f32);
                        }
                    }

                    let dist = if let (Some(tables), Some(cfg), Some(codes)) = (
                        adc_cache.as_ref(),
                        self.pq_config.as_ref(),
                        self.pq_codes.as_ref(),
                    ) {
                        let code_offset = lid * cfg.num_subspaces;
                        let code = &codes[code_offset..code_offset + cfg.num_subspaces];
                        PqOps::approx_l2_with_tables(cfg, tables, code)
                    } else if self.quantized_storage.is_some() {
                        self.approx_distance(lid, query, centroid_dists[cluster])
                    } else {
                        self.full_precision_distance(lid, query)
                    };

                    results.push(ScoredVectorId::new(self.ids[lid], dist));
                }
            }
        }

        let mut sorted = results.into_sorted_vec();
        sorted.reverse();
        sorted
    }

    fn search_with_filter_and_nprobe<F: Fn(VectorId, &Payload) -> bool + Send + Sync>(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&F>,
        nprobe_override: usize,
    ) -> Vec<ScoredVectorId> {
        if query.len() != self.dimension || !self.is_built {
            return Vec::new();
        }

        if self.is_empty() {
            return Vec::new();
        }

        let k_clusters = self
            .params
            .num_clusters
            .min(self.inverted_lists.len());
        let centroid_dists = self.compute_all_centroid_dists(query);
        let top_clusters = self.select_top_nprobe(&centroid_dists, nprobe_override, k_clusters);

        let mut results = BoundedMaxHeap::new(k);

        for &cluster in &top_clusters {
            if let Some(centroid) = self.centroid_slice(cluster) {
                let adc_cache = self.pq_config.as_ref().map(|cfg| {
                    let qr: Vec<f32> = query
                        .iter()
                        .zip(centroid.iter())
                        .map(|(q, c)| q - c)
                        .collect();
                    PqOps::build_adc_tables(cfg, &qr)
                });

                for (j, &vid) in self.inverted_lists[cluster].iter().enumerate() {
                    let lid = vid.as_u64() as usize;
                    if lid >= self.deleted.len() || self.deleted[lid] {
                        continue;
                    }

                    if let Some(f) = filter
                        && !f(self.ids[lid], &self.payloads[lid])
                    {
                        continue;
                    }

                    // Prefetch only for full-precision path
                    if self.pq_config.is_none() && self.quantized_storage.is_none() {
                        let pf_j = j + 4;
                        if pf_j < self.inverted_lists[cluster].len() {
                            let pf_lid = self.inverted_lists[cluster][pf_j].as_u64() as usize;
                            let pf_off = pf_lid * self.dimension;
                            prefetch(&self.vectors[pf_off] as *const f32);
                        }
                    }

                    let dist = if let (Some(tables), Some(cfg), Some(codes)) = (
                        adc_cache.as_ref(),
                        self.pq_config.as_ref(),
                        self.pq_codes.as_ref(),
                    ) {
                        let code_offset = lid * cfg.num_subspaces;
                        let code = &codes[code_offset..code_offset + cfg.num_subspaces];
                        PqOps::approx_l2_with_tables(cfg, tables, code)
                    } else if self.quantized_storage.is_some() {
                        self.approx_distance(lid, query, centroid_dists[cluster])
                    } else {
                        self.full_precision_distance(lid, query)
                    };

                    results.push(ScoredVectorId::new(self.ids[lid], dist));
                }
            }
        }

        let mut sorted = results.into_sorted_vec();
        sorted.reverse();
        sorted
    }

    pub fn train_quantization(
        &mut self,
        config: &QuantizationConfig,
    ) -> Result<(), IndexError> {
        if !self.is_built || self.vectors.is_empty() {
            return Err(IndexError::IndexNotReady);
        }

        let dim = self.dimension;
        config
            .dimension()
            .and_then(|d| {
                if d == dim {
                    Some(())
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                IndexError::Quantization(format!(
                    "dimension mismatch: index={}, config={:?}",
                    dim,
                    config.dimension()
                ))
            })?;

        let n = self.vectors.len() / dim;
        let all_vectors: Vec<&[f32]> = (0..n)
            .filter(|&i| !self.deleted[i])
            .map(|i| {
                let offset = i * dim;
                &self.vectors[offset..offset + dim]
            })
            .collect();

        let storage = encode_vectors(config, &all_vectors);
        self.quantized_storage = Some(storage);
        self.quantization_config = Some(config.clone());

        Ok(())
    }

    pub fn set_quantization_config(
        &mut self,
        config: QuantizationConfig,
    ) -> Result<(), IndexError> {
        if config.is_none() {
            self.quantization_config = None;
            self.quantized_storage = None;
            return Ok(());
        }

        self.train_quantization(&config)
    }

    pub fn train_sq8(&mut self) -> Result<(), IndexError> {
        if !self.is_built || self.vectors.is_empty() {
            return Err(IndexError::IndexNotReady);
        }

        let n = self.vectors.len() / self.dimension;
        let all_vectors: Vec<&[f32]> = (0..n)
            .filter(|&i| !self.deleted[i])
            .map(|i| {
                let offset = i * self.dimension;
                &self.vectors[offset..offset + self.dimension]
            })
            .collect();

        let stats = Sq8Stats::compute(&all_vectors);
        let config = QuantizationConfig::Sq8(stats.into_config());
        self.train_quantization(&config)
    }

    pub fn rescore(
        &self,
        candidates: &[VectorId],
        query: &[f32],
        rescore_factor: usize,
    ) -> Vec<ScoredVectorId> {
        if candidates.is_empty() || query.len() != self.dimension {
            return Vec::new();
        }

        let target = candidates
            .len()
            .min(candidates.len() * rescore_factor.max(1));
        let mut all_candidates = Vec::with_capacity(target);

        for &vid in candidates.iter().take(target) {
            if let Some(local_id) = self.ids.iter().position(|&x| x == vid) {
                if local_id >= self.deleted.len() || self.deleted[local_id] {
                    continue;
                }

                let dist = if self.pq_config.is_some() && self.residuals.is_some() {
                    let cluster = self.assign_to_cluster_for_id(local_id);
                    if let Some(centroid) = self.centroid_slice(cluster) {
                        if let (Some(residuals), Some(pq_config)) =
                            (&self.residuals, &self.pq_config)
                        {
                            let code_offset = local_id * pq_config.num_subspaces;
                            let _code = &self.pq_codes.as_ref().unwrap()
                                [code_offset..code_offset + pq_config.num_subspaces];

                            let residual = &residuals[local_id * self.dimension
                                ..local_id * self.dimension + self.dimension];
                            let reconstructed: Vec<f32> = residual
                                .iter()
                                .zip(centroid.iter())
                                .map(|(r, c)| r + c)
                                .collect();

                            (self.distance_fn)(query, &reconstructed)
                        } else {
                            self.full_precision_distance(local_id, query)
                        }
                    } else {
                        self.full_precision_distance(local_id, query)
                    }
                } else {
                    self.full_precision_distance(local_id, query)
                };

                all_candidates.push(ScoredVectorId::new(vid, dist));
            }
        }

        all_candidates.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_candidates
    }

    fn assign_to_cluster_for_id(
        &self,
        local_id: usize,
    ) -> usize {
        let vector = match self.vector_slice(local_id) {
            Some(v) => v,
            None => return 0,
        };
        self.assign_to_cluster(vector)
    }

    pub fn is_ivf_pq(&self) -> bool {
        self.pq_config.is_some()
    }

    pub fn quantization_config(&self) -> Option<&QuantizationConfig> {
        self.quantization_config.as_ref()
    }

    pub fn quantized_storage(&self) -> Option<&QuantizedStorage> {
        self.quantized_storage.as_ref()
    }

    pub fn len(&self) -> usize {
        self.ids.len() - self.deleted.iter().filter(|&&d| d).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    pub fn size_bytes(&self) -> usize {
        let vector_bytes = self.vectors.capacity() * std::mem::size_of::<f32>();
        let id_bytes = self.ids.capacity() * std::mem::size_of::<VectorId>();
        let payload_bytes = self.payloads.capacity() * std::mem::size_of::<Payload>();
        let centroid_bytes = self.centroids.capacity() * std::mem::size_of::<f32>();
        let deleted_bytes = self.deleted.capacity() * std::mem::size_of::<bool>();
        let list_overhead: usize = self
            .inverted_lists
            .iter()
            .map(|l| l.capacity() * std::mem::size_of::<VectorId>())
            .sum();
        let quant_bytes = self
            .quantized_storage
            .as_ref()
            .map(|s| s.total_bytes())
            .unwrap_or(0);
        let residual_bytes = self
            .residuals
            .as_ref()
            .map(|r| r.capacity() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        let pq_code_bytes = self
            .pq_codes
            .as_ref()
            .map(|c| c.capacity() * std::mem::size_of::<u8>())
            .unwrap_or(0);
        let pq_config_bytes = self
            .pq_config
            .as_ref()
            .map(|c| c.codebook.capacity() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        vector_bytes
            + id_bytes
            + payload_bytes
            + centroid_bytes
            + deleted_bytes
            + list_overhead
            + quant_bytes
            + residual_bytes
            + pq_code_bytes
            + pq_config_bytes
    }

    pub fn num_clusters(&self) -> usize {
        self.inverted_lists.len()
    }

    pub fn avg_cluster_size(&self) -> f64 {
        let k = self.inverted_lists.len();
        if k == 0 {
            return 0.0;
        }
        let total: usize = self.inverted_lists.iter().map(|l| l.len()).sum();
        total as f64 / k as f64
    }

    pub fn serialize<W: Write>(
        &self,
        writer: &mut W,
    ) -> Result<(), IndexError> {
        writer
            .write_all(&IVF_MAGIC)
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        writer
            .write_all(&IVF_VERSION.to_le_bytes())
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        let params_bytes =
            serde_json::to_string(&self.params).map_err(|e| IndexError::Internal(e.to_string()))?;
        let params_len = params_bytes.len() as u32;
        writer
            .write_all(&params_len.to_le_bytes())
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        writer
            .write_all(params_bytes.as_bytes())
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        let metric_byte = match self.metric {
            DistanceMetric::Cosine => 0u8,
            DistanceMetric::L2 => 1u8,
            DistanceMetric::DotProduct => 2u8,
            DistanceMetric::Hamming => 3u8,
        };
        writer
            .write_all(&[metric_byte])
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        writer
            .write_all(&(self.dimension as u32).to_le_bytes())
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        let n = self.vectors.len() / self.dimension;
        writer
            .write_all(&(n as u32).to_le_bytes())
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        writer
            .write_all(&[self.is_built as u8])
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        let vectors_bytes: Vec<u8> = self
            .vectors
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        writer
            .write_all(&vectors_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        for &id in &self.ids {
            writer
                .write_all(&id.as_u64().to_le_bytes())
                .map_err(|e| IndexError::Internal(e.to_string()))?;
        }

        let payload_bytes =
            serde_json::to_vec(&self.payloads).map_err(|e| IndexError::Internal(e.to_string()))?;
        let payload_len = payload_bytes.len() as u32;
        writer
            .write_all(&payload_len.to_le_bytes())
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        writer
            .write_all(&payload_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        let centroids_bytes: Vec<u8> = self
            .centroids
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        writer
            .write_all(&centroids_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        for list in &self.inverted_lists {
            let list_len = list.len() as u32;
            writer
                .write_all(&list_len.to_le_bytes())
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            for &vid in list {
                writer
                    .write_all(&vid.as_u64().to_le_bytes())
                    .map_err(|e| IndexError::Internal(e.to_string()))?;
            }
        }

        let deleted_bytes: Vec<u8> = self.deleted.iter().map(|&b| b as u8).collect();
        writer
            .write_all(&deleted_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        let quant_config_json = serde_json::to_string(&self.quantization_config)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let qc_len = quant_config_json.len() as u32;
        writer
            .write_all(&qc_len.to_le_bytes())
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        writer
            .write_all(quant_config_json.as_bytes())
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        if let Some(ref storage) = self.quantized_storage {
            let storage_bytes = storage.as_codes().to_vec();
            let storage_len = storage_bytes.len() as u32;
            writer
                .write_all(&storage_len.to_le_bytes())
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            writer
                .write_all(&storage_bytes)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
        } else {
            writer
                .write_all(&(0u32).to_le_bytes())
                .map_err(|e| IndexError::Internal(e.to_string()))?;
        }

        if let Some(ref config) = self.pq_config {
            let config_bytes = config.serialize();
            let config_len = u32::try_from(config_bytes.len())
                .map_err(|_| IndexError::Internal("PQ config too large to serialize".into()))?;
            writer
                .write_all(&config_len.to_le_bytes())
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            writer
                .write_all(&config_bytes)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
        } else {
            writer
                .write_all(&(0u32).to_le_bytes())
                .map_err(|e| IndexError::Internal(e.to_string()))?;
        }

        if let Some(ref codes) = self.pq_codes {
            let codes_len = u64::try_from(codes.len())
                .map_err(|_| IndexError::Internal("PQ codes too large to serialize".into()))?;
            writer
                .write_all(&codes_len.to_le_bytes())
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            writer
                .write_all(codes)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
        } else {
            writer
                .write_all(&0u64.to_le_bytes())
                .map_err(|e| IndexError::Internal(e.to_string()))?;
        }

        Ok(())
    }

    pub fn deserialize<R: Read>(reader: &mut R) -> Result<Self, IndexError> {
        let mut magic = [0u8; 4];
        reader
            .read_exact(&mut magic)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        if magic != IVF_MAGIC {
            return Err(IndexError::Internal("invalid IVF magic bytes".into()));
        }

        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let version = u32::from_le_bytes(version_bytes);
        if !(IVF_MIN_SUPPORTED_VERSION..=IVF_VERSION).contains(&version) {
            return Err(IndexError::Internal(format!(
                "unsupported IVF version: {}",
                version
            )));
        }

        let mut params_len_bytes = [0u8; 4];
        reader
            .read_exact(&mut params_len_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let params_len = u32::from_le_bytes(params_len_bytes) as usize;
        let mut params_buf = vec![0u8; params_len];
        reader
            .read_exact(&mut params_buf)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let params_str =
            String::from_utf8(params_buf).map_err(|e| IndexError::Internal(e.to_string()))?;
        let params: IvfParams =
            serde_json::from_str(&params_str).map_err(|e| IndexError::Internal(e.to_string()))?;

        let mut metric_byte = [0u8; 1];
        reader
            .read_exact(&mut metric_byte)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let metric = match metric_byte[0] {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::L2,
            2 => DistanceMetric::DotProduct,
            3 => DistanceMetric::Hamming,
            _ => return Err(IndexError::Internal("invalid metric byte".into())),
        };

        let mut dim_bytes = [0u8; 4];
        reader
            .read_exact(&mut dim_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let dimension = u32::from_le_bytes(dim_bytes) as usize;

        let mut n_bytes = [0u8; 4];
        reader
            .read_exact(&mut n_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let n = u32::from_le_bytes(n_bytes) as usize;

        let mut built_byte = [0u8; 1];
        reader
            .read_exact(&mut built_byte)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let is_built = built_byte[0] != 0;

        let vectors_byte_count = n * dimension * 4;
        let mut vectors_raw = vec![0u8; vectors_byte_count];
        reader
            .read_exact(&mut vectors_raw)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let vectors: Vec<f32> = vectors_raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let mut id_bytes = [0u8; 8];
            reader
                .read_exact(&mut id_bytes)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            ids.push(VectorId::new(u64::from_le_bytes(id_bytes)));
        }

        let mut payload_len_bytes = [0u8; 4];
        reader
            .read_exact(&mut payload_len_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let payload_len = u32::from_le_bytes(payload_len_bytes) as usize;
        let mut payload_buf = vec![0u8; payload_len];
        reader
            .read_exact(&mut payload_buf)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let payloads: Vec<Payload> = serde_json::from_slice(&payload_buf)
            .map_err(|e| IndexError::Internal(e.to_string()))?;

        let k = params.num_clusters.min(n.max(1));
        let centroids_byte_count = k * dimension * 4;
        let mut centroids_raw = vec![0u8; centroids_byte_count];
        reader
            .read_exact(&mut centroids_raw)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let centroids: Vec<f32> = centroids_raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let mut inverted_lists = Vec::with_capacity(k);
        for _ in 0..k {
            let mut list_len_bytes = [0u8; 4];
            reader
                .read_exact(&mut list_len_bytes)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            let list_len = u32::from_le_bytes(list_len_bytes) as usize;

            let mut list = Vec::with_capacity(list_len);
            for _ in 0..list_len {
                let mut vid_bytes = [0u8; 8];
                reader
                    .read_exact(&mut vid_bytes)
                    .map_err(|e| IndexError::Internal(e.to_string()))?;
                list.push(VectorId::new(u64::from_le_bytes(vid_bytes)));
            }
            inverted_lists.push(list);
        }

        let mut deleted = vec![false; n];
        for del in deleted.iter_mut() {
            let mut byte = [0u8; 1];
            reader
                .read_exact(&mut byte)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            *del = byte[0] != 0;
        }

        let mut qc_len_bytes = [0u8; 4];
        reader
            .read_exact(&mut qc_len_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let qc_len = u32::from_le_bytes(qc_len_bytes) as usize;

        let quantization_config = if qc_len > 0 {
            let mut qc_buf = vec![0u8; qc_len];
            reader
                .read_exact(&mut qc_buf)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            let qc: Option<QuantizationConfig> =
                serde_json::from_slice(&qc_buf).map_err(|e| IndexError::Internal(e.to_string()))?;
            qc
        } else {
            None
        };

        let mut storage_len_bytes = [0u8; 4];
        reader
            .read_exact(&mut storage_len_bytes)
            .map_err(|e| IndexError::Internal(e.to_string()))?;
        let storage_len = u32::from_le_bytes(storage_len_bytes) as usize;

        let quantized_storage = if storage_len > 0 {
            let mut storage_buf = vec![0u8; storage_len];
            reader
                .read_exact(&mut storage_buf)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            quantization_config
                .as_ref()
                .map(|config| QuantizedStorage::new(storage_buf, config, n))
        } else {
            None
        };

        let (pq_config, pq_codes) = if version >= 2 {
            let mut pq_config_len_bytes = [0u8; 4];
            reader
                .read_exact(&mut pq_config_len_bytes)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            let pq_config_len = u32::from_le_bytes(pq_config_len_bytes) as usize;
            let pq_config = if pq_config_len > 0 {
                let mut pq_config_buf = vec![0u8; pq_config_len];
                reader
                    .read_exact(&mut pq_config_buf)
                    .map_err(|e| IndexError::Internal(e.to_string()))?;
                Some(
                    PqConfig::deserialize(&pq_config_buf)
                        .map_err(|e| IndexError::Internal(e.to_string()))?,
                )
            } else {
                None
            };

            let mut pq_codes_len_bytes = [0u8; 8];
            reader
                .read_exact(&mut pq_codes_len_bytes)
                .map_err(|e| IndexError::Internal(e.to_string()))?;
            let pq_codes_len = u64::from_le_bytes(pq_codes_len_bytes) as usize;
            let pq_codes = if pq_codes_len > 0 {
                let mut pq_codes_buf = vec![0u8; pq_codes_len];
                reader
                    .read_exact(&mut pq_codes_buf)
                    .map_err(|e| IndexError::Internal(e.to_string()))?;
                Some(pq_codes_buf)
            } else {
                None
            };

            if let (Some(config), Some(codes)) = (&pq_config, &pq_codes) {
                let expected = n.saturating_mul(config.num_subspaces);
                if codes.len() != expected {
                    return Err(IndexError::Internal(format!(
                        "invalid IVF-PQ code length: expected {}, got {}",
                        expected,
                        codes.len()
                    )));
                }
            }

            (pq_config, pq_codes)
        } else {
            (None, None)
        };

        let residuals = pq_config.as_ref().map(|_| {
            Self::recompute_residuals_from_lists(
                dimension,
                n,
                &vectors,
                &centroids,
                &inverted_lists,
            )
        });

        let distance_fn = match metric {
            DistanceMetric::Cosine => get_distance_fn(Metric::Cosine),
            DistanceMetric::L2 => get_distance_fn(Metric::L2),
            DistanceMetric::DotProduct => get_distance_fn(Metric::Dot),
            DistanceMetric::Hamming => get_distance_fn(Metric::Cosine),
        };

        Ok(Self {
            dimension,
            vectors,
            ids,
            payloads,
            centroids,
            inverted_lists,
            deleted,
            params,
            metric,
            distance_fn,
            quantization_config,
            quantized_storage,
            is_built,
            residuals,
            pq_config,
            pq_codes,
        })
    }

    fn recompute_residuals_from_lists(
        dimension: usize,
        n: usize,
        vectors: &[f32],
        centroids: &[f32],
        inverted_lists: &[Vec<VectorId>],
    ) -> Vec<f32> {
        let mut residuals = vec![0.0_f32; n * dimension];

        for (cluster, list) in inverted_lists.iter().enumerate() {
            let centroid_offset = cluster * dimension;
            if centroid_offset + dimension > centroids.len() {
                continue;
            }
            let centroid = &centroids[centroid_offset..centroid_offset + dimension];

            for &vid in list {
                let local_id = vid.as_u64() as usize;
                if local_id >= n {
                    continue;
                }
                let vector_offset = local_id * dimension;
                for j in 0..dimension {
                    residuals[vector_offset + j] = vectors[vector_offset + j] - centroid[j];
                }
            }
        }

        residuals
    }
}

impl VectorIndex for IvfIndex {
    fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<ScoredVectorId> {
        IvfIndex::search(self, query, k)
    }

    fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&FilterFn>,
    ) -> Vec<ScoredVectorId> {
        match filter {
            Some(f) => {
                let filter_fn = |id: VectorId, payload: &Payload| f(id, payload);
                IvfIndex::search_with_filter(self, query, k, Some(&filter_fn))
            },
            None => IvfIndex::search_with_filter(
                self,
                query,
                k,
                None::<&fn(VectorId, &Payload) -> bool>,
            ),
        }
    }

    fn search_with_params(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Vec<ScoredVectorId> {
        match params {
            SearchParams::Ivf(p) => IvfIndex::search_with_nprobe(self, query, k, p.nprobe),
            SearchParams::IvfPq(p) => {
                let rf = p.rescore_factor.max(1);
                let candidates = IvfIndex::search_with_nprobe(self, query, k * rf, p.nprobe);
                if rf > 1 && !candidates.is_empty() {
                    let ids: Vec<VectorId> = candidates.iter().map(|r| r.id).collect();
                    IvfIndex::rescore(self, &ids, query, rf)
                        .into_iter()
                        .take(k)
                        .collect()
                } else {
                    candidates.into_iter().take(k).collect()
                }
            },
            _ => IvfIndex::search(self, query, k),
        }
    }

    fn search_with_params_and_filter(
        &self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        filter: Option<&FilterFn>,
    ) -> Vec<ScoredVectorId> {
        match (params, filter) {
            (SearchParams::Ivf(p), Some(f)) => {
                let filter_fn = |id: VectorId, payload: &Payload| f(id, payload);
                IvfIndex::search_with_filter_and_nprobe(self, query, k, Some(&filter_fn), p.nprobe)
            },
            (SearchParams::Ivf(p), None) => IvfIndex::search_with_nprobe(self, query, k, p.nprobe),
            (SearchParams::IvfPq(p), Some(f)) => {
                let rf = p.rescore_factor.max(1);
                let filter_fn = |id: VectorId, payload: &Payload| f(id, payload);
                let candidates = IvfIndex::search_with_filter_and_nprobe(
                    self,
                    query,
                    k * rf,
                    Some(&filter_fn),
                    p.nprobe,
                );
                if rf > 1 && !candidates.is_empty() {
                    let ids: Vec<VectorId> = candidates.iter().map(|r| r.id).collect();
                    IvfIndex::rescore(self, &ids, query, rf)
                        .into_iter()
                        .take(k)
                        .collect()
                } else {
                    candidates.into_iter().take(k).collect()
                }
            },
            (SearchParams::IvfPq(p), None) => {
                let rf = p.rescore_factor.max(1);
                let candidates = IvfIndex::search_with_nprobe(self, query, k * rf, p.nprobe);
                if rf > 1 && !candidates.is_empty() {
                    let ids: Vec<VectorId> = candidates.iter().map(|r| r.id).collect();
                    IvfIndex::rescore(self, &ids, query, rf)
                        .into_iter()
                        .take(k)
                        .collect()
                } else {
                    candidates.into_iter().take(k).collect()
                }
            },
            _ => VectorIndex::search_with_params_and_filter(self, query, k, params, filter),
        }
    }

    fn insert(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError> {
        IvfIndex::insert(self, id, vector, payload)
    }

    fn delete(
        &mut self,
        id: VectorId,
    ) -> Result<(), IndexError> {
        IvfIndex::delete(self, id)
    }

    fn update(
        &mut self,
        id: VectorId,
        vector: &[f32],
        payload: Payload,
    ) -> Result<(), IndexError> {
        IvfIndex::delete(self, id)?;
        IvfIndex::insert(self, id, vector, payload)
    }

    fn len(&self) -> usize {
        IvfIndex::len(self)
    }

    fn is_empty(&self) -> bool {
        IvfIndex::is_empty(self)
    }

    fn dimension(&self) -> usize {
        IvfIndex::dimension(self)
    }

    fn metric(&self) -> DistanceMetric {
        IvfIndex::metric(self)
    }

    fn size_bytes(&self) -> usize {
        IvfIndex::size_bytes(self)
    }

    fn quantization_config(&self) -> Option<&QuantizationConfig> {
        IvfIndex::quantization_config(self)
    }

    fn quantized_storage(&self) -> Option<&QuantizedStorage> {
        IvfIndex::quantized_storage(self)
    }
}

impl std::fmt::Debug for IvfIndex {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("IvfIndex")
            .field("dimension", &self.dimension)
            .field("total_vectors", &self.ids.len())
            .field("active_vectors", &self.len())
            .field("num_clusters", &self.inverted_lists.len())
            .field("avg_cluster_size", &self.avg_cluster_size())
            .field("metric", &self.metric)
            .field("size_bytes", &self.size_bytes())
            .field("is_built", &self.is_built)
            .field("quantized", &self.quantization_config.is_some())
            .finish()
    }
}
