//! HNSW binary serialization format.
//!
//! # File Format (Version 2)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │ MAGIC: 0x57534E48 ("HNSW")                    (4 bytes) │
//! │ VERSION = 2                                   (4 bytes) │
//! ├─────────────────────────────────────────────────────────┤
//! │ PARAMS (varint encoded)                                 │
//! │   m, ef_construction, ef_search, ml, max_level,         │
//! │   dimension                                             │
//! ├─────────────────────────────────────────────────────────┤
//! │ ENTRY POINT (varint)                                    │
//! │   node_id, level                                        │
//! ├─────────────────────────────────────────────────────────┤
//! │ NUM_ELEMENTS (varint)                                   │
//! │   actual number of nodes                                │  
//! ├─────────────────────────────────────────────────────────┤
//! │ VECTORS                                                 │
//! │   [f32; dimension * num_elements]                       │
//! ├─────────────────────────────────────────────────────────┤
//! │ LAYERS                                                  │
//! │   For each layer L (0 to num_layers-1):                 │
//! │     actual_nodes_in_layer                      (varint) │
//! │     For each actual node:                               │
//! │       node_id                                  (varint) │
//! │       neighbor_count                           (varint) │
//! │       [neighbor_ids...]                     (varint ea) │
//! ├─────────────────────────────────────────────────────────┤
//! │ CHECKSUM (xxhash3)                           (16 bytes) │
//! └─────────────────────────────────────────────────────────┘
//! ```

use std::io::{Read, Write};

use kideta_core::utils::hash::xxhash3_128;
use kideta_core::utils::varint::{MAX_VARINT_LEN64, decode_u32, encode_u32};

use crate::hnsw::graph::HnswGraph;
use crate::hnsw::params::HnswParams;

const MAGIC: u32 = 0x57534E48;
const VERSION: u32 = 2;
const CHECKSUM_SIZE: usize = 16;

pub struct Serializer;

pub struct Deserializer;

fn as_bytes<T: Sized>(slice: &[T]) -> &[u8] {
    let ptr = slice.as_ptr() as *const u8;
    let len = std::mem::size_of_val(slice);
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

impl Serializer {
    pub fn serialize<W: Write>(
        &self,
        graph: &HnswGraph,
        params: &HnswParams,
        mut w: W,
    ) -> Result<(), SerialError> {
        w.write_all(&MAGIC.to_le_bytes())?;
        w.write_all(&VERSION.to_le_bytes())?;

        let mut data_buf = Vec::new();

        {
            let mut varint_buf = [0u8; MAX_VARINT_LEN64];
            let n = encode_u32(params.m as u32, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
            let n = encode_u32(params.ef_construction as u32, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
            let n = encode_u32(params.ef_search as u32, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
        }

        data_buf.write_all(&params.ml.to_le_bytes())?;

        {
            let mut varint_buf = [0u8; MAX_VARINT_LEN64];
            let n = encode_u32(params.max_level as u32, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
            let n = encode_u32(graph.dimension() as u32, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
        }

        if let Some(ep) = graph.get_entry_point() {
            let mut varint_buf = [0u8; MAX_VARINT_LEN64];
            let n = encode_u32(ep.node_id as u32, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
            let n = encode_u32(ep.level as u32, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
        } else {
            let mut varint_buf = [0u8; MAX_VARINT_LEN64];
            let n = encode_u32(u32::MAX, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
            let n = encode_u32(0u32, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
        }

        let num_elements = graph.len() as u32;
        {
            let mut varint_buf = [0u8; MAX_VARINT_LEN64];
            let n = encode_u32(num_elements, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;
        }

        for i in 0..graph.len() {
            if let Some(v) = graph.get_vector(i) {
                data_buf.write_all(as_bytes(v))?;
            }
        }

        let mut varint_buf = [0u8; MAX_VARINT_LEN64];
        let actual_num_layers = graph.layers_ref().len() as u32;
        let n = encode_u32(actual_num_layers, &mut varint_buf);
        data_buf.write_all(&varint_buf[..n])?;

        for layer in graph.layers_ref() {
            let active_nodes: Vec<usize> = layer.iter_active_nodes().collect();
            let actual_count = active_nodes.len() as u32;
            let n = encode_u32(actual_count, &mut varint_buf);
            data_buf.write_all(&varint_buf[..n])?;

            for node_id in &active_nodes {
                let n = encode_u32(*node_id as u32, &mut varint_buf);
                data_buf.write_all(&varint_buf[..n])?;

                let neighbor_vec = &layer.neighbors[*node_id];
                let count = neighbor_vec.len() as u32;
                let n = encode_u32(count, &mut varint_buf);
                data_buf.write_all(&varint_buf[..n])?;
                for &neighbor_id in neighbor_vec.iter() {
                    let n = encode_u32(neighbor_id, &mut varint_buf);
                    data_buf.write_all(&varint_buf[..n])?;
                }
            }
        }

        let checksum = xxhash3_128(&data_buf, 0);
        data_buf.write_all(&checksum.0.to_le_bytes())?;
        data_buf.write_all(&checksum.1.to_le_bytes())?;

        w.write_all(&data_buf)?;
        w.flush()?;
        Ok(())
    }
}

impl Deserializer {
    pub fn deserialize<R: Read>(
        &self,
        mut r: R,
    ) -> Result<(HnswGraph, HnswParams), SerialError> {
        let mut magic_bytes = [0u8; 4];
        r.read_exact(&mut magic_bytes)?;
        let magic = u32::from_le_bytes(magic_bytes);
        if magic != MAGIC {
            return Err(SerialError::InvalidMagic(magic));
        }

        let mut version_bytes = [0u8; 4];
        r.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != VERSION {
            return Err(SerialError::UnsupportedVersion(version));
        }

        let mut header_and_data = Vec::new();
        r.read_to_end(&mut header_and_data)?;

        if header_and_data.len() < CHECKSUM_SIZE {
            return Err(SerialError::TruncatedData);
        }

        let data_len = header_and_data.len() - CHECKSUM_SIZE;
        let (data, stored_checksum) = header_and_data.split_at(data_len);

        let stored_lo = u64::from_le_bytes(stored_checksum[0..8].try_into().unwrap());
        let stored_hi = u64::from_le_bytes(stored_checksum[8..16].try_into().unwrap());
        let (computed_lo, computed_hi) = xxhash3_128(data, 0);

        if computed_lo != stored_lo || computed_hi != stored_hi {
            return Err(SerialError::ChecksumMismatch);
        }

        let mut pos = 0;

        let m = decode_varint_at(data, &mut pos)? as usize;
        let ef_construction = decode_varint_at(data, &mut pos)? as usize;
        let ef_search = decode_varint_at(data, &mut pos)? as usize;

        let ml = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let max_level = decode_varint_at(data, &mut pos)? as usize;
        let dimension = decode_varint_at(data, &mut pos)? as usize;

        let ep_node = decode_varint_at(data, &mut pos)?;
        let ep_level = decode_varint_at(data, &mut pos)?;

        let num_elements = decode_varint_at(data, &mut pos)? as usize;

        let num_f32 = num_elements * dimension;
        let vectors_end = pos + num_f32 * 4;
        let vectors_bytes = &data[pos..vectors_end];
        let vectors: Vec<f32> = vectors_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        pos = vectors_end;

        let params = HnswParams {
            m,
            ef_construction,
            ef_search,
            ml,
            max_level,
        };

        let mut graph = HnswGraph::new(dimension, m, num_elements);

        if ep_node != u32::MAX {
            graph.update_entry_point(ep_node as usize, ep_level as usize);
        }

        let actual_num_layers = decode_varint_at(data, &mut pos)? as usize;

        for layer_idx in 0..actual_num_layers {
            while graph.num_layers() <= layer_idx {
                graph.add_layer();
            }

            let actual_nodes_in_layer = decode_varint_at(data, &mut pos)? as usize;

            for _ in 0..actual_nodes_in_layer {
                let node_id = decode_varint_at(data, &mut pos)? as usize;
                let neighbor_count = decode_varint_at(data, &mut pos)? as usize;
                for _ in 0..neighbor_count {
                    let neighbor_id = decode_varint_at(data, &mut pos)?;
                    graph.add_neighbor(layer_idx, node_id, neighbor_id);
                }
            }
        }

        graph.set_vectors(vectors);

        Ok((graph, params))
    }
}

fn decode_varint_at(
    data: &[u8],
    pos: &mut usize,
) -> Result<u32, SerialError> {
    let remaining = &data[*pos..];
    match decode_u32(remaining) {
        Some((value, bytes_consumed)) => {
            *pos += bytes_consumed;
            Ok(value)
        },
        None => Err(SerialError::InvalidVarint),
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SerialError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid magic number: expected 0x48595357, got 0x{0:08x}")]
    InvalidMagic(u32),

    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),

    #[error("Checksum mismatch")]
    ChecksumMismatch,

    #[error("Truncated data")]
    TruncatedData,

    #[error("Invalid varint encoding")]
    InvalidVarint,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_number() {
        assert_eq!(MAGIC, 0x57534E48);
    }

    #[test]
    fn test_version() {
        assert_eq!(VERSION, 2);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let params = HnswParams::new(16, 200, 50);
        let mut graph = HnswGraph::new(4, 16, 10);

        graph.add_node(&[1.0, 0.0, 0.0, 0.0]);
        graph.add_node(&[0.0, 1.0, 0.0, 0.0]);
        graph.add_node(&[0.0, 0.0, 1.0, 0.0]);
        graph.add_node(&[0.0, 0.0, 0.0, 1.0]);

        graph.add_edge(0, 1, 0);
        graph.add_edge(1, 0, 0);
        graph.add_edge(1, 2, 0);
        graph.add_edge(2, 1, 0);
        graph.add_edge(2, 3, 0);
        graph.add_edge(3, 2, 0);

        graph.update_entry_point(0, 0);

        let mut serialized = Vec::new();
        let serializer = Serializer;
        serializer
            .serialize(&graph, &params, &mut serialized)
            .unwrap();

        assert!(!serialized.is_empty());

        let deserializer = Deserializer;
        let (deserialized_graph, deserialized_params) =
            deserializer.deserialize(&serialized[..]).unwrap();

        assert_eq!(deserialized_params.m, params.m);
        assert_eq!(deserialized_params.ef_construction, params.ef_construction);
        assert_eq!(deserialized_params.ef_search, params.ef_search);
        assert_eq!(deserialized_graph.len(), graph.len());
        assert_eq!(deserialized_graph.dimension(), graph.dimension());

        assert_eq!(deserialized_graph.get_vector(0), graph.get_vector(0));
        assert_eq!(deserialized_graph.get_vector(1), graph.get_vector(1));
        assert_eq!(deserialized_graph.get_vector(2), graph.get_vector(2));
        assert_eq!(deserialized_graph.get_vector(3), graph.get_vector(3));

        assert_eq!(
            deserialized_graph.neighbor_count(0, 0),
            graph.neighbor_count(0, 0)
        );
        assert_eq!(
            deserialized_graph.neighbor_count(1, 0),
            graph.neighbor_count(1, 0)
        );
        assert_eq!(
            deserialized_graph.neighbor_count(2, 0),
            graph.neighbor_count(2, 0)
        );
        assert_eq!(
            deserialized_graph.neighbor_count(3, 0),
            graph.neighbor_count(3, 0)
        );

        let ep = graph.get_entry_point();
        let dep = deserialized_graph.get_entry_point();
        assert_eq!(dep, ep);
    }

    #[test]
    fn test_checksum_detection() {
        let params = HnswParams::new(16, 200, 50);
        let mut graph = HnswGraph::new(4, 16, 10);
        graph.add_node(&[1.0, 0.0, 0.0, 0.0]);

        let mut serialized = Vec::new();
        let serializer = Serializer;
        serializer
            .serialize(&graph, &params, &mut serialized)
            .unwrap();

        serialized[10] ^= 0xFF;

        let deserializer = Deserializer;
        let result = deserializer.deserialize(&serialized[..]);
        assert!(matches!(result, Err(SerialError::ChecksumMismatch)));
    }

    #[test]
    fn test_empty_graph() {
        let params = HnswParams::new(16, 200, 50);
        let graph = HnswGraph::new(4, 16, 0);

        let mut serialized = Vec::new();
        let serializer = Serializer;
        serializer
            .serialize(&graph, &params, &mut serialized)
            .unwrap();

        let deserializer = Deserializer;
        let (deserialized_graph, deserialized_params) =
            deserializer.deserialize(&serialized[..]).unwrap();

        assert_eq!(deserialized_params.m, params.m);
        assert_eq!(deserialized_graph.len(), 0);
    }
}
