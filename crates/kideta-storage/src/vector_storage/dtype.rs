//! Vector dtype enumeration — storage format for vector components.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorDtype {
    F32,
    F16,
    BF16,
    I8,
    BINARY,
}

impl VectorDtype {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => VectorDtype::F32,
            1 => VectorDtype::F16,
            2 => VectorDtype::BF16,
            3 => VectorDtype::I8,
            4 => VectorDtype::BINARY,
            _ => VectorDtype::F32,
        }
    }

    pub fn as_u8(&self) -> u8 {
        match self {
            VectorDtype::F32 => 0,
            VectorDtype::F16 => 1,
            VectorDtype::BF16 => 2,
            VectorDtype::I8 => 3,
            VectorDtype::BINARY => 4,
        }
    }

    pub fn bytes_per_component(&self) -> usize {
        match self {
            VectorDtype::F32 => 4,
            VectorDtype::F16 | VectorDtype::BF16 => 2,
            VectorDtype::I8 => 1,
            VectorDtype::BINARY => 0,
        }
    }

    pub fn byte_size(
        &self,
        dim: u32,
    ) -> usize {
        match self {
            VectorDtype::BINARY => (dim as usize).div_ceil(8),
            _ => self.bytes_per_component() * dim as usize,
        }
    }

    pub fn cast_to_f32(
        &self,
        bytes: &[u8],
        dim: u32,
    ) -> Vec<f32> {
        match self {
            VectorDtype::F32 => {
                let mut f32_vec = vec![0.0f32; dim as usize];
                for (i, chunk) in bytes.chunks(4).enumerate() {
                    if i < dim as usize && chunk.len() == 4 {
                        f32_vec[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    }
                }
                f32_vec
            },
            VectorDtype::F16 => {
                let mut f32_vec = Vec::with_capacity(dim as usize);
                for chunk in bytes.chunks(2) {
                    if chunk.len() == 2 {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f32_vec.push(half::f16::from_bits(bits).to_f32());
                    }
                }
                f32_vec
            },
            VectorDtype::BF16 => {
                let mut f32_vec = Vec::with_capacity(dim as usize);
                for chunk in bytes.chunks(2) {
                    if chunk.len() == 2 {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f32_vec.push(half::bf16::from_bits(bits).to_f32());
                    }
                }
                f32_vec
            },
            VectorDtype::I8 => bytes.iter().map(|&b| b as i8 as f32).collect(),
            VectorDtype::BINARY => {
                let mut f32_vec = Vec::with_capacity(dim as usize);
                for i in 0..dim as usize {
                    let byte_idx = i / 8;
                    let bit_idx = i % 8;
                    if byte_idx < bytes.len() {
                        let bit = (bytes[byte_idx] >> bit_idx) & 1;
                        f32_vec.push(if bit != 0 {
                            1.0
                        } else {
                            -1.0
                        });
                    } else {
                        f32_vec.push(0.0);
                    }
                }
                f32_vec
            },
        }
    }

    pub fn pack_binary_vector(f32_input: &[f32]) -> Vec<u8> {
        let n_bits = f32_input.len();
        let n_bytes = n_bits.div_ceil(8);
        let mut bytes = vec![0u8; n_bytes];
        for (i, &val) in f32_input.iter().enumerate() {
            if val >= 0.0 {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        bytes
    }

    pub fn convert_f32_to_bytes(
        &self,
        f32_input: &[f32],
    ) -> Vec<u8> {
        match self {
            VectorDtype::F32 => {
                let mut bytes = Vec::with_capacity(f32_input.len() * 4);
                for &val in f32_input {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
                bytes
            },
            VectorDtype::F16 => {
                let mut bytes = Vec::with_capacity(f32_input.len() * 2);
                for &val in f32_input {
                    let h = half::f16::from_f32(val);
                    bytes.extend_from_slice(&h.to_le_bytes());
                }
                bytes
            },
            VectorDtype::BF16 => {
                let mut bytes = Vec::with_capacity(f32_input.len() * 2);
                for &val in f32_input {
                    let h = half::bf16::from_f32(val);
                    bytes.extend_from_slice(&h.to_le_bytes());
                }
                bytes
            },
            VectorDtype::I8 => f32_input
                .iter()
                .map(|&val| {
                    let saturated = val.clamp(-128.0, 127.0);
                    saturated.round() as i8 as u8
                })
                .collect(),
            VectorDtype::BINARY => Self::pack_binary_vector(f32_input),
        }
    }
}

impl std::fmt::Display for VectorDtype {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            VectorDtype::F32 => write!(f, "f32"),
            VectorDtype::F16 => write!(f, "f16"),
            VectorDtype::BF16 => write!(f, "bf16"),
            VectorDtype::I8 => write!(f, "i8"),
            VectorDtype::BINARY => write!(f, "binary"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_bytes_per_component() {
        assert_eq!(VectorDtype::F32.bytes_per_component(), 4);
    }

    #[test]
    fn f32_byte_size() {
        assert_eq!(VectorDtype::F32.byte_size(128), 512);
    }

    #[test]
    fn dtype_roundtrip() {
        let dt = VectorDtype::F32;
        let b = dt.as_u8();
        let restored = VectorDtype::from_u8(b);
        assert_eq!(dt, restored);
    }

    #[test]
    fn display() {
        assert_eq!(format!("{}", VectorDtype::F32), "f32");
    }

    #[test]
    fn f16_byte_size() {
        assert_eq!(VectorDtype::F16.byte_size(128), 256);
    }

    #[test]
    fn bf16_byte_size() {
        assert_eq!(VectorDtype::BF16.byte_size(128), 256);
    }

    #[test]
    fn i8_byte_size() {
        assert_eq!(VectorDtype::I8.byte_size(128), 128);
    }

    #[test]
    fn binary_byte_size_aligned() {
        assert_eq!(VectorDtype::BINARY.byte_size(128), 16);
    }

    #[test]
    fn binary_byte_size_unaligned() {
        assert_eq!(VectorDtype::BINARY.byte_size(129), 17);
    }

    #[test]
    fn f32_roundtrip() {
        let input: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5];
        let dtype = VectorDtype::F32;
        let bytes = dtype.convert_f32_to_bytes(&input);
        assert_eq!(bytes.len(), 16);
        let restored = dtype.cast_to_f32(&bytes, 4);
        for (a, b) in input.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn f16_roundtrip() {
        let input: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5];
        let dtype = VectorDtype::F16;
        let bytes = dtype.convert_f32_to_bytes(&input);
        assert_eq!(bytes.len(), 8);
        let restored = dtype.cast_to_f32(&bytes, 4);
        for (a, b) in input.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn bf16_roundtrip() {
        let input: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5];
        let dtype = VectorDtype::BF16;
        let bytes = dtype.convert_f32_to_bytes(&input);
        assert_eq!(bytes.len(), 8);
        let restored = dtype.cast_to_f32(&bytes, 4);
        for (a, b) in input.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn i8_roundtrip() {
        let input: Vec<f32> = vec![0.0, 1.0, -1.0, 127.0, -128.0, 50.0, -50.0];
        let dtype = VectorDtype::I8;
        let bytes = dtype.convert_f32_to_bytes(&input);
        assert_eq!(bytes.len(), 7);
        let restored = dtype.cast_to_f32(&bytes, 7);
        let expected: Vec<f32> = vec![0.0, 1.0, -1.0, 127.0, -128.0, 50.0, -50.0];
        for (a, b) in expected.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-4, "expected {}, got {}", a, b);
        }
    }

    #[test]
    fn i8_saturation() {
        let input: Vec<f32> = vec![300.0, -300.0];
        let dtype = VectorDtype::I8;
        let bytes = dtype.convert_f32_to_bytes(&input);
        assert_eq!(bytes, vec![127, 128]);
    }

    #[test]
    fn binary_pack_and_cast() {
        let input: Vec<f32> = vec![1.0, -0.5, 0.0, -2.0, 0.1];
        let dtype = VectorDtype::BINARY;
        let bytes = dtype.convert_f32_to_bytes(&input);
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0b10101);
        let restored = dtype.cast_to_f32(&bytes, 5);
        assert_eq!(restored, vec![1.0, -1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn all_dtype_variants_display() {
        assert_eq!(format!("{}", VectorDtype::F32), "f32");
        assert_eq!(format!("{}", VectorDtype::F16), "f16");
        assert_eq!(format!("{}", VectorDtype::BF16), "bf16");
        assert_eq!(format!("{}", VectorDtype::I8), "i8");
        assert_eq!(format!("{}", VectorDtype::BINARY), "binary");
    }

    #[test]
    fn all_dtype_variants_as_u8() {
        assert_eq!(VectorDtype::F32.as_u8(), 0);
        assert_eq!(VectorDtype::F16.as_u8(), 1);
        assert_eq!(VectorDtype::BF16.as_u8(), 2);
        assert_eq!(VectorDtype::I8.as_u8(), 3);
        assert_eq!(VectorDtype::BINARY.as_u8(), 4);
    }

    #[test]
    fn all_dtype_variants_from_u8() {
        assert_eq!(VectorDtype::from_u8(0), VectorDtype::F32);
        assert_eq!(VectorDtype::from_u8(1), VectorDtype::F16);
        assert_eq!(VectorDtype::from_u8(2), VectorDtype::BF16);
        assert_eq!(VectorDtype::from_u8(3), VectorDtype::I8);
        assert_eq!(VectorDtype::from_u8(4), VectorDtype::BINARY);
        assert_eq!(VectorDtype::from_u8(255), VectorDtype::F32);
    }
}
