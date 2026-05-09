//! SQ8 operations — encode, decode, approximate distance.
//!
//! All operations are designed to be branch-free in the hot path where possible
//! to enable efficient vectorization.

use crate::quantization::config::Sq8Config;

/// SQ8 operations: encode, decode, approx distance.
pub struct Sq8Ops;

impl Sq8Ops {
    /// Encode an f32 vector into a u8 code vector.
    ///
    /// ```ignore
    /// code[i] = round((vector[i] + offset[i]) * scale[i]) as u8
    /// ```
    ///
    /// Clamps values to [0, 255] range.
    #[inline]
    pub fn encode(
        config: &Sq8Config,
        vector: &[f32],
    ) -> Vec<u8> {
        let mut code = vec![0u8; config.dimension];
        Self::encode_to_slice(config, vector, &mut code);
        code
    }

    /// Encode an f32 vector directly into a pre-allocated u8 slice.
    ///
    /// # Panics
    ///
    /// Panics if `code.len() != config.dimension`.
    #[inline]
    pub fn encode_to_slice(
        config: &Sq8Config,
        vector: &[f32],
        code: &mut [u8],
    ) {
        assert_eq!(code.len(), config.dimension);
        assert_eq!(vector.len(), config.dimension);

        for i in 0..config.dimension {
            let val = (vector[i] + config.offset[i]) * config.scale[i];
            let clamped = val.clamp(0.0, 255.0);
            code[i] = clamped.round() as u8;
        }
    }

    /// Decode a u8 code back to f32 (approximate, for rescore only).
    ///
    /// ```ignore
    /// vector[i] ≈ code[i] / scale[i] - offset[i]
    /// ```
    #[inline]
    pub fn decode(
        config: &Sq8Config,
        code: &[u8],
    ) -> Vec<f32> {
        let mut vector = vec![0.0_f32; config.dimension];
        Self::decode_to_slice(config, code, &mut vector);
        vector
    }

    /// Decode a u8 code directly into a pre-allocated f32 slice.
    #[inline]
    pub fn decode_to_slice(
        config: &Sq8Config,
        code: &[u8],
        vector: &mut [f32],
    ) {
        assert_eq!(code.len(), config.dimension);
        assert_eq!(vector.len(), config.dimension);

        for i in 0..config.dimension {
            vector[i] = (code[i] as f32) / config.scale[i] - config.offset[i];
        }
    }

    /// Compute approximate L2² distance between query (f32) and stored code (u8).
    ///
    /// Avoids full decode by quantizing the query the same way:
    ///
    /// ```ignore
    /// q[i] = round((query[i] + offset[i]) * scale[i])
    /// dist² = Σ (q[i] - code[i])²
    /// ```
    ///
    /// This is the standard asymmetric distance computation for SQ.
    /// Returns L2² (squared distance), not L2.
    #[inline]
    pub fn approx_l2_distance(
        config: &Sq8Config,
        query: &[f32],
        code: &[u8],
    ) -> f32 {
        assert_eq!(query.len(), config.dimension);
        assert_eq!(code.len(), config.dimension);

        let mut sum_sq: f32 = 0.0;
        for i in 0..config.dimension {
            let q = (query[i] + config.offset[i]) * config.scale[i];
            let q_clamped = q.clamp(0.0, 255.0).round();
            let c = code[i] as f32;
            let diff = q_clamped - c;
            sum_sq += diff * diff;
        }
        sum_sq
    }

    /// Compute approximate L2 distance (square root of approx_l2_distance).
    #[inline]
    pub fn approx_l2(
        config: &Sq8Config,
        query: &[f32],
        code: &[u8],
    ) -> f32 {
        Self::approx_l2_distance(config, query, code).sqrt()
    }

    /// Compute approximate cosine similarity between query (f32) and stored code (u8).
    ///
    /// First decodes both to f32, then computes cosine. For production use,
    /// consider using Binary quantization if cosine is the primary metric.
    #[inline]
    pub fn approx_cosine(
        config: &Sq8Config,
        query: &[f32],
        code: &[u8],
    ) -> f32 {
        let decoded = Self::decode(config, code);
        Self::_cosine_similarity(query, &decoded)
    }

    #[inline]
    fn _cosine_similarity(
        a: &[f32],
        b: &[f32],
    ) -> f32 {
        let mut dot: f32 = 0.0;
        let mut norm_a: f32 = 0.0;
        let mut norm_b: f32 = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        let denom = (norm_a.sqrt() * norm_b.sqrt()).max(f32::EPSILON);
        dot / denom
    }

    /// Compute approximate dot product between query (f32) and stored code (u8).
    ///
    /// Note: This decodes to f32 first, so it's not truly "approximate" in the
    /// compute sense — the approximation is the SQ encoding itself.
    #[inline]
    pub fn approx_dot(
        config: &Sq8Config,
        query: &[f32],
        code: &[u8],
    ) -> f32 {
        let decoded = Self::decode(config, code);
        query
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> Sq8Config {
        Sq8Config::with_stats(vec![0.0_f32, -10.0_f32], vec![100.0_f32, 50.0_f32])
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let config = make_test_config();
        let original = vec![50.0_f32, 20.0_f32];
        let code = Sq8Ops::encode(&config, &original);
        let decoded = Sq8Ops::decode(&config, &code);

        let max_err = original
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(max_err < 0.5);
    }

    #[test]
    fn test_encode_clamping() {
        let config = make_test_config();
        let too_high = vec![200.0_f32, 100.0_f32];
        let code = Sq8Ops::encode(&config, &too_high);
        assert_eq!(code[0], 255);
        assert_eq!(code[1], 255);

        let too_low = vec![-10.0_f32, -20.0_f32];
        let code = Sq8Ops::encode(&config, &too_low);
        assert_eq!(code[0], 0);
        assert_eq!(code[1], 0);
    }

    #[test]
    fn test_approx_l2_identical() {
        let config = make_test_config();
        let vector = vec![50.0_f32, 20.0_f32];
        let code = Sq8Ops::encode(&config, &vector);
        let dist = Sq8Ops::approx_l2_distance(&config, &vector, &code);

        assert!(dist < 1.0);
    }

    #[test]
    fn test_approx_l2_different() {
        let config = make_test_config();
        let v1 = vec![50.0_f32, 20.0_f32];
        let v2 = vec![10.0_f32, 40.0_f32];
        let code1 = Sq8Ops::encode(&config, &v1);
        let code2 = Sq8Ops::encode(&config, &v2);

        let dist1 = Sq8Ops::approx_l2_distance(&config, &v1, &code1);
        let dist2 = Sq8Ops::approx_l2_distance(&config, &v2, &code2);

        assert!(dist1 >= 0.0);
        assert!(dist2 >= 0.0);
    }

    #[test]
    fn test_approx_l2_zero_range() {
        let config = Sq8Config::with_stats(vec![5.0_f32, 5.0_f32], vec![5.0_f32, 5.0_f32]);
        let v = vec![5.0_f32, 5.0_f32];
        let code = Sq8Ops::encode(&config, &v);
        let dist = Sq8Ops::approx_l2_distance(&config, &v, &code);

        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_encode_to_slice_exact() {
        let config = make_test_config();
        let vector = vec![25.0_f32, 5.0_f32];
        let mut code = vec![0u8; 2];
        Sq8Ops::encode_to_slice(&config, &vector, &mut code);
        assert_eq!(code[0], 64);
        assert_eq!(code[1], 64);
    }
}
