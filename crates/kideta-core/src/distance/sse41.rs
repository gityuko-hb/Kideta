//! SSE4.1 distance kernels — 4 × f32 per iteration.
//!
//! All functions are `unsafe` and require the `sse4.1` target feature.
//! Call only after `is_x86_feature_detected!("sse4.1")` returns true,
//! or via the safe dispatchers in `distance::mod`.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Horizontal sum of a 128-bit register (a0+a1+a2+a3).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
unsafe fn hsum_ps_128(v: __m128) -> f32 {
    let shuf = _mm_movehdup_ps(v); // [a1, a1, a3, a3]
    let sums = _mm_add_ps(v, shuf); // [a0+a1, _, a2+a3, _]
    let shuf = _mm_movehl_ps(sums, sums); // [a2+a3, _, ...]
    _mm_cvtss_f32(_mm_add_ss(sums, shuf)) // (a0+a1)+(a2+a3)
}

// ── L2 squared ───────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 4;
        let mut acc = _mm_setzero_ps();

        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            let diff = _mm_sub_ps(va, vb);
            acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
        }

        let mut result = hsum_ps_128(acc);
        for i in (chunks * 4)..len {
            let d = a[i] - b[i];
            result += d * d;
        }
        result
    }
}

// ── Dot product ───────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 4;
        let mut acc = _mm_setzero_ps();

        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
        }

        let mut result = hsum_ps_128(acc);
        for i in (chunks * 4)..len {
            result += a[i] * b[i];
        }
        result
    }
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 4;
        let mut vdot = _mm_setzero_ps();
        let mut vnorm_a = _mm_setzero_ps();
        let mut vnorm_b = _mm_setzero_ps();

        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            vdot = _mm_add_ps(vdot, _mm_mul_ps(va, vb));
            vnorm_a = _mm_add_ps(vnorm_a, _mm_mul_ps(va, va));
            vnorm_b = _mm_add_ps(vnorm_b, _mm_mul_ps(vb, vb));
        }

        let mut dot = hsum_ps_128(vdot);
        let mut norm_a = hsum_ps_128(vnorm_a);
        let mut norm_b = hsum_ps_128(vnorm_b);

        for i in (chunks * 4)..len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 { 0.0 } else { dot / denom }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn cosine_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe { 1.0 - cosine_similarity_f32(a, b) }
}

// ── Manhattan (L1) ───────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn manhattan_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 4;
        let mut acc = _mm_setzero_ps();
        let sign_mask = _mm_set1_ps(-0.0_f32);

        for i in 0..chunks {
            let va = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            let diff = _mm_sub_ps(va, vb);
            let abs_diff = _mm_andnot_ps(sign_mask, diff);
            acc = _mm_add_ps(acc, abs_diff);
        }

        let mut result = hsum_ps_128(acc);
        for i in (chunks * 4)..len {
            result += (a[i] - b[i]).abs();
        }
        result
    }
}
