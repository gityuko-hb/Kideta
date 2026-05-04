//! AVX-512F distance kernels — 16 × f32 per iteration.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ── L2 squared ───────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn l2_squared_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 16;
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }

        let mut result = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            let d = a[i] - b[i];
            result += d * d;
        }
        result
    }
}

// ── Dot product ───────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 16;
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }

        let mut result = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            result += a[i] * b[i];
        }
        result
    }
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn cosine_similarity_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 16;
        let mut vdot = _mm512_setzero_ps();
        let mut vnorm_a = _mm512_setzero_ps();
        let mut vnorm_b = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            vdot = _mm512_fmadd_ps(va, vb, vdot);
            vnorm_a = _mm512_fmadd_ps(va, va, vnorm_a);
            vnorm_b = _mm512_fmadd_ps(vb, vb, vnorm_b);
        }

        let mut dot = _mm512_reduce_add_ps(vdot);
        let mut norm_a = _mm512_reduce_add_ps(vnorm_a);
        let mut norm_b = _mm512_reduce_add_ps(vnorm_b);

        for i in (chunks * 16)..len {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            dot / denom
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn cosine_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe { 1.0 - cosine_similarity_f32(a, b) }
}

// ── Manhattan (L1) ───────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn manhattan_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 16;
        let mut acc = _mm512_setzero_ps();
        let sign_mask = _mm512_set1_ps(-0.0_f32);

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(va, vb);
            let abs_diff = _mm512_andnot_ps(sign_mask, diff);
            acc = _mm512_add_ps(acc, abs_diff);
        }

        let mut result = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            result += (a[i] - b[i]).abs();
        }
        result
    }
}
