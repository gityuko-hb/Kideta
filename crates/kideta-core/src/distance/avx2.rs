//! AVX2 + FMA distance kernels — 8 × f32 per iteration.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Reduce a 256-bit register to a scalar f32 sum.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum_ps_256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf))
}

// ── L2 squared ───────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_squared_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        let mut result = hsum_ps_256(acc);
        for i in (chunks * 8)..len {
            let d = a[i] - b[i];
            result += d * d;
        }
        result
    }
}

// ── Dot product ───────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let mut result = hsum_ps_256(acc);
        for i in (chunks * 8)..len {
            result += a[i] * b[i];
        }
        result
    }
}

// ── Cosine similarity ─────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_similarity_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let mut vdot = _mm256_setzero_ps();
        let mut vnorm_a = _mm256_setzero_ps();
        let mut vnorm_b = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            vdot = _mm256_fmadd_ps(va, vb, vdot);
            vnorm_a = _mm256_fmadd_ps(va, va, vnorm_a);
            vnorm_b = _mm256_fmadd_ps(vb, vb, vnorm_b);
        }

        let mut dot = hsum_ps_256(vdot);
        let mut norm_a = hsum_ps_256(vnorm_a);
        let mut norm_b = hsum_ps_256(vnorm_b);

        for i in (chunks * 8)..len {
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
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe { 1.0 - cosine_similarity_f32(a, b) }
}

// ── Manhattan (L1) ───────────────────────────────────────────────────────────
// Uses only AVX (no FMA needed), dispatched under has_avx2().

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn manhattan_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    unsafe {
        let len = a.len();
        let chunks = len / 8;
        let mut acc = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0_f32);

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            acc = _mm256_add_ps(acc, abs_diff);
        }

        let mut result = hsum_ps_256(acc);
        for i in (chunks * 8)..len {
            result += (a[i] - b[i]).abs();
        }
        result
    }
}
