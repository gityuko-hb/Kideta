//! ARM NEON distance kernels — 4 × f32 per iteration.
//!
//! All functions are `unsafe` and require the `neon` target feature.
//! Call only after `std::arch::is_aarch64_feature_detected!("neon")` returns true,
//! or via the safe dispatchers in `distance::mod`.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Horizontal sum of a float32x4_t register (a0+a1+a2+a3).
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    // Pairwise add: [a0+a1, a2+a3, a0+a1, a2+a3]
    let pair = vpaddq_f32(v, v);
    // Second pairwise: [(a0+a1)+(a2+a3), ...]
    let quad = vpaddq_f32(pair, pair);
    vgetq_lane_f32(quad, 0)
}

// ── L2 squared ────────────────────────────────────────────────────────

/// Squared Euclidean distance: Σ(aᵢ − bᵢ)².
///
/// Processes 4 × f32 per iteration using `vld1q_f32`, `vsubq_f32`, and `vmlaq_f32`.
///
/// # Safety
/// This function requires the `neon` target feature to be available.
/// The caller must ensure the CPU supports NEON before calling.
/// Additionally, both slices must have the same length.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn l2_squared_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        let diff = vsubq_f32(va, vb);
        acc = vmlaq_f32(acc, diff, diff); // acc += diff * diff
    }

    let mut result = hsum_f32x4(acc);
    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}

// ── Dot product (1.14) ────────────────────────────────────────────────────────

/// Inner product: Σ aᵢ·bᵢ.
///
/// # Safety
/// This function requires the `neon` target feature to be available.
/// The caller must ensure the CPU supports NEON before calling.
/// Additionally, both slices must have the same length.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn dot_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        acc = vmlaq_f32(acc, va, vb); // acc += va * vb
    }

    let mut result = hsum_f32x4(acc);
    for i in (chunks * 4)..len {
        result += a[i] * b[i];
    }
    result
}

// ── Cosine similarity (1.13) ──────────────────────────────────────────────────

/// Cosine similarity = dot(a,b) / (‖a‖·‖b‖), returns ∈ [−1, 1].
/// Returns 0.0 when either vector is a zero-vector.
///
/// # Safety
/// This function requires the `neon` target feature to be available.
/// The caller must ensure the CPU supports NEON before calling.
/// Additionally, both slices must have the same length.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_similarity_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut vdot = vdupq_n_f32(0.0);
    let mut vnorm_a = vdupq_n_f32(0.0);
    let mut vnorm_b = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        vdot = vmlaq_f32(vdot, va, vb);
        vnorm_a = vmlaq_f32(vnorm_a, va, va);
        vnorm_b = vmlaq_f32(vnorm_b, vb, vb);
    }

    let mut dot = hsum_f32x4(vdot);
    let mut norm_a = hsum_f32x4(vnorm_a);
    let mut norm_b = hsum_f32x4(vnorm_b);

    for i in (chunks * 4)..len {
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

/// Cosine distance = 1 − cosine_similarity (∈ [0, 2]).
///
/// # Safety
/// This function requires the `neon` target feature to be available.
/// The caller must ensure the CPU supports NEON before calling.
/// Additionally, both slices must have the same length.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn cosine_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    1.0 - cosine_similarity_f32(a, b)
}

// ── Manhattan (L1) ───────────────────────────────────────────────────────────

/// Computes the Manhattan (L1) distance between two f32 vectors using NEON.
///
/// # Safety
/// This function requires the `neon` target feature to be available.
/// The caller must ensure the CPU supports NEON before calling.
/// Additionally, both slices must have the same length.
#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn manhattan_f32(
    a: &[f32],
    b: &[f32],
) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        let diff = vsubq_f32(va, vb);
        acc = vaddq_f32(acc, vabsq_f32(diff));
    }

    let mut result = hsum_f32x4(acc);
    for i in (chunks * 4)..len {
        result += (a[i] - b[i]).abs();
    }
    result
}
