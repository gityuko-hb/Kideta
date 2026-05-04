//! Runtime CPU feature detection — no compile-time gating.
//!
//! This module provides functions to detect available SIMD instructions at runtime.
//! All functions return `true` if the feature is available, `false` otherwise.
//!
//! ## Usage
//!
//! ```
//! use kideta_core::distance::detection::{
//!     has_avx512f, has_avx2, has_fma, has_sse4_1, has_neon, best_simd
//! };
//!
//! // Check for specific features (these depend on your CPU)
//! let avx512 = has_avx512f();
//! let avx2 = has_avx2();
//! let fma = has_fma();
//! let sse41 = has_sse4_1();
//! let neon = has_neon();
//!
//! // Get the best available SIMD tier
//! let best = best_simd();
//! println!("Best SIMD: {best}");
//! // Possible values: "avx512f", "avx2+fma", "avx2", "sse4.1", "neon", "scalar"
//! ```
//!
//! ## Tier Ordering
//!
//! The `best_simd()` function returns the highest tier available:
//! 1. `avx512f` — AVX-512 Foundation (16 floats/cycle)
//! 2. `avx2+fma` — AVX2 with FMA (8 floats/cycle)
//! 3. `avx2` — AVX2 without FMA (legacy)
//! 4. `sse4.1` — SSE4.1 (4 floats/cycle)
//! 5. `neon` — ARM NEON (4 floats/cycle on AArch64)
//! 6. `scalar` — No SIMD support (1 float/cycle)

#[inline]
pub fn has_avx512f() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub fn has_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub fn has_sse4_1() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("sse4.1")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
pub fn has_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// Best SIMD tier available on this CPU.
pub fn best_simd() -> &'static str {
    if has_avx512f() {
        "avx512f"
    } else if has_avx2() && has_fma() {
        "avx2+fma"
    } else if has_avx2() {
        "avx2"
    } else if has_sse4_1() {
        "sse4.1"
    } else if has_neon() {
        "neon"
    } else {
        "scalar"
    }
}
