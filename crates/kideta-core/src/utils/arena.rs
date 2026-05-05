//! Bump allocator (`Arena`) and thread-local arena for per-query hot paths.
//!
//! An Arena (bump allocator) provides extremely fast memory allocation for
//! scenarios where you can free all memory at once. This is perfect for
//! query processing: allocate during the query, reset when done.
//!
//! ## Arena
//! Allocates from a contiguous buffer; `reset()` in O(1) reclaims everything.
//!
//! ## Thread-local arena
//! Each thread gets its own `Arena`; no lock contention during search.
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::arena::Arena;
//!
//! // Create an arena with 1KB capacity
//! let mut arena = Arena::new(1024);
//!
//! // Allocate individual values
//! let x = arena.alloc(42u32).unwrap();
//! assert_eq!(*x, 42);
//!
//! // Allocate slices
//! let slice = arena.alloc_slice::<u64>(3, 0).unwrap();
//! assert_eq!(slice, &[0, 0, 0]);
//!
//! // Check usage
//! assert!(arena.used() > 0);
//! assert!(arena.remaining() < 1024);
//!
//! // Reset - reclaim all memory in O(1)
//! arena.reset();
//! assert_eq!(arena.used(), 0);
//!
//! // Allocate more after reset
//! let y = arena.alloc(100u64).unwrap();
//! assert_eq!(*y, 100);
//! ```
//!
//! ## Thread-Local Arena
//!
//! For high-performance query processing, use the thread-local arena:
//!
//! ```
//! use kideta_core::utils::arena::with_thread_arena;
//!
//! // The arena is automatically reset before your closure runs
//! with_thread_arena(|arena| {
//!     // Allocations here are fast and automatically reclaimed
//!     let _ = arena.alloc(1i32);
//!     let _ = arena.alloc_slice::<u8>(100, 0);
//!
//!     // After this closure returns, the arena is reset
//! });
//!
//! // Next call starts with fresh memory
//! with_thread_arena(|arena| {
//!     assert_eq!(arena.used(), 0);
//! });
//! ```
//!
//! ## Use Case: Query Scratch Space
//!
//! Use an arena for temporary allocations during search:
//!
//! ```
//! use kideta_core::utils::arena::Arena;
//!
//! fn process_query_scratch(a: &mut Arena) -> usize {
//!     // Allocate temporary structures without tracking each allocation
//!     let buffer = a.alloc_slice::<u8>(1024, 0).unwrap();
//!     // Do work...
//!     buffer.len() // Return some result
//! }
//!
//! let mut arena = Arena::new(4096);
//! let result = process_query_scratch(&mut arena);
//! // All temporary allocations are reclaimed with one reset()
//! arena.reset();
//! ```

use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;

// ── Arena ─────────────────────────────────────────────────────────────────────

/// Bump allocator — alloc is O(1), free-all is O(1).
///
/// Useful for per-query scratch space: allocate freely during the query, then
/// call `reset()` to reclaim everything at once.
pub struct Arena {
    buf: NonNull<u8>,
    layout: Layout,
    offset: usize,
}

impl Arena {
    /// Create a new `Arena` with `capacity` bytes of backing storage.
    pub fn new(capacity: usize) -> Self {
        let layout = Layout::from_size_align(capacity.max(1), 16).unwrap();
        // SAFETY: layout has non-zero size and valid alignment.
        let buf = unsafe { NonNull::new(alloc(layout)).expect("allocation failed") };
        Self {
            buf,
            layout,
            offset: 0,
        }
    }

    /// Total capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.layout.size()
    }

    /// Bytes currently allocated.
    #[inline]
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Bytes remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.layout.size() - self.offset
    }

    /// Allocate `size` bytes aligned to `align`.
    ///
    /// Returns `None` if there is not enough space.
    #[inline]
    pub fn alloc_raw(
        &mut self,
        size: usize,
        align: usize,
    ) -> Option<NonNull<u8>> {
        let aligned = self.offset.next_multiple_of(align);
        let end = aligned.checked_add(size)?;
        if end > self.layout.size() {
            return None;
        }
        self.offset = end;
        // SAFETY: `aligned` < end ≤ capacity, and the entire range is within
        // the allocated buffer `self.buf`. The pointer returned is aligned to `align`.
        Some(unsafe { NonNull::new_unchecked(self.buf.as_ptr().add(aligned)) })
    }

    /// Allocate a `T` from the arena (uninitialized).
    ///
    /// Returns `None` if there is not enough space.
    #[inline]
    pub fn alloc<T>(
        &mut self,
        value: T,
    ) -> Option<&mut T> {
        let ptr = self.alloc_raw(std::mem::size_of::<T>(), std::mem::align_of::<T>())?;
        // SAFETY: `ptr` is properly aligned for `T` and has enough space for
        // one `T`. The arena ensures this pointer is unique.
        let t = unsafe { &mut *(ptr.as_ptr() as *mut T) };
        *t = value;
        Some(t)
    }

    /// Allocate a slice of `len` copies of `T` (uninitialized).
    #[inline]
    pub fn alloc_slice<T: Copy>(
        &mut self,
        len: usize,
        fill: T,
    ) -> Option<&mut [T]> {
        let size = std::mem::size_of::<T>().checked_mul(len)?;
        let ptr = self.alloc_raw(size, std::mem::align_of::<T>())?;
        // SAFETY: `ptr` is properly aligned for `T`, and `size` bytes are
        // available in the buffer. The slice does not outlive the arena.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, len) };
        slice.fill(fill);
        Some(slice)
    }

    /// Reset the arena — reclaim all allocations in O(1).
    ///
    /// All previously returned references become invalid after this call.
    /// The caller must ensure no live references exist.
    #[inline]
    pub fn reset(&mut self) {
        self.offset = 0;
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: `buf` was allocated with this exact layout.
        unsafe {
            dealloc(self.buf.as_ptr(), self.layout);
        }
    }
}

// SAFETY: Arena is Send because it owns its buffer exclusively.
unsafe impl Send for Arena {}

// ── Thread-local arena ─────────────────────────────────────────────────

/// Default capacity for the thread-local arena: 4 MiB.
const TLS_ARENA_CAPACITY: usize = 4 * 1024 * 1024;

thread_local! {
    static TLS_ARENA: std::cell::RefCell<Arena> =
        std::cell::RefCell::new(Arena::new(TLS_ARENA_CAPACITY));
}

/// Run `f` with exclusive access to the current thread's arena.
///
/// The arena is **reset before** `f` runs, so all memory it allocates is
/// automatically reclaimed when the closure returns.
pub fn with_thread_arena<F, R>(f: F) -> R
where
    F: FnOnce(&mut Arena) -> R,
{
    TLS_ARENA.with(|cell| {
        let mut arena = cell.borrow_mut();
        arena.reset();
        f(&mut arena)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc() {
        let mut a = Arena::new(1024);
        let x = a.alloc(42u64).unwrap();
        assert_eq!(*x, 42);
        assert_eq!(a.used(), 8);
    }

    #[test]
    fn alloc_slice() {
        let mut a = Arena::new(1024);
        let s = a.alloc_slice::<u32>(4, 99).unwrap();
        assert_eq!(s, [99u32; 4]);
    }

    #[test]
    fn reset_reclaims() {
        let mut a = Arena::new(64);
        a.alloc(1u64).unwrap();
        a.alloc(2u64).unwrap();
        assert!(a.used() > 0);
        a.reset();
        assert_eq!(a.used(), 0);
        // Can allocate again after reset.
        a.alloc(3u32).unwrap();
    }

    #[test]
    fn out_of_space_returns_none() {
        let mut a = Arena::new(8);
        a.alloc(0u64).unwrap(); // 8 bytes used
        assert!(a.alloc(1u8).is_none()); // no room
    }

    #[test]
    fn alignment_respected() {
        let mut a = Arena::new(128);
        let _b = a.alloc(1u8).unwrap();
        // Next alloc of u64 should be 8-byte aligned.
        let p = a.alloc(42u64).unwrap() as *mut u64;
        assert_eq!(p as usize % 8, 0);
    }

    #[test]
    fn thread_local_arena() {
        with_thread_arena(|a| {
            let x = a.alloc(777u32).unwrap();
            assert_eq!(*x, 777);
        });
        // After closure the arena is still accessible (reset at next call).
        with_thread_arena(|a| {
            assert_eq!(a.used(), 0); // reset by with_thread_arena
        });
    }
}
