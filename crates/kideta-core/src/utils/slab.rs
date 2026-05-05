//! `SlabAllocator<T>` — fixed-size object pool.
//!
//! A slab allocator manages a fixed pool of objects of the same type. This
//! eliminates the overhead of repeated heap allocations in high-throughput
//! scenarios like search processing where you need to create and destroy
//! many temporary objects.
//!
//! ## How It Works
//!
//! The allocator pre-allocates a fixed number of slots upfront. When you
//! acquire an object, it takes a slot from the free list. When you release
//! it, the slot goes back to the free list for reuse. No heap allocation
//! happens after initial setup.
//!
//! ## Usage
//!
//! ```
//! use kideta_core::utils::slab::SlabAllocator;
//!
//! // Create a pool that can hold 100 u64 values
//! let mut slab: SlabAllocator<u64> = SlabAllocator::new(100);
//!
//! // Acquire a slot and initialize it
//! let idx = slab.acquire(42).unwrap();
//!
//! // Access the object safely
//! unsafe {
//!     let value = slab.get(idx);
//!     assert_eq!(*value, 42);
//! }
//!
//! // Release it back to the pool for reuse
//! unsafe {
//!     slab.release(idx);
//! }
//!
//! // Acquire again - gets the same slot back (reused!)
//! let idx2 = slab.acquire(100).unwrap();
//! assert!(idx2 < 100); // still within capacity
//! ```
//!
//! ## Use Case: HNSW Graph Node Pool
//!
//! ```
//! use kideta_core::utils::slab::SlabAllocator;
//!
//! #[derive(Debug, Clone)]
//! struct SearchNode {
//!     id: u64,
//!     distance: f32,
//!     level: usize,
//! }
//!
//! // Pool for search frontier nodes
//! let mut node_pool: SlabAllocator<SearchNode> = SlabAllocator::new(1000);
//!
//! // Simulate processing search frontier
//! fn process_neighbors(pool: &mut SlabAllocator<SearchNode>, neighbor_ids: &[u64]) -> usize {
//!     let mut processed = 0;
//!     for &id in neighbor_ids {
//!         if let Some(idx) = pool.acquire(SearchNode { id, distance: 0.0, level: 0 }) {
//!             // Do work with node...
//!             processed += 1;
//!             // Release back to pool
//!             unsafe { pool.release(idx); }
//!         }
//!     }
//!     processed
//! }
//! ```

use std::mem::MaybeUninit;

/// A fixed-capacity pool of `T` objects.
///
/// Slots are either **live** (returned by `acquire`) or **free** (on the free
/// list).  The pool itself never shrinks; it grows up to `capacity` slots.
pub struct SlabAllocator<T> {
    /// Backing storage — every slot is either initialized or on the free list.
    slots: Vec<MaybeUninit<T>>,
    /// Stack of free slot indices.
    free: Vec<usize>,
    /// Number of live (acquired) objects.
    live: usize,
    capacity: usize,
}

impl<T> SlabAllocator<T> {
    /// Create a pool that can hold at most `capacity` objects.
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: (0..capacity)
                .map(|_| MaybeUninit::uninit())
                .collect(),
            free: (0..capacity).collect(),
            live: 0,
            capacity,
        }
    }

    /// Total capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of live (acquired) objects.
    #[inline]
    pub fn live(&self) -> usize {
        self.live
    }

    /// Number of available free slots.
    #[inline]
    pub fn available(&self) -> usize {
        self.free.len()
    }

    /// Acquire a slot and initialize it with `value`.
    ///
    /// Returns the slot index, or `None` if the pool is full.
    pub fn acquire(
        &mut self,
        value: T,
    ) -> Option<usize> {
        let idx = self.free.pop()?;
        self.slots[idx].write(value);
        self.live += 1;
        Some(idx)
    }

    /// Get a shared reference to the object in slot `idx`.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// 1. `idx` is within the capacity of the slab.
    /// 2. The slot at `idx` is currently initialized (i.e., it has been returned
    ///    by `acquire` and not yet `release`d).
    #[inline]
    pub unsafe fn get(
        &self,
        idx: usize,
    ) -> &T {
        // SAFETY: The requirements are guaranteed by the caller.
        unsafe { self.slots[idx].assume_init_ref() }
    }

    /// Get a mutable reference to the object in slot `idx`.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// 1. `idx` is within the capacity of the slab.
    /// 2. The slot at `idx` is currently initialized.
    /// 3. No other references to this slot exist.
    #[inline]
    pub unsafe fn get_mut(
        &mut self,
        idx: usize,
    ) -> &mut T {
        // SAFETY: The requirements are guaranteed by the caller.
        unsafe { self.slots[idx].assume_init_mut() }
    }

    /// Release slot `idx` back to the pool, dropping the contained value.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// 1. `idx` is within the capacity of the slab.
    /// 2. The slot at `idx` is currently initialized.
    /// 3. The slot must not be accessed again after this call.
    pub unsafe fn release(
        &mut self,
        idx: usize,
    ) {
        // SAFETY: We drop the initialized value in the slot.
        unsafe {
            self.slots[idx].assume_init_drop();
        }
        self.free.push(idx);
        self.live -= 1;
    }

    /// Release all live slots, dropping every contained value.
    pub fn clear(&mut self) {
        // Walk every slot; live ones are those NOT on the free list.
        let mut on_free: Vec<bool> = vec![false; self.capacity];
        for &fi in &self.free {
            on_free[fi] = true;
        }
        for (i, &is_free) in on_free.iter().enumerate().take(self.capacity) {
            if !is_free {
                // SAFETY: We know slot i is live (not in the free list) and
                // thus contains an initialized value.
                unsafe {
                    self.slots[i].assume_init_drop();
                }
            }
        }
        self.free.clear();
        self.free.extend(0..self.capacity);
        self.live = 0;
    }
}

impl<T> Drop for SlabAllocator<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_and_get() {
        let mut slab: SlabAllocator<u64> = SlabAllocator::new(4);
        let idx = slab.acquire(42).unwrap();
        assert_eq!(unsafe { *slab.get(idx) }, 42);
        assert_eq!(slab.live(), 1);
    }

    #[test]
    fn release_returns_slot() {
        let mut slab: SlabAllocator<String> = SlabAllocator::new(2);
        let a = slab.acquire("hello".to_string()).unwrap();
        let b = slab.acquire("world".to_string()).unwrap();
        assert!(slab.acquire("overflow".to_string()).is_none()); // full
        unsafe {
            slab.release(a);
        }
        assert_eq!(slab.live(), 1);
        // Slot is reusable.
        let c = slab.acquire("reused".to_string()).unwrap();
        assert_eq!(unsafe { slab.get(c) }, "reused");
        unsafe {
            slab.release(b);
        }
        unsafe {
            slab.release(c);
        }
    }

    #[test]
    fn clear_resets_pool() {
        let mut slab: SlabAllocator<i32> = SlabAllocator::new(3);
        slab.acquire(1).unwrap();
        slab.acquire(2).unwrap();
        slab.clear();
        assert_eq!(slab.live(), 0);
        assert_eq!(slab.available(), 3);
        slab.acquire(99).unwrap(); // can allocate again
    }

    #[test]
    fn get_mut() {
        let mut slab: SlabAllocator<Vec<u32>> = SlabAllocator::new(2);
        let idx = slab.acquire(vec![1, 2, 3]).unwrap();
        unsafe {
            slab.get_mut(idx).push(4);
        }
        assert_eq!(unsafe { slab.get(idx) }, &[1, 2, 3, 4]);
        unsafe {
            slab.release(idx);
        }
    }
}
