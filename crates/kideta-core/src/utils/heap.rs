//! Binary heaps — `MinHeap`, `MaxHeap`, and `BoundedMaxHeap`.
//!
//! All three are backed by a `Vec` and generic over any type implementing
//! `PartialOrd`.  NaN comparisons are resolved by treating NaN as "less than
//! everything" so they sink to the bottom of a `MinHeap` / are evicted first
//! from a `BoundedMaxHeap`.
//!
//! ## Usage
//!
//! ### MinHeap - always get the smallest first
//!
//! ```
//! use kideta_core::utils::heap::MinHeap;
//!
//! let mut heap = MinHeap::new();
//! heap.push(5);
//! heap.push(1);
//! heap.push(3);
//!
//! assert_eq!(heap.pop(), Some(1));
//! assert_eq!(heap.pop(), Some(3));
//! assert_eq!(heap.pop(), Some(5));
//! assert_eq!(heap.pop(), None);
//!
//! // Peek without removing
//! heap.push(2);
//! assert_eq!(heap.peek(), Some(&2));
//! ```
//!
//! ### MaxHeap - always get the largest first
//!
//! ```
//! use kideta_core::utils::heap::MaxHeap;
//!
//! let mut heap = MaxHeap::new();
//! heap.push(3);
//! heap.push(1);
//! heap.push(4);
//!
//! assert_eq!(heap.pop(), Some(4));
//! assert_eq!(heap.pop(), Some(3));
//! assert_eq!(heap.pop(), Some(1));
//! ```
//!
//! ### BoundedMaxHeap - keep top-k smallest elements
//!
//! This is essential for HNSW search where you need the k-nearest neighbors.
//!
//! ```
//! use kideta_core::utils::heap::BoundedMaxHeap;
//!
//! // Keep only top 3 smallest distances
//! let mut top_k = BoundedMaxHeap::new(3);
//!
//! // Add candidates with their distances
//! top_k.push(5.0);
//! top_k.push(1.0);
//! top_k.push(4.0);
//! top_k.push(2.0);
//! top_k.push(3.0);
//!
//! // Heap now contains {1.0, 2.0, 3.0} - the 3 smallest
//! let results = top_k.into_sorted_asc();
//! assert_eq!(results, vec![1.0, 2.0, 3.0]);
//! ```
//!
//! ### Rejecting worse candidates efficiently
//!
//! ```
//! use kideta_core::utils::heap::BoundedMaxHeap;
//!
//! let mut top_k = BoundedMaxHeap::new(2);
//! top_k.push(1.0);
//! top_k.push(2.0);
//!
//! // This candidate is worse than the worst in heap (2.0), reject it
//! let inserted = top_k.push(3.0);
//! assert!(!inserted);
//! assert_eq!(top_k.len(), 2);
//!
//! // This candidate is better, replace the worst
//! let inserted = top_k.push(0.5);
//! assert!(inserted);
//! assert_eq!(top_k.len(), 2);
//!
//! let results = top_k.into_sorted_asc();
//! assert_eq!(results, vec![0.5, 1.0]); // 2.0 was evicted
//! ```
//!
//! ## Use Case: HNSW Search
//!
//! In HNSW, you need two priority queues:
//! - **MinHeap**: For the search frontier (process nearest first)
//! - **BoundedMaxHeap**: For results (keep only k nearest)
//!
//! ```
//! use kideta_core::utils::heap::{MinHeap, BoundedMaxHeap};
//!
//! // Search frontier - process closest nodes first
//! let mut frontier: MinHeap<(usize, f32)> = MinHeap::new();
//! frontier.push((0, 0.5)); // (node_id, distance)
//! frontier.push((1, 0.8));
//!
//! // Results - keep only k nearest
//! let mut results: BoundedMaxHeap<f32> = BoundedMaxHeap::new(10);
//! for dist in [0.1, 0.3, 0.5, 0.7, 0.9].iter().copied() {
//!     let _ = results.push(dist);
//! }
//! let nearest = results.into_sorted_asc(); // worst is at root for fast rejection
//! ```

use std::cmp::Ordering;

// ── comparison helper ─────────────────────────────────────────────────────────

#[inline]
fn cmp_partial<T: PartialOrd>(
    a: &T,
    b: &T,
) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Less)
}

// ── MinHeap ───────────────────────────────────────────────────────────────────

/// Binary min-heap: `peek` / `pop` returns the **smallest** element.
///
/// Insertion: O(log n).  Pop: O(log n).  Peek: O(1).
pub struct MinHeap<T: PartialOrd> {
    data: Vec<T>,
}

impl<T: PartialOrd> MinHeap<T> {
    /// Create an empty heap.
    #[inline]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create a heap with a pre-allocated capacity.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true` if the heap is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// View the minimum element without removing it.
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    /// Push a new element.
    #[inline]
    pub fn push(
        &mut self,
        item: T,
    ) {
        self.data.push(item);
        self.sift_up(self.data.len() - 1);
    }

    /// Remove and return the minimum element.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let last = self.data.len() - 1;
        self.data.swap(0, last);
        let min = self.data.pop();
        if !self.data.is_empty() {
            self.sift_down(0);
        }
        min
    }

    /// Drain all elements into a `Vec` in **ascending** order.
    pub fn into_sorted_vec(mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.data.len());
        while let Some(v) = self.pop() {
            out.push(v);
        }
        out
    }

    fn sift_up(
        &mut self,
        mut i: usize,
    ) {
        while i > 0 {
            let p = (i - 1) / 2;
            if cmp_partial(&self.data[i], &self.data[p]) == Ordering::Less {
                self.data.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    fn sift_down(
        &mut self,
        mut i: usize,
    ) {
        let n = self.data.len();
        loop {
            let mut min = i;
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            if l < n && cmp_partial(&self.data[l], &self.data[min]) == Ordering::Less {
                min = l;
            }
            if r < n && cmp_partial(&self.data[r], &self.data[min]) == Ordering::Less {
                min = r;
            }
            if min == i {
                break;
            }
            self.data.swap(i, min);
            i = min;
        }
    }
}

impl<T: PartialOrd> Default for MinHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── MaxHeap ───────────────────────────────────────────────────────────────────

/// Binary max-heap: `peek` / `pop` returns the **largest** element.
pub struct MaxHeap<T: PartialOrd> {
    data: Vec<T>,
}

impl<T: PartialOrd> MaxHeap<T> {
    #[inline]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// View the maximum element (root).
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    #[inline]
    pub fn push(
        &mut self,
        item: T,
    ) {
        self.data.push(item);
        self.sift_up(self.data.len() - 1);
    }

    /// Remove and return the maximum element.
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let last = self.data.len() - 1;
        self.data.swap(0, last);
        let max = self.data.pop();
        if !self.data.is_empty() {
            self.sift_down(0);
        }
        max
    }

    /// Drain all elements into a `Vec` in **descending** order.
    pub fn into_sorted_vec(mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.data.len());
        while let Some(v) = self.pop() {
            out.push(v);
        }
        out
    }

    fn sift_up(
        &mut self,
        mut i: usize,
    ) {
        while i > 0 {
            let p = (i - 1) / 2;
            if cmp_partial(&self.data[i], &self.data[p]) == Ordering::Greater {
                self.data.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    fn sift_down(
        &mut self,
        mut i: usize,
    ) {
        let n = self.data.len();
        loop {
            let mut max = i;
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            if l < n && cmp_partial(&self.data[l], &self.data[max]) == Ordering::Greater {
                max = l;
            }
            if r < n && cmp_partial(&self.data[r], &self.data[max]) == Ordering::Greater {
                max = r;
            }
            if max == i {
                break;
            }
            self.data.swap(i, max);
            i = max;
        }
    }
}

impl<T: PartialOrd> Default for MaxHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── BoundedMaxHeap ────────────────────────────────────────────────────────────

/// A max-heap with a fixed capacity that automatically evicts the largest
/// element when full — thus maintaining the **smallest k** elements seen.
///
/// Use this as the `results` set in HNSW search: the root is always the
/// *worst* result in the current top-k, so you can quickly check whether a
/// new candidate is better than the worst.
pub struct BoundedMaxHeap<T: PartialOrd> {
    inner: MaxHeap<T>,
    cap: usize,
}

impl<T: PartialOrd> BoundedMaxHeap<T> {
    /// Create a bounded heap that keeps at most `cap` elements.
    #[inline]
    pub fn new(cap: usize) -> Self {
        Self {
            inner: MaxHeap::with_capacity(cap + 1),
            cap,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.inner.len() >= self.cap
    }

    /// The worst (largest) element among the current k — `None` if empty.
    #[inline]
    pub fn peek_worst(&self) -> Option<&T> {
        self.inner.peek()
    }

    /// Push `item`.  If the heap is already at capacity and `item` is greater
    /// than the current worst, the worst is evicted first.
    ///
    /// Returns `true` if the item was actually inserted.
    #[inline]
    pub fn push(
        &mut self,
        item: T,
    ) -> bool {
        if self.inner.len() < self.cap {
            self.inner.push(item);
            true
        } else if let Some(worst) = self.inner.peek() {
            if cmp_partial(&item, worst) == Ordering::Less {
                self.inner.pop();
                self.inner.push(item);
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Drain all elements in **descending** order (worst first).
    pub fn into_sorted_vec(self) -> Vec<T> {
        self.inner.into_sorted_vec()
    }

    /// Drain all elements in **ascending** order (best first — nearest first).
    pub fn into_sorted_asc(self) -> Vec<T> {
        let mut v = self.into_sorted_vec();
        v.reverse();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MinHeap ───────────────────────────────────────────────────────────────

    #[test]
    fn min_heap_basic() {
        let mut h = MinHeap::new();
        h.push(5.0_f32);
        h.push(1.0);
        h.push(3.0);
        h.push(2.0);
        assert_eq!(h.peek(), Some(&1.0_f32));
        assert_eq!(h.pop(), Some(1.0));
        assert_eq!(h.pop(), Some(2.0));
        assert_eq!(h.pop(), Some(3.0));
        assert_eq!(h.pop(), Some(5.0));
        assert!(h.pop().is_none());
    }

    #[test]
    fn min_heap_sorted() {
        let mut h = MinHeap::new();
        for x in [9, 3, 7, 1, 5, 2, 8, 4, 6] {
            h.push(x);
        }
        assert_eq!(h.into_sorted_vec(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn min_heap_single() {
        let mut h = MinHeap::new();
        h.push(42_i64);
        assert_eq!(h.pop(), Some(42));
    }

    // ── MaxHeap ───────────────────────────────────────────────────────────────

    #[test]
    fn max_heap_basic() {
        let mut h = MaxHeap::new();
        h.push(3_i32);
        h.push(1);
        h.push(4);
        h.push(1);
        h.push(5);
        assert_eq!(h.pop(), Some(5));
        assert_eq!(h.pop(), Some(4));
        assert_eq!(h.pop(), Some(3));
    }

    #[test]
    fn max_heap_sorted() {
        let mut h: MaxHeap<i32> = MaxHeap::new();
        for x in [4_i32, 2, 7, 1, 9, 3] {
            h.push(x);
        }
        assert_eq!(h.into_sorted_vec(), vec![9, 7, 4, 3, 2, 1]);
    }

    // ── BoundedMaxHeap ────────────────────────────────────────────────────────

    #[test]
    fn bounded_keeps_k_smallest() {
        let mut h = BoundedMaxHeap::new(3);
        for x in [5.0_f32, 1.0, 4.0, 2.0, 3.0] {
            h.push(x);
        }
        // Should contain {1, 2, 3}
        let v = h.into_sorted_asc();
        assert_eq!(v, vec![1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn bounded_full_rejects_worse() {
        let mut h = BoundedMaxHeap::new(2);
        h.push(1.0_f32);
        h.push(2.0);
        let inserted = h.push(3.0); // worse than worst (2.0)
        assert!(!inserted);
        assert_eq!(h.len(), 2);
        let v = h.into_sorted_asc();
        assert_eq!(v, vec![1.0_f32, 2.0]);
    }

    #[test]
    fn bounded_replaces_worst() {
        let mut h = BoundedMaxHeap::new(2);
        h.push(5.0_f32);
        h.push(3.0);
        let inserted = h.push(1.0); // better than worst (5.0) → replace
        assert!(inserted);
        let v = h.into_sorted_asc();
        assert_eq!(v, vec![1.0_f32, 3.0]);
    }
}
