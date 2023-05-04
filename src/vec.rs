use core::{
    borrow::{Borrow, BorrowMut},
    cmp,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    mem::{self, size_of, ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut, Index, IndexMut, RangeBounds},
    ptr::{self, copy_nonoverlapping, drop_in_place},
    slice::{self, SliceIndex},
};

use super::*;
use crate::test_log::trace;
use crate::{
    boxed::Box,
    marker::impose_lifetime_mut,
    scope::{AllocMarker, AllocSelector},
};

pub struct Vec<T: 'static> {
    ptr: ManuallyDrop<ScopePtr<[T]>>,
    len: usize,
}

impl<T: 'static> Vec<T> {
    // Tiny Vecs are dumb. Skip to:
    // - 8 if the element size is 1, because any heap allocators is likely
    //   to round up a request of less than 8 bytes to at least 8 bytes.
    // - 4 if elements are moderate-sized (<= 1 KiB).
    // - 1 otherwise, to avoid wasting too much space for very short Vecs.
    pub(crate) const MIN_NON_ZERO_CAP: usize = if size_of::<T>() == 1 {
        8
    } else if size_of::<T>() <= 1024 {
        4
    } else {
        1
    };

    #[inline]
    pub fn new() -> Self {
        Self::with_marker(Default::default())
    }

    #[inline]
    pub fn with_marker(marker: AllocMarker) -> Self {
        let alloc = scope::current(&()).alloc_for(AllocSelector::with_marker::<T>(marker));
        // SAFETY: when capacity is set to zero ptr will never be accessed and may safely
        // remain dangling.
        //
        // Note: we cannot create dangling ScopeAccess to unsized items yet, so no [T] here
        // but we can cast later to ScopeAccess<[T]>.
        let ptr = unsafe { ScopePtr::<T>::dangling(alloc, Default::default()) };
        Vec {
            ptr: unsafe { ManuallyDrop::new(ptr.cast_to_slice(0)) },
            len: 0,
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::try_with_capacity(capacity).unwrap()
    }

    #[inline]
    pub fn with_capacity_and_marker(capacity: usize, marker: AllocMarker) -> Self {
        Self::try_with_capacity_and_marker(capacity, marker).unwrap()
    }

    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, ArrayAllocError> {
        Self::try_with_capacity_and_marker(capacity, Default::default())
    }

    #[inline]
    pub fn try_with_capacity_and_marker(
        capacity: usize,
        marker: AllocMarker,
    ) -> Result<Self, ArrayAllocError> {
        if capacity == 0 || size_of::<T>() == 0 {
            Ok(Vec::with_marker(marker))
        } else {
            trace!(
                "try alloc for type size {} with capacity {}",
                size_of::<T>(),
                capacity
            );
            let selector = AllocSelector::with_marker::<T>(marker);
            let ptr = ManuallyDrop::new(ScopePtr::alloc_array_uninit(capacity, selector)?);
            Ok(Vec { ptr, len: 0 })
        }
    }

    #[inline]
    pub fn marker(&self) -> AllocMarker {
        self.ptr.marker()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        if size_of::<T>() == 0 {
            usize::MAX
        } else {
            self.ptr.len()
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// This is a low-level operation that maintains none of the normal
    /// invariants of the type. Normally changing the length of a vector
    /// is done using one of the safe operations instead, such as
    /// [`truncate`], [`resize`], [`extend`], or [`clear`].
    ///
    /// [`truncate`]: Vec::truncate
    /// [`resize`]: Vec::resize
    /// [`extend`]: Extend::extend
    /// [`clear`]: Vec::clear
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`].
    /// - The elements at `old_len..new_len` must be initialized.
    ///
    /// [`capacity()`]: Vec::capacity
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(self.capacity() >= new_len);
        self.len = new_len;
    }

    #[inline]
    unsafe fn as_slice_with_len_mut(&mut self, len: usize) -> &mut [T] {
        let ptr = self.ptr.as_mut_ptr();
        slice::from_raw_parts_mut(ptr, len)
    }

    #[inline]
    unsafe fn as_slice_with_len(&self, len: usize) -> &[T] {
        let ptr = self.ptr.as_ptr();
        slice::from_raw_parts(ptr, len)
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { self.as_slice_with_len(self.len) }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { self.as_slice_with_len_mut(self.len) }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    #[inline]
    pub fn as_mut(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn spare_capacity(&self) -> &[MaybeUninit<T>] {
        unsafe {
            let slice_ptr = self.as_ptr();
            let after_slice_ptr = slice_ptr.add(self.len()) as _;
            slice::from_raw_parts(after_slice_ptr, self.spare_len())
        }
    }

    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe {
            let slice_ptr = self.as_mut_ptr();
            let after_slice_ptr = slice_ptr.add(self.len()) as _;
            slice::from_raw_parts_mut(after_slice_ptr, self.spare_len())
        }
    }

    #[inline]
    pub fn spare_len(&self) -> usize {
        self.capacity() - self.len()
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.try_reserve_exact(additional).unwrap();
    }

    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let required_cap = self.len.overflow_guarded_add(additional);
        if self.capacity() < required_cap {
            self.grow_exact(required_cap).map_err(|e| e.into())
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.try_reserve(additional).unwrap();
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let required_cap = self.len.overflow_guarded_add(additional);
        if self.capacity() < required_cap {
            self.grow_amortized(required_cap).map_err(|e| e.into())
        } else {
            Ok(())
        }
    }

    #[cold]
    fn grow_amortized(&mut self, new_capacity: usize) -> Result<(), ArrayAllocError> {
        // This is ensured by the calling contexts.
        debug_assert!(self.capacity() < new_capacity);

        let new_capacity = cmp::max(new_capacity, Self::MIN_NON_ZERO_CAP);
        let new_capacity = cmp::max(new_capacity, self.capacity() * 2);

        self.grow_exact(new_capacity)
    }

    #[cold]
    fn grow_exact(&mut self, new_capacity: usize) -> Result<(), ArrayAllocError> {
        // This is ensured by the calling contexts.
        debug_assert!(self.capacity() < new_capacity);

        trace!("grow from {} to {}", self.capacity(), new_capacity);
        if self.capacity() == 0 {
            *self = Vec::try_with_capacity_and_marker(new_capacity, self.ptr.marker())?;
            Ok(())
        } else {
            unsafe { self.ptr.realloc_array(new_capacity) }
        }
    }

    #[cold]
    fn shrink(&mut self, capacity: usize) -> Result<(), ArrayAllocError> {
        // This is ensured by the calling contexts.
        debug_assert!(self.capacity() > capacity);
        debug_assert!(self.len <= capacity);
        if size_of::<T>() == 0 {
            return Ok(());
        }

        unsafe { self.ptr.realloc_array(capacity) }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if self.len() < self.capacity() {
            self.shrink(self.len()).unwrap();
        }
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        let capacity = cmp::max(self.len(), min_capacity);
        if self.capacity() > capacity {
            self.shrink(capacity).unwrap();
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        for i in 0..self.len() {
            // SAFETY: we iterate in the initialized range bound by length parameter and
            // so the values are correct.
            let ptr = unsafe { self.get_unchecked_mut(i) };
            // SAFETY: value in the array is valid to be dropped at this point.
            unsafe { drop_in_place(ptr) };
        }
        // SAFETY: length is for sure less than or equal to the capacity here.
        unsafe { self.set_len(0) };
    }

    #[inline]
    pub fn push(&mut self, element: T) {
        self.reserve_exact(1);
        let slice_ptr = self.as_mut_ptr();
        // SAFETY: offset address is assured to be correct by reserve function.
        // Reserve would otherwise fail if it was not possible to locate the value in that address
        // range.
        let insert_ptr = unsafe { slice_ptr.add(self.len()) };
        // SAFETY: ptr is within recently reserved memory region.
        unsafe { ptr::write(insert_ptr, element) };
        // SAFETY: we reserved at least one element so this length will be correct and will
        // cover recently stored value.
        unsafe { self.set_len(self.len() + 1) };
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            // SAFETY: len is positive as we checked array is not empty. Index is within
            // valid range as it is below len.
            unsafe {
                let ptr = self.get_unchecked(self.len() - 1);
                let val = ptr::read(ptr);
                self.set_len(self.len() - 1);
                Some(val)
            }
        }
    }

    #[inline]
    pub fn leak(mut self) -> &'static mut [T] {
        // SAFETY: Vec is used in some scope and slice is 'static in it as it will not be
        // ever deallocated when leaked within the scope lifetime.
        unsafe { impose_lifetime_mut(&mut self) }
    }

    /// Deallocate Vec without running Drop for the elements.
    #[inline]
    pub fn dealloc_without_drop(mut self) {
        unsafe { self.ptr.dealloc() };
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.reserve_exact(other.len());
        // SAFETY: ptr is within valid reserved range.
        let self_end_ptr = unsafe { self.as_mut_ptr().add(self.len()) as *mut T };
        let other_ptr = other.as_ptr();
        // SAFETY: memory ranges of different Vec are non-overlapping, aligned, and
        // we reserved required space in current Vec.
        unsafe {
            copy_nonoverlapping(other_ptr, self_end_ptr, other.len());
            self.set_len(self.len() + other.len());
            other.set_len(0);
        }
    }

    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T])
    where
        T: Clone,
    {
        self.reserve_exact(slice.len());
        for i in slice {
            self.push(i.clone());
        }
    }

    pub fn extend_from_iter(&mut self, iter: impl IntoIterator<Item = T>) {
        let iter = iter.into_iter();
        self.reserve_exact(iter.size_hint().0);

        for item in iter {
            self.push(item);
        }
    }

    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(self.len() > index);

        if self.len() == 1 {
            self.pop().unwrap()
        } else {
            // SAFETY: index was asserted to be within array range.
            let elem_ptr: *mut T = unsafe { self.get_unchecked_mut(index) };
            // SAFETY: index is within array range and does not underflow as len is positive.
            let last = self.len() - 1;
            let last_ptr: *mut T = unsafe { self.get_unchecked_mut(last) };
            // SAFETY: pointers are valid and non-overlapping.
            unsafe {
                let val = ptr::read(elem_ptr);
                copy_nonoverlapping(last_ptr, elem_ptr, 1);
                self.set_len(self.len() - 1);
                val
            }
        }
    }

    #[inline]
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        // Make slice be the same length as capacity. This would split off all possibly
        // uninitialized values that are there because of mismatch of length and
        // capacity.
        self.shrink_to_fit();

        let access = unsafe { self.ptr.clone() };
        // Avoid normal Drop of Vec.
        mem::forget(self);

        Box(access)
    }

    #[inline]
    pub fn split_off(&mut self, at: usize) -> Self {
        if self.len() <= at {
            Vec::new()
        } else {
            let count = self.len() - at;
            let mut vec = Vec::with_capacity(count);
            let vec_ptr = vec.as_mut_ptr();
            // SAFETY: we checked above that `at` is within array range.
            let at_ptr = unsafe { self.get_unchecked(at) };

            unsafe { copy_nonoverlapping(at_ptr, vec_ptr, count) };
            vec
        }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        assert!(index <= self.len());

        if self.is_full() {
            self.reserve_exact(1);
        }

        unsafe {
            let insert_ptr: *mut T = self
                .as_slice_with_len_mut(self.len() + 1)
                .get_unchecked_mut(index);
            // Shift everything over to make space. (Duplicating the
            // `index`th element into two consecutive places.)
            ptr::copy(insert_ptr, insert_ptr.add(1), self.len() - index);
            ptr::write(insert_ptr, element);

            self.set_len(self.len() + 1);
        }
    }

    pub fn truncate(&mut self, len: usize) {
        // Note: It's intentional that this is `>` and not `>=`.
        //       Changing it to `>=` has negative performance
        //       implications in some cases. See #78884 for more.
        if len > self.len() {
            return;
        }

        // This is safe because:
        //
        // * the slice passed to `drop_in_place` is valid; the `len > self.len`
        //   case avoids creating an invalid slice, and
        // * the `len` of the vector is shrunk before calling `drop_in_place`,
        //   such that no value will be dropped twice in case `drop_in_place`
        //   were to panic once (if it panics twice, the program aborts).
        unsafe {
            let remaining_len = self.len() - len;
            let s = ptr::slice_from_raw_parts_mut(self.as_mut_ptr().add(len), remaining_len);
            self.set_len(len);
            ptr::drop_in_place(s);
        }
    }

    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len());

        unsafe {
            let elem_ptr: *mut T = self.get_unchecked_mut(index);
            let next_ptr = elem_ptr.add(1);

            // Copy element out, unsafely having a copy of the value on
            // the stack and in the vector at the same time.
            let elem = ptr::read(elem_ptr);

            // Shift everything down to fill that spot in the vector.
            ptr::copy(next_ptr, elem_ptr, self.len() - index - 1);

            // Now only stack has the value.
            // Decrease length to account for the removed element.
            self.set_len(self.len() - 1);

            elem
        }
    }

    pub fn resize(&mut self, len: usize, val: T)
    where
        T: Clone,
    {
        ResizeWithVal(val).do_resize(self, len)
    }

    pub fn resize_with(&mut self, len: usize, f: impl FnMut() -> T) {
        ResizeWithFn(f).do_resize(self, len)
    }

    pub fn retain(&mut self, retain: impl FnMut(&T) -> bool) {
        RetainIter::retain(self, retain)
    }

    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.dedup_by(|a, b| PartialEq::eq(a, b))
    }

    pub fn dedup_by_key<K: PartialEq>(&mut self, key: impl FnMut(&mut T) -> K) {
        RetainIter::dedup_by_key(self, key)
    }

    pub fn dedup_by(&mut self, same_bucket: impl FnMut(&mut T, &mut T) -> bool) {
        RetainIter::dedup_by(self, same_bucket)
    }

    pub fn drain(&mut self, range: impl RangeBounds<usize>) -> Drain<T> {
        Drain(DrainInner::new(self, range))
    }
}

pub struct Drain<'vec, T: 'static>(DrainInner<'vec, T, Vec<T>>);

pub(crate) struct DrainInner<'vec, T: 'static, V: Vecx<T>> {
    slice: &'vec [T],
    vec_ptr: *mut V,
    start: usize,
    end: usize,
    original_len: usize,
}

impl<'vec, T: 'static, V: Vecx<T>> DrainInner<'vec, T, V> {
    pub(crate) fn new(vec: &'vec mut V, range: impl RangeBounds<usize>) -> Self {
        use core::ops::Bound::*;
        let start = match range.start_bound() {
            Unbounded => 0,
            Included(&v) => v,
            Excluded(&v) => v + 1,
        };
        let end = match range.end_bound() {
            Unbounded => vec.len(),
            Included(&v) => v + 1,
            Excluded(&v) => v,
        };
        // Decrease length so Vec does not overlap with drained slice.
        // In case Drop panics or Drain gets leaked
        // this will leave only valid values in the Vec though those elements
        // located after the slice will be leaked.
        unsafe { vec.set_len(start) };

        DrainInner {
            vec_ptr: vec,
            start,
            end,
            slice: &vec.as_slice()[start..end],
            original_len: vec.len(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        self.slice
    }
}

impl<'vec, T: 'static, V: Vecx<T>> Drop for DrainInner<'vec, T, V> {
    fn drop(&mut self) {
        // `count` reads all remaining elements out and drops them.
        self.count();

        // Regain access to the original Vec.
        // SAFETY: `slice` no longer will be accessed nor is valid. All elements of it are dropped.
        let vec = unsafe { &mut *self.vec_ptr };

        unsafe {
            // Remove gap in `vec` where `slice` has been.
            let slice_start: *mut T = vec.get_unchecked_mut(self.start);
            let slice_end = vec.get_unchecked(self.end);
            let count = self.end - self.start;
            ptr::copy(slice_end, slice_start, count);

            // Restore valid length of Vec.
            vec.set_len(self.original_len - count);
        }
    }
}

impl<'vec, T: 'static, V: Vecx<T>> Iterator for DrainInner<'vec, T, V> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(ptr) = self.slice.first().map(|v| v as *const T) {
            unsafe {
                let value = ptr::read(ptr);
                self.slice = slice::from_raw_parts(ptr.add(1), self.slice.len() - 1);
                Some(value)
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.slice.len();
        (len, Some(len))
    }
}

impl<'vec, T: 'static, V: Vecx<T>> DoubleEndedIterator for DrainInner<'vec, T, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if let Some(ptr) = self.slice.last().map(|v| v as *const T) {
            unsafe {
                let value = ptr::read(ptr);
                self.slice = slice::from_raw_parts(ptr, self.slice.len() - 1);
                Some(value)
            }
        } else {
            None
        }
    }
}

impl<'vec, T: 'static, V: Vecx<T>> ExactSizeIterator for DrainInner<'vec, T, V> {}

impl<'vec, T: 'static, V: Vecx<T>> FusedIterator for DrainInner<'vec, T, V> {}

macro_rules! impl_drain {
    ($t:ty, $($gen:tt)*) => {
        impl<$($gen)*> $t {
            pub fn as_slice(&self) -> &[T] {
                self.0.as_slice()
            }
        }

        impl<$($gen)*> Iterator for $t {
            type Item = T;

            fn next(&mut self) -> Option<Self::Item> {
                self.0.next()
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                self.0.size_hint()
            }
        }

        impl<$($gen)*> DoubleEndedIterator for $t {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.0.next_back()
            }
        }

        impl<$($gen)*> ExactSizeIterator for $t {}

        impl<$($gen)*> FusedIterator for $t {}
    };
}
pub(crate) use impl_drain;
impl_drain!(Drain<'vec, T>, 'vec, T: 'static);

pub(crate) trait ResizeWith<T: 'static>: Sized {
    fn next(&mut self) -> T;
    fn last(self) -> T;

    #[inline]
    fn do_resize(mut self, vec: &mut dyn Vecx<T>, len: usize) {
        if vec.len() >= len {
            vec.truncate(len);
        } else {
            let additional = len - vec.len();
            vec.reserve_exact(additional);

            // Add all elements except last.
            for _ in 0..(additional - 1) {
                vec.push(self.next());
            }
            // Add last element. For `resize` it will add last value without unnecessary
            // clonning it. For `resize_with` it will just call the Fn again.
            vec.push(self.last());
        }
    }
}

pub(crate) struct ResizeWithFn<T, F: FnMut() -> T>(pub(crate) F);
impl<T: 'static, F: FnMut() -> T> ResizeWith<T> for ResizeWithFn<T, F> {
    fn next(&mut self) -> T {
        self.0()
    }

    fn last(mut self) -> T {
        self.0()
    }
}

pub(crate) struct ResizeWithVal<T: Clone>(pub(crate) T);
impl<T: Clone + 'static> ResizeWith<T> for ResizeWithVal<T> {
    fn next(&mut self) -> T {
        self.0.clone()
    }

    fn last(self) -> T {
        self.0
    }
}

// To facilitate iterating through the Vec in `retain` and `dedup_*` fns:
pub(crate) struct RetainIter<'a, T: 'static> {
    vec: &'a mut dyn Vecx<T>,
    write: usize,
    read: usize,
    initial_len: usize,
}

impl<'a, T: 'static> RetainIter<'a, T> {
    /// Create new iterator and instruct whether first element should be discarded
    /// or retained.
    fn new(vec: &'a mut dyn Vecx<T>, discard_first: bool) -> Self {
        // This is ensured by the calling contexts.
        debug_assert!(!vec.is_empty());

        let initial_len = vec.len();
        if discard_first {
            // Set len to 0 in case drop panics so that invalid elements could not be
            // accessed. Len will increase gradually as new elements gets processed.
            unsafe { vec.set_len(0) };
            unsafe { drop_in_place(vec.as_slice_with_len_mut(initial_len).get_unchecked_mut(0)) };
            RetainIter {
                write: 0,
                read: 1,
                vec,
                initial_len,
            }
        } else {
            unsafe { vec.set_len(1) };
            RetainIter {
                write: 1,
                read: 2,
                vec,
                initial_len,
            }
        }
    }

    fn has_next(&self) -> bool {
        self.read < self.initial_len
    }

    fn peek(&self) -> &T {
        unsafe { self.get_unchecked(self.read) }
    }

    fn peek_mut(&mut self) -> &mut T {
        unsafe { self.get_unchecked_mut(self.read) }
    }

    unsafe fn get_unchecked(&self, index: usize) -> &T {
        self.vec
            .as_slice_with_len(self.initial_len)
            .get_unchecked(index)
    }

    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        self.vec
            .as_slice_with_len_mut(self.initial_len)
            .get_unchecked_mut(index)
    }

    /// Copy current read element from `read` to `write`.
    ///
    /// # Safety
    /// Care should be taken to forget the element in `read` and only use one stored
    /// in `write` after copying.
    unsafe fn copy(&mut self) {
        debug_assert_ne!(self.read, self.write);

        let read: *const T = self.get_unchecked(self.read);
        let write: *mut T = self.get_unchecked_mut(self.write);
        ptr::copy_nonoverlapping(read, write, 1);
    }

    /// Increment `len` in controlled `vec` to indicate processed element.
    ///
    /// # Safety
    /// Progress `len` when the valid element is in that next position. We better
    /// keep `vec` valid.
    unsafe fn increment_vec_len(&mut self) {
        self.vec.set_len(self.vec.len() + 1);
    }

    /// Retain read element and move to next.
    fn keep_next(&mut self) {
        unsafe { self.copy() };
        unsafe { self.increment_vec_len() };
        self.read += 1;
        self.write += 1;
    }

    /// Drop read element and move to next.
    fn discard_next(&mut self) {
        unsafe { drop_in_place(self.get_unchecked_mut(self.read)) };
        self.read += 1;
    }

    #[inline]
    pub fn dedup_by(vec: &'a mut dyn Vecx<T>, mut same_bucket: impl FnMut(&mut T, &mut T) -> bool) {
        let mut keeped: *mut T = if let Some(first) = vec.get_mut(0) {
            first
        } else {
            // Return on empty array.
            return;
        };
        // Shadow `same_bucket` to accept pointers. We use those to avoid errors of
        // borrowing `vec` both mutable and immutable at the same time.
        //
        // SAFETY: no references/pointers do not access the same data in this context.
        // Race condition cannot occur.
        let mut same_bucket =
            |tested: *mut T, keeped: *mut T| unsafe { same_bucket(&mut *tested, &mut *keeped) };

        let mut iter = RetainIter::new(vec, false);
        while iter.has_next() {
            let tested: *mut T = iter.peek_mut();
            if same_bucket(tested, keeped) {
                iter.discard_next();
            } else {
                iter.keep_next();
                keeped = tested;
            }
        }
    }

    #[inline]
    pub fn dedup_by_key<K: PartialEq>(vec: &mut dyn Vecx<T>, mut key: impl FnMut(&mut T) -> K) {
        let mut prev_key = if let Some(first) = vec.get_mut(0) {
            key(first)
        } else {
            // Return on empty array.
            return;
        };

        let mut iter = RetainIter::new(vec, false);
        while iter.has_next() {
            let cur_key = key(iter.peek_mut());
            if prev_key == cur_key {
                iter.discard_next();
            } else {
                iter.keep_next();
                prev_key = cur_key;
            }
        }
    }

    #[inline]
    pub fn retain(vec: &mut dyn Vecx<T>, mut retain: impl FnMut(&T) -> bool) {
        let discard_first = if let Some(first) = vec.get(0) {
            !retain(first)
        } else {
            // Return on empty array.
            return;
        };

        let mut iter = RetainIter::new(vec, discard_first);
        while iter.has_next() {
            if retain(iter.peek()) {
                iter.keep_next();
            } else {
                iter.discard_next();
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TryReserveError(ArrayAllocError);

impl From<ArrayAllocError> for TryReserveError {
    fn from(e: ArrayAllocError) -> Self {
        TryReserveError(e)
    }
}

impl<T: 'static> Default for Vec<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> Deref for Vec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: 'static> DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: 'static> Borrow<[T]> for Vec<T> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: 'static> BorrowMut<[T]> for Vec<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: 'static> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: 'static> AsMut<[T]> for Vec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: Clone + 'static> Clone for Vec<T> {
    fn clone(&self) -> Self {
        let mut vec = Vec::new();
        vec.extend_from_slice(self);
        vec
    }
}

impl<T: 'static, I: SliceIndex<[T]>> Index<I> for Vec<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.as_slice().index(index)
    }
}

impl<T: 'static, I: SliceIndex<[T]>> IndexMut<I> for Vec<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.as_mut_slice().index_mut(index)
    }
}

impl<T: PartialEq + 'static> PartialEq for Vec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq + Eq + 'static> Eq for Vec<T> {}

impl<T: PartialOrd + 'static> PartialOrd for Vec<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Ord + 'static> Ord for Vec<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: Hash + 'static> Hash for Vec<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

impl<T: 'static> Drop for Vec<T> {
    fn drop(&mut self) {
        if self.capacity() > 0 {
            trace!("dropping Vec");
            for i in 0..self.len() {
                unsafe { drop_in_place(self.get_unchecked_mut(i)) };
            }
            unsafe { self.ptr.dealloc() };
        } else {
            trace!("dropped Vec (unallocated)");
            // Not allocated. Nothing to drop.
        }
    }
}

impl<T: Clone> From<&[T]> for Vec<T> {
    fn from(slice: &[T]) -> Self {
        let mut vec = Vec::new();
        vec.clone_from_slice(slice);
        vec
    }
}

impl<T: Clone> From<&mut [T]> for Vec<T> {
    fn from(slice: &mut [T]) -> Self {
        Vec::from(&*slice)
    }
}

impl<T, const N: usize> From<[T; N]> for Vec<T> {
    fn from(slice: [T; N]) -> Self {
        let mut vec = Vec::new();
        let slice_ptr = slice.as_ptr();
        let vec_ptr = vec.as_mut_ptr();

        unsafe {
            // Resize Vec to be able to hold new copied elements.
            vec.reserve_exact(N);
            // Copy all data from slice to Vec.
            ptr::copy_nonoverlapping(slice_ptr, vec_ptr, N);
            // Set correct amount of elements.
            vec.set_len(N);
            // Forget the slice to avoid dropping. All this effectively just moves the elements
            // to Vec.
            mem::forget(slice);
        }

        vec
    }
}

impl<T> From<Box<[T]>> for Vec<T> {
    fn from(boxed: Box<[T]>) -> Self {
        let len = boxed.0.len();
        Vec {
            ptr: ManuallyDrop::new(boxed.0),
            len,
        }
    }
}

impl From<Box<str>> for Vec<u8> {
    fn from(bx: Box<str>) -> Self {
        let len = bx.len();
        let bx: Box<[u8]> = bx.into();
        Vec {
            ptr: ManuallyDrop::new(bx.0),
            len,
        }
    }
}

impl<T> FromIterator<T> for Vec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut vec = Vec::with_capacity(iter.size_hint().0);

        for item in iter {
            vec.push(item);
        }

        vec
    }
}

impl<T: 'static> IntoIterator for Vec<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            vec: self,
            index: 0,
        }
    }
}

pub struct IntoIter<T: 'static> {
    vec: Vec<T>,
    index: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(value) = self.vec.get(self.index) {
            self.index += 1;
            // SAFETY: element is moved out of Vec, will never be accessed through it.
            unsafe { Some(ptr::read(value)) }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.vec.len(), Some(self.vec.len()))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.vec.pop()
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> FusedIterator for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&mut self) {
        unsafe {
            let len = self.vec.len();

            // Disallow Vec to deallocate any remaining elements.
            self.vec.set_len(0);

            // Get elements that were not moved out of Vec.
            let unmoved_elements = slice::from_raw_parts_mut(
                self.vec.as_mut_ptr().add(self.index),
                len - self.index,
            );
            // Drop each of elements still owned by Vec.
            for element in unmoved_elements {
                drop_in_place(element);
            }
        }
    }
}

impl<T> IntoIter<T> {
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            let len = self.len() - self.index;
            let data = self.vec.as_ptr().add(self.index);
            slice::from_raw_parts(data, len)
        }
    }

    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe {
            let len = self.len() - self.index;
            let data = self.vec.as_mut_ptr().add(self.index);
            slice::from_raw_parts_mut(data, len)
        }
    }
}

impl<T> AsRef<[T]> for IntoIter<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: 'static> Vecx<T> for Vec<T> {
    #[inline(always)]
    unsafe fn set_len(&mut self, len: usize) {
        self.set_len(len)
    }

    #[inline(always)]
    unsafe fn as_slice_with_len_mut(&mut self, len: usize) -> &mut [T] {
        self.as_slice_with_len_mut(len)
    }

    #[inline(always)]
    unsafe fn as_slice_with_len(&self, len: usize) -> &[T] {
        self.as_slice_with_len(len)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }

    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }

    #[inline(always)]
    fn truncate(&mut self, len: usize) {
        self.truncate(len)
    }

    #[inline(always)]
    fn push(&mut self, val: T) {
        self.push(val)
    }

    #[inline(always)]
    fn reserve_exact(&mut self, additional: usize) {
        self.reserve_exact(additional)
    }
}

pub(crate) trait Vecx<T: 'static>: Deref<Target = [T]> + DerefMut {
    unsafe fn set_len(&mut self, len: usize);
    unsafe fn as_slice_with_len_mut(&mut self, len: usize) -> &mut [T];
    unsafe fn as_slice_with_len(&self, len: usize) -> &[T];
    fn len(&self) -> usize;
    fn as_slice(&self) -> &[T];
    fn truncate(&mut self, len: usize);
    fn push(&mut self, val: T);
    fn reserve_exact(&mut self, additional: usize);
}

pub(crate) trait OverflowGuardedAdd {
    /// Guard the value from overflowing. Use when calculating array capacity or length
    /// to be safe from overflowing. If overflow occurs the add will panic.
    fn overflow_guarded_add(&self, rhs: Self) -> Self;
}

impl OverflowGuardedAdd for usize {
    #[track_caller]
    #[inline]
    fn overflow_guarded_add(&self, rhs: Self) -> Self {
        self.checked_add(rhs)
            .unwrap_or_else(|| panic!("{:?}", ArrayAllocError::CapacityOverflow))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::init;
    use crate::test_log::info;

    #[test]
    fn create_vec() {
        init();
        Vec::<usize>::new();
    }

    #[test]
    fn create_with_capacity() {
        init();
        let vec = Vec::<u8>::with_capacity(4);
        assert!(vec.capacity() >= 4);
        info!("capacity {}", vec.capacity());
    }

    #[test]
    fn push() {
        init();
        let mut vec = Vec::with_capacity(4);
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.as_slice(), [1, 2, 3]);
    }

    #[test]
    fn push_pop() {
        init();
        let mut vec = Vec::with_capacity(4);
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert!(!vec.is_empty());
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.as_slice(), [1]);

        vec.pop();
        assert!(vec.is_empty());
    }

    #[test]
    fn reserve() {
        init();
        let mut vec = Vec::<usize>::new();
        vec.reserve(4);
        assert!(vec.capacity() >= 4);
    }

    #[test]
    fn reserve_exact_small() {
        init();
        let mut vec = Vec::<u8>::new();
        vec.reserve_exact(1);
        assert_eq!(vec.capacity(), 1);
    }

    #[test]
    fn extend_from_slice() {
        init();
        let mut vec = Vec::new();
        let slice = [1, 2, 3];
        vec.extend_from_slice(&slice);
        assert_eq!(vec.as_slice(), slice);
    }

    #[test]
    fn insert() {
        init();
        let mut vec = Vec::new();
        vec.extend_from_slice(&[1, 3, 4]);
        vec.insert(1, 2);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
        assert_eq!(vec.len(), 4);

        vec.insert(4, 5);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4, 5]);
        assert_eq!(vec.len(), 5);
    }

    #[test]
    fn delete() {
        init();
        let mut vec = Vec::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);

        vec.remove(2);
        assert_eq!(vec.as_slice(), [1, 2, 4]);
        vec.remove(0);
        assert_eq!(vec.as_slice(), [2, 4]);
        vec.remove(1);
        assert_eq!(vec.as_slice(), [2]);
        vec.remove(0);
        assert!(vec.is_empty());
    }

    #[test]
    fn clear() {
        init();
        let mut vec = Vec::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);

        vec.clear();
        assert!(vec.is_empty());
    }

    #[test]
    fn resize_more() {
        init();
        let mut vec = Vec::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);

        vec.resize(6, 2);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4, 2, 2]);
    }

    #[test]
    fn resize_less() {
        init();
        let mut vec = Vec::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);

        vec.resize(2, 0);
        assert_eq!(vec.as_slice(), [1, 2]);
    }

    #[test]
    fn retain() {
        init();
        let mut vec = Vec::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);

        vec.retain(|&x| x % 2 == 0);
        assert_eq!(vec.as_slice(), [2, 4]);
    }

    #[test]
    fn shrink_to_fit() {
        init();
        let mut vec = Vec::with_capacity(128);
        vec.extend_from_slice(&[1, 2, 3, 4]);
        vec.shrink_to_fit();
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
        assert_eq!(vec.capacity(), 4);
    }

    #[test]
    fn shrink_to() {
        init();
        let mut vec = Vec::with_capacity(128);
        vec.extend_from_slice(&[1, 2, 3, 4]);
        vec.shrink_to(8);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
        assert_eq!(vec.capacity(), 8);
    }

    #[test]
    fn shrink_to_too_small() {
        init();
        let mut vec = Vec::with_capacity(128);
        vec.extend_from_slice(&[1, 2, 3, 4]);
        vec.shrink_to(1);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
        assert_eq!(vec.capacity(), 4);
    }

    #[test]
    fn swap_remove() {
        init();
        let mut vec = Vec::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);
        let two = vec.swap_remove(1);
        assert_eq!(two, 2);
        assert_eq!(vec.as_slice(), [1, 4, 3]);
    }

    #[test]
    fn truncate() {
        init();
        let mut vec = Vec::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);
        vec.truncate(2);
        assert_eq!(vec.as_slice(), [1, 2]);
    }

    #[test]
    fn spare_capacity() {
        init();
        let mut vec = Vec::with_capacity(4);
        vec.extend_from_slice(&[1, 2, 3, 4]);
        vec.truncate(2);

        let slice = vec.spare_capacity();
        assert_eq!(slice.len(), 2);
        unsafe {
            assert_eq!(slice[0].assume_init(), 3);
            assert_eq!(slice[1].assume_init(), 4);
        }

        let slice = vec.spare_capacity_mut();
        assert_eq!(slice.len(), 2);
        unsafe {
            assert_eq!(slice[0].assume_init(), 3);
            assert_eq!(slice[1].assume_init(), 4);
        }
    }

    #[test]
    fn append() {
        init();
        let mut this = Vec::with_capacity(4);
        this.extend_from_slice(&[1, 2, 3, 4]);
        let mut other = Vec::with_capacity(4);
        other.extend_from_slice(&[5, 6, 7, 8]);

        this.append(&mut other);
        assert_eq!(this.as_slice(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(other.is_empty());
    }

    #[test]
    fn dedup() {
        init();
        let mut vec = Vec::new();
        vec.extend_from_slice(&[1, 2, 2, 2, 3, 3, 4]);
        vec.dedup();
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
    }
}
