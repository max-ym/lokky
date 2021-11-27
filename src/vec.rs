use core::{
    borrow::{Borrow, BorrowMut},
    cmp,
    hash::{Hash, Hasher},
    mem::{size_of, MaybeUninit},
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr::{self, copy_nonoverlapping, drop_in_place},
    slice::{self, SliceIndex},
};

use super::*;
use crate::{marker::impose_lifetime_mut, scope::AllocSelector};

pub struct Vec<T: 'static> {
    ptr: ScopeAccess<[T]>,
    len: usize,
}

impl<T: 'static> Vec<T> {
    // Tiny Vecs are dumb. Skip to:
    // - 8 if the element size is 1, because any heap allocators is likely
    //   to round up a request of less than 8 bytes to at least 8 bytes.
    // - 4 if elements are moderate-sized (<= 1 KiB).
    // - 1 otherwise, to avoid wasting too much space for very short Vecs.
    const MIN_NON_ZERO_CAP: usize = if core::mem::size_of::<T>() == 1 {
        8
    } else if core::mem::size_of::<T>() <= 1024 {
        4
    } else {
        1
    };

    #[inline]
    pub fn new() -> Self {
        let alloc = scope::current().alloc_for(AllocSelector::new::<T>());
        // SAFETY: when capacity is set to zero ptr will never be accessed and may safely
        // remain dangling.
        let ptr = unsafe { ScopeAccess::<T>::dangling(alloc, Default::default()) };
        Vec {
            ptr: unsafe { ptr.cast_to_slice(0) },
            len: 0,
        }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::try_with_capacity(capacity).unwrap()
    }

    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, ArrayAllocError> {
        if capacity == 0 || size_of::<T>() == 0 {
            Ok(Vec::new())
        } else {
            let selector = AllocSelector::new::<T>();
            let ptr = ScopeAccess::alloc_array_uninit(capacity, selector)?;
            Ok(Vec { ptr, len: 0 })
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        if size_of::<T>() == 0 {
            usize::MAX
        } else {
            self.ptr.access().len()
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
        debug_assert!(self.capacity() > new_len);
        self.len = new_len;
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
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
            let slice_ptr = self.ptr.access().as_ptr();
            let after_slice_ptr = slice_ptr.add(self.len()) as _;
            slice::from_raw_parts(after_slice_ptr, self.spare_len())
        }
    }

    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe {
            let slice_ptr = self.ptr.access_mut().as_mut_ptr();
            let after_slice_ptr = slice_ptr.add(self.len()) as _;
            slice::from_raw_parts_mut(after_slice_ptr, self.spare_len())
        }
    }

    #[inline]
    pub fn spare_len(&self) -> usize {
        self.capacity() - self.len()
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
        unsafe { self.ptr.realloc_array(new_capacity) }
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
            unsafe {
                drop_in_place(ptr);
            }
        }
        // SAFETY: length is for sure less than or equal to the capacity here.
        unsafe {
            self.set_len(0);
        }
    }

    #[inline]
    pub fn push(&mut self, val: T) {
        self.reserve(1);
        let slice_ptr = self.as_mut_ptr();
        // SAFETY: offset address is assured to be correct by reserve function.
        // Reserve would otherwise fail if it was not possible to locate the value in that address
        // range.
        let insert_ptr = unsafe { slice_ptr.add(self.len()) };
        // SAFETY: ptr is within recently reserved memory region.
        unsafe {
            *insert_ptr = val;
        }
        // SAFETY: we reserved at least one element so this length will be correct and will
        // cover recently stored value.
        unsafe {
            self.set_len(self.len() + 1);
        }
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

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.reserve(other.len());
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
        self.reserve(slice.len());
        for i in slice {
            self.push(i.clone());
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
                val
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
        self.ptr.access()
    }
}

impl<T: 'static> DerefMut for Vec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ptr.access_mut()
    }
}

impl<T: 'static> Borrow<[T]> for Vec<T> {
    fn borrow(&self) -> &[T] {
        self.ptr.access()
    }
}

impl<T: 'static> BorrowMut<[T]> for Vec<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.ptr.access_mut()
    }
}

impl<T: 'static> AsRef<[T]> for Vec<T> {
    fn as_ref(&self) -> &[T] {
        self.ptr.access()
    }
}

impl<T: 'static> AsMut<[T]> for Vec<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.ptr.access_mut()
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
        for i in 0..self.len() {
            unsafe { drop_in_place(self.get_unchecked_mut(i)) }
        }
        unsafe { self.ptr.dealloc(); }
    }
}

trait OverflowGuardedAdd {
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
