use core::borrow::{Borrow, BorrowMut};
use core::fmt::{self, Display};
use core::hash::Hash;
use core::iter::FusedIterator;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};
use core::ptr::{self, drop_in_place};
use core::ptr::copy_nonoverlapping;
use core::slice;

use crate::boxed::Box;
use crate::scope::AllocMarker;
use crate::vec::{
    DrainInner, OverflowGuardedAdd, ResizeWith, ResizeWithFn, ResizeWithVal, RetainIter,
    TryReserveError, Vec, Vecx, impl_drain,
};
use crate::ArrayAllocError;

/// The same as `Vec` but also allows storing small slices in the stack without allocating
/// space on heap.
pub struct SmallVec<T: 'static, const N: usize> {
    storage: Storage<T, N>,
    marker: AllocMarker,
}

enum Storage<T: 'static, const N: usize> {
    Stack(StackVec<T, N>),
    Heap(Vec<T>),
}

struct StackVec<T: 'static, const N: usize> {
    slice: [MaybeUninit<T>; N],
    len: usize,
}

// Allows to redirect call to either stack or heap based function depending on
// current storage type.
macro_rules! redirect_fn {
    ($v:vis fn $f:ident (&self $(,$var:ident : $var_ty:ty)*) $(-> $t:ty)?) => {
        #[inline]
        $v fn $f(&self $(,$var : $var_ty)*) $(-> $t)? {
            use Storage::*;
            match &self.storage {
                Stack(vec) => vec.$f($($var),*),
                Heap(vec) => vec.$f($($var),*),
            }
        }
    };
    ($v:vis fn $f:ident (&mut self $(,$var:ident : $var_ty:ty)*) $(-> $t:ty)?) => {
        #[inline]
        $v fn $f(&mut self $(,$var : $var_ty)*) $(-> $t)? {
            use Storage::*;
            match &mut self.storage {
                Stack(vec) => vec.$f($($var),*),
                Heap(vec) => vec.$f($($var),*),
            }
        }
    };
    ($v:vis unsafe fn $f:ident (&mut self $(,$var:ident : $var_ty:ty)*) $(-> $t:ty)?) => {
        #[inline]
        $v unsafe fn $f(&mut self $(,$var : $var_ty)*) $(-> $t)? {
            use Storage::*;
            match &mut self.storage {
                Stack(vec) => vec.$f($($var),*),
                Heap(vec) => vec.$f($($var),*),
            }
        }
    };
    ($v:vis unsafe fn $f:ident (&self $(,$var:ident : $var_ty:ty)*) $(-> $t:ty)?) => {
        #[inline]
        $v unsafe fn $f(&self $(,$var : $var_ty)*) $(-> $t)? {
            use Storage::*;
            match &self.storage {
                Stack(vec) => vec.$f($($var),*),
                Heap(vec) => vec.$f($($var),*),
            }
        }
    };

    ($v:vis fn $f:ident (&mut self $(,$var:ident : $var_ty:ty)*) $(-> $t:ty)?; $($tt:tt)*) => {
        redirect_fn!($v fn $f (&mut self $(,$var : $var_ty)*) $(-> $t)?);
        redirect_fn!($($tt)*);
    };
    ($v:vis fn $f:ident (&self $(,$var:ident : $var_ty:ty)*) $(-> $t:ty)?; $($tt:tt)*) => {
        redirect_fn!($v fn $f (&self $(,$var : $var_ty)*) $(-> $t)?);
        redirect_fn!($($tt)*);
    };
    ($v:vis unsafe fn $f:ident (&mut self $(,$var:ident : $var_ty:ty)*) $(-> $t:ty)?; $($tt:tt)*) => {
        redirect_fn!($v unsafe fn $f (&mut self $(,$var : $var_ty)*) $(-> $t)?);
        redirect_fn!($($tt)*);
    };
    ($v:vis unsafe fn $f:ident (&self $(,$var:ident : $var_ty:ty)*) $(-> $t:ty)?; $($tt:tt)*) => {
        redirect_fn!($v unsafe fn $f (&self $(,$var : $var_ty)*) $(-> $t)?);
        redirect_fn!($($tt)*);
    };
    () => {};
}

impl<T: 'static, const N: usize> StackVec<T, N> {
    #[inline]
    pub fn new() -> Self {
        StackVec {
            // TODO: when stabilized, use uninit_slice instead.
            slice: unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() },
            len: 0,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        // Stack-based version has const size.
        N
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(N >= len);

        self.len = len;
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { self.as_slice_with_len(self.len()) }
    }

    #[inline]
    unsafe fn as_slice_with_len_mut(&mut self, len: usize) -> &mut [T] {
        let ptr = self.slice.as_mut_ptr() as *mut T;
        slice::from_raw_parts_mut(ptr, len)
    }

    #[inline]
    unsafe fn as_slice_with_len(&self, len: usize) -> &[T] {
        let ptr = self.slice.as_ptr() as *const T;
        slice::from_raw_parts(ptr, len)
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { self.as_slice_with_len_mut(self.len()) }
    }

    pub fn clear(&mut self) {
        let ptr = self.slice.as_mut_ptr() as *mut T;
        let slice = unsafe { slice::from_raw_parts_mut(ptr, self.len()) };
        unsafe { self.set_len(0) };
        for element in slice {
            unsafe { drop_in_place(element) };
        }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    #[inline]
    pub unsafe fn unchecked_push(&mut self, value: T) {
        assert!(!self.is_full());

        let index = self.len();

        // SAFETY: we just asserted that the index is in bounds.
        let extended_slice = slice::from_raw_parts_mut(self.slice.as_mut_ptr(), index + 1);
        let new_elem_place = extended_slice.get_unchecked_mut(index);

        new_elem_place.write(value);
        self.set_len(self.len() + 1);
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            unsafe {
                let last = self.len() - 1;
                let val = ptr::read(self.get_unchecked(last));
                self.set_len(self.len() - 1);
                Some(val)
            }
        }
    }

    pub unsafe fn unchecked_insert(&mut self, index: usize, value: T) {
        debug_assert!(self.len() >= index);

        let write: *mut T = self.get_unchecked_mut(index);
        // Shift everything over to make space. (Duplicating the
        // `index`th element into two consecutive places.)
        ptr::copy(write, write.add(1), self.len() - index);
        ptr::write(write, value);

        self.set_len(self.len() + 1);
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        assert!(index < self.len());
        unsafe { self.unchecked_remove(index) }
    }

    pub unsafe fn unchecked_remove(&mut self, index: usize) -> T {
        debug_assert!(index < self.len());

        // Read out value at `index`.
        let read: *mut T = self.get_unchecked_mut(index);
        let value = ptr::read(read);

        // Move elements to fill read-out gap.
        let count = self.len() - index;
        ptr::copy(
            self.as_ptr().add(index + 1),
            self.as_mut_ptr().add(index),
            count,
        );

        self.set_len(self.len() - 1);

        value
    }

    fn shrink_to_fit(&mut self) {
        // Nothing. Stack-based version has fixed capacity.
    }

    fn shrink_to(&mut self, _min_capacity: usize) {
        // Nothing.
    }

    #[inline]
    pub fn spare_len(&self) -> usize {
        self.capacity() - self.len()
    }

    pub fn truncate(&mut self, len: usize) {
        if self.len() > len {
            let old_len = self.len();
            // Reduce len in case Drop panics so that dropped data does not
            // get potentially accessed.
            unsafe { self.set_len(len) };
            for to_drop in old_len..len {
                unsafe {
                    let to_drop = self.get_unchecked_mut(to_drop);
                    drop_in_place(to_drop);
                }
            }
        }
    }

    pub fn swap_remove(&mut self, index: usize) -> T {
        assert!(self.len() > index);

        unsafe {
            // Read-out value by index.
            let remove: *mut T = self.get_unchecked_mut(index);
            let value = ptr::read(remove);

            // Copy last element in place of removed one.
            let last = self.get_unchecked(self.len() - 1);
            ptr::copy_nonoverlapping(last, remove, 1);

            // We have removed the element so reduce len.
            self.set_len(self.len() - 1);
            value
        }
    }

    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        self.dedup_by(|a, b| PartialEq::eq(a, b))
    }

    pub fn dedup_by(&mut self, same_bucket: impl FnMut(&mut T, &mut T) -> bool) {
        RetainIter::dedup_by(self, same_bucket)
    }

    pub fn dedup_by_key<K: PartialEq>(&mut self, key: impl FnMut(&mut T) -> K) {
        RetainIter::dedup_by_key(self, key)
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
}

impl<T: 'static, const N: usize> SmallVec<T, N> {
    #[inline]
    pub fn new() -> Self {
        Self::with_marker(Default::default())
    }

    #[inline]
    pub fn try_with_capacity_and_marker(
        capacity: usize,
        marker: AllocMarker,
    ) -> Result<Self, ArrayAllocError> {
        let vec = if capacity > N {
            SmallVec {
                storage: Storage::Heap(Vec::try_with_capacity_and_marker(capacity, marker)?),
                marker,
            }
        } else {
            Self::with_marker(marker)
        };
        Ok(vec)
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
    pub fn with_capacity(capacity: usize) -> Self {
        Self::try_with_capacity(capacity).unwrap()
    }

    #[inline]
    pub fn with_marker(marker: AllocMarker) -> Self {
        SmallVec {
            storage: Storage::Stack(StackVec::new()),
            marker,
        }
    }

    redirect_fn! {
        pub fn capacity(&self) -> usize;
        pub fn len(&self) -> usize;
        unsafe fn set_len(&mut self, len: usize);
        unsafe fn as_slice_with_len_mut(&mut self, len: usize) -> &mut [T];
        unsafe fn as_slice_with_len(&self, len: usize) -> &[T];
        pub fn clear(&mut self);
        pub fn is_full(&self) -> bool;
        pub fn is_empty(&self) -> bool;
        pub fn as_slice(&self) -> &[T];
        pub fn as_mut_slice(&mut self) -> &mut [T];
        pub fn shrink_to_fit(&mut self);
        pub fn shrink_to(&mut self, min_capacity: usize);
        pub fn pop(&mut self) -> Option<T>;
        pub fn remove(&mut self, index: usize) -> T;
        pub fn as_ptr(&self) -> *const T;
        pub fn as_mut_ptr(&mut self) -> *mut T;
        pub fn spare_len(&self) -> usize;
        pub fn truncate(&mut self, len: usize);
        pub fn swap_remove(&mut self, index: usize) -> T;
        pub fn dedup_by(&mut self, same_bucket: impl FnMut(&mut T, &mut T) -> bool);
        pub fn resize_with(&mut self, len: usize, f: impl FnMut() -> T);
        pub fn retain(&mut self, retain: impl FnMut(&T) -> bool);
    }

    #[inline]
    pub fn dedup(&mut self)
    where
        T: PartialEq,
    {
        use Storage::*;
        match &mut self.storage {
            Stack(vec) => vec.dedup(),
            Heap(vec) => vec.dedup(),
        }
    }

    #[inline]
    pub fn dedup_by_key<K: PartialEq>(&mut self, key: impl FnMut(&mut T) -> K) {
        use Storage::*;
        match &mut self.storage {
            Stack(vec) => vec.dedup_by_key(key),
            Heap(vec) => vec.dedup_by_key(key),
        }
    }

    #[inline]
    pub fn resize(&mut self, len: usize, val: T)
    where
        T: Clone,
    {
        if len > N {
            // Move to heap to be able to store all new elements.
            self.reserve(len - N);
        }

        use Storage::*;
        match &mut self.storage {
            Stack(vec) => vec.resize(len, val),
            Heap(vec) => vec.resize(len, val),
        }
    }

    #[inline]
    pub fn is_heap(&self) -> bool {
        use Storage::*;
        match &self.storage {
            Stack { .. } => false,
            Heap(_) => true,
        }
    }

    #[inline]
    pub fn is_stack(&self) -> bool {
        !self.is_heap()
    }

    #[inline]
    pub fn move_to_heap(&mut self) {
        self.try_move_to_heap_and_reserve_exact(0).unwrap()
    }

    pub fn try_move_to_heap_and_reserve_exact(
        &mut self,
        additional: usize,
    ) -> Result<(), ArrayAllocError> {
        use Storage::*;
        match &mut self.storage {
            Stack(vec) => {
                let old_vec = vec;
                let reserve = usize::max(old_vec.len() + additional, Vec::<T>::MIN_NON_ZERO_CAP);
                let mut new_vec = Vec::try_with_capacity_and_marker(reserve, self.marker)?;

                unsafe {
                    let read = old_vec.as_slice();
                    let write = new_vec.as_mut_ptr();
                    ptr::copy_nonoverlapping(read.as_ptr(), write, read.len());
                }
                unsafe {
                    new_vec.set_len(old_vec.len());
                    old_vec.set_len(0);
                }

                self.storage = Heap(new_vec);
            }
            Heap(vec) => vec.reserve_exact(additional),
        }
        Ok(())
    }

    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        use Storage::*;
        match &mut self.storage {
            Stack(_) => {
                if self.len().overflow_guarded_add(additional) > self.capacity() {
                    self.try_move_to_heap_and_reserve_exact(additional)
                        .map_err(|e| e.into())
                } else {
                    Ok(())
                }
            }
            Heap(vec) => vec.try_reserve_exact(additional),
        }
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        let new_capacity = self.capacity().overflow_guarded_add(additional);
        let new_capacity = usize::max(Vec::<T>::MIN_NON_ZERO_CAP, new_capacity);

        let reserve = new_capacity - self.capacity();
        self.try_reserve_exact(reserve)
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.try_reserve(additional).unwrap();
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.try_reserve_exact(additional).unwrap();
    }

    pub fn try_move_to_stack(&mut self) -> Result<(), LenTooBigError> {
        if self.is_stack() {
            return Ok(());
        }

        if self.len() <= N {
            let mut stack_vec = StackVec::new();

            // Copy all elements to stack vec.
            unsafe {
                let heap_ptr = self.as_slice().as_ptr();
                copy_nonoverlapping(heap_ptr, stack_vec.as_mut_ptr(), self.len());
                stack_vec.set_len(self.len());
            }

            // Dealloc heap Vec without dropping the values as those are moved to the stack now.
            let mut storage = Storage::Stack(stack_vec);
            core::mem::swap(&mut storage, &mut self.storage);
            if let Storage::Heap(vec) = storage {
                vec.dealloc_without_drop();
            } else {
                // `storage` should hold heap Vec here.
                unreachable!();
            }

            Ok(())
        } else {
            Err(LenTooBigError)
        }
    }

    #[inline]
    pub fn into_boxed_slice(mut self) -> Box<[T]> {
        self.move_to_heap();

        use Storage::*;
        match self.storage {
            Heap(vec) => vec.into_boxed_slice(),
            Stack(_) => unreachable!(),
        }
    }

    pub fn into_vec(mut self) -> Vec<T> {
        self.move_to_heap();

        use Storage::*;
        match self.storage {
            Heap(vec) => vec,
            Stack(_) => unreachable!(),
        }
    }

    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.reserve(other.len());

        let self_end_ptr = unsafe { self.as_mut_ptr().add(self.len()) as *mut T };
        let other_ptr = other.as_ptr();
        unsafe {
            // SAFETY: we reserved enough space to copy all the elements.
            copy_nonoverlapping(other_ptr, self_end_ptr, other.len());

            self.set_len(self.len() + other.len());
            other.set_len(0);
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        // Ensure we can store one more element. This will move to heap if stack is full.
        self.reserve(1);

        use Storage::*;
        match &mut self.storage {
            Heap(vec) => vec.push(value),
            Stack(vec) => unsafe { vec.unchecked_push(value) },
        }
    }

    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        // Ensure we can store one more element. This will move to heap if stack is full.
        self.reserve(1);

        use Storage::*;
        match &mut self.storage {
            Heap(vec) => vec.insert(index, value),
            Stack(vec) => unsafe { vec.unchecked_insert(index, value) },
        }
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

    pub fn drain(&mut self, range: impl RangeBounds<usize>) -> Drain<T, N> {
        Drain(DrainInner::new(self, range))
    }

    pub fn extend_from_slice(&mut self, slice: &[T])
    where
        T: Clone,
    {
        self.reserve(slice.len());
        for elem in slice {
            self.push(elem.clone());
        }
    }

    #[inline]
    pub fn split_off(&mut self, at: usize) -> Self {
        self.split_off_into(at)
    }

    pub fn split_off_into<const LEN: usize>(&mut self, at: usize) -> SmallVec<T, LEN> {
        let slice = &mut self[at] as *mut T;
        let count = self.len() - at;
        let mut into = SmallVec::with_capacity(count);
        unsafe {
            copy_nonoverlapping(slice, into.as_mut_ptr(), count);
            self.set_len(self.len() - count);
            into.set_len(count);
        }
        into
    }
}

pub struct Drain<'vec, T: 'static, const N: usize>(DrainInner<'vec, T, SmallVec<T, N>>);
impl_drain!(Drain<'vec, T, N>, 'vec, T, const N: usize);

impl<T: 'static, const N: usize> Vecx<T> for SmallVec<T, N> {
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

#[derive(Clone, Copy, Debug)]
pub struct LenTooBigError;

impl Display for LenTooBigError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "failed to move Vec to stack as len is too big")
    }
}

impl<T: 'static, const N: usize> Deref for StackVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: 'static, const N: usize> DerefMut for StackVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Clone + 'static, const N: usize> Clone for StackVec<T, N> {
    fn clone(&self) -> Self {
        let mut vec = Self::new();
        for value in self.as_slice() {
            unsafe { vec.unchecked_push(value.clone()) };
        }
        vec
    }
}

impl<T: 'static, const N: usize> Default for StackVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static, const N: usize> Drop for StackVec<T, N> {
    fn drop(&mut self) {
        for value in self.as_mut_slice() {
            unsafe { drop_in_place(value) };
        }
    }
}

impl<T: Eq + 'static, const N: usize> Eq for StackVec<T, N> {}

impl<T: PartialEq + 'static, const N: usize> PartialEq for StackVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialOrd + 'static, const N: usize> PartialOrd for StackVec<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Ord + 'static, const N: usize> Ord for StackVec<T, N> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: Hash + 'static, const N: usize> Hash for StackVec<T, N> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

impl<T: 'static, const N: usize> Index<usize> for StackVec<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T: 'static, const N: usize> Vecx<T> for StackVec<T, N> {
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
        debug_assert!(!self.is_full());
        unsafe { self.unchecked_push(val) };
    }

    #[inline(always)]
    fn reserve_exact(&mut self, additional: usize) {
        debug_assert!(self.len() + additional <= N);
    }
}

impl<T: 'static, const N: usize> Default for SmallVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static, const N: usize> Deref for SmallVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: 'static, const N: usize> DerefMut for SmallVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: 'static, const N: usize> Borrow<[T]> for SmallVec<T, N> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: 'static, const N: usize> BorrowMut<[T]> for SmallVec<T, N> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: 'static, const N: usize> AsRef<[T]> for SmallVec<T, N> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: 'static, const N: usize> AsMut<[T]> for SmallVec<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: 'static, const N: usize> AsRef<SmallVec<T, N>> for SmallVec<T, N> {
    fn as_ref(&self) -> &SmallVec<T, N> {
        self
    }
}

impl<T: 'static, const N: usize> AsMut<SmallVec<T, N>> for SmallVec<T, N> {
    fn as_mut(&mut self) -> &mut SmallVec<T, N> {
        self
    }
}

impl<T: Clone + 'static, const N: usize> Clone for SmallVec<T, N> {
    fn clone(&self) -> Self {
        let data = self.as_slice();
        Self::from(data)
    }
}

impl<T: fmt::Debug + 'static, const N: usize> fmt::Debug for SmallVec<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: PartialEq + 'static, const N: usize> PartialEq for SmallVec<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq + 'static, const N: usize> Eq for SmallVec<T, N> {}

impl<T: PartialOrd + 'static, const N: usize> PartialOrd for SmallVec<T, N> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Ord + 'static, const N: usize> Ord for SmallVec<T, N> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: 'static, const N: usize> Index<usize> for SmallVec<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T: 'static, const N: usize> IndexMut<usize> for SmallVec<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

impl<T: Hash + 'static, const N: usize> Hash for SmallVec<T, N> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state)
    }
}

impl<T: Clone + 'static, const N: usize> From<&[T]> for SmallVec<T, N> {
    fn from(slice: &[T]) -> Self {
        let mut vec = Self::with_capacity(slice.len());
        vec.extend_from_slice(slice);
        vec
    }
}

impl<T: Clone + 'static, const N: usize> From<&mut [T]> for SmallVec<T, N> {
    fn from(slice: &mut [T]) -> Self {
        Self::from(&*slice)
    }
}

impl<T: 'static, const N: usize, const L: usize> From<[T; L]> for SmallVec<T, N> {
    fn from(slice: [T; L]) -> Self {
        if L <= N {
            let mut vec = StackVec::new();
            let slice = slice.map(MaybeUninit::new);
            // SAFETY: we checked that slice is smaller than target StackVec and will fit.
            unsafe {
                ptr::copy_nonoverlapping(
                    &slice as *const MaybeUninit<T>,
                    &mut vec.slice as *mut MaybeUninit<T>,
                    L,
                )
            };
            SmallVec {
                storage: Storage::Stack(vec),
                marker: Default::default(),
            }
        } else {
            SmallVec {
                storage: Storage::Heap(Vec::from(slice)),
                marker: Default::default(),
            }
        }
    }
}

impl<T: 'static, const N: usize> From<Vec<T>> for SmallVec<T, N> {
    fn from(vec: Vec<T>) -> Self {
        let marker = vec.marker();
        SmallVec {
            storage: Storage::Heap(vec),
            marker,
        }
    }
}

impl<T: 'static, const N: usize> From<Box<[T]>> for SmallVec<T, N> {
    fn from(b: Box<[T]>) -> Self {
        Self::from(Vec::from(b))
    }
}

impl<T: Copy + 'static, const N: usize> Extend<&'static T> for SmallVec<T, N> {
    fn extend<I: IntoIterator<Item = &'static T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for &elem in iter {
            self.push(elem);
        }
    }
}

impl<T: 'static, const N: usize> Extend<T> for SmallVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for elem in iter {
            self.push(elem);
        }
    }
}

impl<T: 'static, const N: usize> FromIterator<T> for SmallVec<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut vec = SmallVec::new();
        vec.extend(iter);
        vec
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::test::init;

    #[test]
    fn create_vec() {
        init();
        SmallVec::<usize, 10>::new();
    }

    #[test]
    fn alloc_vec() {
        init();
        let vec = SmallVec::<usize, 10>::with_capacity(11);
        assert!(vec.capacity() >= 11);
    }

    #[test]
    fn push() {
        init();
        let mut vec = SmallVec::<usize, 3>::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.as_slice(), [1, 2, 3]);
    }

    #[test]
    fn push_heap() {
        init();
        let mut vec = SmallVec::<usize, 1>::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.as_slice(), [1, 2, 3]);
    }

    #[test]
    fn push_pop() {
        init();
        let mut vec = SmallVec::<usize, 1>::new();
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
        let mut vec = SmallVec::<usize, 1>::new();
        vec.reserve_exact(4);
        assert!(vec.capacity() >= 4);
    }

    #[test]
    fn reserve_too_small() {
        init();
        let mut vec = SmallVec::<u8, 0>::new();
        vec.reserve_exact(1);
        assert!(vec.capacity() > 1);
    }

    #[test]
    fn extend_from_slice() {
        init();
        let mut vec = SmallVec::<i32, 2>::new();
        let slice = [1, 2, 3];
        vec.extend_from_slice(&slice);
        assert_eq!(vec.as_slice(), slice);
    }

    #[test]
    fn insert() {
        init();
        let mut vec = SmallVec::<i32, 4>::new();
        vec.extend_from_slice(&[1, 3, 4]);
        vec.insert(1, 2);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
        assert_eq!(vec.len(), 4);
        assert!(vec.is_stack());

        vec.insert(4, 5);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4, 5]);
        assert_eq!(vec.len(), 5);
        assert!(vec.is_heap());
    }

    #[test]
    fn delete() {
        init();
        let mut vec = SmallVec::<i32, 4>::new();
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
        let mut vec = SmallVec::<i32, 4>::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);

        vec.clear();
        assert!(vec.is_empty());
    }

    #[test]
    fn resize_more() {
        init();
        let mut vec = SmallVec::<i32, 4>::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);

        vec.resize(6, 2);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4, 2, 2]);
    }

    #[test]
    fn resize_less() {
        init();
        let mut vec = SmallVec::<i32, 4>::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);

        vec.resize(2, 0);
        assert_eq!(vec.as_slice(), [1, 2]);
    }

    #[test]
    fn retain() {
        init();
        let mut vec = SmallVec::<i32, 4>::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);

        vec.retain(|&x| x % 2 == 0);
        assert_eq!(vec.as_slice(), [2, 4]);
    }

    #[test]
    fn shrink_to_fit() {
        init();
        let mut vec = SmallVec::<i32, 4>::with_capacity(128);
        vec.extend_from_slice(&[1, 2, 3, 4]);
        vec.shrink_to_fit();
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
        assert_eq!(vec.capacity(), 4);
    }

    #[test]
    fn shrink_to() {
        init();
        let mut vec = SmallVec::<i32, 4>::with_capacity(128);
        vec.extend_from_slice(&[1, 2, 3, 4]);
        vec.shrink_to(8);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
        assert_eq!(vec.capacity(), 8);
    }

    #[test]
    fn shrink_to_too_small() {
        init();
        let mut vec = SmallVec::<i32, 4>::with_capacity(128);
        vec.extend_from_slice(&[1, 2, 3, 4]);
        vec.shrink_to(1);
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
        assert_eq!(vec.capacity(), 4);
    }

    #[test]
    fn swap_remove() {
        init();
        let mut vec = SmallVec::<i32, 4>::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);
        let two = vec.swap_remove(1);
        assert_eq!(two, 2);
        assert_eq!(vec.as_slice(), [1, 4, 3]);
    }

    #[test]
    fn truncate() {
        init();
        let mut vec = SmallVec::<i32, 4>::new();
        vec.extend_from_slice(&[1, 2, 3, 4]);
        vec.truncate(2);
        assert_eq!(vec.as_slice(), [1, 2]);
    }

    #[test]
    fn spare_capacity() {
        init();
        let mut vec = SmallVec::<i32, 4>::with_capacity(4);
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
        let mut this = SmallVec::<i32, 4>::with_capacity(4);
        this.extend_from_slice(&[1, 2, 3, 4]);
        let mut other = SmallVec::<i32, 4>::with_capacity(4);
        other.extend_from_slice(&[5, 6, 7, 8]);

        this.append(&mut other);
        assert_eq!(this.as_slice(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(other.is_empty());
    }

    #[test]
    fn dedup() {
        init();
        let mut vec = SmallVec::<i32, 4>::new();
        vec.extend_from_slice(&[1, 2, 2, 2, 3, 3, 4]);
        vec.dedup();
        assert_eq!(vec.as_slice(), [1, 2, 3, 4]);
    }
}
