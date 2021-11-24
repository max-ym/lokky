use core::alloc::Layout;
use super::*;
use core::{cmp, mem, ptr, slice};
use core::ptr::drop_in_place;
use crate::scope::AllocSelector;

pub struct Vec<T> {
    ptr: ScopeAccess<T>,
    cap: usize,
    len: usize,
}

impl<T: 'static> Default for Vec<T> {
    fn default() -> Self {
        Vec::new()
    }
}

macro_rules! assert_index(
    ($msg:expr, $index:expr, $len:expr) => {
        if $index >= $len {
            #[cold]
            #[track_caller]
            #[inline(never)]
            fn cold(index: usize, len: usize) { panic!($msg, index, len) }
            cold($index, $len);
        }
    };
);

impl<T: 'static> Vec<T> {
    // Tiny Vecs are dumb. Skip to:
    // - 8 if the element size is 1, because any heap allocators is likely
    //   to round up a request of less than 8 bytes to at least 8 bytes.
    // - 4 if elements are moderate-sized (<= 1 KiB).
    // - 1 otherwise, to avoid wasting too much space for very short Vecs.
    const MIN_NON_ZERO_CAP: usize = if mem::size_of::<T>() == 1 {
        8
    } else if mem::size_of::<T>() <= 1024 {
        4
    } else {
        1
    };

    #[inline]
    pub fn new() -> Self {
        Vec {
            // This is safe as we do not access the memory when capacity is 0. 
            ptr: unsafe { ScopeAccess::dangling(
                crate::scope::current().alloc_for(AllocSelector::new::<T>()),
                Default::default(),
            )},
            cap: 0,
            len: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            Vec::new()
        } else if core::mem::size_of::<T>() != 0 {
            Layout::array::<T>(capacity)
                .unwrap_or_else(|_| capacity_overflow());
            alloc_guard(capacity)
                .unwrap_or_else(|_| capacity_overflow());
            let ptr = ScopeAccess::alloc_array_uninit(
                capacity, AllocSelector::new::<T>()
            );
            Vec {
                ptr,
                cap: capacity,
                len: 0,
            }
        } else {
            Vec {
                ptr: unsafe { ScopeAccess::dangling(
                    crate::scope::current().alloc_for(AllocSelector::new::<T>()),
                    Default::default(),
                )},
                cap: capacity,
                len: 0,
            }
        }
    }

    pub const fn capacity(&self) -> usize {
        if mem::size_of::<T>() == 0 {
            usize::MAX
        } else {
            self.cap
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let required_cap = self.len.checked_add(additional)
            .unwrap_or_else(|| capacity_overflow());

        if self.cap < required_cap {
            self.do_reserve(required_cap);
        }
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        #[cold]
        fn do_realloc<T: 'static>(slf: &mut Vec<T>, c: usize) {
            slf.do_realloc(c)
        }

        let required_cap = self.len.checked_add(additional)
            .unwrap_or_else(|| capacity_overflow());

        if self.cap < required_cap {
            do_realloc(self, required_cap);
        }
    }

    #[cold]
    fn do_reserve(&mut self, required_cap: usize) {
        let cap = cmp::max(self.cap * 2, required_cap);
        let cap = cmp::max(Vec::<T>::MIN_NON_ZERO_CAP, cap);
        self.do_realloc(cap);
    }

    fn do_realloc(&mut self, required_cap: usize) {
        if mem::size_of::<T>() == 0 {
            capacity_overflow();
        }

        unsafe { self.ptr.realloc_array(self.cap, required_cap); }
        self.cap = required_cap;
    }

    pub fn shrink_to_fit(&mut self) {
        if self.len < self.cap {
            self.do_realloc(self.len);
        }
    }

    pub fn shrink_to(&mut self, min_capacity: usize) {
        if self.cap > min_capacity {
            if self.len < min_capacity {
                self.do_realloc(min_capacity);
            } else {
                self.shrink_to_fit();
            }
        }
    }

    #[inline]
    pub fn into_boxed_slice(mut self) -> crate::boxed::Box<[T]> {
        self.shrink_to_fit();
        unsafe { crate::boxed::Box(self.ptr.cast_to_slice(self.len)) }
    }

    pub fn truncate(&mut self, len: usize) {
        if self.len > len {
            let remainder_len = self.len - len;
            let remainder = unsafe {
                slice::from_raw_parts_mut(
                    (self.ptr.access_mut() as *mut T).add(len),
                    remainder_len,
                )
            };

            for i in remainder {
                unsafe { drop_in_place(i); }
            }

            self.len = len;
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.access(), self.len) }
    }

    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.access_mut(), self.len) }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.access()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.access_mut()
    }

    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(self.cap >= len);
        self.len = len;
    }

    #[inline]
    pub fn swap_remove(&mut self, index: usize) -> T {
        assert_index!("swap_remove index (is {}) should be < len (is {})", index, self.len);

        unsafe {
            let last = ptr::read(self.as_ptr().add(self.len - 1));
            let hole = self.as_mut_ptr().add(index);
            self.set_len(self.len - 1);
            ptr::replace(hole, last)
        }
    }

    #[inline]
    pub fn insert(&mut self, index: usize, element: T) {
        assert_index!("insertion index (is {}) should be <= len (is {})", index, self.len);

        if self.len == self.capacity() {
            self.reserve(1);
        }

        unsafe {
            let p = self.as_mut_ptr().add(index);
            ptr::copy(p, p.add(1), self.len - index);
            ptr::write(p, element);
            self.set_len(self.len + 1);
        }
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        assert_index!("removal index (is {}) should be < len (is {})", index, self.len);

        unsafe {
            let ptr = self.as_mut_ptr().add(index);
            let ret = ptr::read(ptr);
            ptr::copy(ptr.add(1), ptr, self.len - index - 1);
            ret
        }
    }

    pub fn retain(&mut self, mut f: impl FnMut(&T) -> bool) {
        // Vec: [Kept, Kept, Hole, Hole, Hole, Hole, Unchecked, Unchecked]
        //      |<-              processed len   ->| ^- next to check
        //                  |<-  deleted cnt     ->|
        //      |<-              original_len                          ->|
        // Kept: Elements which predicate returns true on.
        // Hole: Moved or dropped element slot.
        // Unchecked: Unchecked valid elements.
        //
        // This drop guard will be invoked when predicate or `drop` of element panicked.
        // It shifts unchecked elements to cover holes and `set_len` to the correct length.
        // In cases when predicate and `drop` never panick, it will be optimized out.
        struct BackshiftOnDrop<'a, T: 'static> {
            vec: &'a mut Vec<T>,
            processed_count: usize,
            original_len: usize,
            deleted_count: usize,
        }

        impl<'a, T: 'static> Drop for BackshiftOnDrop<'a, T> {
            fn drop(&mut self) {
                if self.deleted_count > 0 {
                    // SAFETY: Trailing unchecked items must be valid since we never touch them.
                    unsafe {
                        ptr::copy(
                            self.vec.as_ptr().add(self.processed_count),
                            self.vec.as_mut_ptr().add(self.processed_count - self.deleted_count),
                            self.original_len - self.processed_count,
                        )
                    }
                }
                // SAFETY: After filling holes, all items are in contiguous memory.
                unsafe {
                    self.vec.set_len(self.original_len - self.deleted_count);
                }
            }
        }

        let original_len = self.len();
        let mut g = BackshiftOnDrop {
            vec: self,
            processed_count: 0,
            deleted_count: 0,
            original_len,
        };

        // Avoid double drop if the drop guard is not executed,
        // since we may make some holes during the process.
        unsafe { g.vec.set_len(0) };

        while g.processed_count < original_len {
            unsafe {
                // SAFETY: Unchecked element must be valid.
                let cur = g.vec.as_mut_ptr().add(g.processed_count);

                if !f(&*cur) {
                    // Advance early to avoid double drop if `drop_in_place` panicked.
                    g.processed_count += 1;
                    g.deleted_count += 1;

                    // SAFETY: We never touch this element again after dropped.
                    ptr::drop_in_place(cur);
                    continue;
                }
                if g.deleted_count > 0 {
                    // SAFETY: `deleted_count` > 0, so the hole slot must not overlap with current
                    // element. We use copy for move, and never touch this element again.
                    let hole = g.vec.as_mut_ptr().add(g.processed_count - g.deleted_count);
                    ptr::copy_nonoverlapping(cur, hole, 1);
                }
                g.processed_count += 1;
            }
        }

        // All item are processed. This can be optimized to `set_len` by LLVM.
        drop(g);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    pub fn dedup_by(&mut self, mut same_bucket: impl FnMut(&mut T, &mut T) -> bool) {
        if self.len() <= 1 {
            return;
        }

        // Drop guard that will remove holes in Vec in case same_bucket fn panics.
        struct FillGapOnDrop<'a, T: 'static> {
            vec: &'a mut Vec<T>,

            // Element index to test and drop if duplicate.
            test: usize,

            // Element index to compare with and that will remain.
            retain: usize,
        }

        impl<'a, T: 'static> Drop for FillGapOnDrop<'a, T> {
            fn drop(&mut self) {
                // The space between test and retain indices indicates the hole count.
                let delete_count = self.test - self.retain - 1;
                unsafe {
                    let hole = self.vec.as_mut_ptr().add(self.retain + 1);
                    let remainder = self.vec.as_ptr().add(self.test);
                    ptr::copy(remainder, hole, self.vec.len() - self.test);
                    self.vec.set_len(self.vec.len() - delete_count);
                }
            }
        }

        let mut g = FillGapOnDrop {
            vec: self,
            test: 1,
            retain: 0,
        };
        while g.test < g.vec.len() {
            unsafe {
                let test_ptr = g.vec.as_mut_ptr().add(g.test);
                let retain_ptr = g.vec.as_mut_ptr().add(g.retain);
                if same_bucket(&mut *test_ptr, &mut *retain_ptr) {
                    // Advance index now to correctly clean holes if Drop panic.
                    // This will mark dropped element as a hole.
                    g.test += 1;

                    drop_in_place(test_ptr);
                } else {
                    let hole = g.vec.as_mut_ptr().add(g.retain + 1);
                    if g.retain + 1 != g.test {
                        ptr::copy_nonoverlapping(test_ptr, hole, 1);
                    }

                    g.retain += 1;
                    g.test += 1;
                }
            }
        }

        drop(g);
    }

    #[inline]
    pub fn dedup_by_key<K: PartialEq>(&mut self, mut key: impl FnMut(&mut T) -> K) {
        self.dedup_by(|a, b| key(a) == key(b))
    }

    #[inline]
    pub fn push(&mut self, element: T) {
        if self.len == self.cap {
            self.reserve(1);
        }

        unsafe {
            let end = self.as_mut_ptr().add(self.len);
            ptr::write(end, element);
            self.len += 1;
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.len > 0 {
            unsafe {
                self.len -= 1;
                let end = self.as_ptr().add(self.len);
                let element = ptr::read(end);
                Some(element)
            }
        } else {
            None
        }
    }

    pub fn append(&mut self, other: &mut Self) {
        #[cold]
        fn panic_overflow() -> ! {
            panic!("after append the length of the final Vec would overflow usize");
        }

        if let Some(new_len) = self.len.checked_add(other.len) {
            unsafe {
                self.reserve(other.len);
                let other_ptr = other.as_ptr();
                let self_ptr = self.as_mut_ptr().add(self.len);
                ptr::copy_nonoverlapping(other_ptr, self_ptr, other.len);
                self.set_len(new_len);
                other.set_len(0);
            }
        } else {
            panic_overflow()
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn dedup(&mut self) where T: PartialEq {
        self.dedup_by(|a, b| a == b)
    }
}

#[cold]
fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

struct CapacityOverflowError;

#[inline]
fn alloc_guard(alloc_size: usize) -> Result<(), CapacityOverflowError> {
    if usize::BITS < 64 && alloc_size > isize::MAX as usize {
        Err(CapacityOverflowError)
    } else {
        Ok(())
    }
}
