//! Universal Reference Counter.
//! It use efficient non-atomic counters for sharing the reference in the code
//! of a single thread even when the data is actually shared between several threads.
//! Atomic counters are used only to create a `Master` which can be safely sent over the thread
//! boundaries and from which normal `Urc` can be derived. This essentially speeds up the code
//! by not using (unlike Arc) atomic counter synchronization for each clone/drop operation.
//! This basically has both benefits from the thread-safe Arc and the un-`Send`-able Rc.
//!
//! The downside is that you cannot know how much actual references there exist for an object
//! if it is referred to by more than one thread since one thread does not have any means to
//! synchronize such data with each another.

use core::cell::Cell;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::Deref;
use core::pin::Pin;
use core::ptr::drop_in_place;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use core::{fmt, mem, ptr};

use crate::marker::{MaybeDropped, Scoped, UnsafeFrom, UnsafeInto};
use crate::scope::AllocSelector;
use crate::{Access, AccessMut, AllocError, ScopeAccess};

/// Urc that can be sent over the thread boundaries.
pub struct Fence<T: ?Sized> {
    strong: Cell<usize>,
    weak: Cell<usize>,
    master: ManuallyDrop<ScopeAccess<Master<T>>>,
}

impl<T: ?Sized> Fence<T> {
    #[inline]
    pub fn is_strong(&self) -> bool {
        self.strong.get() > 0
    }

    #[inline]
    pub fn strong_count(&self) -> usize {
        self.strong.get()
    }

    #[inline]
    pub fn is_weak(&self) -> bool {
        !self.is_strong()
    }

    #[inline]
    pub fn weak_count(&self) -> usize {
        self.weak.get()
    }

    #[inline]
    fn inc_weak(&self) {
        let v = self.weak.get();
        self.weak.set(v + 1);
    }

    #[inline(always)]
    fn inc_strong(&self) {
        let v = self.strong.get();
        self.strong.set(v + 1);
    }

    #[inline]
    fn dec_weak(&self) {
        let v = self.weak.get();
        debug_assert!(v > 0);
        self.weak.set(v - 1);
    }

    #[inline(always)]
    fn dec_strong(&self) {
        let v = self.strong.get();
        debug_assert!(v > 0);
        self.strong.set(v - 1);
    }
}

struct Master<T: ?Sized> {
    /// Count of `Fence` that has at least one strong reference.
    strong_fence: AtomicUsize,

    /// Count of `Fence` that has no strong references but has at least one weak reference.
    weak_fence: AtomicUsize,

    /// Actual data to which the `Urc`s are referring to.
    /// The data is dropped once there are no more strong fences.
    data: MaybeDropped<T>,
}

pub struct Urc<T: ?Sized + 'static> {
    /// A fence of this Urc.
    fence: ManuallyDrop<ScopeAccess<Fence<T>>>,

    /// Data reference shortcut to prevent multiple dereferences and speed up access.
    data: Scoped<T>,
}

pub struct Weak<T: ?Sized + 'static> {
    fence: Scoped<Fence<T>>,

    /// Master reference shortcut to prevent multiple dereferences and speed up access.
    master: Scoped<Master<T>>,
}

impl<T: ?Sized> Urc<T> {
    pub fn fence_of(urc: &Self) -> &Fence<T> {
        urc.fence.access()
    }

    #[inline]
    pub fn fence_weak_count(this: &Self) -> usize {
        this.fence.access().weak_count()
    }

    #[inline]
    pub fn fence_strong_count(this: &Self) -> usize {
        this.fence.access().strong_count()
    }

    /// Whether only one strong fence exists for this data.
    pub fn is_sole_strong_fence(this: &Self) -> bool {
        let cnt = this.master().strong_fence.load(Acquire);
        cnt == 1
    }

    pub fn as_ptr(this: &Self) -> *const T {
        this.as_ref()
    }

    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::eq(this.as_ref(), other.as_ref())
    }

    #[must_use = "this returns a new `Weak` pointer, without modifying the original `Urc`"]
    pub fn downgrade(this: &Self) -> Weak<T> {
        this.fence().inc_weak();
        unsafe {
            Weak {
                fence: Scoped::unsafe_from(this.fence()),
                master: Scoped::unsafe_from(this.master()),
            }
        }
    }

    #[inline(always)]
    fn master(&self) -> &Master<T> {
        self.fence.access().master.access()
    }

    #[inline(always)]
    fn master_mut(&mut self) -> &mut Master<T> {
        self.fence.access_mut().master.access_mut()
    }

    #[inline(always)]
    fn fence(&self) -> &Fence<T> {
        self.fence.access()
    }

    #[inline(always)]
    fn fence_mut(&mut self) -> &mut Fence<T> {
        self.fence.access_mut()
    }
}

impl<T: 'static> Urc<T> {
    #[inline]
    pub fn new(data: T) -> Self {
        Self::try_new(data).unwrap()
    }

    #[inline]
    pub fn try_new(data: T) -> Result<Self, AllocError> {
        let master = Master {
            strong_fence: 1.into(),
            weak_fence: 0.into(),
            data: data.into(),
        };
        let data = unsafe { Scoped::unsafe_from(&master.data).unsafe_into() };
        let fence = Fence {
            strong: 1.into(),
            weak: 0.into(),
            master: ManuallyDrop::new(ScopeAccess::alloc(
                master,
                AllocSelector::new::<Master<T>>(),
            )?),
        };
        let urc = Urc {
            fence: ManuallyDrop::new(ScopeAccess::alloc(fence, AllocSelector::new::<T>())?),
            data,
        };

        Ok(urc)
    }

    #[inline]
    pub fn new_uninit() -> Urc<MaybeUninit<T>> {
        Self::try_new_uninit().unwrap()
    }

    #[inline]
    pub fn try_new_uninit() -> Result<Urc<MaybeUninit<T>>, AllocError> {
        Urc::try_new(MaybeUninit::uninit())
    }

    #[inline]
    pub fn pin(data: T) -> Pin<Self> {
        Self::try_pin(data).unwrap()
    }

    #[inline]
    pub fn try_pin(data: T) -> Result<Pin<Self>, AllocError> {
        unsafe { Ok(Pin::new_unchecked(Self::try_new(data)?)) }
    }
}

impl<T> Urc<T> {
    pub fn try_unwrap(mut this: Self) -> Result<T, Self> {
        if this
            .master()
            .strong_fence
            .compare_exchange(1, 0, Acquire, Relaxed)
            .is_err()
        {
            // Exactly one strong reference is required but several strong fences exist.
            // And the strong reference count cannot be less than strong fence count.
            return Err(this);
        }

        if this.fence().strong_count() != 1 {
            // Exactly one strong reference is required but several strong references exist.
            return Err(this);
        }

        unsafe {
            // Move out the data.
            let elem = ptr::read(MaybeDropped::as_ref(&this.master().data));

            // If nothing refers to the Fence no more then release it.
            if this.fence().weak_count() == 0 {
                let master: &mut ManuallyDrop<ScopeAccess<Master<T>>> =
                    &mut *(&mut this.fence_mut().master as *mut _);
                ManuallyDrop::drop(&mut this.fence);
                // If nothing refers to the master no more then release it.
                if master.access().weak_fence.load(Acquire) == 0 {
                    ManuallyDrop::drop(&mut *master);
                }
            }

            // Prevent normal Drop execution - we already managed resources here.
            mem::forget(this);

            Ok(elem)
        }
    }
}

impl<T: ?Sized> Weak<T> {
    #[inline(always)]
    fn fence(&self) -> &Fence<T> {
        &self.fence
    }

    #[inline(always)]
    fn master(&self) -> &Master<T> {
        &self.master
    }

    #[inline(always)]
    fn fence_weak_count(&self) -> usize {
        self.fence().weak_count()
    }

    #[inline(always)]
    fn fence_strong_count(&self) -> usize {
        self.fence().strong_count()
    }
}

impl<T: ?Sized> AsRef<T> for Urc<T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        self.data.as_ref()
    }
}

impl<T: ?Sized> Deref for Urc<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        self.as_ref()
    }
}

impl<T: ?Sized> core::borrow::Borrow<T> for Urc<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

impl<T: ?Sized> Clone for Urc<T> {
    fn clone(&self) -> Self {
        self.fence().inc_strong();
        unsafe {
            Urc {
                fence: ManuallyDrop::new(self.fence.clone()),
                data: self.data.clone(),
            }
        }
    }
}

impl<T: ?Sized> Drop for Urc<T> {
    fn drop(&mut self) {
        if Urc::fence_strong_count(self) == 1 && Urc::fence_weak_count(self) == 0 {
            // The fence is no more referenced so drop it.
            let master = unsafe { &mut *(self.master_mut() as *mut Master<T>) };
            unsafe { ManuallyDrop::drop(&mut self.fence) };

            if master.strong_fence.fetch_sub(1, Release) == 1 {
                core::sync::atomic::fence(Acquire);

                // Destroy the data at this time, even though we must not free the box
                // allocation itself (there might still be weak pointers lying around).
                unsafe { drop_in_place(&mut master.data) };

                // Release the Master if Weaks are not alive.
                if master.weak_fence.load(Acquire) == 0 {
                    unsafe { drop_in_place(master) };
                }
            }
        } else {
            // There are still either other strong or some weak references.
            // Just decrement the counter and keep the Fence.
            self.fence().dec_strong()
        }
    }
}

impl<T: ?Sized> Drop for Weak<T> {
    fn drop(&mut self) {
        if self.fence_weak_count() == 1 && self.fence_strong_count() == 0 {
            // The fence is no more referenced so drop it.
            let master = self.master();
            unsafe { drop_in_place(self.fence() as *const _ as *mut Fence<T>) };

            if master.weak_fence.fetch_sub(1, Release) == 1 {
                core::sync::atomic::fence(Acquire);
                // Release the Master if Strongs are not alive.
                if master.strong_fence.load(Acquire) == 0 {
                    unsafe { drop_in_place(master as *const _ as *mut Master<T>) };
                }
            }
        } else {
            // There are still either some strong or other weak references.
            // Just decrement the counter and keep the Fence.
            self.fence().dec_weak()
        }
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Urc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Urc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for Urc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

impl<T: 'static + Default> Default for Urc<T> {
    fn default() -> Urc<T> {
        Urc::new(Default::default())
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Urc<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: ?Sized + Eq + PartialEq> Eq for Urc<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Urc<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        (**self).partial_cmp(&**other)
    }

    fn lt(&self, other: &Self) -> bool {
        *(*self) < *(*other)
    }

    fn le(&self, other: &Self) -> bool {
        *(*self) <= *(*other)
    }

    fn gt(&self, other: &Self) -> bool {
        *(*self) > *(*other)
    }

    fn ge(&self, other: &Self) -> bool {
        *(*self) >= *(*other)
    }
}

impl<T: ?Sized + Ord + PartialOrd> Ord for Urc<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + core::hash::Hash> core::hash::Hash for Urc<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}
