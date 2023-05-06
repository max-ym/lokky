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

use crate::arc::AtomicCounter;
use crate::marker::{MaybeDropped, Scoped, UnsafeFrom, UnsafeInto};
use crate::scope::{AllocSelector, Envelope, ScopeRecv, Envelop};
use crate::{AllocError, ScopePtr};
use core::any::Any;
use core::cell::Cell;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::Deref;
use core::pin::Pin;
use core::ptr::drop_in_place;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use core::{fmt, mem, ptr};
use std::panic::{RefUnwindSafe, UnwindSafe};
use crate::test_log::trace;

// TODO update this docs, it seems we wont share Fence directly, instead use Envelopes!!!
/// `Fence` allows to share the data behind `Urc` between threads.
/// Each `Urc` in a single thread points to it's corresponding `Fence`.
/// Each same thread cloning of `Urc` is then performed as on plain `Rc`
/// with no performance penalty of synchronization as in `Arc`. The actual
/// synchronization is performed only when the the `Fence` itself is manipulated on.
/// For example, if all `Urc` and `Weak` are dropped for this `Fence` then
/// it is deallocated and only then the synchronization actually is performed to update
/// the global counters in the master behind the `Fences` for the data.
pub struct Fence<T: ?Sized> {
    strong: Cell<usize>,
    weak: Cell<usize>,
    master: ManuallyDrop<ScopePtr<Master<T>>>,
}

unsafe impl<T: ?Sized> Send for Fence<T> {}

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

impl<T> Fence<T> {
    /// Create `Urc` for this `Fence`.
    pub fn into_urc(self) -> Urc<T> {
        use crate::boxed::Box;

        let data = unsafe { Scoped::unsafe_from(&self.master.data).unsafe_into() };
        Urc {
            fence: ManuallyDrop::new(Box::new(self).0),
            data,
        }
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
    fence: ManuallyDrop<ScopePtr<Fence<T>>>,

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
        &urc.fence
    }

    #[inline]
    pub fn fence_weak_count(this: &Self) -> usize {
        this.fence.weak_count()
    }

    #[inline]
    pub fn fence_strong_count(this: &Self) -> usize {
        this.fence.strong_count()
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
        &self.fence.master
    }

    #[inline(always)]
    fn master_mut(&mut self) -> &mut Master<T> {
        &mut self.fence.master
    }

    #[inline(always)]
    fn fence(&self) -> &Fence<T> {
        &self.fence
    }

    #[inline(always)]
    fn fence_mut(&mut self) -> &mut Fence<T> {
        &mut self.fence
    }

    #[inline]
    pub fn try_into_fence(mut this: Self) -> Result<ScopePtr<Fence<T>>, Self> {
        if this.fence().strong_count() == 1 && this.fence().weak_count() == 0 {
            // Take out `Fence` from old Urc and "forget" it to avoid running Drop.
            let fence = unsafe {
                let dangling = ScopePtr::dangling_in(&this.fence);
                mem::replace(&mut this.fence, ManuallyDrop::new(dangling))
            };
            mem::forget(this);
            Ok(ManuallyDrop::into_inner(fence))
        } else {
            Err(this)
        }
    }
}

impl<T> Urc<T> {
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
        let fence = Fence {
            strong: 1.into(),
            weak: 0.into(),
            master: ManuallyDrop::new(ScopePtr::alloc(master, AllocSelector::new::<Master<T>>())?),
        };
        let data = unsafe { Scoped::unsafe_from(&fence.master.data).unsafe_into() };
        let urc = Urc {
            fence: ManuallyDrop::new(ScopePtr::alloc(fence, AllocSelector::new::<T>())?),
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
                let master: &mut ManuallyDrop<ScopePtr<Master<T>>> =
                    &mut *(&mut this.fence_mut().master as *mut _);
                ManuallyDrop::drop(&mut this.fence);
                // If nothing refers to the master no more then release it.
                if master.weak_fence.load(Acquire) == 0 {
                    ManuallyDrop::drop(&mut *master);
                }
            }

            // Prevent normal Drop execution - we already managed resources here.
            mem::forget(this);

            Ok(elem)
        }
    }

    pub fn make_mut(this: &mut Self) -> &mut T
    where
        T: Clone,
    {
        // Clone the data into new Urc.
        macro_rules! clone(
            () => {{
                *this = Self::new((*this.data).clone());
                unsafe { Urc::get_mut_unchecked(this) }
            }};
        );

        // Move out the data to another completely new Master.
        macro_rules! move_out(
            () => {{
                let data = unsafe { ptr::read(this.as_ref()) };
                // Set strong to zero to indicate this Fence/Master
                // does not own any data no more.
                this.fence_mut().strong.set(0);
                this.master_mut().strong_fence.store(0, Release);

                *this = Self::new(data);
                unsafe { Urc::get_mut_unchecked(this) }
            }};
        );

        if this.master().strong_fence.load(Acquire) == 1 {
            // This is the only thread that has strong references.

            if this.fence().weak_count() != 0 {
                if this.master().weak_fence.load(Acquire) == 1 {
                    // This is the only thread that has weak references.
                    //
                    // Since this is the only thread holding the data
                    // (strong and weak fences == 1)
                    // we are safe to proceed without synchronization.

                    if this.fence().strong_count() == 1 {
                        // Disassociate weak pointers from the data.
                        move_out!()
                    } else {
                        clone!()
                    }
                } else {
                    // This is _not_ the only thread that has weak references.
                    // We need to keep synchronization for data access.

                    // Move out the data to a new Urc.
                    // Since this thread still holds some Weaks, we have no fear
                    // the Master not the Fence will get deallocated by other thread
                    // during this process.
                    move_out!()
                }
            } else {
                // The other thread still owns Weaks.
                // Create a new Weak in this thread to prevent Master from deallocation
                // during the data move here. It will also clean any resources at the end
                // if needed by its Drop implementation.

                let mut weak_fence = AtomicCounter::new(&this.master().weak_fence, Acquire);
                weak_fence.increment(Release);
                let _ = unsafe {
                    Weak {
                        fence: Scoped::unsafe_from(this.fence()),
                        master: Scoped::unsafe_from(this.master()),
                    }
                };

                move_out!()
            }
        } else {
            clone!()
        }
    }

    /// # Safety
    /// There can be multiple mutable accesses, which may cause bad things.
    #[inline]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        // Accessing the master here since `data` is not mutable.
        MaybeDropped::as_mut(&mut this.master_mut().data)
    }

    #[inline]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            // This unsafety is ok because we're guaranteed that the pointer
            // returned is the *only* pointer that will ever be returned to T. Our
            // reference count is guaranteed to be 1 at this point, and we required
            // the Urc itself to be `mut`, so we're returning the only possible
            // reference to the inner data.
            unsafe { Some(Urc::get_mut_unchecked(this)) }
        } else {
            None
        }
    }

    /// Determine whether this is the unique reference (including weak refs) to
    /// the underlying data.
    #[inline]
    fn is_unique(&self) -> bool {
        // See 'Arc' impl of the same function for more details.

        if self
            .master()
            .weak_fence
            .compare_exchange(0, usize::MAX, Acquire, Relaxed)
            .is_ok()
        {
            let unique_fence = self.master().strong_fence.load(Acquire) == 1;
            if unique_fence {
                self.master().weak_fence.store(0, Release);
                // Since there is only one Fence then only this thread owns the data
                // so it is safe to proceed without synchronization.
                self.fence().strong_count() == 1 && self.fence().weak_count() == 0
            } else {
                false
            }
        } else {
            false
        }
    }
}

impl Urc<dyn Any + Send + Sync> {
    #[inline]
    pub fn downcast<T>(self) -> Result<Urc<T>, Self>
    where
        T: Any + Send + Sync,
    {
        if (*self).is::<T>() {
            unsafe {
                let fence = ManuallyDrop::new(self.fence.clone().cast::<Fence<T>>());
                let data = Scoped::unsafe_from(&fence.master.data).unsafe_into();
                mem::forget(self);
                Ok(Urc { fence, data })
            }
        } else {
            Err(self)
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
        self
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
        trace!("Dropping Urc");
        if Urc::fence_strong_count(self) == 1 && Urc::fence_weak_count(self) == 0 {
            // The fence is no more referenced so drop it.
            trace!("Will drop Fence as it is the only Urc for it");
            let master = unsafe { &mut *(self.master_mut() as *mut Master<T>) };
            unsafe { ManuallyDrop::drop(&mut self.fence) };

            if master.strong_fence.fetch_sub(1, Release) == 1 {
                trace!("Will drop Master as this also was the only Fence");
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

impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Urc<T> {}

impl<'env, T: 'env> Envelop<'env> for Urc<T> {
    type Send = Urc<T>;

    fn envelop(&self, scope: &impl ScopeRecv<'env>) -> Envelope<'env, Self::Send> {
        trace!("Enveloping Urc");
        let master = unsafe { self.fence().master.clone() };
        let cnt = master.strong_fence.fetch_add(1, Acquire);
        trace!("Previous strong fence count: {cnt}; new fence created");

        let fence = Fence {
            strong: 1.into(),
            weak: 0.into(),
            master: ManuallyDrop::new(master),
        };

        unsafe {
            Envelope::new(
                Urc {
                    fence: ManuallyDrop::new(ScopePtr::alloc(fence, AllocSelector::new::<T>()).unwrap()),
                    data: self.data.clone(),
                }
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::channel;
    use std::thread;
    use crate::scope::ScopeRecv;
    use crate::test::init;
    use crate::test_log::info;
    use super::*;

    #[test]
    fn manually_share_urc() {
        init();
        use crate::vec::Vec;

        let v = Vec::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let urc_v = Urc::new(v);
        let (tx, rx) = channel();

        thread::scope(|s| {
            info!("Spawning thread");
            let _t = s.spawn(move || {
                let urc_v: Urc<Vec<i32>> = s.recv(rx.recv().unwrap());
                assert_eq!((*urc_v)[3], 4);
            });

            info!("Sending Urc");
            tx.send(urc_v.envelop(s)).unwrap();

            assert_eq!((*urc_v)[2], 3);
            assert_eq!((*urc_v)[4], 5);
        });
    }
}
