use crate::{marker::MaybeDropped, scope::AllocSelector, AllocError, ScopePtr};
use core::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use core::{
    any::Any,
    fmt, hint,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops,
    pin::Pin,
    ptr::{self, drop_in_place},
    sync::atomic::{AtomicUsize, Ordering},
};

const MAX_REFCOUNT: usize = isize::MAX as _;

struct ArcInner<T: ?Sized> {
    // If `strong` is equal to usize::MAX mean a locked value. This should
    // prevent changes to `ArcInner` or `Drop` execution until unlocked.
    strong: AtomicUsize,
    weak: AtomicUsize,
    data: MaybeDropped<T>,
}

unsafe impl<T: ?Sized + Sync + Send> Send for ArcInner<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for ArcInner<T> {}

pub struct Arc<T: ?Sized>(ManuallyDrop<ScopePtr<ArcInner<T>>>);

unsafe impl<T: ?Sized + Sync + Send> Send for Arc<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for Arc<T> {}

pub struct Weak<T: ?Sized>(ManuallyDrop<ScopePtr<ArcInner<T>>>);

unsafe impl<T: ?Sized + Sync + Send> Send for Weak<T> {}
unsafe impl<T: ?Sized + Sync + Send> Sync for Weak<T> {}

impl<T: ?Sized> ArcInner<T> {
    #[inline]
    fn load_weak(&self, order: Ordering) -> AtomicCounter {
        AtomicCounter::new(&self.weak, order)
    }

    #[inline]
    fn load_strong(&self, order: Ordering) -> AtomicCounter {
        AtomicCounter::new(&self.strong, order)
    }
}

pub(crate) struct AtomicCounter<'a> {
    pub(crate) val: usize,
    pub(crate) inner: &'a AtomicUsize,
}

impl<'a> AtomicCounter<'a> {
    #[inline]
    pub(crate) fn new(inner: &'a AtomicUsize, order: Ordering) -> Self {
        AtomicCounter {
            val: inner.load(order),
            inner,
        }
    }

    /// Check if the counter is currently "locked"; if so, spin.
    #[inline]
    pub(crate) fn sync_lock(&mut self) {
        while self.is_locked() {
            hint::spin_loop();
            self.val = self.inner.load(Relaxed);
        }
    }

    #[inline]
    pub(crate) fn increment(&mut self, order: Ordering) -> usize {
        loop {
            match self.try_increment(order) {
                Ok(val) => return val,
                Err(val) => {
                    self.val = val;
                    hint::spin_loop();
                }
            }
        }
    }

    #[inline]
    pub(crate) fn try_increment(&mut self, order: Ordering) -> Result<usize, usize> {
        // TODO: this code currently ignores the possibility of overflow
        // into usize::MAX; in general both Rc and Arc need to be adjusted
        // to deal with overflow.
        let current = self.val;
        let new = current + 1;
        self.inner
            .compare_exchange_weak(current, new, order, Relaxed)
    }

    #[inline]
    pub(crate) fn try_decrement(&mut self, order: Ordering) -> Result<usize, usize> {
        debug_assert!(self.val > 0);
        self.inner
            .compare_exchange_weak(self.val, self.val - 1, order, Relaxed)
    }

    #[inline]
    pub(crate) fn try_set_zero(&mut self, order: Ordering) -> Result<usize, usize> {
        debug_assert!(self.val > 0);
        self.inner
            .compare_exchange_weak(self.val, 0, order, Relaxed)
    }

    /// Whether the counter was locked when last read.
    #[inline]
    pub(crate) fn is_locked(&self) -> bool {
        self.val == usize::MAX
    }

    #[inline]
    pub(crate) fn get(&self) -> Result<usize, ()> {
        if self.is_locked() {
            Err(())
        } else {
            Ok(self.val)
        }
    }
}

impl<'a> PartialEq<usize> for AtomicCounter<'a> {
    fn eq(&self, other: &usize) -> bool {
        self.val == *other
    }
}

impl<T: 'static> Arc<T> {
    #[inline]
    pub fn new(data: T) -> Self {
        Self::try_new(data).unwrap()
    }

    #[inline]
    pub fn try_new(data: T) -> Result<Self, AllocError> {
        let inner = ArcInner {
            strong: AtomicUsize::new(1),
            weak: AtomicUsize::new(0),
            data: data.into(),
        };
        Ok(Arc(ManuallyDrop::new(ScopePtr::alloc(
            inner,
            AllocSelector::new::<T>(),
        )?)))
    }

    #[inline]
    pub fn new_uninit() -> Arc<MaybeUninit<T>> {
        Self::try_new_uninit().unwrap()
    }

    #[inline]
    pub fn try_new_uninit() -> Result<Arc<MaybeUninit<T>>, AllocError> {
        Arc::try_new(MaybeUninit::uninit())
    }

    #[inline]
    pub fn pin(data: T) -> Pin<Self> {
        Self::try_pin(data).unwrap()
    }

    #[inline]
    pub fn try_pin(data: T) -> Result<Pin<Self>, AllocError> {
        unsafe { Ok(Pin::new_unchecked(Self::try_new(data)?)) }
    }

    pub fn make_mut(this: &mut Self) -> &mut T
    where
        T: Clone,
    {
        let mut strong = this.inner().load_strong(Acquire);
        let mut weak = this.inner().load_weak(Acquire);

        if strong == 1 {
            if weak == 0 {
                // No weaks and only one strong. We own the data and no other thread
                // cannot possibly Clone the data. We can safely return a mutable reference
                // to current Arc's data.
            } else {
                // Disassociate present weak pointers from the data.

                // Create new Weak to prevent ArcInner from deallocating while we use it here.
                // This binding also will automatically clean resources afterwards.
                //
                // The deallocation is possible because we will set `strong` to zero and
                // if all Weaks will be Dropped in the other threads, the last Weak will
                // deallocate ArcInner.
                weak.increment(Release);
                let _weak_struct = unsafe { Weak(ManuallyDrop::new(this.0.clone())) };

                // Set strong to zero to indicate ArcInner does not own any data no more.
                match strong.try_set_zero(Acquire) {
                    Ok(_) => {
                        // Move out the data.
                        let data = unsafe { ptr::read(this.as_ref()) };
                        // Create a new Arc and store the data there.
                        *this = Arc::new(data);
                    }
                    Err(_) => {
                        // Probably, `Arc` was cloned before we have disassociated.
                        // Need to create a new Arc instead without decrementing and clone the
                        // data.
                        let clone = this.as_ref().clone();
                        *this = Arc::new(clone);
                    }
                }
            }
        } else {
            // There are several `strong`s so just clone the data and create a new Arc.
            let clone = this.as_ref().clone();
            *this = Arc::new(clone);
        }

        unsafe { Arc::get_mut_unchecked(this) }
    }
}

impl<T> Arc<T> {
    pub fn try_unwrap(mut this: Self) -> Result<T, Self> {
        if this
            .inner()
            .strong
            .compare_exchange(1, 0, Acquire, Relaxed)
            .is_err()
        {
            // Exactly one strong reference is required but several exist.
            return Err(this);
        }

        unsafe {
            // Move out the data.
            let elem = ptr::read(MaybeDropped::as_ref(&this.0.data));

            // If nothing refers to the ArcInner no more then release it.
            if this.inner().weak.load(Acquire) == 0 {
                ManuallyDrop::drop(&mut this.0);
            }

            // Prevent normal Drop execution - we already managed resources here.
            mem::forget(this);

            Ok(elem)
        }
    }
}

impl<T: ?Sized> Arc<T> {
    #[inline]
    pub fn as_ptr(this: &Self) -> *const T {
        &**this
    }

    #[must_use = "this returns a new `Weak` pointer, without modifying the original `Arc`"]
    pub fn downgrade(this: &Self) -> Weak<T> {
        let mut weak = this.inner().load_weak(Relaxed);
        loop {
            weak.sync_lock();
            if weak.try_increment(Acquire).is_ok() {
                return Weak(ManuallyDrop::new(unsafe { this.0.clone() }));
            }
        }
    }

    #[inline]
    pub fn weak_count(this: &Self) -> usize {
        let cnt = this.inner().load_weak(SeqCst);
        // If the weak count is currently locked, the value of the
        // count was 0 just before taking the lock.
        cnt.get().unwrap_or(0)
    }

    #[inline]
    pub fn strong_count(this: &Self) -> usize {
        this.inner().strong.load(SeqCst)
    }

    #[inline]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        Arc::as_ptr(this) == Arc::as_ptr(other)
    }

    /// # Safety
    /// This function is unsafe because improper use may lead to memory problems.
    /// For example, two threads can acquire mutable references to the same value
    /// at the same time.
    #[inline]
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        MaybeDropped::as_mut(&mut this.0.data)
    }

    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.is_unique() {
            // This unsafety is ok because we're guaranteed that the pointer
            // returned is the *only* pointer that will ever be returned to T. Our
            // reference count is guaranteed to be 1 at this point, and we required
            // the Arc itself to be `mut`, so we're returning the only possible
            // reference to the inner data.
            unsafe { Some(Arc::get_mut_unchecked(this)) }
        } else {
            None
        }
    }

    /// Determine whether this is the unique reference (including weak refs) to
    /// the underlying data.
    ///
    /// Note that this requires locking the weak ref count.
    #[inline]
    fn is_unique(&self) -> bool {
        // Lock the weak pointer count if no Weaks exist.
        //
        // The acquire label here ensures a happens-before relationship with any
        // writes to `strong` (in particular in `Weak::upgrade`) prior to decrements
        // of the `weak` count (via `Weak::drop`, which uses release). If the upgraded
        // weak ref was never dropped, the CAS here will fail so we do not care to synchronize.
        if self
            .inner()
            .weak
            .compare_exchange(0, usize::MAX, Acquire, Relaxed)
            .is_ok()
        {
            // This needs to be an `Acquire` to synchronize with the decrement of the `strong`
            // counter in `drop` -- the only access that happens when any but the last reference
            // is being dropped.
            let unique = self.inner().strong.load(Acquire) == 1;

            // The release write here synchronizes with a read in `downgrade`,
            // effectively preventing the above read of `strong` from happening
            // after the write.
            self.inner().weak.store(0, Release); // release the lock.
            unique
        } else {
            false
        }
    }

    #[inline]
    fn inner(&self) -> &ArcInner<T> {
        &self.0
    }

    // Non-inlined part of `drop`.
    #[inline(never)]
    unsafe fn drop_slow(&mut self) {
        // Destroy the data at this time, even though we must not free the box
        // allocation itself (there might still be weak pointers lying around).
        drop_in_place(Self::get_mut_unchecked(self));

        // Release the data if Weaks are not alive.
        if self.inner().load_weak(Acquire) == 0 {
            ManuallyDrop::drop(&mut self.0);
        }
    }
}

impl Arc<dyn Any + Send + Sync> {
    #[inline]
    pub fn downcast<T>(self) -> Result<Arc<T>, Self>
    where
        T: Any + Send + Sync + 'static,
    {
        if (*self).is::<T>() {
            let ptr = unsafe { self.0.clone().cast::<ArcInner<T>>() };
            mem::forget(self);
            Ok(Arc(ManuallyDrop::new(ptr)))
        } else {
            Err(self)
        }
    }
}

impl<T: ?Sized> ops::Deref for Arc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: Arc holds a strong reference and so the data must be alive here.
        unsafe { MaybeDropped::as_ref(&self.0.data) }
    }
}

impl<T: ?Sized> AsRef<T> for Arc<T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: ?Sized> core::borrow::Borrow<T> for Arc<T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Debug> fmt::Debug for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for Arc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&(&**self as *const T), f)
    }
}

impl<T: 'static + Default> Default for Arc<T> {
    fn default() -> Arc<T> {
        Arc::new(Default::default())
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Arc<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: ?Sized + Eq + PartialEq> Eq for Arc<T> {}

impl<T: ?Sized + PartialOrd> PartialOrd for Arc<T> {
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

impl<T: ?Sized + Ord + PartialOrd> Ord for Arc<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (**self).cmp(&**other)
    }
}

impl<T: ?Sized + core::hash::Hash> core::hash::Hash for Arc<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: ?Sized + 'static> Clone for Arc<T> {
    #[inline]
    fn clone(&self) -> Self {
        // Using a relaxed ordering is alright here, as knowledge of the
        // original reference prevents other threads from erroneously deleting
        // the object.
        //
        // As explained in the [Boost documentation][1], Increasing the
        // reference counter can always be done with memory_order_relaxed: New
        // references to an object can only be formed from an existing
        // reference, and passing an existing reference from one thread to
        // another must already provide any required synchronization.
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        let old_size = self.inner().strong.fetch_add(1, Relaxed);

        // However we need to guard against massive refcounts in case someone
        // is `mem::forget`ing Arcs. If we don't do this the count can overflow
        // and users will use-after free. We racily saturate to `isize::MAX` on
        // the assumption that there aren't ~2 billion threads incrementing
        // the reference count at once. This branch will never be taken in
        // any realistic program.
        //
        // We abort because such a program is incredibly degenerate, and we
        // don't care to support it.
        if old_size > MAX_REFCOUNT {
            panic!("Arc Clone exceeded MAX_REFCOUNT of {}", MAX_REFCOUNT);
        }

        unsafe { Arc(ManuallyDrop::new(self.0.clone())) }
    }
}

impl<T: ?Sized> Drop for Arc<T> {
    #[inline]
    fn drop(&mut self) {
        // Because `fetch_sub` is already atomic, we do not need to synchronize
        // with other threads unless we are going to delete the object. This
        // same logic applies to the below `fetch_sub` to the `weak` count.
        if self.inner().strong.fetch_sub(1, Release) != 1 {
            return;
        }

        // This fence is needed to prevent reordering of use of the data and
        // deletion of the data. Because it is marked `Release`, the decreasing
        // of the reference count synchronizes with this `Acquire` fence. This
        // means that use of the data happens before decreasing the reference
        // count, which happens before this fence, which happens before the
        // deletion of the data.
        //
        // As explained in the [Boost documentation][1],
        //
        // > It is important to enforce any possible access to the object in one
        // > thread (through an existing reference) to *happen before* deleting
        // > the object in a different thread. This is achieved by a "release"
        // > operation after dropping a reference (any access to the object
        // > through this reference must obviously happened before), and an
        // > "acquire" operation before deleting the object.
        //
        // In particular, while the contents of an Arc are usually immutable, it's
        // possible to have interior writes to something like a Mutex<T>. Since a
        // Mutex is not acquired when it is deleted, we can't rely on its
        // synchronization logic to make writes in thread A visible to a destructor
        // running in thread B.
        //
        // Also note that the Acquire fence here could probably be replaced with an
        // Acquire load, which could improve performance in highly-contended
        // situations. See [2].
        //
        // [1]: (www.boost.org/doc/libs/1_55_0/doc/html/atomic/usage_examples.html)
        // [2]: (https://github.com/rust-lang/rust/pull/41714)
        core::sync::atomic::fence(Acquire);

        unsafe { self.drop_slow() }
    }
}
