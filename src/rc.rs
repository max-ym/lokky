use crate::boxed::Box;
use crate::marker::{clone_scope_access, MaybeDropped};
use crate::scope::AllocSelector;
use crate::*;
use core::cell::Cell;
use core::cmp::Ordering;
use core::fmt::Formatter;
use core::hash::{Hash, Hasher};
use core::mem::ManuallyDrop;
use core::ptr::drop_in_place;
use core::{fmt, ptr};

/// Internal allocated holder for value and counters of Rc.
struct RcBox<T: ?Sized> {
    strong: Cell<usize>,
    weak: Cell<usize>,
    val: MaybeDropped<T>,
}

impl<T: ?Sized> RcBox<T> {
    /// Whether only one reference for a value exists.
    fn is_single_ref(&self) -> bool {
        self.strong.get() == 1 && self.weak.get() == 0
    }

    /// Whether the value is deallocated (no strong references).
    fn is_dealloc(&self) -> bool {
        self.strong.get() == 0
    }

    /// Decrease strong counter.
    fn dec_strong(&self) -> usize {
        self.strong.replace(self.strong.get() - 1)
    }

    /// Set strong counter to zero.
    fn set_zero_strong(&self) {
        self.strong.set(0)
    }

    /// Increase strong counter.
    fn inc_strong(&self) -> usize {
        self.strong.replace(self.strong.get() + 1)
    }

    /// Decrease weak counter.
    fn dec_weak(&self) -> usize {
        self.weak.replace(self.weak.get() - 1)
    }

    /// Increase weak counter.
    fn inc_weak(&self) -> usize {
        self.weak.replace(self.weak.get() + 1)
    }
}

pub struct Rc<T: ?Sized>(ManuallyDrop<ScopePtr<RcBox<T>>>);

impl<T: ?Sized> Rc<T> {
    /// Access inner `RcBox`.
    fn inner(&self) -> &RcBox<T> {
        &self.0
    }

    /// Mutable access to inner `RcBox`.
    fn inner_mut(&mut self) -> &mut RcBox<T> {
        &mut self.0
    }

    pub fn as_ptr(this: &Rc<T>) -> *const T {
        this.as_ref()
    }

    pub fn downgrade(this: &Rc<T>) -> Weak<T> {
        this.inner().inc_weak();
        unsafe { Weak(Some(ManuallyDrop::new((*this.0).clone()))) }
    }

    pub fn weak_count(this: &Rc<T>) -> usize {
        this.inner().weak.get()
    }

    pub fn strong_count(this: &Rc<T>) -> usize {
        this.inner().strong.get()
    }

    pub fn get_mut(this: &mut Rc<T>) -> Option<&mut T> {
        if this.inner().is_single_ref() {
            unsafe { Some(this.as_mut()) }
        } else {
            None
        }
    }

    pub fn ptr_eq(this: &Rc<T>, other: &Rc<T>) -> bool {
        ptr::eq(this.inner(), other.inner())
    }

    /// Mutable reference for the value.
    ///
    /// # Safety
    /// This function does not check whether `RcBox` still holds the value or whether it is
    /// deallocated. So it is possible to get a reference to invalid data.
    unsafe fn as_mut(&mut self) -> &mut T {
        MaybeDropped::as_mut(&mut self.inner_mut().val)
    }
}

impl<T: 'static> Rc<T> {
    pub fn new(val: T) -> Self {
        let b = RcBox {
            strong: Cell::new(1),
            weak: Cell::new(0),
            val: val.into(),
        };
        Rc(ManuallyDrop::new(
            ScopePtr::alloc(b, AllocSelector::new::<T>()).unwrap(),
        ))
    }

    pub fn try_unwrap(this: Rc<T>) -> Result<T, Self> {
        if Rc::strong_count(&this) == 1 {
            let val = unsafe { ptr::read(&this.0.val) };
            Ok(unsafe { MaybeDropped::into_inner(val) })
        } else {
            Err(this)
        }
    }

    pub fn make_mut(this: &mut Rc<T>) -> &mut T
    where
        T: Clone,
    {
        if Rc::strong_count(this) == 1 {
            if Rc::weak_count(this) != 0 {
                // Disassociate weak pointers from the data.
                let data = unsafe { ptr::read(this.as_ref()) };
                // Set strong to zero to indicate this RcBox does not own any data no more.
                this.inner_mut().set_zero_strong();

                *this = Rc::new(data);
            }
        } else {
            let clone = this.as_ref().clone();
            *this = Rc::new(clone);
        }
        unsafe { this.as_mut() }
    }
}

impl Rc<dyn core::any::Any + 'static> {
    pub fn downcast<T: core::any::Any>(self) -> Result<Rc<T>, Self> {
        if (*self).is::<T>() {
            unsafe { Ok(Rc(ManuallyDrop::new(self.0.clone().cast()))) }
        } else {
            Err(self)
        }
    }
}

impl<T: ?Sized> Drop for Rc<T> {
    fn drop(&mut self) {
        if Rc::strong_count(self) == 1 {
            // This was the last `Rc` so drop the value.
            unsafe { drop_in_place(&mut self.inner_mut().val) }
        } else {
            self.inner().dec_strong();
        }

        if Rc::strong_count(self) == 0 && Rc::weak_count(self) == 0 {
            // Nothing refers to RcBox now so drop it.
            unsafe { drop_in_place(&mut self.0) }
        }
    }
}

impl<T: ?Sized> AsRef<T> for Rc<T> {
    fn as_ref(&self) -> &T {
        // SAFETY: value is valid as at least one Rc points to it here.
        unsafe { MaybeDropped::as_ref(&self.inner().val) }
    }
}

impl<T: ?Sized> core::ops::Deref for Rc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T: ?Sized> core::borrow::Borrow<T> for Rc<T> {
    fn borrow(&self) -> &T {
        self.as_ref()
    }
}

impl<T: ?Sized> Clone for Rc<T> {
    fn clone(&self) -> Self {
        self.inner().inc_strong();
        Rc(clone_scope_access(&self.0))
    }
}

impl<T: fmt::Debug + ?Sized> fmt::Debug for Rc<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: fmt::Display + ?Sized> fmt::Display for Rc<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: 'static> From<crate::boxed::Box<T>> for Rc<T> {
    fn from(boxed: Box<T>) -> Self {
        Rc::new(boxed.into_inner())
    }
}

impl<T: Hash + ?Sized> Hash for Rc<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: PartialEq + ?Sized> PartialEq<Rc<T>> for Rc<T> {
    fn eq(&self, other: &Rc<T>) -> bool {
        (**self).eq(&**other)
    }
}

impl<T: Eq + PartialEq + ?Sized> Eq for Rc<T> {}

impl<T: PartialOrd + ?Sized> PartialOrd<Rc<T>> for Rc<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: PartialOrd + Ord + ?Sized> Ord for Rc<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

pub struct Weak<T: ?Sized>(Option<ManuallyDrop<ScopePtr<RcBox<T>>>>);

impl<T: ?Sized> Weak<T> {
    pub fn new() -> Self {
        Weak(None)
    }

    fn inner(&self) -> Option<&RcBox<T>> {
        if let Some(v) = &self.0 {
            Some(v)
        } else {
            None
        }
    }

    pub fn upgrade(&self) -> Option<Rc<T>> {
        if let Some(v) = self.inner() {
            if !v.is_dealloc() {
                v.inc_strong();
                // Construct Rc refering to this Weak's RcBox.
                unsafe { Some(Rc(ManuallyDrop::new((*self.0.as_ref().unwrap()).clone()))) }
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn strong_count(&self) -> usize {
        if let Some(v) = self.inner() {
            v.strong.get()
        } else {
            0
        }
    }

    pub fn weak_count(&self) -> usize {
        if let Some(v) = self.inner() {
            v.weak.get()
        } else {
            0
        }
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        let this = self.inner().map(|v| v as *const _);
        let other = other.inner().map(|v| v as *const _);
        this == other
    }
}

impl<T> Weak<T> {
    pub fn as_ptr(&self) -> *const T {
        if let Some(v) = self.inner() {
            // SAFETY: don't care here as we return not a reference but a pointer which will be
            // used in unsafe code anyway.
            unsafe { MaybeDropped::as_ref(&v.val) }
        } else {
            core::ptr::null()
        }
    }
}

impl<T: ?Sized> Clone for Weak<T> {
    fn clone(&self) -> Self {
        if let Some(v) = self.inner() {
            v.inc_weak();
            unsafe { Weak(Some(ManuallyDrop::new((*self.0.as_ref().unwrap()).clone()))) }
        } else {
            Weak::new()
        }
    }
}

impl<T: fmt::Debug + ?Sized> fmt::Debug for Weak<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if let Some(v) = self.inner() {
            if v.is_dealloc() {
                write!(f, "(Weak deallocated)")
            } else {
                unsafe { write!(f, "(Weak({:?}))", MaybeDropped::as_ref(&v.val)) }
            }
        } else {
            write!(f, "(Weak null)")
        }
    }
}

impl<T: ?Sized> Default for Weak<T> {
    fn default() -> Self {
        Weak::new()
    }
}

impl<T: ?Sized> Drop for Weak<T> {
    fn drop(&mut self) {
        if let Some(v) = self.inner() {
            let weak = v.dec_weak();
            if weak == 0 && self.strong_count() == 0 {
                unsafe {
                    drop_in_place(&mut self.0);
                }
            }
        }
    }
}
