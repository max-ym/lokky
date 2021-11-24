use core::mem::{ManuallyDrop};
use crate::ScopeAccess;

/// Marker to indicate value that should be alive for current scope in which it is used.
/// It wraps a reference to the scope's object. It is unsafe to create this wrapper as
/// it should be guaranteed that this value indeed will outlive the scope (be scope-static).
#[repr(transparent)]
pub struct ExpectSurvive<T: ?Sized + 'static>(&'static T);
pub type Scoped<T> = ExpectSurvive<T>;

/// The same as ExpectSurvive but with mutable reference.
#[repr(transparent)]
pub struct ExpectSurviveMut<T: ?Sized + 'static>(&'static mut T);
pub type ScopedMut<T> = ExpectSurviveMut<T>;

/// Indicates that a value wrapped might be dropped. This wrapper enforces a value to be
/// always dropped manually. If the wrapper goes out of scope, destructor would not be
/// automatically run.
#[repr(transparent)]
pub struct MaybeDropped<T: ?Sized>(ManuallyDrop<T>);

pub unsafe fn impose_lifetime<'new, 'old, T: ?Sized>(t: &'old T) -> &'new T {
    &*(t as *const T)
}

pub unsafe fn impose_lifetime_mut<'new, 'old, T: ?Sized>(t: &'old mut T) -> &'new mut T {
    &mut *(t as *mut T)
}

/// Indicates a value which is not safe to convert into an other type. For example it is not
/// safe to wrap a type into an ExpectSurvive wrapper as the value should be guaranteed to outlive
/// the scope. Otherwise, it might end up referring to the invalid value.
pub trait UnsafeFrom<T: ?Sized> {
    unsafe fn unsafe_from(value: T) -> Self;
}

pub trait UnsafeInto<T> {
    unsafe fn unsafe_into(self) -> T;
}

impl<T, U> UnsafeInto<U> for T
where U: UnsafeFrom<T> {
    unsafe fn unsafe_into(self) -> U {
        U::unsafe_from(self)
    }
}

impl<T: ?Sized> AsRef<T> for ExpectSurvive<T> {
    fn as_ref(&self) -> &T {
        self.0
    }
}

impl<T: ?Sized> core::borrow::Borrow<T> for ExpectSurvive<T> {
    fn borrow(&self) -> &T {
        self.0
    }
}

impl<T: ?Sized> core::ops::Deref for ExpectSurvive<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0
    }
}

impl<T: ?Sized> UnsafeFrom<&T> for ExpectSurvive<T> {
    unsafe fn unsafe_from(inner: &T) -> Self {
        ExpectSurvive(core::mem::transmute_copy(&inner))
    }
}

impl<T: ?Sized> ExpectSurvive<T> {
    pub fn into_ref(self) -> &'static T {
        self.0
    }
}

impl<T: ?Sized> Clone for ExpectSurvive<T> {
    fn clone(&self) -> Self {
        ExpectSurvive(self.0)
    }
}

impl<T: ?Sized> AsRef<T> for ExpectSurviveMut<T> {
    fn as_ref(&self) -> &T {
        self.0
    }
}

impl<T: ?Sized> AsMut<T> for ExpectSurviveMut<T> {
    fn as_mut(&mut self) -> &mut T {
        self.0
    }
}

impl<T: ?Sized> core::borrow::Borrow<T> for ExpectSurviveMut<T> {
    fn borrow(&self) -> &T {
        self.0
    }
}

impl<T: ?Sized> core::borrow::BorrowMut<T> for ExpectSurviveMut<T> {
    fn borrow_mut(&mut self) -> &mut T {
        self.0
    }
}

impl<T: ?Sized> UnsafeFrom<&mut T> for ExpectSurviveMut<T> {
    unsafe fn unsafe_from(inner: &mut T) -> Self {
        ExpectSurviveMut(core::mem::transmute_copy(&inner))
    }
}

impl<T: ?Sized> core::ops::Deref for ExpectSurviveMut<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.0
    }
}

impl<T: ?Sized> core::ops::DerefMut for ExpectSurviveMut<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0
    }
}

impl<T: ?Sized> ExpectSurviveMut<T> {
    pub fn into_ref(self) -> &'static mut T {
        self.0
    }
}

impl<T> From<T> for MaybeDropped<T> {
    fn from(value: T) -> Self {
        MaybeDropped(ManuallyDrop::new(value))
    }
}

impl<T: Clone> Clone for MaybeDropped<T> {
    fn clone(&self) -> Self {
        MaybeDropped(self.0.clone())
    }
}

impl<T: ?Sized> MaybeDropped<T> {
    /// Accessing the value is not safe as it might be dropped.
    pub unsafe fn as_ref(this: &Self) -> &T {
        &this.0
    }

    pub unsafe fn as_mut(this: &mut Self) -> &mut T {
        &mut this.0
    }
}

impl<T> MaybeDropped<T> {
    pub unsafe fn into_inner(this: Self) -> T {
        ManuallyDrop::into_inner(this.0)
    }
}

pub(crate) fn clone_scope_access<T: ?Sized>(access: &ManuallyDrop<ScopeAccess<T>>)
    -> ManuallyDrop<ScopeAccess<T>> {
    unsafe { ManuallyDrop::new((*access).clone()) }
}
