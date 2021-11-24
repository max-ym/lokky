use crate::scope::{AllocMarker, AllocSelector};
use crate::*;
use core::cmp::Ordering;
use core::hash::Hasher;
use core::{fmt, hash, iter};

pub struct Box<T: ?Sized>(pub(crate) ScopeAccess<T>);

impl<T: 'static> Box<T> {
    pub fn new(x: T) -> Self {
        Box(ScopeAccess::alloc(x, AllocSelector::new::<T>()))
    }

    pub fn new_with(x: T, marker: AllocMarker) -> Self {
        Box(ScopeAccess::alloc(
            x,
            AllocSelector::with_marker::<T>(marker),
        ))
    }

    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

impl Box<dyn core::any::Any + 'static> {
    pub fn downcast<T: core::any::Any>(self) -> Result<Box<T>, Self> {
        if self.is::<T>() {
            unsafe { Ok(Box(self.0.cast())) }
        } else {
            Err(self)
        }
    }
}

impl<T: ?Sized> AsMut<T> for Box<T> {
    fn as_mut(&mut self) -> &mut T {
        self.0.access_mut()
    }
}

impl<T: ?Sized> AsRef<T> for Box<T> {
    fn as_ref(&self) -> &T {
        self.0.access()
    }
}

impl<T> core::borrow::Borrow<T> for Box<T> {
    fn borrow(&self) -> &T {
        self.0.access()
    }
}

impl<T> core::borrow::BorrowMut<T> for Box<T> {
    fn borrow_mut(&mut self) -> &mut T {
        self.0.access_mut()
    }
}

impl<T: fmt::Debug + ?Sized> fmt::Debug for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.access().fmt(f)
    }
}

impl<T: fmt::Display + ?Sized> fmt::Display for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.access().fmt(f)
    }
}

impl<T: ?Sized> core::ops::Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.access()
    }
}

impl<T: ?Sized> core::ops::DerefMut for Box<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.access_mut()
    }
}

impl<T: iter::Iterator + ?Sized> iter::Iterator for Box<T> {
    type Item = T::Item;

    fn next(&mut self) -> Option<T::Item> {
        (**self).next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }

    fn nth(&mut self, n: usize) -> Option<T::Item> {
        (**self).nth(n)
    }
}

impl<T: iter::DoubleEndedIterator + ?Sized> iter::DoubleEndedIterator for Box<T> {
    fn next_back(&mut self) -> Option<T::Item> {
        (**self).next_back()
    }

    fn nth_back(&mut self, n: usize) -> Option<T::Item> {
        (**self).nth_back(n)
    }
}

impl<T: iter::ExactSizeIterator + ?Sized> iter::ExactSizeIterator for Box<T> {
    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<T: iter::FusedIterator + ?Sized> iter::FusedIterator for Box<T> {}

impl<T: PartialEq + ?Sized> PartialEq for Box<T> {
    fn eq(&self, other: &Self) -> bool {
        (**self).eq(other)
    }
}

impl<T: PartialEq + Eq + ?Sized> Eq for Box<T> {}

impl<T: hash::Hash + ?Sized> hash::Hash for Box<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T: hash::Hasher + ?Sized> hash::Hasher for Box<T> {
    fn finish(&self) -> u64 {
        (**self).finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        (**self).write(bytes)
    }
}

impl<T: PartialOrd + ?Sized> PartialOrd for Box<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(other)
    }
}

impl<T: PartialEq + Eq + Ord + PartialOrd + ?Sized> Ord for Box<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(other)
    }
}
