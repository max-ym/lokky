use crate::marker::Scoped;
use crate::scope::{AllocMarker, AllocSelector};
use crate::trace;
use core::alloc::{GlobalAlloc, Layout};
use core::marker::PhantomData;
use core::mem::{self, size_of, transmute};
use core::ptr::NonNull;
use core::{ptr, slice};

/// The access to the memory location that stores some type `T`.
pub trait Access<T: ?Sized> {
    fn access(&self) -> &T;
}

/// The mutable access to the memory that stores some type `T`.
pub trait AccessMut<T: ?Sized>: Access<T> {
    fn access_mut(&mut self) -> &mut T;
}

/// A wrapper type for a reference to type `T`.
pub struct RefAccess<'a, T: ?Sized>(&'a T);

/// A wrapper type for a mutable reference to type `T`.
pub struct MutAccess<'a, T: ?Sized>(&'a mut T);

/// The access that is guaranteed to be valid in some scope.
pub struct ScopeAccess<T: ?Sized> {
    ptr: NonNull<T>,
    alloc: Scoped<dyn GlobalAlloc>,
    alloc_marker: AllocMarker,
    _marker: PhantomData<T>,
}

impl<'a, T: ?Sized> Access<T> for RefAccess<'a, T> {
    fn access(&self) -> &T {
        self.0
    }
}

impl<'a, T: ?Sized> Access<T> for MutAccess<'a, T> {
    fn access(&self) -> &T {
        self.0
    }
}

impl<'a, T: ?Sized> AccessMut<T> for MutAccess<'a, T> {
    fn access_mut(&mut self) -> &mut T {
        self.0
    }
}

impl<T: ?Sized> Access<T> for ScopeAccess<T> {
    fn access(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> AccessMut<T> for ScopeAccess<T> {
    fn access_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T: ?Sized> Drop for ScopeAccess<T> {
    fn drop(&mut self) {
        trace!("dropping ScopeAccess ({:#?})", self.ptr.as_ptr());
        unsafe {
            self.ptr.as_ptr().drop_in_place();
            self.dealloc();
        }
    }
}

impl<T: ?Sized> ScopeAccess<T> {
    /// Create scope access from raw parts.
    ///
    /// # Safety
    /// Creating scope access from raw parts is unsafe as the caller should guarantee that:
    /// * pointer is valid
    /// * provided allocator was actually the one used to allocate given pointed-to value
    /// * allocator marker should be valid and actually used for given allocation
    pub unsafe fn new(ptr: NonNull<T>, alloc: &dyn GlobalAlloc, alloc_marker: AllocMarker) -> Self {
        trace!("new ScopeAccess {:#?}", ptr.as_ptr());
        ScopeAccess {
            ptr,
            alloc: transmute(alloc),
            alloc_marker,
            _marker: Default::default(),
        }
    }

    /// Cast the type for scope access. Only pointer type is changed but actual data
    /// is not modified.
    ///
    /// # Safety
    /// The data should be correctly aligned. The pointed-to data will be reinterpreted as if it
    /// has the other type which may lead to undefined behaviour.
    pub unsafe fn cast<O>(self) -> ScopeAccess<O> {
        let access = ScopeAccess {
            ptr: self.ptr.cast(),
            alloc: self.alloc.clone(),
            alloc_marker: self.alloc_marker,
            _marker: Default::default(),
        };
        mem::forget(self);
        access
    }

    /// Change the access to the value to access to the array of values.
    ///
    /// # Safety
    /// The length of the slice should not be larger than the actual slice length so that
    /// memory accesses will be valid.
    pub unsafe fn cast_to_slice(self, len: usize) -> ScopeAccess<[T]>
    where
        T: Sized,
    {
        let access = ScopeAccess {
            ptr: NonNull::new_unchecked(slice::from_raw_parts_mut(self.ptr.as_ptr(), len)),
            alloc: self.alloc.clone(),
            alloc_marker: self.alloc_marker,
            _marker: Default::default(),
        };
        mem::forget(self);
        access
    }

    /// Change the access to the value to access to the array of values.
    ///
    /// # Safety
    /// The length of the slice should not be larger than the actual slice length so that
    /// memory accesses will be valid.
    pub unsafe fn slice_ref(&self, len: usize) -> RefAccess<[T]>
    where
        T: Sized,
    {
        RefAccess(slice::from_raw_parts(self.ptr.as_ptr(), len))
    }

    /// Change the access to the value to access to the array of values.
    ///
    /// # Safety
    /// The length of the slice should not be larger than the actual slice length so that
    /// memory accesses will be valid.
    pub unsafe fn slice_mut(&mut self, len: usize) -> MutAccess<[T]>
    where
        T: Sized,
    {
        MutAccess(slice::from_raw_parts_mut(self.ptr.as_ptr(), len))
    }

    /// Create the clone of the access.
    ///
    /// # Safety
    /// This may lead to double-free if both the original and cloned access will get dropped.
    pub unsafe fn clone(&self) -> Self {
        ScopeAccess {
            ptr: self.ptr,
            alloc: self.alloc.clone(),
            alloc_marker: self.alloc_marker,
            _marker: Default::default(),
        }
    }

    /// The marker by which the value allocator was selected.
    pub fn marker(&self) -> AllocMarker {
        self.alloc_marker
    }

    /// Forget the value without running any destructor or deallocating memory.
    pub fn forget(self) {
        mem::forget(self)
    }
}

impl<T: 'static + ?Sized> ScopeAccess<T> {
    /// Get allocation query for given access that was used to select the allocator.
    pub fn alloc_selector(&self) -> AllocSelector {
        AllocSelector::with_marker::<T>(self.marker())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArrayAllocError {
    CapacityOverflow,
    AllocError(AllocError),
}

impl<T> ScopeAccess<T> {
    /// Allocate the given value on the heap of the current scope. Given query will be used
    /// to select appropriate allocator.
    pub fn alloc(value: T, selector: AllocSelector) -> Result<Self, AllocError> {
        let mut access = Self::alloc_layout(Layout::for_value(&value), selector)?;
        *access.access_mut() = value;
        Ok(access)
    }

    /// Allocate memory using given `Layout` and `AllocQuery`.
    fn alloc_layout(layout: Layout, selector: AllocSelector) -> Result<Self, AllocError> {
        let alloc = crate::scope::current().alloc_for(selector);
        trace!("alloc with {:?}", &layout);
        let mem = unsafe { alloc.alloc(layout) as *mut T };
        if let Some(ptr) = NonNull::new(mem) {
            // SAFETY: all parameters are guaranteed to be correct in the code above.
            let alloc = unsafe { ScopeAccess::new(ptr, alloc, selector.marker()) };
            Ok(alloc)
        } else {
            Err(AllocError)
        }
    }

    /// Read out the accessed value and deallocate the memory.
    pub fn into_inner(mut self) -> T {
        unsafe {
            let v = ptr::read(self.ptr.as_ptr());
            self.dealloc();
            v
        }
    }

    /// Create the dangling access.
    ///
    /// # Safety
    /// The pointed-to memory location should not be accessed. Access destructor should
    /// be prevented from running as this would attempt to deallocate memory which was never
    /// allocated for this access.
    pub unsafe fn dangling(alloc: &dyn GlobalAlloc, alloc_marker: AllocMarker) -> Self {
        ScopeAccess {
            ptr: NonNull::dangling(),
            alloc: transmute(alloc),
            alloc_marker,
            _marker: Default::default(),
        }
    }
}

impl<T: ?Sized> ScopeAccess<T> {
    /// Deallocate memory.
    ///
    /// # Safety
    /// Memory should not be accessed and Drop execution should be prevented.
    pub unsafe fn dealloc(&mut self) {
        trace!("dealloc ScopeAccess {:#?}", self.ptr.as_ptr());
        self.alloc
            .dealloc(self.ptr.as_ptr() as _, Layout::for_value(self.ptr.as_ref()));
    }
}

impl<T> ScopeAccess<[T]> {
    /// Allocate uninitialized array. Given query will be used
    /// to select appropriate allocator.
    pub fn alloc_array_uninit(
        capacity: usize,
        selector: AllocSelector,
    ) -> Result<Self, ArrayAllocError> {
        use ArrayAllocError::*;
        ScopeAccess::<T>::alloc_layout(
            Layout::array::<T>(capacity).map_err(|_| CapacityOverflow)?,
            selector,
        )
        .map_err(AllocError)
        .map(|v| unsafe { v.cast_to_slice(capacity) })
    }

    /// Reallocate array.
    ///
    /// # Safety
    /// If new capacity is greater than previous then array will be populated with uninitialized
    /// values and caller must ensure those are not read from or else undefined behaviour.
    pub unsafe fn realloc_array(&mut self, new_capacity: usize) -> Result<(), ArrayAllocError> {
        use ArrayAllocError::*;
        trace!("realloc array");
        let ptr = self.alloc.realloc(
            self.ptr.as_ptr() as _,
            // TODO: use unwrap_unchecked instead when stable (issue 81383).
            // SAFETY: we already constructed such Layout before to allocate - it will not fail.
            Layout::array::<T>(self.access().len()).map_err(|_| CapacityOverflow)?,
            new_capacity * size_of::<T>(),
        );
        if ptr.is_null() {
            Err(AllocError(super::AllocError))
        } else {
            self.ptr = NonNull::new_unchecked(slice::from_raw_parts_mut(ptr as _, new_capacity));
            Ok(())
        }
    }
}
