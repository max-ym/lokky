use crate::marker::Scoped;
use crate::scope::{AllocMarker, AllocSelector};
use core::alloc::{GlobalAlloc, Layout};
use core::marker::PhantomData;
use core::mem::transmute;
use core::ptr::NonNull;
use core::{mem, ptr, slice};

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
        unsafe {
            self.ptr.as_ptr().drop_in_place();
            self.alloc
                .dealloc(self.ptr.as_ptr() as _, Layout::for_value(self.ptr.as_ref()));
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
        ScopeAccess {
            ptr: self.ptr.cast(),
            alloc: self.alloc.clone(),
            alloc_marker: self.alloc_marker,
            _marker: Default::default(),
        }
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
        ScopeAccess {
            ptr: NonNull::new_unchecked(slice::from_raw_parts_mut(self.ptr.as_ptr(), len)),
            alloc: self.alloc.clone(),
            alloc_marker: self.alloc_marker,
            _marker: Default::default(),
        }
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
    pub fn forget(self) {}
}

impl<T> ScopeAccess<T> {
    /// Read out the accessed value and deallocate the memory.
    pub fn into_inner(self) -> T {
        unsafe {
            let v = ptr::read(self.ptr.as_ptr());
            self.alloc
                .dealloc(self.ptr.as_ptr() as _, Layout::for_value(self.ptr.as_ref()));
            // Forget access to avoid destructor run.
            mem::forget(self);
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

impl<T: 'static + ?Sized> ScopeAccess<T> {
    /// Get allocation query for given access that was used to select the allocator.
    pub fn alloc_query(&self) -> AllocSelector {
        AllocSelector::with_marker::<T>(self.marker())
    }
}

impl<T> ScopeAccess<T> {
    /// Allocate the given value on the heap of the current scope. Given query will be used
    /// to select appropriate allocator.
    pub fn alloc(value: T, query: AllocSelector) -> Self {
        let mut access = Self::alloc_layout(Layout::for_value(&value), query);
        *access.access_mut() = value;
        access
    }

    /// Allocate uninitialized array. Given query will be used
    /// to select appropriate allocator.
    pub fn alloc_array_uninit(capacity: usize, query: AllocSelector) -> Self {
        Self::alloc_layout(Layout::array::<T>(capacity).unwrap(), query)
    }

    /// Reallocate array.
    ///
    /// # Safety
    /// Old capacity should be the same as the one used to allocate the array.
    pub unsafe fn realloc_array(&mut self, old_capacity: usize, new_capacity: usize) {
        let ptr = self.alloc.realloc(
            self.ptr.as_ptr() as _,
            Layout::array::<T>(old_capacity).unwrap(),
            new_capacity,
        );
        self.ptr = NonNull::new(ptr as _).expect("array reallocation failed");
    }

    /// Allocate memory using given `Layout` and `AllocQuery`.
    fn alloc_layout(layout: Layout, query: AllocSelector) -> Self {
        let alloc = crate::scope::current().alloc_for(query);
        let mem = unsafe { alloc.alloc(layout) as *mut T };
        if let Some(ptr) = NonNull::new(mem) {
            // SAFETY: all parameters are guaranteed to be correct in the code above.
            unsafe { ScopeAccess::new(ptr, alloc, query.marker()) }
        } else {
            panic!("Allocator returned null pointer");
        }
    }
}
