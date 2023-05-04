use crate::marker::{impose_lifetime_mut, ScopedMut, UnsafeInto};
use core::alloc::GlobalAlloc;
use core::any::TypeId;
use core::mem::transmute_copy;
use std::marker::PhantomData;

/// Function to resolve Env for current application. It should be initialized before using
/// scopes and types that rely on scope information.
pub static mut ENV: Option<fn() -> &'static mut Env> = None;

/// Environment of scopes. Provides types information about current variables and allocators.
pub struct Env {
    cur: ScopedMut<dyn Scope>,
}

/// Allocator Marker provides information to indicate which allocator to select for given
/// allocation.
#[derive(Default, Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum AllocMarker {
    /// No marker used.
    #[default]
    None,

    /// Type-based marker. It is expected to use unit structures, for example,
    /// MonsterMarker or NpcMarker.
    Type(TypeId),
}

impl AllocMarker {
    /// Create marker based on a type. It is expected to use unit structures, for example,
    /// MonsterMarker or NpcMarker.
    pub fn new<T: 'static>() -> Self {
        AllocMarker::Type(TypeId::of::<T>())
    }
}

/// Allocation selector provides information needed to select appropriate allocator
/// for some allocation.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct AllocSelector {
    marker: AllocMarker,
    ty: TypeId,
}

impl AllocSelector {
    /// Create new selector for given type.
    pub fn new<T: 'static + ?Sized>() -> Self {
        AllocSelector {
            ty: TypeId::of::<T>(),
            marker: AllocMarker::None,
        }
    }

    /// Create selector for given type with attached marker.
    pub fn with_marker<T: 'static + ?Sized>(marker: AllocMarker) -> Self {
        AllocSelector {
            ty: TypeId::of::<T>(),
            marker,
        }
    }

    /// Current allocation marker.
    pub fn marker(&self) -> AllocMarker {
        self.marker
    }

    /// Type for which the selector was created.
    pub fn ty(&self) -> TypeId {
        self.ty
    }
}

/// Scope encapsulates a group of allocators and variables for some piece of code. It
/// guarantees that all those variables and allocated objects will be valid during
/// scope lifetime.
pub trait Scope {
    /// Get allocator that can satisfy given allocation request.
    fn alloc_for(&self, selector: AllocSelector) -> &dyn GlobalAlloc;
}

impl Env {
    pub fn new(scope: &mut (dyn Scope + 'static)) -> Self {
        Env {
            cur: unsafe { scope.unsafe_into() },
        }
    }

    /// Spawn new scope to execute function in it.
    pub fn spawn<T, S>(
        &mut self,
        mut scope_init: impl FnMut(&dyn Scope) -> S,
        mut f: impl FnMut(&mut S) -> T,
    ) -> T
    where
        S: Scope + 'static,
    {
        use crate::marker::UnsafeFrom;
        unsafe {
            // SAFETY: we definitely know the lifetime will be valid during scope execution.
            let prev: &mut dyn Scope = impose_lifetime_mut(self.cur.as_mut());
            let mut new_scope = scope_init(&*prev);
            // SAFETY: new scope will survive scope execution by definition.
            self.cur = ScopedMut::unsafe_from(&mut new_scope as &mut (dyn Scope + 'static));

            let t = f(&mut new_scope);
            // SAFETY: we will give up execution to previous scope so restore previous
            // scope's lifetime by unsafe_into.
            self.cur = prev.unsafe_into();
            t
        }
    }

    /// Get current scope. The scope's lifetime is bound to the lifetime of current function
    /// (or more exactly, to the value passed to create a bound to). The lifetime
    /// is required so that the scope reference will not be returned outside of the
    /// local execution, which may as well otherwise be
    /// outside the scope that this function is returning.
    pub fn current<'local, T>(&self, _bound: &'local T) -> &'local (dyn Scope + 'local) {
        unsafe { transmute_copy(&self.cur) }
    }
}

#[inline]
fn env() -> &'static Env {
    unsafe { ENV.unwrap()() }
}

#[inline]
fn env_mut() -> &'static mut Env {
    unsafe { ENV.unwrap()() }
}

/// Get current scope. The scope's lifetime is bound to the lifetime of current function
/// (or more exactly, to the value passed to create a bound to). The lifetime
/// is required so that the scope reference will not be returned outside of the
/// local execution, which may as well be outside the scope that this function is returning.
#[inline]
pub fn current<'local, T>(bound: &'local T) -> &'local (dyn Scope + 'local) {
    env().current(bound)
}

pub fn spawn<T, S: Scope + 'static>(
    scope_init: impl FnMut(&dyn Scope) -> S,
    f: impl FnMut(&mut S) -> T,
) -> T {
    env_mut().spawn(scope_init, f)
}

/// Envelope type that can be sent into the inner scope into the another thread.
/// Envelope wraps up the normally un-[Send]-able types like [Vec] or [Arc] from
/// this library, and allows to send them into the inner scope of the another thread.
/// The value is guaranteed to be valid during the lifetime of the scope where it was
/// originally created due to lifetime constraints declared.
pub struct Envelope<'scope, T: 'scope> {
    val: T,
    _scope: PhantomData<&'scope mut T>,
}

unsafe impl<'scope, T: 'scope> Send for Envelope<'scope, T> {}

/// Trait to indicate the other thread's scope that can receive [Envelope] types.
/// The bounds in this trait are designed as such that the data is guaranteed to be valid
/// during the lifetime of the scope where they were originally created.
pub trait ScopeRecv<'inner> {
    fn recv<T: 'inner>(&self, v: Envelope<'inner, T>) -> T {
        v.val
    }

    /// Envelop the given data to be passed inside the scope to another thread.
    fn envelop<T: 'inner>(&self, val: T) -> Envelope<'inner, T> {
        Envelope {
            val,
            _scope: PhantomData,
        }
    }
}

#[cfg(not(feature = "no_std"))]
impl<'env> ScopeRecv<'env> for std::thread::Scope<'_, 'env> {}
