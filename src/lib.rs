#![no_std]

#[cfg(not(feature = "no_std"))]
extern crate std;

mod test_log;

mod access;
pub use access::*;

pub mod scope;

pub mod boxed;

pub mod rc;

pub mod vec;

pub mod small_vec;

pub mod string;

mod marker;

#[cfg(test)]
mod test {
    use core::mem::MaybeUninit;

    use jemallocator::Jemalloc;
    use log::LevelFilter;

    use crate::scope::AllocSelector;
    use crate::*;

    struct DummyScope(Jemalloc);
    impl scope::Scope for DummyScope {
        fn alloc_for(&self, _selector: AllocSelector) -> &dyn core::alloc::GlobalAlloc {
            &self.0
        }
    }

    static mut SCOPE: MaybeUninit<DummyScope> = MaybeUninit::uninit();
    static mut ENV: MaybeUninit<scope::Env> = MaybeUninit::uninit();

    fn env() -> &'static mut scope::Env {
        unsafe { ENV.assume_init_mut() }
    }

    pub fn init() {
        use simplelog::{ColorChoice, Config, TermLogger, TerminalMode};
        let _ = TermLogger::init(
            LevelFilter::Trace,
            Config::default(),
            TerminalMode::Stdout,
            ColorChoice::Auto,
        );
        unsafe {
            SCOPE = MaybeUninit::new(DummyScope(Jemalloc));
            ENV = MaybeUninit::new(scope::Env::new(SCOPE.assume_init_mut()));
            scope::ENV = Some(env);
        }
    }
}
