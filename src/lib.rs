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

mod marker;
