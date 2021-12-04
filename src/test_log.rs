#[cfg(test)]
#[macro_export(crate)]
macro_rules! trace {
    (target: $target:expr, $($arg:tt)+) => (
        log::trace!(target: $target, $($arg)+)
    );
    ($($arg:tt)+) => (
        log::trace!($($arg)+)
    )
}

#[cfg(not(test))]
#[macro_export(crate)]
macro_rules! trace {
    (target: $target:expr, $($arg:tt)+) => (
        // Nothing.
    );
    ($($arg:tt)+) => (
        // Nothing.
    )
}

#[cfg(test)]
#[macro_export(crate)]
macro_rules! info {
    (target: $target:expr, $($arg:tt)+) => (
        log::info!(target: $target, $($arg)+)
    );
    ($($arg:tt)+) => (
        log::info!($($arg)+)
    )
}

#[cfg(not(test))]
#[macro_export(crate)]
macro_rules! info {
    (target: $target:expr, $($arg:tt)+) => (
        // Nothing.
    );
    ($($arg:tt)+) => (
        // Nothing.
    )
}
