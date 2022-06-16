//! Types to track the lifetime of types which may be cloned.
//!
//! This module defines two types:
//! - [`Liveness`] - Used to test if all of it's handles have been dropped.
//! - [`Alive`] - A handle which while alive ensures [`Liveness`] does not report being dropped.

use std::{
    fmt,
    sync::{Arc, Weak},
};

pub struct Liveness(Weak<()>);

impl Liveness {
    pub fn new() -> (Liveness, Alive) {
        let arc = Arc::new(());
        let weak = Arc::downgrade(&arc);

        (Liveness(weak), Alive(arc))
    }

    /// Returns whether all handles have been dropped.
    pub fn is_dropped(&self) -> bool {
        self.0.strong_count() == 0
    }
}

impl fmt::Debug for Liveness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Liveness(dropped: {})", self.is_dropped())
    }
}

#[derive(Clone)]
pub struct Alive(Arc<()>);

impl fmt::Debug for Alive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(Alive)")
    }
}
