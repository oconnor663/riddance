//! error types

use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum InsertReservedErrorKind {
    Exists,
    Dangling,
    GenerationTooNew,
    IndexOutOfBounds,
}

#[derive(Copy, Clone)]
pub struct InsertReservedError<T> {
    pub(crate) kind: InsertReservedErrorKind,
    pub(crate) inner: T,
}

impl<T> InsertReservedError<T> {
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> fmt::Debug for InsertReservedError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FillEmptyReservationError")
            .field("kind", &self.kind)
            .finish()
    }
}

impl<T> fmt::Display for InsertReservedError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let message = match self.kind {
            InsertReservedErrorKind::Exists => "entry with this ID already exists",
            InsertReservedErrorKind::Dangling => "this ID has been removed",
            InsertReservedErrorKind::GenerationTooNew => {
                "ID generation too new (dangling ID retained across recycle?)"
            }
            InsertReservedErrorKind::IndexOutOfBounds => "ID index out of bounds",
        };
        write!(f, "{}", message)
    }
}

impl<T> std::error::Error for InsertReservedError<T> {}
