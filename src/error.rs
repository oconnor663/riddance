//! error types

use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum FillEmptyReservationErrorKind {
    Exists,
    Dangling,
    GenerationTooNew,
    IndexOutOfBounds,
}

#[derive(Copy, Clone)]
pub struct FillEmptyReservationError<T> {
    pub(crate) kind: FillEmptyReservationErrorKind,
    pub(crate) inner: T,
}

impl<T> FillEmptyReservationError<T> {
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> fmt::Debug for FillEmptyReservationError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FillEmptyReservationError")
            .field("kind", &self.kind)
            .finish()
    }
}

impl<T> fmt::Display for FillEmptyReservationError<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let message = match self.kind {
            FillEmptyReservationErrorKind::Exists => "entry with this ID already exists",
            FillEmptyReservationErrorKind::Dangling => "this ID has been removed",
            FillEmptyReservationErrorKind::GenerationTooNew => {
                "ID generation too new (dangling ID retained across recycle?)"
            }
            FillEmptyReservationErrorKind::IndexOutOfBounds => "ID index out of bounds",
        };
        write!(f, "{}", message)
    }
}

impl<T> std::error::Error for FillEmptyReservationError<T> {}
