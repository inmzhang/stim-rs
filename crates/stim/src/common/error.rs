use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// A type alias for results that may contain a [`StimError`].
///
/// This is the standard return type for all fallible operations in the `stim`
/// crate. It is a convenience alias so that callers do not need to import
/// `StimError` separately when propagating errors with `?`.
///
/// # Examples
///
/// ```
/// fn parse_circuit(text: &str) -> stim::Result<stim::Circuit> {
///     text.parse::<stim::Circuit>()
///         .map_err(|e| stim::StimError::new(e.to_string()))
/// }
/// ```
pub type Result<T> = std::result::Result<T, StimError>;

/// An error produced by the Stim library.
///
/// This is the unified error type for all fallible operations in the `stim`
/// crate. It wraps a human-readable message string describing what went wrong.
///
/// Most functions and methods in this crate return [`Result<T>`](crate::Result),
/// which is an alias for `std::result::Result<T, StimError>`. When an operation
/// fails -- for example because a circuit contains an invalid gate name, a
/// target index is out of range, a file cannot be read, or a stabilizer
/// operation is mathematically invalid -- the error message explains the cause.
///
/// `StimError` implements the standard [`Error`](std::error::Error) trait, so it
/// integrates smoothly with the Rust error-handling ecosystem (including
/// `anyhow`, `eyre`, and the `?` operator). It also converts automatically
/// from C++ exceptions raised by the underlying Stim C++ library via the
/// `From<cxx::Exception>` implementation.
///
/// # Examples
///
/// ```
/// let err = stim::StimError::new("invalid circuit syntax");
/// assert_eq!(err.message(), "invalid circuit syntax");
/// assert_eq!(err.to_string(), "invalid circuit syntax");
///
/// // StimError implements std::error::Error.
/// let dyn_err: Box<dyn std::error::Error> = Box::new(err);
/// assert_eq!(dyn_err.to_string(), "invalid circuit syntax");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StimError {
    message: String,
}

impl StimError {
    /// Creates a new error with the given message.
    ///
    /// The message should describe what went wrong in a way that helps the
    /// caller understand how to fix the problem. It will be displayed by the
    /// [`Display`](std::fmt::Display) implementation and returned by
    /// [`message`](Self::message).
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the human-readable error message.
    ///
    /// This is the same string that was passed to [`new`](Self::new) and that
    /// is displayed by the [`Display`](std::fmt::Display) implementation.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for StimError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for StimError {}

impl From<cxx::Exception> for StimError {
    fn from(value: cxx::Exception) -> Self {
        Self::new(value.what())
    }
}
