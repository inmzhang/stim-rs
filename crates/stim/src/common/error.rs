use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// A type alias for results that may contain a [`StimError`].
pub type Result<T> = std::result::Result<T, StimError>;

/// An error produced by the Stim library.
///
/// Wraps a human-readable message describing what went wrong. This type is
/// returned by most fallible operations in the crate, either directly or
/// through the [`Result`] type alias.
///
/// # Examples
///
/// ```
/// let err = stim::StimError::new("invalid circuit syntax");
/// assert_eq!(err.message(), "invalid circuit syntax");
/// assert_eq!(err.to_string(), "invalid circuit syntax");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StimError {
    message: String,
}

impl StimError {
    /// Creates a new error with the given message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the error message.
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
