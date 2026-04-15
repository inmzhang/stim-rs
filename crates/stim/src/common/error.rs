use std::error::Error;
use std::fmt::{self, Display, Formatter};

pub type Result<T> = std::result::Result<T, StimError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StimError {
    message: String,
}

impl StimError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

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
