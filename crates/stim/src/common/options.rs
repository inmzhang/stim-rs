use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::StimError;

/// Qubit-ordering convention used by matrix and state-vector conversions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Endian {
    /// Qubit 0 is the least-significant bit.
    Little,
    /// Qubit 0 is the most-significant bit.
    Big,
}

impl Endian {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Little => "little",
            Self::Big => "big",
        }
    }

    pub(crate) const fn is_little(self) -> bool {
        matches!(self, Self::Little)
    }
}

impl Display for Endian {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Endian {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "little" => Ok(Self::Little),
            "big" => Ok(Self::Big),
            _ => Err(StimError::new("endian not in ['little', 'big']")),
        }
    }
}

/// OpenQASM major version supported by Stim exports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpenQasmVersion {
    V2,
    V3,
}

impl OpenQasmVersion {
    #[must_use]
    pub const fn as_i32(self) -> i32 {
        match self {
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }
}

impl Display for OpenQasmVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_i32())
    }
}

impl FromStr for OpenQasmVersion {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "2" => Ok(Self::V2),
            "3" => Ok(Self::V3),
            _ => Err(StimError::new("open_qasm_version not in [2, 3]")),
        }
    }
}

/// SAT/maxSAT format supported by Stim search exports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SatProblemFormat {
    Wdimacs,
}

impl SatProblemFormat {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wdimacs => "WDIMACS",
        }
    }
}

impl Display for SatProblemFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SatProblemFormat {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("WDIMACS") {
            Ok(Self::Wdimacs)
        } else {
            Err(StimError::new("sat problem format not in ['WDIMACS']"))
        }
    }
}
