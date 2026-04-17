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

/// Shot-data file/sample format supported by Stim I/O APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShotDataFormat {
    Bits01,
    B8,
    R8,
    Ptb64,
    Hits,
    Dets,
}

impl ShotDataFormat {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bits01 => "01",
            Self::B8 => "b8",
            Self::R8 => "r8",
            Self::Ptb64 => "ptb64",
            Self::Hits => "hits",
            Self::Dets => "dets",
        }
    }
}

impl Display for ShotDataFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ShotDataFormat {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "01" => Ok(Self::Bits01),
            "b8" => Ok(Self::B8),
            "r8" => Ok(Self::R8),
            "ptb64" => Ok(Self::Ptb64),
            "hits" => Ok(Self::Hits),
            "dets" => Ok(Self::Dets),
            _ => Err(StimError::new(
                "shot data format not in ['01', 'b8', 'r8', 'ptb64', 'hits', 'dets']",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Endian, OpenQasmVersion, SatProblemFormat, ShotDataFormat};
    use std::str::FromStr;

    #[test]
    fn options_display_and_parse_cover_all_variants_and_errors() {
        assert_eq!(Endian::Little.to_string(), "little");
        assert_eq!(Endian::Big.to_string(), "big");
        assert!(Endian::Little.is_little());
        assert!(!Endian::Big.is_little());
        assert_eq!("little".parse::<Endian>().unwrap(), Endian::Little);
        assert_eq!("big".parse::<Endian>().unwrap(), Endian::Big);
        assert!("middle".parse::<Endian>().is_err());

        assert_eq!(OpenQasmVersion::V2.as_i32(), 2);
        assert_eq!(OpenQasmVersion::V3.as_i32(), 3);
        assert_eq!(OpenQasmVersion::V2.to_string(), "2");
        assert_eq!(OpenQasmVersion::V3.to_string(), "3");
        assert_eq!(OpenQasmVersion::from_str("2").unwrap(), OpenQasmVersion::V2);
        assert_eq!(OpenQasmVersion::from_str("3").unwrap(), OpenQasmVersion::V3);
        assert!(OpenQasmVersion::from_str("4").is_err());

        assert_eq!(SatProblemFormat::Wdimacs.as_str(), "WDIMACS");
        assert_eq!(SatProblemFormat::Wdimacs.to_string(), "WDIMACS");
        assert_eq!(
            SatProblemFormat::from_str("WDIMACS").unwrap(),
            SatProblemFormat::Wdimacs
        );
        assert_eq!(
            SatProblemFormat::from_str("wdimacs").unwrap(),
            SatProblemFormat::Wdimacs
        );
        assert!(SatProblemFormat::from_str("cnf").is_err());

        assert_eq!(ShotDataFormat::Bits01.as_str(), "01");
        assert_eq!(ShotDataFormat::B8.as_str(), "b8");
        assert_eq!(ShotDataFormat::R8.as_str(), "r8");
        assert_eq!(ShotDataFormat::Ptb64.as_str(), "ptb64");
        assert_eq!(ShotDataFormat::Hits.as_str(), "hits");
        assert_eq!(ShotDataFormat::Dets.as_str(), "dets");
        assert_eq!(ShotDataFormat::Bits01.to_string(), "01");
        assert_eq!(
            "01".parse::<ShotDataFormat>().unwrap(),
            ShotDataFormat::Bits01
        );
        assert_eq!("b8".parse::<ShotDataFormat>().unwrap(), ShotDataFormat::B8);
        assert_eq!("r8".parse::<ShotDataFormat>().unwrap(), ShotDataFormat::R8);
        assert_eq!(
            "ptb64".parse::<ShotDataFormat>().unwrap(),
            ShotDataFormat::Ptb64
        );
        assert_eq!(
            "hits".parse::<ShotDataFormat>().unwrap(),
            ShotDataFormat::Hits
        );
        assert_eq!(
            "dets".parse::<ShotDataFormat>().unwrap(),
            ShotDataFormat::Dets
        );
        assert!("csv".parse::<ShotDataFormat>().is_err());
    }
}
