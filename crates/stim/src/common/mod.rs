pub(crate) mod bit_packing;
pub(crate) mod bridge;
pub(crate) mod error;
pub(crate) mod io;
pub(crate) mod parse;
pub(crate) mod slicing;

/// A 32-bit complex number.
///
/// This is a type alias for [`num_complex::Complex<f32>`], matching the
/// precision Stim uses internally for gate arguments and unitary matrix
/// entries.
///
/// # Examples
///
/// ```
/// let z = stim::Complex32::new(1.0, -0.5);
/// assert_eq!(z.re, 1.0);
/// assert_eq!(z.im, -0.5);
/// ```
pub type Complex32 = num_complex::Complex<f32>;

pub use bridge::upstream_commit;
pub use error::{Result, StimError};
pub use io::{read_shot_data_file, write_shot_data_file};
