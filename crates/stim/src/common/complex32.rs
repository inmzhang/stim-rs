/// A 32-bit complex number.
///
/// Each component is an [`f32`], matching the precision Stim uses internally
/// for gate arguments and Pauli coefficients.
///
/// # Examples
///
/// ```
/// let z = stim::Complex32::new(1.0, -0.5);
/// assert_eq!(z.re, 1.0);
/// assert_eq!(z.im, -0.5);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex32 {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex32 {
    /// Creates a new complex number from real and imaginary parts.
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}
