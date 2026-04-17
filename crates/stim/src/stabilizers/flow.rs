use std::fmt::{self, Display, Formatter};
use std::ops::Mul;
use std::str::FromStr;

/// A stabilizer flow relating an input Pauli frame to an output Pauli frame,
/// optionally mediated by measurement records and observable indices.
///
/// Stabilizer circuits implement, and can be defined by, how they turn input
/// stabilizers into output stabilizers mediated by measurements. These
/// relationships are called stabilizer flows: if a circuit has flow `P -> Q`,
/// then it maps the instantaneous stabilizer `P` at the start of the circuit to
/// the instantaneous stabilizer `Q` at the end.
///
/// Flows can include measurement record lookbacks (`rec[-k]`) and observable
/// indices (`obs[k]`), XOR-ed with the output Pauli string. This allows
/// expressing preparation, measurement, and check flows:
///
/// - `P -> Q` means the circuit transforms `P` into `Q`.
/// - `1 -> P` means the circuit prepares `P`.
/// - `P -> 1` means the circuit measures `P` (output is purely classical).
/// - `1 -> 1` means the circuit contains a check (e.g. a `DETECTOR`).
///
/// Flows are used to verify that circuits preserve stabilizer properties. For
/// example, a `Flow` can be passed to `stim::Circuit::has_flow` to check that
/// a circuit implements the flow.
///
/// The flow text format uses `->` to separate input from output, and `xor` to
/// combine the output Pauli string with measurement records and observable
/// references. Identical terms cancel: XOR-ing a measurement twice removes it,
/// and multiplying two identical Pauli terms yields identity.
///
/// # References
///
/// Stim's gate documentation includes the stabilizer flows of each gate.
/// Appendix A of <https://arxiv.org/abs/2302.02192> describes how flows are
/// defined and provides a circuit construction for experimentally verifying
/// their presence.
///
/// # Examples
///
/// ```
/// use stim::Flow;
///
/// // A CNOT gate has the flow X__ -> X_X (X on control propagates to target).
/// let flow = Flow::new("__X__ -> __X_X").unwrap();
/// assert_eq!(flow.to_string(), "__X__ -> __X_X");
///
/// // Flows involving measurements use `xor rec[-k]` syntax.
/// let flow = Flow::new("X -> rec[-1]").unwrap();
/// assert_eq!(flow.measurements(), &[-1]);
///
/// // Flows involving observables use `xor obs[k]` syntax.
/// let flow = Flow::new("X -> Y xor obs[3]").unwrap();
/// assert_eq!(flow.included_observables(), &[3]);
/// ```
#[derive(Clone)]
pub struct Flow {
    text: String,
    input: crate::PauliString,
    output: crate::PauliString,
    measurements: Vec<i32>,
    observables: Vec<u64>,
}

impl PartialEq for Flow {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text
    }
}

impl Eq for Flow {}

impl PartialOrd for Flow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Flow {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.text.cmp(&other.text)
    }
}

impl std::hash::Hash for Flow {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.text.hash(state);
    }
}

impl Flow {
    pub(crate) fn from_canonical_text(text: String) -> Self {
        let (input, output, measurements, observables) = parse_flow_text(&text);
        Self {
            text,
            input,
            output,
            measurements,
            observables,
        }
    }

    /// Creates a flow by parsing and canonicalizing Stim flow shorthand text.
    ///
    /// The input string is parsed according to Stim's flow shorthand format,
    /// where `->` separates the input Pauli string from the output, and `xor`
    /// joins measurement records or observable indices. The text is
    /// canonicalized: qubit-indexed Pauli terms like `X2` are expanded to
    /// explicit underscore-padded strings, and duplicate terms cancel.
    ///
    /// # Errors
    ///
    /// Returns an error if `text` is not valid Stim flow shorthand.
    ///
    /// # Examples
    ///
    /// ```
    /// let flow = stim::Flow::new("X2 -> obs[3]").unwrap();
    /// assert_eq!(flow.to_string(), "__X -> obs[3]");
    /// ```
    pub fn new(text: &str) -> crate::Result<Self> {
        stim_cxx::canonicalize_flow_text(text)
            .map(Self::from_canonical_text)
            .map_err(crate::StimError::from)
    }

    /// Returns the flow's input Pauli string (the stabilizer before the circuit acts).
    ///
    /// The input Pauli string is the left-hand side of the `->` arrow in the
    /// flow text. A flow like `X -> Y` has input `+X`. A preparation flow like
    /// `1 -> Z` has an empty (zero-qubit) input Pauli string representing the
    /// identity.
    ///
    /// # Examples
    ///
    /// ```
    /// let flow = stim::Flow::new("X -> Y xor obs[3]").unwrap();
    /// assert_eq!(flow.input(), &"X".parse::<stim::PauliString>().unwrap());
    ///
    /// // Preparation flows have an empty input.
    /// let flow = stim::Flow::new("1 -> X xor rec[-1] xor obs[2]").unwrap();
    /// assert_eq!(flow.input(), &stim::PauliString::new(0));
    /// ```
    #[must_use]
    pub fn input(&self) -> &crate::PauliString {
        &self.input
    }

    /// Returns the flow's output Pauli string (the stabilizer after the circuit acts).
    ///
    /// The output Pauli string is the Pauli term on the right-hand side of the
    /// `->` arrow, excluding any `rec[...]` or `obs[...]` terms joined by `xor`.
    /// A measurement flow like `Z -> rec[-1]` has an empty (zero-qubit) output
    /// Pauli string, because the output is purely classical.
    ///
    /// # Examples
    ///
    /// ```
    /// let flow = stim::Flow::new("X -> Y xor obs[3]").unwrap();
    /// assert_eq!(flow.output(), &"Y".parse::<stim::PauliString>().unwrap());
    ///
    /// // Measurement-only flows have an empty output.
    /// let flow = stim::Flow::new("X -> rec[-1]").unwrap();
    /// assert_eq!(flow.output(), &stim::PauliString::new(0));
    /// ```
    #[must_use]
    pub fn output(&self) -> &crate::PauliString {
        &self.output
    }

    /// Returns the flow's referenced measurement record lookbacks as a list
    /// of signed indices.
    ///
    /// Measurement indices follow the convention where negative values are
    /// lookbacks relative to the end of the measurement record: `-1` is the
    /// most recent measurement, `-2` is the one before that, etc. Positive
    /// values index from the start of the circuit's measurement record.
    ///
    /// # Examples
    ///
    /// ```
    /// let flow = stim::Flow::new("X -> rec[-1]").unwrap();
    /// assert_eq!(flow.measurements(), &[-1]);
    ///
    /// // Multiple measurement records are listed in order.
    /// let flow = stim::Flow::new("X -> rec[-2] xor rec[-1]").unwrap();
    /// assert_eq!(flow.measurements(), &[-2, -1]);
    ///
    /// // A flow with no measurements returns an empty list.
    /// let flow = stim::Flow::new("X -> Y").unwrap();
    /// assert!(flow.measurements().is_empty());
    /// ```
    #[must_use]
    pub fn measurements(&self) -> &[i32] {
        &self.measurements
    }

    /// Returns the observable indices included by this flow.
    ///
    /// When an observable is included in a flow, the flow implicitly incorporates
    /// all measurements and Pauli terms from `OBSERVABLE_INCLUDE` instructions
    /// targeting that observable index. For example, the flow `X5 -> obs[3]` says
    /// "at the start of the circuit, observable 3 should be an X term on qubit 5;
    /// by the end of the circuit it will be measured, and the `OBSERVABLE_INCLUDE(3)`
    /// instructions in the circuit explain how."
    ///
    /// # Examples
    ///
    /// ```
    /// let flow = stim::Flow::new("X -> Y xor obs[3]").unwrap();
    /// assert_eq!(flow.included_observables(), &[3]);
    ///
    /// // Duplicate observable references cancel (XOR is self-inverse).
    /// let flow = stim::Flow::new("X -> Y xor obs[3] xor obs[3] xor obs[3]").unwrap();
    /// assert_eq!(flow.included_observables(), &[3]);
    ///
    /// // A flow with no observables returns an empty list.
    /// let flow = stim::Flow::new("X -> Y").unwrap();
    /// assert!(flow.included_observables().is_empty());
    /// ```
    #[must_use]
    pub fn included_observables(&self) -> &[u64] {
        &self.observables
    }
}

impl FromStr for Flow {
    type Err = crate::StimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl Display for Flow {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl fmt::Debug for Flow {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "stim::Flow({:?})", self.text)
    }
}

/// Computes the product of two stabilizer flows.
///
/// Multiplying flows composes their input Pauli strings and output Pauli
/// strings independently via the Pauli group product, and XOR-combines their
/// measurement records and observable indices. This corresponds to the fact
/// that if a circuit has flows `P1 -> Q1` and `P2 -> Q2`, it also has flow
/// `P1*P2 -> Q1*Q2` (with any mediating measurements XOR-ed together).
///
/// # Panics
///
/// Panics if the inputs anti-commute, because the product would be
/// anti-Hermitian. For example, `1 -> X` times `1 -> Y` fails because
/// it would yield `1 -> iZ`.
///
/// # Examples
///
/// ```
/// use stim::Flow;
///
/// // Pauli products: X * Z = Y (up to phase).
/// let xy = Flow::new("X -> X").unwrap() * Flow::new("Z -> Z").unwrap();
/// assert_eq!(xy, Flow::new("Y -> Y").unwrap());
///
/// // Anti-commuting outputs acquire a sign flip.
/// let yy = Flow::new("1 -> XX").unwrap() * Flow::new("1 -> ZZ").unwrap();
/// assert_eq!(yy, Flow::new("1 -> -YY").unwrap());
///
/// // Measurement records are XOR-combined.
/// let combined = Flow::new("X -> rec[-1]").unwrap() * Flow::new("X -> rec[-2]").unwrap();
/// assert_eq!(combined, Flow::new("_ -> rec[-2] xor rec[-1]").unwrap());
/// ```
impl Mul for Flow {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let text = stim_cxx::multiply_flow_texts(&self.text, &rhs.text)
            .expect("flow multiplication between previously valid canonical flows should succeed");
        Self::from_canonical_text(text)
    }
}

fn parse_flow_text(text: &str) -> (crate::PauliString, crate::PauliString, Vec<i32>, Vec<u64>) {
    let (input_text, rest) = text
        .split_once(" -> ")
        .expect("canonical flow text should contain an arrow");
    let input = parse_flow_pauli(input_text);
    let mut output = crate::PauliString::new(0);
    let mut measurements = Vec::new();
    let mut observables = Vec::new();

    for segment in rest.split(" xor ") {
        if let Some(value) = segment
            .strip_prefix("rec[")
            .and_then(|tail| tail.strip_suffix(']'))
        {
            measurements.push(
                value
                    .parse::<i32>()
                    .expect("canonical rec target should contain an integer"),
            );
        } else if let Some(value) = segment
            .strip_prefix("obs[")
            .and_then(|tail| tail.strip_suffix(']'))
        {
            observables.push(
                value
                    .parse::<u64>()
                    .expect("canonical obs target should contain an integer"),
            );
        } else {
            output = parse_flow_pauli(segment);
        }
    }

    (input, output, measurements, observables)
}

fn parse_flow_pauli(text: &str) -> crate::PauliString {
    if text == "1" {
        crate::PauliString::new(0)
    } else {
        text.parse::<crate::PauliString>()
            .expect("canonical flow pauli term should be parseable as a PauliString")
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::Flow;

    #[test]
    fn flow_from_text_canonicalizes_documented_examples() {
        assert_eq!(
            Flow::new("X2 -> -Y2*Z4 xor rec[-1]").unwrap().to_string(),
            "__X -> -__Y_Z xor rec[-1]"
        );
        assert_eq!(
            Flow::new("Z -> 1 xor rec[-1]").unwrap().to_string(),
            "Z -> rec[-1]"
        );
        assert_eq!(
            Flow::new("X2 -> Y2*Y2 xor rec[-2] xor rec[-2]")
                .unwrap()
                .to_string(),
            "__X -> ___"
        );
        assert_eq!(
            Flow::new("X -> Y xor obs[3] xor obs[3] xor obs[3]")
                .unwrap()
                .to_string(),
            "X -> Y xor obs[3]"
        );
    }

    #[test]
    fn flow_display_debug_and_from_str_match() {
        let flow = Flow::from_str("X2 -> obs[3]").unwrap();
        assert_eq!(flow.to_string(), "__X -> obs[3]");
        assert_eq!(format!("{flow:?}"), "stim::Flow(\"__X -> obs[3]\")");
        assert_eq!(Flow::new("X2 -> obs[3]").unwrap(), flow);
    }

    #[test]
    fn flow_mul_matches_documented_examples() {
        assert_eq!(
            Flow::new("X -> X").unwrap() * Flow::new("Z -> Z").unwrap(),
            Flow::new("Y -> Y").unwrap()
        );
        assert_eq!(
            Flow::new("1 -> XX").unwrap() * Flow::new("1 -> ZZ").unwrap(),
            Flow::new("1 -> -YY").unwrap()
        );
        assert_eq!(
            Flow::new("X -> rec[-1]").unwrap() * Flow::new("X -> rec[-2]").unwrap(),
            Flow::new("_ -> rec[-2] xor rec[-1]").unwrap()
        );
    }

    #[test]
    fn flow_copy_accessors_match_documented_examples() {
        let flow = Flow::new("X -> Y xor obs[3]").unwrap();
        assert_eq!(flow.input(), &"X".parse::<crate::PauliString>().unwrap());
        assert_eq!(flow.output(), &"Y".parse::<crate::PauliString>().unwrap());
        assert_eq!(flow.included_observables(), &[3]);

        let flow = Flow::new("X -> rec[-1]").unwrap();
        assert_eq!(flow.output(), &crate::PauliString::new(0));
        assert_eq!(flow.measurements(), &[-1]);

        let flow = Flow::new("1 -> X xor rec[-1] xor obs[2]").unwrap();
        assert_eq!(flow.input(), &crate::PauliString::new(0));
        assert_eq!(flow.output(), &"X".parse::<crate::PauliString>().unwrap());
        assert_eq!(flow.measurements(), &[-1]);
        assert_eq!(flow.included_observables(), &[2]);
    }
}
