use std::fmt::{self, Display, Formatter};
use std::ops::Mul;
use std::str::FromStr;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// A stabilizer flow relating an input Pauli frame, output frame, and classical data.
pub struct Flow {
    text: String,
}

impl Flow {
    /// Creates a flow from canonicalizable Stim flow text.
    ///
    /// # Examples
    ///
    /// ```
    /// let flow = stim::Flow::new("X2 -> obs[3]").unwrap();
    /// assert_eq!(flow.to_string(), "__X -> obs[3]");
    /// ```
    pub fn new(text: &str) -> crate::Result<Self> {
        Self::from_text(text)
    }

    /// Parses and canonicalizes flow text.
    ///
    /// # Examples
    ///
    /// ```
    /// let flow = stim::Flow::from_text("X2 -> -Y2*Z4 xor rec[-1]").unwrap();
    /// assert_eq!(flow.to_string(), "__X -> -__Y_Z xor rec[-1]");
    /// ```
    pub fn from_text(text: &str) -> crate::Result<Self> {
        stim_cxx::canonicalize_flow_text(text)
            .map(|text| Self { text })
            .map_err(crate::StimError::from)
    }

    /// Returns a copy of the flow's input Pauli string.
    ///
    /// # Examples
    ///
    /// ```
    /// let flow = stim::Flow::from_text("X -> Y xor obs[3]").unwrap();
    /// assert_eq!(flow.input_copy(), stim::PauliString::from_text("X").unwrap());
    /// ```
    #[must_use]
    pub fn input_copy(&self) -> crate::PauliString {
        let (input, _, _, _) = parse_flow_text(&self.text);
        input
    }

    /// Returns a copy of the flow's output Pauli string.
    #[must_use]
    pub fn output_copy(&self) -> crate::PauliString {
        let (_, output, _, _) = parse_flow_text(&self.text);
        output
    }

    /// Returns the flow's referenced measurement record lookbacks.
    #[must_use]
    pub fn measurements_copy(&self) -> Vec<i32> {
        let (_, _, measurements, _) = parse_flow_text(&self.text);
        measurements
    }

    /// Returns the observables included by the flow.
    #[must_use]
    pub fn included_observables_copy(&self) -> Vec<u64> {
        let (_, _, _, observables) = parse_flow_text(&self.text);
        observables
    }
}

impl FromStr for Flow {
    type Err = crate::StimError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_text(s)
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

impl Mul for Flow {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let text = stim_cxx::multiply_flow_texts(&self.text, &rhs.text)
            .expect("flow multiplication between previously valid canonical flows should succeed");
        Self { text }
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
        crate::PauliString::from_text(text)
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
            Flow::from_text("X2 -> -Y2*Z4 xor rec[-1]")
                .unwrap()
                .to_string(),
            "__X -> -__Y_Z xor rec[-1]"
        );
        assert_eq!(
            Flow::from_text("Z -> 1 xor rec[-1]").unwrap().to_string(),
            "Z -> rec[-1]"
        );
        assert_eq!(
            Flow::from_text("X2 -> Y2*Y2 xor rec[-2] xor rec[-2]")
                .unwrap()
                .to_string(),
            "__X -> ___"
        );
        assert_eq!(
            Flow::from_text("X -> Y xor obs[3] xor obs[3] xor obs[3]")
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
            Flow::from_text("X -> X").unwrap() * Flow::from_text("Z -> Z").unwrap(),
            Flow::from_text("Y -> Y").unwrap()
        );
        assert_eq!(
            Flow::from_text("1 -> XX").unwrap() * Flow::from_text("1 -> ZZ").unwrap(),
            Flow::from_text("1 -> -YY").unwrap()
        );
        assert_eq!(
            Flow::from_text("X -> rec[-1]").unwrap() * Flow::from_text("X -> rec[-2]").unwrap(),
            Flow::from_text("_ -> rec[-2] xor rec[-1]").unwrap()
        );
    }

    #[test]
    fn flow_copy_accessors_match_documented_examples() {
        let flow = Flow::from_text("X -> Y xor obs[3]").unwrap();
        assert_eq!(
            flow.input_copy(),
            crate::PauliString::from_text("X").unwrap()
        );
        assert_eq!(
            flow.output_copy(),
            crate::PauliString::from_text("Y").unwrap()
        );
        assert_eq!(flow.included_observables_copy(), vec![3]);

        let flow = Flow::from_text("X -> rec[-1]").unwrap();
        assert_eq!(flow.output_copy(), crate::PauliString::new(0));
        assert_eq!(flow.measurements_copy(), vec![-1]);

        let flow = Flow::from_text("1 -> X xor rec[-1] xor obs[2]").unwrap();
        assert_eq!(flow.input_copy(), crate::PauliString::new(0));
        assert_eq!(
            flow.output_copy(),
            crate::PauliString::from_text("X").unwrap()
        );
        assert_eq!(flow.measurements_copy(), vec![-1]);
        assert_eq!(flow.included_observables_copy(), vec![2]);
    }
}
