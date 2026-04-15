use std::fmt::{self, Display, Formatter};
use std::hash::{Hash, Hasher};

use crate::{Circuit, Result, StimError};

/// A `REPEAT` block from a circuit, representing a sub-circuit that is
/// executed a fixed number of times.
///
/// This is the Rust equivalent of Python's `stim.CircuitRepeatBlock`.
/// Repeat blocks are the primary looping construct in Stim circuits.
/// They contain a body [`Circuit`] and a repetition count, and may carry
/// an optional string tag for annotation.
///
/// In Stim circuit text, repeat blocks look like:
///
/// ```text
/// REPEAT 100 {
///     CX 0 1
///     M 0
/// }
/// ```
///
/// # Duck-typing with `CircuitInstruction`
///
/// Both `CircuitRepeatBlock` and [`CircuitInstruction`](crate::CircuitInstruction)
/// expose a [`name()`](Self::name) method and a [`tag()`](Self::tag) method,
/// which enables code that iterates over heterogeneous circuit items
/// (via [`CircuitItem`](crate::CircuitItem)) to inspect the name without
/// needing a type check first. For repeat blocks, `name()` always returns
/// `"REPEAT"`.
///
/// # Measurement counting
///
/// The total number of measurements produced by a repeat block equals the
/// body circuit's measurement count multiplied by the repeat count. For
/// example, a body containing `M 0 1` (2 measurements) repeated 25 times
/// produces 50 measurements total.
///
/// # Ordering and hashing
///
/// `CircuitRepeatBlock` implements [`Eq`], [`Ord`], and [`Hash`].
/// Ordering is lexicographic by repeat count, then tag, then the string
/// representation of the body circuit.
///
/// # Examples
///
/// ```
/// let body: stim::Circuit = "CX 0 1\nM 0".parse().expect("valid body");
/// let block = stim::CircuitRepeatBlock::new(100, &body, "")
///     .expect("repeat count must be > 0");
/// assert_eq!(block.repeat_count(), 100);
/// assert_eq!(block.num_measurements(), 100);
/// ```
///
/// Appending a repeat block to a circuit:
///
/// ```
/// let mut circuit = stim::Circuit::new();
/// let body: stim::Circuit = "M 0".parse().expect("valid body");
/// circuit.append_repeat_block(25, &body, "").expect("nonzero count");
/// assert_eq!(circuit.num_measurements(), 25);
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct CircuitRepeatBlock {
    repeat_count: u64,
    body: Circuit,
    tag: String,
}

impl CircuitRepeatBlock {
    /// Creates a new repeat block that executes `body` exactly
    /// `repeat_count` times.
    ///
    /// This is the Rust equivalent of Python's
    /// `stim.CircuitRepeatBlock.__init__`.
    ///
    /// # Arguments
    ///
    /// * `repeat_count` — The number of times the body circuit will be
    ///   executed. Must be at least 1.
    /// * `body` — The sub-circuit to repeat. The body is cloned into the
    ///   block, so subsequent mutations of the original `body` circuit do
    ///   not affect this block.
    /// * `tag` — An arbitrary string annotation attached to the `REPEAT`
    ///   instruction. Use `""` for no tag.
    ///
    /// # Errors
    ///
    /// Returns an error if `repeat_count` is zero, since a zero-iteration
    /// loop is not meaningful in Stim circuits.
    ///
    /// # Examples
    ///
    /// ```
    /// let body: stim::Circuit = "H 0\nM 0".parse().expect("valid body");
    /// let block = stim::CircuitRepeatBlock::new(5, &body, "my-tag")
    ///     .expect("nonzero repeat count");
    /// assert_eq!(block.tag(), "my-tag");
    /// ```
    pub fn new(repeat_count: u64, body: &Circuit, tag: impl Into<String>) -> Result<Self> {
        if repeat_count == 0 {
            return Err(StimError::new("Can't repeat 0 times."));
        }
        Ok(Self {
            repeat_count,
            body: body.clone(),
            tag: tag.into(),
        })
    }

    /// Returns `"REPEAT"`, the fixed name for all repeat blocks.
    ///
    /// This is a duck-typing convenience method. It exists so that code
    /// iterating over mixed [`CircuitItem`](crate::CircuitItem) values
    /// (which may be either a [`CircuitInstruction`](crate::CircuitInstruction)
    /// or a `CircuitRepeatBlock`) can check the item's name without
    /// performing a type check first.
    #[must_use]
    pub fn name(&self) -> &str {
        "REPEAT"
    }

    /// Returns the block's tag string, or `""` if no tag was set.
    ///
    /// The tag is an arbitrary custom string attached to the `REPEAT`
    /// instruction. Tags appear in Stim syntax as bracketed annotations,
    /// e.g. `REPEAT[my-tag] 5 { ... }`. Stim will attempt to propagate
    /// tags across circuit transformations but otherwise ignores them.
    /// The default tag, when none is specified, is the empty string.
    #[must_use]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Returns the number of times the body circuit is repeated.
    ///
    /// This value is always at least 1, since the constructor rejects
    /// zero-iteration loops.
    #[must_use]
    pub fn repeat_count(&self) -> u64 {
        self.repeat_count
    }

    /// Returns the total number of measurement results (bits) produced by
    /// the block, which equals the body's measurement count multiplied by
    /// the repeat count.
    ///
    /// For example, if the body contains `M 0 1` (2 measurements) and the
    /// repeat count is 25, this method returns 50.
    #[must_use]
    pub fn num_measurements(&self) -> u64 {
        self.body.num_measurements() * self.repeat_count
    }

    /// Returns a clone of the body circuit.
    ///
    /// A fresh copy is returned each time to make it clear that editing the
    /// result will not change the block's body. This follows the same
    /// semantics as Python's `stim.CircuitRepeatBlock.body_copy()`.
    #[must_use]
    pub fn body_copy(&self) -> Circuit {
        self.body.clone()
    }
}

impl Circuit {
    /// Appends a `REPEAT` block to this circuit.
    ///
    /// The block executes `body` exactly `repeat_count` times. An optional
    /// `tag` string is attached as an annotation.
    ///
    /// # Errors
    ///
    /// Returns an error if `repeat_count` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut circuit = stim::Circuit::new();
    /// let body: stim::Circuit = "CX 0 1\nM 0".parse().expect("valid body");
    /// circuit.append_repeat_block(3, &body, "").expect("nonzero count");
    /// assert_eq!(circuit.num_measurements(), 3);
    /// ```
    pub fn append_repeat_block(
        &mut self,
        repeat_count: u64,
        body: &Circuit,
        tag: &str,
    ) -> Result<()> {
        self.inner
            .append_repeat_block(repeat_count, &body.inner, tag)
            .map_err(StimError::from)
    }
}

impl Display for CircuitRepeatBlock {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("REPEAT")?;
        if !self.tag.is_empty() {
            write!(f, "[{}]", self.tag)?;
        }
        write!(f, " {} {{", self.repeat_count)?;
        let body = self.body.to_string();
        if !body.is_empty() {
            f.write_str("\n")?;
            for line in body.lines() {
                writeln!(f, "    {line}")?;
            }
        } else {
            f.write_str("\n")?;
        }
        f.write_str("}")
    }
}

impl fmt::Debug for CircuitRepeatBlock {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stim::CircuitRepeatBlock(repeat_count={}, body={:?}, tag={:?})",
            self.repeat_count, self.body, self.tag
        )
    }
}

impl PartialOrd for CircuitRepeatBlock {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CircuitRepeatBlock {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.repeat_count
            .cmp(&other.repeat_count)
            .then_with(|| self.tag.cmp(&other.tag))
            .then_with(|| self.body.to_string().cmp(&other.body.to_string()))
    }
}

impl Hash for CircuitRepeatBlock {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.repeat_count.hash(state);
        self.tag.hash(state);
        self.body.to_string().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn constructor_and_field_accessors_preserve_values() {
        let body = Circuit::from_str("M 0\nH 1").expect("body should parse");
        let block =
            CircuitRepeatBlock::new(5, &body, "look-at-me").expect("block should construct");

        assert_eq!(block.name(), "REPEAT");
        assert_eq!(block.tag(), "look-at-me");
        assert_eq!(block.repeat_count(), 5);
        assert_eq!(block.num_measurements(), 5);
        assert_eq!(block.body_copy(), body);
    }

    #[test]
    fn clone_and_equality_include_tag() {
        let body = Circuit::from_str("X 0").expect("body should parse");
        let first = CircuitRepeatBlock::new(3, &body, "tagged").expect("block should construct");
        let same = CircuitRepeatBlock::new(3, &body, "tagged").expect("block should construct");
        let different_tag =
            CircuitRepeatBlock::new(3, &body, "other").expect("block should construct");

        assert_eq!(first.clone(), same);
        assert_ne!(first, different_tag);
    }

    #[test]
    fn display_and_debug_match_upstream_like_shapes() {
        let body = Circuit::from_str("X 0\nM 0").expect("body should parse");
        let block = CircuitRepeatBlock::new(3, &body, "tagged").expect("block should construct");

        assert_eq!(block.to_string(), "REPEAT[tagged] 3 {\n    X 0\n    M 0\n}");
        assert_eq!(
            format!("{block:?}"),
            "stim::CircuitRepeatBlock(repeat_count=3, body=stim::Circuit(\"\"\"\nX 0\nM 0\n\"\"\"), tag=\"tagged\")"
        );
    }

    #[test]
    fn new_rejects_zero_repetitions() {
        let error = CircuitRepeatBlock::new(0, &Circuit::new(), "")
            .expect_err("repeat_count=0 should fail");

        assert!(error.message().contains("repeat 0"));
    }

    #[test]
    fn append_repeat_block_roundtrips_through_stim_text() {
        let mut circuit = Circuit::from_str("H 0").expect("circuit should parse");
        let body = Circuit::from_str("CX 0 1\nM 0").expect("body should parse");

        circuit
            .append_repeat_block(2, &body, "tagged")
            .expect("append_repeat_block should succeed");

        let text = circuit.to_string();
        assert_eq!(text, "H 0\nREPEAT[tagged] 2 {\n    CX 0 1\n    M 0\n}");
        assert_eq!(
            Circuit::from_str(&text).expect("text should parse"),
            circuit
        );
    }

    #[test]
    fn append_repeat_block_rejects_zero_repetitions() {
        let mut circuit = Circuit::new();
        let body = Circuit::from_str("M 0").expect("body should parse");
        let error = circuit
            .append_repeat_block(0, &body, "")
            .expect_err("repeat_count=0 should fail");

        assert!(error.message().contains("repeat 0"));
    }
}
