use std::fmt::{self, Display, Formatter};
use std::str::FromStr;

use crate::{DemTarget, DetectorErrorModel, Result, StimError};

/// A detector error model instruction kind.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DemInstructionType {
    Error,
    ShiftDetectors,
    Detector,
    LogicalObservable,
    Repeat,
}

impl DemInstructionType {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::ShiftDetectors => "shift_detectors",
            Self::Detector => "detector",
            Self::LogicalObservable => "logical_observable",
            Self::Repeat => "repeat",
        }
    }

    pub(crate) fn from_name(name: &str) -> Result<Self> {
        match name {
            "error" => Ok(Self::Error),
            "shift_detectors" => Ok(Self::ShiftDetectors),
            "detector" => Ok(Self::Detector),
            "logical_observable" => Ok(Self::LogicalObservable),
            "repeat" => Ok(Self::Repeat),
            _ => Err(StimError::new(format!(
                "unrecognized DEM instruction type: {name}"
            ))),
        }
    }
}

impl Display for DemInstructionType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A target used inside a detector error model (DEM) instruction.
///
/// DEM instructions refer to detectors, logical observables, separators,
/// or raw integer offsets. This enum wraps those two categories:
///
/// - [`DemTarget`](Self::DemTarget) -- a detector (`D5`), logical
///   observable (`L2`), or separator (`^`), represented as a
///   [`crate::DemTarget`].
/// - [`RelativeOffset`](Self::RelativeOffset) -- a raw integer offset
///   used by instructions like `shift_detectors`.
///
/// Values of either variant can be created via the `From` conversions
/// from [`DemTarget`](crate::DemTarget), [`u64`], [`u32`], or
/// [`usize`].
///
/// # Examples
///
/// ```
/// use stim::{DemInstructionTarget, DemTarget};
///
/// let det = stim::DemTarget::relative_detector_id(3).expect("valid id");
/// let target = DemInstructionTarget::from(det);
/// assert!(!target.is_separator());
///
/// let offset = DemInstructionTarget::from(42u64);
/// assert!(!offset.is_separator());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DemInstructionTarget {
    /// A detector, observable, or separator target.
    DemTarget(DemTarget),
    /// A raw integer offset, used by `shift_detectors` and similar
    /// instructions that take plain numeric arguments rather than
    /// detector/observable references.
    RelativeOffset(u64),
}

impl DemInstructionTarget {
    /// Returns `true` if this target is a [`DemTarget`] separator (`^`).
    ///
    /// Separator targets delimit groups within an `error` instruction's
    /// target list, indicating a suggested decomposition of a
    /// multi-component error into independent parts.
    #[must_use]
    pub fn is_separator(self) -> bool {
        matches!(self, Self::DemTarget(target) if target.is_separator())
    }
}

impl From<DemTarget> for DemInstructionTarget {
    fn from(value: DemTarget) -> Self {
        Self::DemTarget(value)
    }
}

impl From<u64> for DemInstructionTarget {
    fn from(value: u64) -> Self {
        Self::RelativeOffset(value)
    }
}

impl From<u32> for DemInstructionTarget {
    fn from(value: u32) -> Self {
        Self::RelativeOffset(value as u64)
    }
}

impl From<usize> for DemInstructionTarget {
    fn from(value: usize) -> Self {
        Self::RelativeOffset(value as u64)
    }
}

impl Display for DemInstructionTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DemTarget(target) => write!(f, "{target}"),
            Self::RelativeOffset(value) => write!(f, "{value}"),
        }
    }
}

impl fmt::Debug for DemInstructionTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::DemTarget(target) => write!(f, "{target:?}"),
            Self::RelativeOffset(value) => write!(f, "{value}"),
        }
    }
}

/// An instruction from a detector error model (DEM).
///
/// A detector error model is Stim's compact representation of the
/// error-propagation structure of a quantum error-correction circuit.
/// Each instruction in a DEM has four components:
///
/// - A **type** ([`DemInstructionType`]) identifying the instruction kind.
/// - An optional **tag** -- the `[...]` annotation after the type name,
///   e.g. `error[my-tag](0.125) D0`. Tags are arbitrary strings that
///   Stim propagates but otherwise ignores.
/// - A list of floating-point **arguments** -- the `(...)` parenthesized
///   values, e.g. the error probability `0.125` in `error(0.125) D0`.
/// - A list of **targets** ([`DemInstructionTarget`] values) -- the
///   detectors, observables, separators, or raw offsets that the
///   instruction references.
///
/// Instructions can be constructed programmatically via
/// [`new`](Self::new) or converted from a string using `str::parse`
/// (the [`FromStr`](std::str::FromStr) implementation).
///
/// # Examples
///
/// ```
/// // Parse from DEM text.
/// let inst: stim::DemInstruction = "error(0.125) D0 D1".parse().expect("valid DEM line");
/// assert_eq!(inst.r#type(), stim::DemInstructionType::Error);
/// assert_eq!(inst.args(), &[0.125]);
/// assert_eq!(inst.targets().len(), 2);
///
/// // Construct programmatically.
/// let inst = stim::DemInstruction::new(
///     stim::DemInstructionType::Error,
///     [0.125],
///     [
///         stim::DemTarget::relative_detector_id(5).expect("valid id"),
///         stim::DemTarget::logical_observable_id(2).expect("valid id"),
///     ],
///     "",
/// ).expect("valid instruction");
/// assert_eq!(inst.to_string(), "error(0.125) D5 L2");
/// ```
#[derive(Clone, PartialEq)]
pub struct DemInstruction {
    instruction_type: DemInstructionType,
    tag: String,
    args: Vec<f64>,
    targets: Vec<DemInstructionTarget>,
}

impl DemInstruction {
    /// Constructs a new DEM instruction from its components.
    ///
    /// The instruction is validated against the Stim DEM parser after
    /// construction; invalid combinations will return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the assembled instruction text is not valid DEM.
    ///
    /// # Examples
    ///
    /// ```
    /// let inst = stim::DemInstruction::new(
    ///     stim::DemInstructionType::Error,
    ///     [0.125],
    ///     [
    ///         stim::DemTarget::relative_detector_id(5).expect("valid id"),
    ///         stim::DemTarget::logical_observable_id(2).expect("valid id"),
    ///     ],
    ///     "",
    /// ).expect("valid instruction");
    /// assert_eq!(inst.to_string(), "error(0.125) D5 L2");
    /// ```
    pub fn new(
        instruction_type: DemInstructionType,
        args: impl IntoIterator<Item = f64>,
        targets: impl IntoIterator<Item = impl Into<DemInstructionTarget>>,
        tag: impl Into<String>,
    ) -> Result<Self> {
        let instruction = Self {
            instruction_type,
            tag: tag.into(),
            args: args.into_iter().collect(),
            targets: targets.into_iter().map(Into::into).collect(),
        };
        instruction.validate()?;
        Ok(instruction)
    }

    /// Parses a single DEM instruction from its textual representation.
    ///
    /// The text must contain exactly one DEM instruction (not a repeat block,
    /// not multiple lines). Comments are stripped.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is empty, contains a repeat block, or
    /// contains more than one instruction.
    ///
    /// # Examples
    ///
    /// ```
    /// let inst: stim::DemInstruction = "error(0.125) D5 L6 ^ D4  # comment"
    ///     .parse()
    ///     .expect("valid DEM text");
    /// assert_eq!(inst.r#type(), stim::DemInstructionType::Error);
    /// assert_eq!(inst.to_string(), "error(0.125) D5 L6 ^ D4");
    /// ```
    fn parse_text(text: &str) -> Result<Self> {
        let normalized = DetectorErrorModel::from_str(text)?.to_string();
        if normalized.is_empty() {
            return Err(StimError::new(
                "expected a single detector error model instruction, got empty text",
            ));
        }
        if normalized.starts_with("repeat ") {
            return Err(StimError::new(
                "DemInstruction cannot represent DEM repeat blocks",
            ));
        }
        if normalized.lines().count() != 1 {
            return Err(StimError::new(
                "expected a single detector error model instruction",
            ));
        }

        let (head, tail) = split_head_and_tail(&normalized);
        let (instruction_type, tag, args) = parse_head(head)?;
        let targets = parse_targets(tail)?;
        Self::new(instruction_type, args, targets, tag)
    }

    /// Returns the instruction type.
    ///
    /// This is a duck-typing convenience method. It exists so that code
    /// that doesn't know whether it has a `DemInstruction` or a
    /// [`DemRepeatBlock`](crate::DemRepeatBlock) can check the type
    /// field without doing a pattern match first.
    #[must_use]
    pub fn r#type(&self) -> DemInstructionType {
        self.instruction_type
    }

    /// Returns the instruction's tag, or an empty string if untagged.
    ///
    /// Tags are the `[...]` annotation appearing after the instruction type
    /// name, e.g. `error[my-tag](0.125) D0`. They are arbitrary strings
    /// that Stim propagates across transformations but otherwise ignores,
    /// allowing user code to attach metadata to specific instructions.
    ///
    /// # Examples
    ///
    /// ```
    /// let tagged: stim::DemInstruction = "error[my-tag](0.125) D0"
    ///     .parse().expect("valid");
    /// assert_eq!(tagged.tag(), "my-tag");
    ///
    /// let untagged: stim::DemInstruction = "error(0.125) D0"
    ///     .parse().expect("valid");
    /// assert_eq!(untagged.tag(), "");
    /// ```
    #[must_use]
    pub fn tag(&self) -> &str {
        &self.tag
    }

    /// Returns the instruction's parenthesized arguments.
    ///
    /// For `error` instructions this is typically a single-element list
    /// containing the error probability. For `detector` instructions it
    /// contains the detector's coordinate data. For `shift_detectors`
    /// it contains the coordinate offsets.
    ///
    /// # Examples
    ///
    /// ```
    /// let inst: stim::DemInstruction = "error(0.125) D0".parse().expect("valid");
    /// assert_eq!(inst.args(), &[0.125]);
    /// ```
    #[must_use]
    pub fn args(&self) -> &[f64] {
        &self.args
    }

    /// Returns the instruction's target list.
    ///
    /// # Examples
    ///
    /// ```
    /// let inst: stim::DemInstruction = "error(0.125) D0 D1".parse().expect("valid");
    /// assert_eq!(inst.targets().len(), 2);
    /// ```
    #[must_use]
    pub fn targets(&self) -> &[DemInstructionTarget] {
        &self.targets
    }

    /// Splits the target list into groups delimited by separator targets
    /// (`^`).
    ///
    /// When a detector error model `error` instruction contains a
    /// suggested decomposition of a multi-component error, its targets
    /// contain separators (`^`). This method splits the targets into
    /// groups based on the separators, similar to how `str::split`
    /// works. Separator targets themselves are not included in the
    /// returned groups. If the instruction has no targets, a single
    /// empty group is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let inst: stim::DemInstruction = "error(0.01) D0 D1 ^ D2".parse().expect("valid");
    /// let groups = inst.target_groups();
    /// assert_eq!(groups.len(), 2);
    /// assert_eq!(groups[0].len(), 2); // D0 D1
    /// assert_eq!(groups[1].len(), 1); // D2
    /// ```
    #[must_use]
    pub fn target_groups(&self) -> Vec<Vec<DemInstructionTarget>> {
        if self.targets.is_empty() {
            return vec![Vec::new()];
        }

        let mut groups = Vec::new();
        let mut current = Vec::new();
        for target in &self.targets {
            if target.is_separator() {
                groups.push(std::mem::take(&mut current));
            } else {
                current.push(*target);
            }
        }
        groups.push(current);
        groups
    }

    fn validate(&self) -> Result<()> {
        let _ = DetectorErrorModel::from_str(&self.to_string())?;
        Ok(())
    }

    fn fmt_repr(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stim::DemInstruction({:?}, {:?}",
            self.instruction_type.name(),
            self.args
        )?;
        f.write_str(", [")?;
        for (index, target) in self.targets.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{target:?}")?;
        }
        f.write_str("]")?;
        if !self.tag.is_empty() {
            write!(f, ", tag={:?}", self.tag)?;
        }
        f.write_str(")")
    }
}

impl Eq for DemInstruction {}

impl PartialOrd for DemInstruction {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DemInstruction {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.instruction_type
            .cmp(&other.instruction_type)
            .then_with(|| self.tag.cmp(&other.tag))
            .then_with(|| compare_f64_slices(&self.args, &other.args))
            .then_with(|| self.targets.cmp(&other.targets))
    }
}

impl std::hash::Hash for DemInstruction {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.instruction_type.hash(state);
        self.tag.hash(state);
        for arg in &self.args {
            arg.to_bits().hash(state);
        }
        self.targets.hash(state);
    }
}

impl Display for DemInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.instruction_type.name())?;
        if !self.tag.is_empty() {
            write!(f, "[{}]", self.tag)?;
        }
        if !self.args.is_empty() {
            f.write_str("(")?;
            for (index, arg) in self.args.iter().enumerate() {
                if index > 0 {
                    f.write_str(",")?;
                }
                write!(f, "{arg}")?;
            }
            f.write_str(")")?;
        }
        for target in &self.targets {
            write!(f, " {target}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for DemInstruction {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.fmt_repr(f)
    }
}

impl FromStr for DemInstruction {
    type Err = StimError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse_text(s)
    }
}

fn parse_head(head: &str) -> Result<(DemInstructionType, String, Vec<f64>)> {
    let name_end = head.find(['[', '(']).unwrap_or(head.len());
    let instruction_type = &head[..name_end];
    let mut rest = &head[name_end..];
    let mut tag = String::new();
    let mut args = Vec::new();

    if let Some(after_open) = rest.strip_prefix('[') {
        let close = after_open
            .find(']')
            .ok_or_else(|| StimError::new("unterminated DEM instruction tag"))?;
        tag = after_open[..close].to_string();
        rest = &after_open[close + 1..];
    }

    if let Some(after_open) = rest.strip_prefix('(') {
        let close = after_open
            .find(')')
            .ok_or_else(|| StimError::new("unterminated DEM instruction argument list"))?;
        let raw_args = &after_open[..close];
        if !raw_args.is_empty() {
            args = raw_args
                .split(',')
                .map(|arg| {
                    arg.trim()
                        .parse::<f64>()
                        .map_err(|_| StimError::new(format!("invalid DEM arg: {arg}")))
                })
                .collect::<Result<Vec<_>>>()?;
        }
    }

    Ok((DemInstructionType::from_name(instruction_type)?, tag, args))
}

fn parse_targets(text: &str) -> Result<Vec<DemInstructionTarget>> {
    text.split_whitespace()
        .map(|token| {
            token
                .parse::<DemTarget>()
                .map(DemInstructionTarget::from)
                .or_else(|_| {
                    token
                        .parse::<u64>()
                        .map(DemInstructionTarget::from)
                        .map_err(|_| {
                            StimError::new(format!(
                                "failed to parse DEM instruction target: {token}"
                            ))
                        })
                })
        })
        .collect()
}

fn split_head_and_tail(text: &str) -> (&str, &str) {
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' => paren_depth += 1,
            ')' => paren_depth = paren_depth.saturating_sub(1),
            '[' => bracket_depth += 1,
            ']' => bracket_depth = bracket_depth.saturating_sub(1),
            ' ' if paren_depth == 0 && bracket_depth == 0 => {
                return (&text[..index], &text[index + 1..]);
            }
            _ => {}
        }
    }
    (text, "")
}

fn compare_f64_slices(left: &[f64], right: &[f64]) -> std::cmp::Ordering {
    for (l, r) in left.iter().zip(right.iter()) {
        let cmp = l.to_bits().cmp(&r.to_bits());
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
    }
    left.len().cmp(&right.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeSet, HashSet};
    use std::str::FromStr;

    #[test]
    fn constructor_and_accessors_preserve_values() {
        let instruction = DemInstruction::new(
            DemInstructionType::Error,
            [0.125],
            [
                DemTarget::relative_detector_id(5).unwrap(),
                DemTarget::logical_observable_id(2).unwrap(),
            ],
            "test-tag",
        )
        .unwrap();

        assert_eq!(instruction.r#type(), DemInstructionType::Error);
        assert_eq!(instruction.tag(), "test-tag");
        assert_eq!(instruction.args(), &[0.125]);
        assert_eq!(
            instruction.targets(),
            [
                DemInstructionTarget::from(DemTarget::relative_detector_id(5).unwrap()),
                DemInstructionTarget::from(DemTarget::logical_observable_id(2).unwrap()),
            ]
        );
        assert_eq!(instruction.to_string(), "error[test-tag](0.125) D5 L2");
    }

    #[test]
    fn parsed_line_roundtrips_and_strips_comments() {
        let instruction = DemInstruction::from_str("error(0.125) D5 L6 ^ D4  # comment").unwrap();

        assert_eq!(instruction.r#type(), DemInstructionType::Error);
        assert_eq!(instruction.tag(), "");
        assert_eq!(instruction.args(), &[0.125]);
        assert_eq!(
            instruction.targets(),
            [
                DemInstructionTarget::from(DemTarget::relative_detector_id(5).unwrap()),
                DemInstructionTarget::from(DemTarget::logical_observable_id(6).unwrap()),
                DemInstructionTarget::from(DemTarget::separator()),
                DemInstructionTarget::from(DemTarget::relative_detector_id(4).unwrap()),
            ]
        );
        assert_eq!(instruction.to_string(), "error(0.125) D5 L6 ^ D4");
    }

    #[test]
    fn equality_hash_and_order_follow_all_fields() {
        let first = DemInstruction::new(
            DemInstructionType::Error,
            [0.125],
            [DemTarget::relative_detector_id(2).unwrap()],
            "",
        )
        .unwrap();
        let same = DemInstruction::new(
            DemInstructionType::Error,
            [0.125],
            [DemTarget::relative_detector_id(2).unwrap()],
            "",
        )
        .unwrap();
        let different = DemInstruction::new(
            DemInstructionType::Error,
            [0.125],
            [DemTarget::relative_detector_id(3).unwrap()],
            "",
        )
        .unwrap();

        assert_eq!(first, same);
        assert_ne!(first, different);

        let ordered = [different.clone(), same.clone(), first.clone()]
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(ordered, vec![first.clone(), different.clone()]);

        let hashed = [first.clone(), same, different.clone()]
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(hashed.len(), 2);
        assert!(hashed.contains(&first));
        assert!(hashed.contains(&different));
    }

    #[test]
    fn target_groups_split_on_separators_and_preserve_empty_groups() {
        let split = DemInstruction::new(
            DemInstructionType::Error,
            [0.01],
            [
                DemTarget::relative_detector_id(0).unwrap(),
                DemTarget::relative_detector_id(1).unwrap(),
                DemTarget::separator(),
                DemTarget::relative_detector_id(2).unwrap(),
            ],
            "",
        )
        .unwrap();
        let single = DemInstruction::new(
            DemInstructionType::Error,
            [0.01],
            [
                DemTarget::relative_detector_id(0).unwrap(),
                DemTarget::logical_observable_id(0).unwrap(),
            ],
            "",
        )
        .unwrap();
        let empty = DemInstruction::new(
            DemInstructionType::Error,
            [0.01],
            Vec::<DemInstructionTarget>::new(),
            "",
        )
        .unwrap();

        assert_eq!(
            split.target_groups(),
            vec![
                vec![
                    DemInstructionTarget::from(DemTarget::relative_detector_id(0).unwrap()),
                    DemInstructionTarget::from(DemTarget::relative_detector_id(1).unwrap())
                ],
                vec![DemInstructionTarget::from(
                    DemTarget::relative_detector_id(2).unwrap()
                )],
            ]
        );
        assert_eq!(
            single.target_groups(),
            vec![vec![
                DemInstructionTarget::from(DemTarget::relative_detector_id(0).unwrap()),
                DemInstructionTarget::from(DemTarget::logical_observable_id(0).unwrap()),
            ]]
        );
        assert_eq!(empty.target_groups(), vec![Vec::new()]);
    }

    #[test]
    fn raw_integer_targets_are_supported_for_shift_detectors() {
        let instruction = DemInstruction::new(
            DemInstructionType::ShiftDetectors,
            [1.0, 2.0, 3.0],
            [5u64],
            "",
        )
        .unwrap();

        assert_eq!(instruction.targets(), [DemInstructionTarget::from(5u64)]);
        assert_eq!(instruction.to_string(), "shift_detectors(1,2,3) 5");
        assert_eq!(
            DemInstruction::from_str("shift_detectors(1,2,3) 5").unwrap(),
            instruction
        );
    }

    #[test]
    fn debug_matches_binding_conventions() {
        let instruction = DemInstruction::new(
            DemInstructionType::Error,
            [0.125],
            [DemTarget::relative_detector_id(5).unwrap()],
            "test-tag",
        )
        .unwrap();

        assert_eq!(
            format!("{instruction:?}"),
            "stim::DemInstruction(\"error\", [0.125], [stim::DemTarget('D5')], tag=\"test-tag\")"
        );

        let shifted =
            DemInstruction::new(DemInstructionType::ShiftDetectors, [1.0], [5u64], "").unwrap();
        assert_eq!(
            format!("{shifted:?}"),
            "stim::DemInstruction(\"shift_detectors\", [1.0], [5])"
        );
    }

    #[test]
    fn parsing_and_helper_error_paths_are_covered() {
        let empty = "".parse::<DemInstruction>().unwrap_err();
        assert!(empty.message().contains("got empty text"));
        let repeat = "repeat 2 {\n    error(0.1) D0\n}"
            .parse::<DemInstruction>()
            .unwrap_err();
        assert!(
            repeat
                .message()
                .contains("cannot represent DEM repeat blocks")
        );
        let multiple = "error(0.1) D0\nerror(0.2) D1"
            .parse::<DemInstruction>()
            .unwrap_err();
        assert!(
            multiple
                .message()
                .contains("expected a single detector error model instruction")
        );

        assert_eq!(DemInstructionTarget::from(5u32).to_string(), "5");
        assert_eq!(DemInstructionTarget::from(6usize).to_string(), "6");

        let shifted =
            DemInstruction::new(DemInstructionType::ShiftDetectors, [1.0], [5u64], "").unwrap();
        assert!(format!("{shifted:?}").contains("[5]"));

        let unterminated_tag = parse_head("error[tag").unwrap_err();
        assert!(
            unterminated_tag
                .message()
                .contains("unterminated DEM instruction tag")
        );
        let bad_target = parse_targets("bad").unwrap_err();
        assert!(
            bad_target
                .message()
                .contains("failed to parse DEM instruction target")
        );
        assert_eq!(
            split_head_and_tail("shift_detectors"),
            ("shift_detectors", "")
        );

        let low = DemInstruction::new(
            DemInstructionType::Error,
            [0.1],
            [DemTarget::relative_detector_id(0).unwrap()],
            "",
        )
        .unwrap();
        let high = DemInstruction::new(
            DemInstructionType::Error,
            [0.2],
            [DemTarget::relative_detector_id(0).unwrap()],
            "",
        )
        .unwrap();
        assert!(low < high);
    }
}
