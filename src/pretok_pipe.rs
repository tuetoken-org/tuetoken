//! Universal composable HuggingFace pre-tokenizer pipeline.
//!
//! Overcomes per-model pre-tokenizer hardcoding by faithfully executing ANY
//! HuggingFace `pre_tokenizer` composition of standard stages in exact order:
//!   * `Split` (`Regex` or `String`, any `behavior`: `Removed`, `Isolated`,
//!     `MergedWithPrevious`, `MergedWithNext`, `Contiguous`, `invert: bool`,
//!     including per-piece `\A` / `\G` / `\z` anchors and atomic groups)
//!   * `Digits` (`individual_digits: bool`, using exact `char::is_numeric`)
//!   * `Punctuation` (any `behavior`, matching HF's ASCII + Unicode `P` set)
//!   * `ByteLevel` (`add_prefix_space`, `use_regex`, composable ANYWHERE in a
//!     `Sequence` — including BEFORE later `Digits`/`Split` stages as in Falcon!)
//!   * `Metaspace` (`replacement`, `prepend_scheme`: `always`/`first`/`never`, `split`)
//!   * `Whitespace` / `WhitespaceSplit`
//!   * `Sequence` (arbitrary nesting)
//!
//! Fast-path models (GPT-2, cl100k, o200k, Llama, Qwen, DeepSeek-V3/R1,
//! Mistral Tekken, etc.) still run on `blockscan` / `cjk` at full SIMD speed;
//! this module gives every other BPE model 100% HF-parity support automatically
//! while feeding every emitted chunk into the shared cache + BPE engine.
//! Any unrecognized stage type fails closed at load time.

use crate::pretok::{next_chunk_end, Pretok};
use crate::unicode::is_hf_punctuation;
use serde_json::Value;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

impl SplitBehavior {
    fn from_json(v: Option<&Value>, default: Self) -> Result<Self, String> {
        match v {
            None | Some(Value::Null) => Ok(default),
            Some(Value::String(s)) => match s.as_str() {
                "Removed" => Ok(Self::Removed),
                "Isolated" => Ok(Self::Isolated),
                "MergedWithPrevious" => Ok(Self::MergedWithPrevious),
                "MergedWithNext" => Ok(Self::MergedWithNext),
                "Contiguous" => Ok(Self::Contiguous),
                other => Err(format!("unsupported Split behavior {other:?}")),
            },
            _ => Err("invalid Split behavior".to_string()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrependScheme {
    Always,
    First,
    Never,
}

impl PrependScheme {
    fn from_json(v: Option<&Value>) -> Result<Self, String> {
        match v {
            None | Some(Value::Null) => Ok(Self::Always),
            Some(Value::String(s)) => match s.as_str() {
                "always" => Ok(Self::Always),
                "first" => Ok(Self::First),
                "never" => Ok(Self::Never),
                other => Err(format!("unsupported Metaspace prepend_scheme {other:?}")),
            },
            _ => Err("invalid Metaspace prepend_scheme".to_string()),
        }
    }
}

#[derive(Clone, Debug)]
pub enum SplitPattern {
    /// Hand-written exact SWAR state machine (used automatically when a `Split`
    /// stage inside a multi-stage `Sequence` matches a proven total grammar).
    FastMachine(Pretok),
    Regex(fancy_regex::Regex),
    Literal(String),
}

#[derive(Clone, Debug)]
pub enum PretokStage {
    Split {
        pattern: SplitPattern,
        behavior: SplitBehavior,
        invert: bool,
    },
    Digits {
        individual_digits: bool,
    },
    Punctuation {
        behavior: SplitBehavior,
    },
    ByteLevel {
        add_prefix_space: bool,
        use_regex: bool,
    },
    Metaspace {
        replacement: char,
        prepend_scheme: PrependScheme,
        split: bool,
    },
    Whitespace,
    WhitespaceSplit,
}

/// Composable HF pre-tokenizer pipeline.
#[derive(Clone, Debug)]
pub struct PretokPipe {
    stages: Vec<PretokStage>,
    /// True when no stage mutates piece text (no `add_prefix_space: true`, no
    /// `Metaspace`, and no stage appears *after* a `ByteLevel` stage). In this
    /// common case we operate purely on `(start, end)` byte spans of `&str`
    /// with zero per-piece `String` allocations.
    slice_only: bool,
}

fn whitespace_regex() -> &'static fancy_regex::Regex {
    static RE: OnceLock<fancy_regex::Regex> = OnceLock::new();
    RE.get_or_init(|| fancy_regex::Regex::new(r"\w+|[^\w\s]+").expect("valid Whitespace regex"))
}

/// Standard GPT-2 ByteLevel `byte -> Unicode char` table (inverse of `c2b`).
fn byte_to_char_table() -> &'static [char; 256] {
    static B2U: OnceLock<[char; 256]> = OnceLock::new();
    B2U.get_or_init(|| {
        let mut table = ['\0'; 256];
        let mut mapped = [false; 256];
        for (lo, hi) in [(0x21usize, 0x7e), (0xa1, 0xac), (0xae, 0xff)] {
            for b in lo..=hi {
                table[b] = char::from_u32(b as u32).unwrap();
                mapped[b] = true;
            }
        }
        let mut n = 0u32;
        for b in 0..256 {
            if !mapped[b] {
                table[b] = char::from_u32(256 + n).unwrap();
                n += 1;
            }
        }
        table
    })
}

fn encode_bytelevel_str(s: &str) -> String {
    let b2u = byte_to_char_table();
    let mut out = String::with_capacity(s.len() * 2);
    for &b in s.as_bytes() {
        out.push(b2u[b as usize]);
    }
    out
}

fn decode_bytelevel_bytes(s: &str, c2b: &[u16; 324], out: &mut Vec<u8>) {
    out.clear();
    for ch in s.chars() {
        let cp = ch as usize;
        if cp < 324 {
            let v = c2b[cp];
            if v != u16::MAX {
                out.push(v as u8);
                continue;
            }
        }
        let mut buf = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
    }
}

/// Translate Oniguruma-specific nested repetition syntax (`X{n}+`, `X{m,n}+`,
/// `X{n}*`, `X{m,n}*`) into PCRE/`fancy-regex` syntax (`(?:X{n})+`, etc.).
///
/// In Oniguruma (used by HF `tokenizers`), `+` or `*` immediately after a `{...}`
/// quantifier stacks a second quantifier onto the `{...}` atom (e.g. Dolphin's
/// `\p{Nd}{3}+` means 1+ groups of 3 digits), whereas `fancy-regex` parses `{3}+`
/// as a possessive fixed repeat of 3. Plain `{m,n}?` stays lazy as in both engines.
fn onig_to_fancy(pat: &str) -> String {
    let b = pat.as_bytes();
    let n = b.len();
    let mut out = String::with_capacity(n + 16);
    let mut group_stack: Vec<usize> = Vec::new();
    let mut last_atom_start: Option<usize> = None;
    let mut i = 0usize;

    while i < n {
        let c = b[i];
        // Escape sequence (including \p{...}, \P{...}, \u{...}, \xHH, \uHHHH)
        if c == b'\\' {
            let atom_start = out.len();
            out.push('\\');
            i += 1;
            if i < n {
                let nc = b[i];
                let ch = pat[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
                if matches!(nc, b'p' | b'P') && i < n && b[i] == b'{' {
                    while i < n {
                        let ch2 = pat[i..].chars().next().unwrap();
                        out.push(ch2);
                        i += ch2.len_utf8();
                        if ch2 == '}' {
                            break;
                        }
                    }
                } else if nc == b'u' && i < n && b[i] == b'{' {
                    while i < n {
                        let ch2 = pat[i..].chars().next().unwrap();
                        out.push(ch2);
                        i += ch2.len_utf8();
                        if ch2 == '}' {
                            break;
                        }
                    }
                }
            }
            last_atom_start = Some(atom_start);
            continue;
        }
        // Character class [...]
        if c == b'[' {
            let atom_start = out.len();
            out.push('[');
            i += 1;
            let mut first = true;
            while i < n {
                let ch = pat[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
                if ch == '\\' && i < n {
                    let ch2 = pat[i..].chars().next().unwrap();
                    out.push(ch2);
                    i += ch2.len_utf8();
                    first = false;
                    continue;
                }
                if ch == '^' && first {
                    continue;
                }
                if ch == ']' && !first {
                    break;
                }
                first = false;
            }
            last_atom_start = Some(atom_start);
            continue;
        }
        if c == b'(' {
            group_stack.push(out.len());
            out.push('(');
            i += 1;
            last_atom_start = None;
            continue;
        }
        if c == b')' {
            out.push(')');
            i += 1;
            last_atom_start = group_stack.pop();
            continue;
        }
        // Quantifier {n} or {m,n} immediately followed by '+' or '*'
        if c == b'{' {
            if let Some(atom_start) = last_atom_start {
                let mut j = i + 1;
                let mut saw_digit = false;
                while j < n && b[j].is_ascii_digit() {
                    saw_digit = true;
                    j += 1;
                }
                let mut valid = saw_digit;
                if valid && j < n && b[j] == b',' {
                    j += 1;
                    while j < n && b[j].is_ascii_digit() {
                        j += 1;
                    }
                }
                if valid && j < n && b[j] == b'}' && j + 1 < n && matches!(b[j + 1], b'+' | b'*') {
                    let q_end = j + 1; // exclusive of '}'
                    let rep = b[q_end] as char;
                    out.insert_str(atom_start, "(?:");
                    out.push_str(&pat[i..q_end]);
                    out.push(')');
                    out.push(rep);
                    i = q_end + 1;
                    last_atom_start = Some(atom_start);
                    continue;
                }
                valid = false;
                let _ = valid;
            }
        }
        let ch = pat[i..].chars().next().unwrap();
        let atom_start = out.len();
        out.push(ch);
        i += ch.len_utf8();
        last_atom_start = if matches!(ch, '|' | '^' | '$') { None } else { Some(atom_start) };
    }
    out
}

fn parse_stages(val: &Value, out: &mut Vec<PretokStage>) -> Result<(), String> {
    if val.is_null() {
        return Ok(());
    }
    let t = val
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "pre_tokenizer missing type".to_string())?;
    match t {
        "Sequence" => {
            if let Some(arr) = val.get("pretokenizers").and_then(|v| v.as_array()) {
                for item in arr {
                    parse_stages(item, out)?;
                }
            }
            Ok(())
        }
        "Split" => {
            let behavior = SplitBehavior::from_json(val.get("behavior"), SplitBehavior::Isolated)?;
            let invert = val.get("invert").and_then(|v| v.as_bool()).unwrap_or(false);
            let pat = val
                .get("pattern")
                .ok_or_else(|| "Split missing pattern".to_string())?;
            let pattern = if let Some(r) = pat.get("Regex").and_then(|v| v.as_str()) {
                let total_ok = (behavior == SplitBehavior::Isolated && !invert)
                    || (behavior == SplitBehavior::Removed && invert);
                if total_ok {
                    if let Some(kind) = crate::match_fast_split_regex(r) {
                        out.push(PretokStage::Split {
                            pattern: SplitPattern::FastMachine(kind),
                            behavior,
                            invert,
                        });
                        return Ok(());
                    }
                }
                let translated = onig_to_fancy(r);
                let re = fancy_regex::Regex::new(&translated)
                    .map_err(|e| format!("cannot compile Split regex {r:?}: {e}"))?;
                SplitPattern::Regex(re)
            } else if let Some(s) = pat.get("String").and_then(|v| v.as_str()) {
                SplitPattern::Literal(s.to_string())
            } else {
                return Err("Split pattern must be Regex or String".to_string());
            };
            out.push(PretokStage::Split { pattern, behavior, invert });
            Ok(())
        }
        "Digits" => {
            let individual_digits = val
                .get("individual_digits")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            out.push(PretokStage::Digits { individual_digits });
            Ok(())
        }
        "Punctuation" => {
            let behavior = SplitBehavior::from_json(val.get("behavior"), SplitBehavior::Isolated)?;
            out.push(PretokStage::Punctuation { behavior });
            Ok(())
        }
        "ByteLevel" => {
            let add_prefix_space = val
                .get("add_prefix_space")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let use_regex = val
                .get("use_regex")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            out.push(PretokStage::ByteLevel { add_prefix_space, use_regex });
            Ok(())
        }
        "Metaspace" => {
            let replacement = val
                .get("replacement")
                .and_then(|v| v.as_str())
                .and_then(|s| {
                    let mut it = s.chars();
                    let c = it.next()?;
                    if it.next().is_none() {
                        Some(c)
                    } else {
                        None
                    }
                })
                .unwrap_or('\u{2581}');
            let prepend_scheme = PrependScheme::from_json(val.get("prepend_scheme"))?;
            let split = val.get("split").and_then(|v| v.as_bool()).unwrap_or(true);
            out.push(PretokStage::Metaspace { replacement, prepend_scheme, split });
            Ok(())
        }
        "Whitespace" => {
            out.push(PretokStage::Whitespace);
            Ok(())
        }
        "WhitespaceSplit" => {
            out.push(PretokStage::WhitespaceSplit);
            Ok(())
        }
        other => Err(format!("unsupported pre_tokenizer stage {other:?}")),
    }
}

/// Apply HF's exact `SplitDelimiterBehavior` to a partition `segs` of
/// `(start, end, is_delim)` over a single piece, pushing resulting `(start, end)`
/// byte spans into `out`.
#[inline]
fn apply_behavior(segs: &[(usize, usize, bool)], behavior: SplitBehavior, out: &mut Vec<(usize, usize)>) {
    match behavior {
        SplitBehavior::Removed => {
            for &(s, e, is_delim) in segs {
                if !is_delim {
                    out.push((s, e));
                }
            }
        }
        SplitBehavior::Isolated => {
            for &(s, e, _) in segs {
                out.push((s, e));
            }
        }
        SplitBehavior::MergedWithPrevious => {
            let mut i = 0;
            while i < segs.len() {
                let (s, mut e, d) = segs[i];
                if !d && i + 1 < segs.len() && segs[i + 1].2 {
                    e = segs[i + 1].1;
                    i += 2;
                } else {
                    i += 1;
                }
                out.push((s, e));
            }
        }
        SplitBehavior::MergedWithNext => {
            let mut i = 0;
            while i < segs.len() {
                let (s, mut e, d) = segs[i];
                if d && i + 1 < segs.len() && !segs[i + 1].2 {
                    e = segs[i + 1].1;
                    i += 2;
                } else {
                    i += 1;
                }
                out.push((s, e));
            }
        }
        SplitBehavior::Contiguous => {
            let mut i = 0;
            while i < segs.len() {
                let (s, mut e, d) = segs[i];
                i += 1;
                while i < segs.len() && segs[i].2 == d {
                    e = segs[i].1;
                    i += 1;
                }
                out.push((s, e));
            }
        }
    }
}

/// Partition `sub` (`text[base..base+sub.len()]`) by a char predicate `is_match`,
/// pushing `(abs_start, abs_end)` pieces according to `behavior` and `invert`.
fn split_by_char_pred(
    sub: &str,
    base: usize,
    pred: impl Fn(char) -> bool,
    behavior: SplitBehavior,
    invert: bool,
    segs: &mut Vec<(usize, usize, bool)>,
    out: &mut Vec<(usize, usize)>,
) {
    if sub.is_empty() {
        return;
    }
    segs.clear();
    let mut non_start: Option<usize> = None;
    for (idx, ch) in sub.char_indices() {
        let len = ch.len_utf8();
        if pred(ch) {
            if let Some(ns) = non_start.take() {
                segs.push((base + ns, base + idx, invert));
            }
            segs.push((base + idx, base + idx + len, !invert));
        } else if non_start.is_none() {
            non_start = Some(idx);
        }
    }
    if let Some(ns) = non_start {
        segs.push((base + ns, base + sub.len(), invert));
    }
    apply_behavior(segs, behavior, out);
}

/// Partition `sub` (`text[base..base+sub.len()]`) by a `fancy_regex::Regex`
/// match iterator, pushing `(abs_start, abs_end)` pieces according to `behavior`
/// and `invert`.
fn split_by_regex(
    sub: &str,
    base: usize,
    re: &fancy_regex::Regex,
    behavior: SplitBehavior,
    invert: bool,
    segs: &mut Vec<(usize, usize, bool)>,
    out: &mut Vec<(usize, usize)>,
) {
    if sub.is_empty() {
        return;
    }
    segs.clear();
    let mut last = 0usize;
    for m in re.find_iter(sub) {
        let m = match m {
            Ok(m) => m,
            Err(_) => break,
        };
        let (ms, me) = (m.start(), m.end());
        if ms == me {
            continue;
        }
        if ms > last {
            segs.push((base + last, base + ms, invert));
        }
        segs.push((base + ms, base + me, !invert));
        last = me;
    }
    if last < sub.len() {
        segs.push((base + last, base + sub.len(), invert));
    }
    apply_behavior(segs, behavior, out);
}

/// Partition `sub` (`text[base..base+sub.len()]`) by literal occurrences of `lit`.
fn split_by_literal(
    sub: &str,
    base: usize,
    lit: &str,
    behavior: SplitBehavior,
    invert: bool,
    segs: &mut Vec<(usize, usize, bool)>,
    out: &mut Vec<(usize, usize)>,
) {
    if sub.is_empty() {
        return;
    }
    if lit.is_empty() {
        if !(behavior == SplitBehavior::Removed && invert) {
            out.push((base, base + sub.len()));
        }
        return;
    }
    segs.clear();
    let mut last = 0usize;
    for (ms, _) in sub.match_indices(lit) {
        let me = ms + lit.len();
        if ms > last {
            segs.push((base + last, base + ms, invert));
        }
        segs.push((base + ms, base + me, !invert));
        last = me;
    }
    if last < sub.len() {
        segs.push((base + last, base + sub.len(), invert));
    }
    apply_behavior(segs, behavior, out);
}

/// Split `sub` (`text[base..base+sub.len()]`) with GPT-2 ByteLevel's regex
/// (`Pretok::Word(false)` state machine — exact match, ~10x faster than regex).
fn split_gpt2_word(sub: &str, base: usize, out: &mut Vec<(usize, usize)>) {
    let b = sub.as_bytes();
    let n = b.len();
    let mut pos = 0usize;
    while pos < n {
        let end = next_chunk_end(Pretok::Word(false), b, pos);
        out.push((base + pos, base + end));
        pos = end;
    }
}

/// Apply one non-mutating stage to a single slice `sub` (`base`-anchored).
fn apply_stage_slice(
    stage: &PretokStage,
    sub: &str,
    base: usize,
    segs: &mut Vec<(usize, usize, bool)>,
    out: &mut Vec<(usize, usize)>,
) {
    match stage {
        PretokStage::Split { pattern, behavior, invert } => match pattern {
            SplitPattern::FastMachine(kind) => {
                let b = sub.as_bytes();
                let n = b.len();
                let mut pos = 0usize;
                while pos < n {
                    let end = next_chunk_end(*kind, b, pos);
                    out.push((base + pos, base + end));
                    pos = end;
                }
            }
            SplitPattern::Regex(re) => split_by_regex(sub, base, re, *behavior, *invert, segs, out),
            SplitPattern::Literal(lit) => {
                split_by_literal(sub, base, lit, *behavior, *invert, segs, out)
            }
        },
        PretokStage::Digits { individual_digits } => {
            let beh = if *individual_digits {
                SplitBehavior::Isolated
            } else {
                SplitBehavior::Contiguous
            };
            split_by_char_pred(sub, base, |c| c.is_numeric(), beh, false, segs, out);
        }
        PretokStage::Punctuation { behavior } => {
            split_by_char_pred(
                sub,
                base,
                |c| is_hf_punctuation(c as u32),
                *behavior,
                false,
                segs,
                out,
            );
        }
        PretokStage::ByteLevel { use_regex, .. } => {
            if *use_regex {
                split_gpt2_word(sub, base, out);
            } else if !sub.is_empty() {
                out.push((base, base + sub.len()));
            }
        }
        PretokStage::Whitespace => {
            split_by_regex(
                sub,
                base,
                whitespace_regex(),
                SplitBehavior::Removed,
                true,
                segs,
                out,
            );
        }
        PretokStage::WhitespaceSplit => {
            split_by_char_pred(
                sub,
                base,
                |c| c.is_whitespace(),
                SplitBehavior::Removed,
                false,
                segs,
                out,
            );
        }
        PretokStage::Metaspace { .. } => unreachable!("Metaspace is not slice_only"),
    }
}

impl PretokPipe {
    /// Parse any HF `pre_tokenizer` JSON value.
    /// Returns:
    ///   * `Ok(None)` for `null` or empty `Sequence` (whole-text single chunk)
    ///   * `Ok(Some(pipe))` for any supported composition of stages
    ///   * `Err(msg)` for any unrecognized/unsupported stage (fail closed!)
    pub fn from_json(val: &Value) -> Result<Option<Self>, String> {
        let mut stages = Vec::new();
        parse_stages(val, &mut stages)?;
        if stages.is_empty() {
            return Ok(None);
        }
        // Slice-only iff no Metaspace, no ByteLevel(add_prefix_space=true), and
        // no stage follows a ByteLevel stage (which transforms UTF-8 bytes into
        // ByteLevel Unicode characters for downstream stages).
        let mut seen_bl = false;
        let mut slice_only = true;
        for st in &stages {
            if seen_bl {
                slice_only = false;
                break;
            }
            match st {
                PretokStage::ByteLevel { add_prefix_space, .. } => {
                    seen_bl = true;
                    if *add_prefix_space {
                        slice_only = false;
                        break;
                    }
                }
                PretokStage::Metaspace { .. } => {
                    slice_only = false;
                    break;
                }
                _ => {}
            }
        }
        Ok(Some(PretokPipe { stages, slice_only }))
    }

    /// True if this pipeline only slices the original `&str` without mutating
    /// bytes (so `encode_with_offsets` can report exact input byte offsets).
    #[inline]
    pub fn is_slice_only(&self) -> bool {
        self.slice_only
    }

    /// If the pipeline is a single unsplit `Metaspace { replacement, split: false }`
    /// stage, return its `replacement` char (used for load-time verified
    /// metaspace chunk-splitting optimization).
    pub fn single_unsplit_metaspace(&self) -> Option<char> {
        if self.stages.len() == 1 {
            if let PretokStage::Metaspace { replacement, split: false, .. } = self.stages[0] {
                return Some(replacement);
            }
        }
        None
    }

    /// Zero-copy byte ranges for `slice_only` pipelines.
    pub fn chunk_ranges(&self, text: &str) -> Vec<(usize, usize)> {
        debug_assert!(self.slice_only);
        if text.is_empty() {
            return Vec::new();
        }
        let mut pieces: Vec<(usize, usize)> = vec![(0, text.len())];
        let mut next: Vec<(usize, usize)> = Vec::new();
        let mut segs: Vec<(usize, usize, bool)> = Vec::new();
        for st in &self.stages {
            next.clear();
            for &(ps, pe) in &pieces {
                apply_stage_slice(st, &text[ps..pe], ps, &mut segs, &mut next);
            }
            std::mem::swap(&mut pieces, &mut next);
            if pieces.is_empty() {
                break;
            }
        }
        pieces
    }

    /// Execute the full composable pipeline on `text`, invoking `emit` with the
    /// exact raw bytes of every final pre-token chunk.
    ///
    /// `at_start`: true for whole-string encodes and for the first fragment of
    /// `encode_special` (controls `Metaspace` `prepend_scheme="first"`).
    /// `c2b`: ByteLevel Unicode-to-byte inverse table (used only when stages run
    /// after a `ByteLevel` stage and transform pieces through ByteLevel chars).
    pub fn for_each_chunk(
        &self,
        text: &str,
        at_start: bool,
        c2b: &[u16; 324],
        mut emit: impl FnMut(&[u8]),
    ) {
        if text.is_empty() {
            return;
        }
        if self.slice_only {
            let bytes = text.as_bytes();
            for (ps, pe) in self.chunk_ranges(text) {
                emit(&bytes[ps..pe]);
            }
            return;
        }

        // General transforming mode (handles ByteLevel(add_prefix_space),
        // stages after ByteLevel like Falcon/Falcon3/Falcon-H1, and Metaspace).
        let mut pieces: Vec<String> = vec![text.to_owned()];
        let mut next: Vec<String> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut segs: Vec<(usize, usize, bool)> = Vec::new();
        let mut in_bytelevel = false;
        let last_idx = self.stages.len() - 1;

        for (si, st) in self.stages.iter().enumerate() {
            next.clear();
            match st {
                PretokStage::ByteLevel { add_prefix_space, use_regex } => {
                    let need_bl_chars = si < last_idx;
                    for p in &pieces {
                        if p.is_empty() {
                            continue;
                        }
                        let prefixed: std::borrow::Cow<'_, str> =
                            if *add_prefix_space && !p.starts_with(' ') {
                                let mut s = String::with_capacity(p.len() + 1);
                                s.push(' ');
                                s.push_str(p);
                                std::borrow::Cow::Owned(s)
                            } else {
                                std::borrow::Cow::Borrowed(p.as_str())
                            };
                        if *use_regex {
                            spans.clear();
                            split_gpt2_word(&prefixed, 0, &mut spans);
                            for &(a, b) in &spans {
                                let sub = &prefixed[a..b];
                                if need_bl_chars && !in_bytelevel {
                                    next.push(encode_bytelevel_str(sub));
                                } else {
                                    next.push(sub.to_owned());
                                }
                            }
                        } else if need_bl_chars && !in_bytelevel {
                            next.push(encode_bytelevel_str(&prefixed));
                        } else {
                            next.push(prefixed.into_owned());
                        }
                    }
                    if need_bl_chars {
                        in_bytelevel = true;
                    }
                }
                PretokStage::Metaspace { replacement, prepend_scheme, split } => {
                    let mut rep_buf = [0u8; 4];
                    let rep_str = replacement.encode_utf8(&mut rep_buf);
                    for (pi, p) in pieces.iter().enumerate() {
                        if p.is_empty() {
                            continue;
                        }
                        let mut s = if p.contains(' ') {
                            p.replace(' ', rep_str)
                        } else {
                            p.clone()
                        };
                        let do_prepend = match prepend_scheme {
                            PrependScheme::Always => true,
                            PrependScheme::First => at_start && pi == 0,
                            PrependScheme::Never => false,
                        };
                        if do_prepend && !s.starts_with(*replacement) {
                            let mut pref = String::with_capacity(rep_str.len() + s.len());
                            pref.push_str(rep_str);
                            pref.push_str(&s);
                            s = pref;
                        }
                        if *split {
                            spans.clear();
                            split_by_char_pred(
                                &s,
                                0,
                                |c| c == *replacement,
                                SplitBehavior::MergedWithNext,
                                false,
                                &mut segs,
                                &mut spans,
                            );
                            for &(a, b) in &spans {
                                next.push(s[a..b].to_owned());
                            }
                        } else {
                            next.push(s);
                        }
                    }
                }
                _ => {
                    for p in &pieces {
                        spans.clear();
                        apply_stage_slice(st, p, 0, &mut segs, &mut spans);
                        for &(a, b) in &spans {
                            next.push(p[a..b].to_owned());
                        }
                    }
                }
            }
            std::mem::swap(&mut pieces, &mut next);
            if pieces.is_empty() {
                break;
            }
        }

        if in_bytelevel {
            let mut raw = Vec::with_capacity(64);
            for p in &pieces {
                decode_bytelevel_bytes(p, c2b, &mut raw);
                if !raw.is_empty() {
                    emit(&raw);
                }
            }
        } else {
            for p in &pieces {
                if !p.is_empty() {
                    emit(p.as_bytes());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn run_strs(cfg: Value, text: &str) -> Vec<String> {
        let pipe = PretokPipe::from_json(&cfg).unwrap().unwrap();
        let c2b = crate::bytes_to_unicode_table();
        let mut out = Vec::new();
        pipe.for_each_chunk(text, true, &c2b, |b| {
            out.push(String::from_utf8_lossy(b).into_owned());
        });
        out
    }

    #[test]
    fn split_behaviors_and_invert() {
        let mk = |beh: &str, inv: bool| {
            json!({
                "type": "Split",
                "pattern": {"Regex": "[.!]"},
                "behavior": beh,
                "invert": inv,
            })
        };
        let s = "..a..b!c!";
        assert_eq!(run_strs(mk("Removed", false), s), vec!["a", "b", "c"]);
        assert_eq!(run_strs(mk("Removed", true), s), vec![".", ".", ".", ".", "!", "!"]);
        assert_eq!(
            run_strs(mk("Isolated", false), s),
            vec![".", ".", "a", ".", ".", "b", "!", "c", "!"]
        );
        assert_eq!(
            run_strs(mk("MergedWithPrevious", false), s),
            vec![".", ".", "a.", ".", "b!", "c!"]
        );
        assert_eq!(
            run_strs(mk("MergedWithPrevious", true), s),
            vec![".", ".a", ".", ".b", "!c", "!"]
        );
        assert_eq!(
            run_strs(mk("MergedWithNext", false), s),
            vec![".", ".a", ".", ".b", "!c", "!"]
        );
        assert_eq!(
            run_strs(mk("MergedWithNext", true), s),
            vec![".", ".", "a.", ".", "b!", "c!"]
        );
        assert_eq!(
            run_strs(mk("Contiguous", false), s),
            vec!["..", "a", "..", "b", "!", "c", "!"]
        );
    }

    #[test]
    fn punctuation_behaviors() {
        let s = "a...b!? c";
        assert_eq!(
            run_strs(json!({"type": "Punctuation", "behavior": "Contiguous"}), s),
            vec!["a", "...", "b", "!?", " c"]
        );
        assert_eq!(
            run_strs(json!({"type": "Punctuation", "behavior": "MergedWithPrevious"}), s),
            vec!["a.", ".", ".", "b!", "?", " c"]
        );
        assert_eq!(
            run_strs(json!({"type": "Punctuation", "behavior": "MergedWithNext"}), s),
            vec!["a", ".", ".", ".b", "!", "? c"]
        );
    }

    #[test]
    fn dolphin_right_aligned_digits() {
        let cfg = json!({
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": false},
                {"type": "Split", "pattern": {"Regex": "\\p{Nd}{1,510}(?=(?>\\p{Nd}{510})*(?:\\P{Nd}|$))|\\G\\p{Nd}{510}"}, "behavior": "Isolated", "invert": false},
                {"type": "Split", "pattern": {"Regex": "\\A\\p{Nd}{1,2}(?=\\p{Nd}{3}+\\z)"}, "behavior": "Isolated", "invert": false},
                {"type": "Split", "pattern": {"Regex": "\\A\\p{Nd}{3}|\\G\\p{Nd}{3}"}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
            ]
        });
        assert_eq!(run_strs(cfg.clone(), "1234"), vec!["1", "234"]);
        assert_eq!(run_strs(cfg.clone(), "12345"), vec!["12", "345"]);
        assert_eq!(run_strs(cfg.clone(), "123456"), vec!["123", "456"]);
        assert_eq!(run_strs(cfg.clone(), "a1234567b"), vec!["a", "1", "234", "567", "b"]);
        let long = "1".repeat(515);
        let parts = run_strs(cfg, &long);
        assert_eq!(parts[0], "11");
        assert_eq!(parts[1], "111");
        assert_eq!(parts.len(), 172);
    }

    #[test]
    fn falcon3_post_bytelevel_cyrillic_bytes() {
        // 'в' is UTF-8 0xD0 0xB2; in ByteLevel 0xB2 -> '²' (numeric!), so
        // Digits running AFTER ByteLevel splits right between 0xD0 and 0xB2.
        let cfg = json!({
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Punctuation", "behavior": "Contiguous"},
                {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
                {"type": "Digits", "individual_digits": true}
            ]
        });
        let pipe = PretokPipe::from_json(&cfg).unwrap().unwrap();
        let c2b = crate::bytes_to_unicode_table();
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        pipe.for_each_chunk(" в 12", true, &c2b, |b| chunks.push(b.to_vec()));
        assert_eq!(
            chunks,
            vec![
                vec![b' ', 0xD0],
                vec![0xB2],
                vec![b' '],
                vec![b'1'],
                vec![b'2'],
            ]
        );
    }

    #[test]
    fn metaspace_variants() {
        for (ps, want_a, want_ab) in [
            ("always", vec!["▁a"], vec!["▁a", "▁", "▁b"]),
            ("first", vec!["▁a"], vec!["▁a", "▁", "▁b"]),
            ("never", vec!["a"], vec!["a", "▁", "▁b"]),
        ] {
            let cfg = json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": ps, "split": true});
            assert_eq!(run_strs(cfg.clone(), "a"), want_a);
            assert_eq!(run_strs(cfg, "a  b"), want_ab);
        }
        // First scheme with at_start=false must NOT prepend
        let pipe = PretokPipe::from_json(&json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false}))
            .unwrap()
            .unwrap();
        let c2b = crate::bytes_to_unicode_table();
        let mut out = Vec::new();
        pipe.for_each_chunk("hello world", false, &c2b, |b| {
            out.push(std::str::from_utf8(b).unwrap().to_owned());
        });
        assert_eq!(out, vec!["hello▁world"]);
    }

    #[test]
    fn fail_closed_on_unknown_stage() {
        assert!(PretokPipe::from_json(&json!({"type": "BertPreTokenizer"})).is_err());
        assert!(PretokPipe::from_json(&json!({"type": "UnicodeScripts"})).is_err());
    }
}
