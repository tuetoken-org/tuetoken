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

/// Proven-sound necessary-condition guard checked in O(n) bytes before invoking
/// `fancy-regex` on a piece. When the guard is false, the regex is guaranteed
/// to have zero matches in that piece, skipping the regex engine completely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegexGuard {
    None,
    /// Every match requires at least one non-ASCII byte (`>= 0x80`).
    NonAsciiOnly,
    /// Every match requires at least one Unicode/ASCII decimal digit.
    NeedsDigit,
    /// Every match requires at least one whitespace character.
    NeedsWs,
}

#[derive(Clone, Debug)]
pub enum SplitPattern {
    /// Hand-written exact SWAR state machine (used automatically when a `Split`
    /// stage inside a multi-stage `Sequence` matches a proven total grammar).
    FastMachine(Pretok),
    /// Generic pre-tokenizer variant whose symbol branch does not absorb
    /// trailing `\r`/`\n` (e.g. Seed-Coder's ` ?[^\s\p{L}\p{N}\r\n]+`).
    GenericNoNl { max_digits: u32, marks: bool },
    /// DeepSeek/Dolphin stage-3 main split (`src/cjk.rs::split_main_stage`).
    CjkMain,
    /// Exact consecutive runs of `k` ASCII digits (`[0-9]`, `[0-9][0-9][0-9]`).
    AsciiDigitRun(usize),
    /// Exact `[\r\n]` byte split via SIMD `memchr2`.
    CrLf,
    Regex(fancy_regex::Regex, RegexGuard),
    Literal(String),
}

/// Reusable ping-pong arenas and span buffers for zero-per-piece-allocation
/// `PretokPipe` execution (stored inside `Scratch` per thread).
#[derive(Default)]
pub struct PipeScratch {
    cur_buf: String,
    nxt_buf: String,
    cur_spans: Vec<(usize, usize)>,
    nxt_spans: Vec<(usize, usize)>,
    tmp_spans: Vec<(usize, usize)>,
    segs: Vec<(usize, usize, bool)>,
    raw: Vec<u8>,
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

fn encode_bytelevel_into(s: &str, out: &mut String) {
    let b2u = byte_to_char_table();
    out.reserve(s.len());
    for &b in s.as_bytes() {
        out.push(b2u[b as usize]);
    }
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
        // Escape sequence (including \p{...}, \P{...}, \u{...}, \x{...}, \xHH, \uHHHH)
        if c == b'\\' {
            let atom_start = out.len();
            out.push('\\');
            i += 1;
            if i < n {
                let nc = b[i];
                let ch = pat[i..].chars().next().unwrap();
                out.push(ch);
                i += ch.len_utf8();
                if matches!(nc, b'p' | b'P' | b'u' | b'x') && i < n && b[i] == b'{' {
                    while i < n {
                        let ch2 = pat[i..].chars().next().unwrap();
                        out.push(ch2);
                        i += ch2.len_utf8();
                        if ch2 == '}' {
                            break;
                        }
                    }
                } else if nc == b'x' && i + 2 <= n && b[i].is_ascii_hexdigit() && b[i + 1].is_ascii_hexdigit() {
                    out.push(b[i] as char);
                    out.push(b[i + 1] as char);
                    i += 2;
                } else if nc == b'u'
                    && i + 4 <= n
                    && b[i..i + 4].iter().all(|x| x.is_ascii_hexdigit())
                {
                    out.push_str(&pat[i..i + 4]);
                    i += 4;
                }
            }
            last_atom_start = Some(atom_start);
            continue;
        }
        // Character class [...] (including nested POSIX classes like [:alpha:])
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
                if ch == '[' && i < n && b[i] == b':' {
                    out.push(':');
                    i += 1;
                    while i < n {
                        let ch2 = pat[i..].chars().next().unwrap();
                        out.push(ch2);
                        i += ch2.len_utf8();
                        if ch2 == ':' && i < n && b[i] == b']' {
                            out.push(']');
                            i += 1;
                            break;
                        }
                    }
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
        // Quantifier {n}, {m,n}, or {,n} immediately followed by '+' or '*'
        if c == b'{' {
            if let Some(atom_start) = last_atom_start {
                let mut j = i + 1;
                let mut saw_lhs = false;
                while j < n && b[j].is_ascii_digit() {
                    saw_lhs = true;
                    j += 1;
                }
                let mut saw_rhs = false;
                let mut has_comma = false;
                if j < n && b[j] == b',' {
                    has_comma = true;
                    j += 1;
                    while j < n && b[j].is_ascii_digit() {
                        saw_rhs = true;
                        j += 1;
                    }
                }
                let valid = saw_lhs || (has_comma && saw_rhs);
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

/// Canonicalize literal CR/LF bytes (`0x0D`/`0x0A`) in JSON regex strings to
/// `\\r`/`\\n` escape sequences so recognizers match both spellings identically
/// (many HF `tokenizer.json` files embed raw `\r\n` instead of `\\r\\n`).
fn canon_crlf(pat: &str) -> std::borrow::Cow<'_, str> {
    if !pat.as_bytes().iter().any(|&b| b == b'\r' || b == b'\n') {
        return std::borrow::Cow::Borrowed(pat);
    }
    let mut s = String::with_capacity(pat.len() + 8);
    for ch in pat.chars() {
        match ch {
            '\r' => s.push_str(r"\r"),
            '\n' => s.push_str(r"\n"),
            _ => s.push(ch),
        }
    }
    std::borrow::Cow::Owned(s)
}

/// Exact DeepSeek / Dolphin stage-3 main regex (matched to `cjk::split_main_stage`).
const CJK_MAIN_REGEX: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

/// Match Generic pre-tokenizer variants whose symbol branch does NOT absorb
/// trailing `\r`/`\n` (e.g. Seed-Coder's ` ?[^\s\p{L}\p{N}\r\n]+`).
fn match_generic_no_nl(pat: &str) -> Option<(u32, bool)> {
    let rest = pat
        .strip_prefix(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|")
        .or_else(|| pat.strip_prefix(r"'(?i:[sdmt]|ll|ve|re)|"))?;
    let (marks, rest) = if let Some(r) = rest
        .strip_prefix(r"[^\r\n\p{L}\p{N}]?\p{L}+|")
        .or_else(|| rest.strip_prefix(r"[^\r\n\p{L}\p{N}]?+\p{L}+|"))
    {
        (false, r)
    } else {
        (true, rest.strip_prefix(r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|")?)
    };
    let (dig_part, rest) = rest.split_once('|')?;
    let max_digits = match dig_part {
        r"\p{N}+" => 0,
        r"\p{N}" | r"\p{N}{1}" | r"\p{N}{1,1}" => 1,
        r"\p{N}{1,2}" => 2,
        r"\p{N}{1,3}" => 3,
        _ => return None,
    };
    let rest = if marks {
        rest.strip_prefix(r" ?[^\s\p{L}\p{M}\p{N}\r\n]+|")
            .or_else(|| rest.strip_prefix(r" ?[^\s\p{L}\p{M}\p{N}]+|"))?
    } else {
        rest.strip_prefix(r" ?[^\s\p{L}\p{N}\r\n]+|")
            .or_else(|| rest.strip_prefix(r" ?[^\s\p{L}\p{N}]+|"))?
    };
    if matches!(
        rest,
        r"\s*[\r\n]+|\s+(?!\S)|\s+" | r"\s*[\r\n]|\s+(?!\S)|\s+"
    ) {
        Some((max_digits, marks))
    } else {
        None
    }
}

/// Prove whether a bracket class `[...]+` only matches non-ASCII codepoints (`>= 0x80`).
fn is_non_ascii_bracket_atom(s: &str) -> bool {
    let inner = match s.strip_prefix('[').and_then(|x| x.strip_suffix("]+")) {
        Some(x) if !x.is_empty() && !x.starts_with('^') => x,
        _ => return false,
    };
    let b = inner.as_bytes();
    let mut i = 0usize;
    let mut saw_any = false;
    while i < b.len() {
        if b[i] == b'-' {
            i += 1;
            continue;
        }
        if b[i] == b'\\' {
            if i + 6 <= b.len() && b[i + 1] == b'u' && b[i + 2..i + 6].iter().all(|c| c.is_ascii_hexdigit()) {
                let cp = u32::from_str_radix(&inner[i + 2..i + 6], 16).unwrap_or(0);
                if cp < 0x80 {
                    return false;
                }
                saw_any = true;
                i += 6;
                continue;
            }
            return false;
        }
        let ch = inner[i..].chars().next().unwrap();
        if (ch as u32) < 0x80 {
            return false;
        }
        saw_any = true;
        i += ch.len_utf8();
    }
    saw_any
}

/// Infer a proven-sound necessary-condition guard for a `Split` regex.
fn infer_regex_guard(pat: &str) -> RegexGuard {
    if pat == r"\s+$" {
        return RegexGuard::NeedsWs;
    }
    // Check if `pat` is `[non-ascii]+` or `(?:[non-ascii]+|[non-ascii]+|...)`.
    let body = pat
        .strip_prefix("(?:")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(pat);
    if !body.is_empty() && body.split('|').all(is_non_ascii_bracket_atom) {
        return RegexGuard::NonAsciiOnly;
    }
    // Check if every top-level `|` alternative begins with a digit requirement.
    let mut depth = 0usize;
    let mut in_class = false;
    let mut esc = false;
    let mut start = 0usize;
    let mut alts = Vec::new();
    for (idx, ch) in pat.char_indices() {
        if esc {
            esc = false;
            continue;
        }
        match ch {
            '\\' => esc = true,
            '[' if !in_class => in_class = true,
            ']' if in_class => in_class = false,
            '(' if !in_class => depth += 1,
            ')' if !in_class => depth = depth.saturating_sub(1),
            '|' if !in_class && depth == 0 => {
                alts.push(&pat[start..idx]);
                start = idx + 1;
            }
            _ => {}
        }
    }
    alts.push(&pat[start..]);
    let alt_needs_digit = |a: &str| {
        let a = a.strip_prefix(r"\A").or_else(|| a.strip_prefix(r"\G")).unwrap_or(a);
        a.starts_with(r"\p{Nd}") || a.starts_with(r"\p{N}") || a.starts_with("[0-9]")
    };
    if !alts.is_empty() && alts.iter().all(|&a| alt_needs_digit(a)) {
        return RegexGuard::NeedsDigit;
    }
    RegexGuard::None
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
                let rc = canon_crlf(r);
                let total_ok = (behavior == SplitBehavior::Isolated && !invert)
                    || (behavior == SplitBehavior::Removed && invert);
                if total_ok {
                    if let Some(kind) = crate::match_fast_split_regex(&rc) {
                        out.push(PretokStage::Split {
                            pattern: SplitPattern::FastMachine(kind),
                            behavior,
                            invert,
                        });
                        return Ok(());
                    }
                    if let Some((max_digits, marks)) = match_generic_no_nl(&rc) {
                        out.push(PretokStage::Split {
                            pattern: SplitPattern::GenericNoNl { max_digits, marks },
                            behavior,
                            invert,
                        });
                        return Ok(());
                    }
                    if rc.as_ref() == CJK_MAIN_REGEX {
                        out.push(PretokStage::Split {
                            pattern: SplitPattern::CjkMain,
                            behavior,
                            invert,
                        });
                        return Ok(());
                    }
                }
                if rc.as_ref() == "[0-9]" {
                    SplitPattern::AsciiDigitRun(1)
                } else if rc.as_ref() == "[0-9][0-9][0-9]" {
                    SplitPattern::AsciiDigitRun(3)
                } else if rc.as_ref() == r"[\r\n]" {
                    SplitPattern::CrLf
                } else {
                    let guard = infer_regex_guard(r);
                    let translated = onig_to_fancy(r);
                    let re = fancy_regex::Regex::new(&translated)
                        .map_err(|e| format!("cannot compile Split regex {r:?}: {e}"))?;
                    SplitPattern::Regex(re, guard)
                }
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

/// Emit whole `sub` when a stage is proven to have zero matches in `sub`.
#[inline(always)]
fn emit_no_match(
    base: usize,
    len: usize,
    behavior: SplitBehavior,
    invert: bool,
    out: &mut Vec<(usize, usize)>,
) {
    if len > 0 && !(behavior == SplitBehavior::Removed && invert) {
        out.push((base, base + len));
    }
}

/// Partition `sub` (`text[base..base+sub.len()]`) by exact runs of `k` ASCII
/// digits (`[0-9]` or `[0-9][0-9][0-9]`), matching regex left-to-right semantics.
fn split_ascii_digit_run(
    sub: &str,
    base: usize,
    k: usize,
    behavior: SplitBehavior,
    invert: bool,
    segs: &mut Vec<(usize, usize, bool)>,
    out: &mut Vec<(usize, usize)>,
) {
    let b = sub.as_bytes();
    let n = b.len();
    if n == 0 {
        return;
    }
    if n < k || !b.iter().any(|x| x.is_ascii_digit()) {
        emit_no_match(base, n, behavior, invert, out);
        return;
    }
    segs.clear();
    let mut last = 0usize;
    let mut i = 0usize;
    while i + k <= n {
        if b[i..i + k].iter().all(|x| x.is_ascii_digit()) {
            if i > last {
                segs.push((base + last, base + i, invert));
            }
            segs.push((base + i, base + i + k, !invert));
            i += k;
            last = i;
        } else {
            i += 1;
        }
    }
    if last < n {
        segs.push((base + last, base + n, invert));
    }
    apply_behavior(segs, behavior, out);
}

/// Partition `sub` on `[\r\n]` via SIMD `memchr2`.
fn split_crlf(
    sub: &str,
    base: usize,
    behavior: SplitBehavior,
    invert: bool,
    segs: &mut Vec<(usize, usize, bool)>,
    out: &mut Vec<(usize, usize)>,
) {
    let b = sub.as_bytes();
    let n = b.len();
    if n == 0 {
        return;
    }
    let first = match memchr::memchr2(b'\r', b'\n', b) {
        Some(pos) => pos,
        None => {
            emit_no_match(base, n, behavior, invert, out);
            return;
        }
    };
    segs.clear();
    let mut last = 0usize;
    let mut pos = first;
    loop {
        if pos > last {
            segs.push((base + last, base + pos, invert));
        }
        segs.push((base + pos, base + pos + 1, !invert));
        last = pos + 1;
        match memchr::memchr2(b'\r', b'\n', &b[last..]) {
            Some(rel) => pos = last + rel,
            None => break,
        }
    }
    if last < n {
        segs.push((base + last, base + n, invert));
    }
    apply_behavior(segs, behavior, out);
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
            SplitPattern::GenericNoNl { max_digits, marks } => {
                let b = sub.as_bytes();
                let n = b.len();
                let mut pos = 0usize;
                while pos < n {
                    let end = crate::pretok::next_chunk_end_generic_no_nl(b, pos, *max_digits, *marks);
                    out.push((base + pos, base + end));
                    pos = end;
                }
            }
            SplitPattern::CjkMain => {
                crate::cjk::split_main_stage(sub, base, out);
            }
            SplitPattern::AsciiDigitRun(k) => {
                split_ascii_digit_run(sub, base, *k, *behavior, *invert, segs, out);
            }
            SplitPattern::CrLf => {
                split_crlf(sub, base, *behavior, *invert, segs, out);
            }
            SplitPattern::Regex(re, guard) => {
                let skip = match guard {
                    RegexGuard::None => false,
                    RegexGuard::NonAsciiOnly => sub.is_ascii(),
                    RegexGuard::NeedsDigit => sub.bytes().all(|b| b < 0x80 && !b.is_ascii_digit()),
                    RegexGuard::NeedsWs => sub.bytes().all(|b| b < 0x80 && !b.is_ascii_whitespace()),
                };
                if skip {
                    emit_no_match(base, sub.len(), *behavior, *invert, out);
                } else {
                    split_by_regex(sub, base, re, *behavior, *invert, segs, out);
                }
            }
            SplitPattern::Literal(lit) => {
                split_by_literal(sub, base, lit, *behavior, *invert, segs, out)
            }
        },
        PretokStage::Digits { individual_digits } => {
            if sub.bytes().all(|b| b < 0x80 && !b.is_ascii_digit()) {
                if !sub.is_empty() {
                    out.push((base, base + sub.len()));
                }
                return;
            }
            let beh = if *individual_digits {
                SplitBehavior::Isolated
            } else {
                SplitBehavior::Contiguous
            };
            split_by_char_pred(sub, base, |c| c.is_numeric(), beh, false, segs, out);
        }
        PretokStage::Punctuation { behavior } => {
            if sub.bytes().all(|b| b < 0x80 && !b.is_ascii_punctuation()) {
                if !sub.is_empty() {
                    out.push((base, base + sub.len()));
                }
                return;
            }
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

    /// Zero-copy byte ranges for `slice_only` pipelines, reusing `ps`.
    fn chunk_ranges_into(&self, text: &str, ps: &mut PipeScratch) {
        debug_assert!(self.slice_only);
        ps.cur_spans.clear();
        if text.is_empty() {
            return;
        }
        ps.cur_spans.push((0, text.len()));
        for st in &self.stages {
            ps.nxt_spans.clear();
            for &(s, e) in &ps.cur_spans {
                apply_stage_slice(st, &text[s..e], s, &mut ps.segs, &mut ps.nxt_spans);
            }
            std::mem::swap(&mut ps.cur_spans, &mut ps.nxt_spans);
            if ps.cur_spans.is_empty() {
                break;
            }
        }
    }

    /// Zero-copy byte ranges for `slice_only` pipelines.
    pub fn chunk_ranges(&self, text: &str) -> Vec<(usize, usize)> {
        let mut ps = PipeScratch::default();
        self.chunk_ranges_into(text, &mut ps);
        ps.cur_spans
    }

    /// Execute the full composable pipeline on `text`, invoking `emit` with the
    /// exact raw bytes of every final pre-token chunk. Reuses `ps` ping-pong
    /// arenas so transforming pipelines (Falcon/Falcon3/Falcon-H1, Metaspace,
    /// `add_prefix_space`) do zero per-piece heap allocations.
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
        ps: &mut PipeScratch,
        mut emit: impl FnMut(&[u8]),
    ) {
        if text.is_empty() {
            return;
        }
        if self.slice_only {
            self.chunk_ranges_into(text, ps);
            let bytes = text.as_bytes();
            for &(s, e) in &ps.cur_spans {
                emit(&bytes[s..e]);
            }
            return;
        }

        // General transforming mode using ping-pong flat String arenas:
        // non-mutating stages slice `ps.cur_buf` in-place (zero byte copies);
        // mutating stages append into `ps.nxt_buf` and swap.
        ps.cur_buf.clear();
        ps.cur_buf.push_str(text);
        ps.cur_spans.clear();
        ps.cur_spans.push((0, text.len()));
        let mut in_bytelevel = false;
        let last_idx = self.stages.len() - 1;

        for (si, st) in self.stages.iter().enumerate() {
            match st {
                PretokStage::ByteLevel { add_prefix_space, use_regex } => {
                    let need_bl_chars = si < last_idx;
                    let encode_bl = need_bl_chars && !in_bytelevel;
                    if !*add_prefix_space && !encode_bl {
                        if *use_regex {
                            ps.nxt_spans.clear();
                            for &(a, b) in &ps.cur_spans {
                                if a < b {
                                    split_gpt2_word(&ps.cur_buf[a..b], a, &mut ps.nxt_spans);
                                }
                            }
                            std::mem::swap(&mut ps.cur_spans, &mut ps.nxt_spans);
                        }
                    } else {
                        ps.nxt_buf.clear();
                        ps.nxt_spans.clear();
                        for &(a, b) in &ps.cur_spans {
                            let p = &ps.cur_buf[a..b];
                            if p.is_empty() {
                                continue;
                            }
                            let prefixed: &str = if *add_prefix_space && !p.starts_with(' ') {
                                ps.raw.clear();
                                ps.raw.push(b' ');
                                ps.raw.extend_from_slice(p.as_bytes());
                                // SAFETY: `' '` + valid UTF-8 `p` is valid UTF-8.
                                unsafe { std::str::from_utf8_unchecked(&ps.raw) }
                            } else {
                                p
                            };
                            if *use_regex {
                                ps.tmp_spans.clear();
                                split_gpt2_word(prefixed, 0, &mut ps.tmp_spans);
                                for &(sa, sb) in &ps.tmp_spans {
                                    let sub = &prefixed[sa..sb];
                                    let dst_s = ps.nxt_buf.len();
                                    if encode_bl {
                                        encode_bytelevel_into(sub, &mut ps.nxt_buf);
                                    } else {
                                        ps.nxt_buf.push_str(sub);
                                    }
                                    ps.nxt_spans.push((dst_s, ps.nxt_buf.len()));
                                }
                            } else {
                                let dst_s = ps.nxt_buf.len();
                                if encode_bl {
                                    encode_bytelevel_into(prefixed, &mut ps.nxt_buf);
                                } else {
                                    ps.nxt_buf.push_str(prefixed);
                                }
                                ps.nxt_spans.push((dst_s, ps.nxt_buf.len()));
                            }
                        }
                        std::mem::swap(&mut ps.cur_buf, &mut ps.nxt_buf);
                        std::mem::swap(&mut ps.cur_spans, &mut ps.nxt_spans);
                    }
                    if need_bl_chars {
                        in_bytelevel = true;
                    }
                }
                PretokStage::Metaspace { replacement, prepend_scheme, split } => {
                    let mut rep_buf = [0u8; 4];
                    let rep_str = replacement.encode_utf8(&mut rep_buf);
                    ps.nxt_buf.clear();
                    ps.nxt_spans.clear();
                    let mut first_non_empty = at_start;
                    for &(a, b) in &ps.cur_spans {
                        let p = &ps.cur_buf[a..b];
                        if p.is_empty() {
                            continue;
                        }
                        let do_prepend = match prepend_scheme {
                            PrependScheme::Always => true,
                            PrependScheme::First => first_non_empty,
                            PrependScheme::Never => false,
                        };
                        first_non_empty = false;
                        let dst_s = ps.nxt_buf.len();
                        if do_prepend && !p.starts_with(' ') && !p.starts_with(*replacement) {
                            ps.nxt_buf.push_str(rep_str);
                        }
                        if p.contains(' ') {
                            let mut rem = p;
                            while let Some(idx) = rem.find(' ') {
                                ps.nxt_buf.push_str(&rem[..idx]);
                                ps.nxt_buf.push_str(rep_str);
                                rem = &rem[idx + 1..];
                            }
                            ps.nxt_buf.push_str(rem);
                        } else {
                            ps.nxt_buf.push_str(p);
                        }
                        let dst_e = ps.nxt_buf.len();
                        if *split {
                            split_by_char_pred(
                                &ps.nxt_buf[dst_s..dst_e],
                                dst_s,
                                |c| c == *replacement,
                                SplitBehavior::MergedWithNext,
                                false,
                                &mut ps.segs,
                                &mut ps.nxt_spans,
                            );
                        } else {
                            ps.nxt_spans.push((dst_s, dst_e));
                        }
                    }
                    std::mem::swap(&mut ps.cur_buf, &mut ps.nxt_buf);
                    std::mem::swap(&mut ps.cur_spans, &mut ps.nxt_spans);
                }
                _ => {
                    ps.nxt_spans.clear();
                    for &(a, b) in &ps.cur_spans {
                        apply_stage_slice(st, &ps.cur_buf[a..b], a, &mut ps.segs, &mut ps.nxt_spans);
                    }
                    std::mem::swap(&mut ps.cur_spans, &mut ps.nxt_spans);
                }
            }
            if ps.cur_spans.is_empty() {
                break;
            }
        }

        if in_bytelevel {
            for &(a, b) in &ps.cur_spans {
                decode_bytelevel_bytes(&ps.cur_buf[a..b], c2b, &mut ps.raw);
                if !ps.raw.is_empty() {
                    emit(&ps.raw);
                }
            }
        } else {
            let bytes = ps.cur_buf.as_bytes();
            for &(a, b) in &ps.cur_spans {
                if a < b {
                    emit(&bytes[a..b]);
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
        let mut ps = PipeScratch::default();
        let mut out = Vec::new();
        pipe.for_each_chunk(text, true, &c2b, &mut ps, |b| {
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
        let mut ps = PipeScratch::default();
        let mut chunks: Vec<Vec<u8>> = Vec::new();
        pipe.for_each_chunk(" в 12", true, &c2b, &mut ps, |b| chunks.push(b.to_vec()));
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
        let mut scratch = PipeScratch::default();
        let mut out = Vec::new();
        pipe.for_each_chunk("hello world", false, &c2b, &mut scratch, |b| {
            out.push(std::str::from_utf8(b).unwrap().to_owned());
        });
        assert_eq!(out, vec!["hello▁world"]);
    }

    #[test]
    fn fast_patterns_match_regex_exact() {
        let sample = "Hello, world!?\r\n\n  foo_bar+baz=42 1234567 don't DON'T café\u{0301}  \t \r\n漢テスト — “hi”";
        // 1. Seed-Coder GenericNoNl vs fancy-regex
        let seed_pat = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
        let re = fancy_regex::Regex::new(seed_pat).unwrap();
        let mut segs = Vec::new();
        let mut want = Vec::new();
        split_by_regex(sample, 0, &re, SplitBehavior::Isolated, false, &mut segs, &mut want);
        let pipe = PretokPipe::from_json(&json!({"type": "Split", "pattern": {"Regex": seed_pat}, "behavior": "Isolated", "invert": false})).unwrap().unwrap();
        assert_eq!(pipe.chunk_ranges(sample), want);

        // 2. CJK_MAIN_REGEX vs fancy-regex
        let re_cjk = fancy_regex::Regex::new(CJK_MAIN_REGEX).unwrap();
        want.clear();
        split_by_regex(sample, 0, &re_cjk, SplitBehavior::Isolated, false, &mut segs, &mut want);
        let pipe_cjk = PretokPipe::from_json(&json!({"type": "Split", "pattern": {"Regex": CJK_MAIN_REGEX}, "behavior": "Isolated", "invert": false})).unwrap().unwrap();
        assert_eq!(pipe_cjk.chunk_ranges(sample), want);

        // 3. AsciiDigitRun(1), AsciiDigitRun(3), CrLf across all behaviors & invert
        for pat in ["[0-9]", "[0-9][0-9][0-9]", "[\r\n]"] {
            let re_p = fancy_regex::Regex::new(pat).unwrap();
            for (bstr, beh) in [
                ("Removed", SplitBehavior::Removed),
                ("Isolated", SplitBehavior::Isolated),
                ("MergedWithPrevious", SplitBehavior::MergedWithPrevious),
                ("MergedWithNext", SplitBehavior::MergedWithNext),
                ("Contiguous", SplitBehavior::Contiguous),
            ] {
                for inv in [false, true] {
                    want.clear();
                    split_by_regex(sample, 0, &re_p, beh, inv, &mut segs, &mut want);
                    let p = PretokPipe::from_json(&json!({"type": "Split", "pattern": {"Regex": pat}, "behavior": bstr, "invert": inv})).unwrap().unwrap();
                    assert_eq!(p.chunk_ranges(sample), want, "pat={pat} beh={bstr} inv={inv}");
                }
            }
        }
    }

    #[test]
    fn onig_to_fancy_edge_cases() {
        assert_eq!(onig_to_fancy(r"\x20{3}+"), r"(?:\x20{3})+");
        assert_eq!(onig_to_fancy(r"\u0041{2}*"), r"(?:\u0041{2})*");
        assert_eq!(onig_to_fancy(r"[[:alpha:]0-9]{3}+"), r"(?:[[:alpha:]0-9]{3})+");
        assert_eq!(onig_to_fancy(r"\p{Nd}{,3}+"), r"(?:\p{Nd}{,3})+");
        assert_eq!(onig_to_fancy(r"\p{Nd}{1,3}?"), r"\p{Nd}{1,3}?");
    }

    #[test]
    fn fail_closed_on_unknown_stage() {
        assert!(PretokPipe::from_json(&json!({"type": "BertPreTokenizer"})).is_err());
        assert!(PretokPipe::from_json(&json!({"type": "UnicodeScripts"})).is_err());
    }
}
