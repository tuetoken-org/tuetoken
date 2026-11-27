//! Composable HuggingFace normalizer pipeline + fast ASCII-skipping Unicode
//! normalization (NFC/NFD/NFKC/NFKD).
//!
//! Key fact for segmented Unicode normalization: splitting a string immediately
//! BEFORE any ASCII character is safe for all four forms — `N(a + b) == N(a) + N(b)`
//! whenever `b` starts with ASCII:
//!   * every ASCII char is a starter (ccc = 0), so canonical reordering never
//!     moves a mark across it;
//!   * no canonical composition has an ASCII char as its SECOND element, so an
//!     ASCII char never composes with what precedes it;
//!   * it is NF*_QC = Yes, i.e. unchanged by (de)composition itself.
//!
//! (An ASCII char CAN be the first element of a composition — `e` + U+0301 ->
//! `é` — so each non-ASCII run is normalized together with the one ASCII char
//! in front of it.)
//!
//! So: skip ASCII 8 bytes at a time, quick-check each non-ASCII run (with its
//! leading ASCII char), and only run the real normalizer on the runs that need
//! it. Already-normalized input returns `Cow::Borrowed` with no allocation; a
//! 100 MB document with one decomposed accent no longer re-normalizes 100 MB.

use serde_json::Value;
use std::borrow::Cow;
use unicode_normalization::{
    is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick, IsNormalized, UnicodeNormalization,
};

/// form: 1 NFC, 2 NFD, 3 NFKC, 4 NFKD; anything else = identity.
fn quick(form: u8, s: &str) -> bool {
    let r = match form {
        1 => is_nfc_quick(s.chars()),
        2 => is_nfd_quick(s.chars()),
        3 => is_nfkc_quick(s.chars()),
        4 => is_nfkd_quick(s.chars()),
        _ => IsNormalized::Yes,
    };
    r == IsNormalized::Yes
}

fn full(form: u8, s: &str, out: &mut String) {
    match form {
        1 => out.extend(s.nfc()),
        2 => out.extend(s.nfd()),
        3 => out.extend(s.nfkc()),
        4 => out.extend(s.nfkd()),
        _ => out.push_str(s),
    }
}

/// Next maximal non-ASCII run at or after `i`: Some((run_start, run_end)).
#[inline]
fn next_run(b: &[u8], mut i: usize) -> Option<(usize, usize)> {
    let n = b.len();
    while i + 8 <= n
        && u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) & 0x8080_8080_8080_8080 == 0
    {
        i += 8;
    }
    while i < n && b[i] < 0x80 {
        i += 1;
    }
    if i >= n {
        return None;
    }
    let st = i;
    while i < n && b[i] >= 0x80 {
        i += 1;
    }
    // Bytes >= 0x80 are exactly lead+continuation bytes of non-ASCII chars, so a
    // maximal run of them starts and ends on char boundaries in valid UTF-8.
    Some((st, i))
}

/// Segment for run [st, en): include the ASCII char before it (if any).
#[inline]
fn seg_start(st: usize) -> usize {
    st.saturating_sub(1)
}

pub fn normalize(form: u8, text: &str) -> Cow<'_, str> {
    if !(1..=4).contains(&form) {
        return Cow::Borrowed(text);
    }
    let b = text.as_bytes();
    // Pass 1: find the first run that is not already normalized.
    let mut i = 0usize;
    let first_bad = loop {
        match next_run(b, i) {
            None => return Cow::Borrowed(text),
            Some((st, en)) => {
                if !quick(form, &text[seg_start(st)..en]) {
                    break (st, en);
                }
                i = en;
            }
        }
    };
    // Pass 2: copy verbatim up to the first bad segment, then segment by segment.
    let mut out = String::with_capacity(text.len() + 16);
    let (st, en) = first_bad;
    let ss = seg_start(st);
    out.push_str(&text[..ss]);
    full(form, &text[ss..en], &mut out);
    let mut copied = en; // text[..copied] is accounted for
    let mut i = en;
    while let Some((st, en)) = next_run(b, i) {
        let ss = seg_start(st);
        if quick(form, &text[ss..en]) {
            out.push_str(&text[copied..en]);
        } else {
            out.push_str(&text[copied..ss]);
            full(form, &text[ss..en], &mut out);
        }
        copied = en;
        i = en;
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
}

/// A single composable HuggingFace normalizer stage.
#[derive(Clone, Debug)]
pub enum NormStage {
    Unicode(u8), // 1 NFC, 2 NFD, 3 NFKC, 4 NFKD
    Lowercase,
    StripAccents,
    Strip { left: bool, right: bool },
    Prepend(String),
    ReplaceStr { from: String, to: String },
    ReplaceRe { re: fancy_regex::Regex, to: String },
}

/// Ordered pipeline of HuggingFace normalizer stages.
///
/// Empty pipelines (`is_identity()`) and single-stage Unicode normalization
/// (`fast_unicode != 0`) take zero-overhead fast paths so standard LLMs pay no
/// extra indirection.
#[derive(Clone, Debug, Default)]
pub struct Normalizer {
    stages: Vec<NormStage>,
    fast_unicode: u8,
    ascii_identity: bool,
}

fn parse_stages(val: &Value, out: &mut Vec<NormStage>) -> Result<(), String> {
    if val.is_null() {
        return Ok(());
    }
    let t = val
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "<unknown>".to_string())?;
    match t {
        "Sequence" => {
            if let Some(arr) = val.get("normalizers").and_then(|v| v.as_array()) {
                for item in arr {
                    parse_stages(item, out)?;
                }
            }
            Ok(())
        }
        "NFC" => {
            out.push(NormStage::Unicode(1));
            Ok(())
        }
        "NFD" => {
            out.push(NormStage::Unicode(2));
            Ok(())
        }
        "NFKC" => {
            out.push(NormStage::Unicode(3));
            Ok(())
        }
        "NFKD" => {
            out.push(NormStage::Unicode(4));
            Ok(())
        }
        "Lowercase" => {
            out.push(NormStage::Lowercase);
            Ok(())
        }
        "StripAccents" => {
            out.push(NormStage::StripAccents);
            Ok(())
        }
        "Strip" => {
            let left = val.get("strip_left").and_then(|v| v.as_bool()).unwrap_or(true);
            let right = val.get("strip_right").and_then(|v| v.as_bool()).unwrap_or(true);
            if left || right {
                out.push(NormStage::Strip { left, right });
            }
            Ok(())
        }
        "Prepend" => {
            let prepend = val
                .get("prepend")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Prepend(missing prepend)".to_string())?;
            if !prepend.is_empty() {
                out.push(NormStage::Prepend(prepend.to_string()));
            }
            Ok(())
        }
        "Replace" => {
            let to = val
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "Replace(missing content)".to_string())?
                .to_string();
            let pat = val
                .get("pattern")
                .ok_or_else(|| "Replace(missing pattern)".to_string())?;
            if let Some(s) = pat.get("String").and_then(|v| v.as_str()) {
                if !s.is_empty() {
                    out.push(NormStage::ReplaceStr { from: s.to_string(), to });
                }
                Ok(())
            } else if let Some(r) = pat.get("Regex").and_then(|v| v.as_str()) {
                let re = fancy_regex::Regex::new(r)
                    .map_err(|e| format!("Replace(Regex {r:?}: {e})"))?;
                out.push(NormStage::ReplaceRe { re, to });
                Ok(())
            } else {
                Err("Replace(unknown pattern)".to_string())
            }
        }
        other => Err(other.to_string()),
    }
}

impl Normalizer {
    pub fn from_json(val: &Value) -> Result<Self, String> {
        let mut stages = Vec::new();
        parse_stages(val, &mut stages)?;
        let fast_unicode = if stages.len() == 1 {
            match stages[0] {
                NormStage::Unicode(code) => code,
                _ => 0,
            }
        } else {
            0
        };
        let ascii_identity = stages
            .iter()
            .all(|s| matches!(s, NormStage::Unicode(_) | NormStage::StripAccents));
        Ok(Normalizer { stages, fast_unicode, ascii_identity })
    }

    /// True when the normalizer is a no-op (null or empty Sequence).
    #[inline(always)]
    pub fn is_identity(&self) -> bool {
        self.stages.is_empty()
    }

    /// True when the normalizer is guaranteed to leave every pure-ASCII string
    /// byte-for-byte unchanged (empty pipeline, NFC/NFD/NFKC/NFKD, StripAccents).
    #[inline(always)]
    pub fn is_ascii_identity(&self) -> bool {
        self.ascii_identity
    }

    /// True if any stage replaces `' '` with `ch` (e.g. SentencePiece `'▁'`).
    pub fn replaces_space_with(&self, ch: char) -> bool {
        let mut buf = [0u8; 4];
        let target = ch.encode_utf8(&mut buf);
        self.stages.iter().any(|s| match s {
            NormStage::ReplaceStr { from, to } => from == " " && to == target,
            _ => false,
        })
    }

    /// Detect exact SentencePiece metaspace normalizer pipelines:
    /// - `1`: `[Prepend("▁"), Replace(" ", "▁")]` (Phi-3.5, Llama-2, SOLAR, Zamba2)
    /// - `2`: `[Replace(" ", "▁")]` (Yi-1.5, Gemma-2/3, ERNIE-4.5)
    /// - `0`: anything else.
    pub fn sp_metaspace_kind(&self) -> u8 {
        match self.stages.as_slice() {
            [NormStage::Prepend(p), NormStage::ReplaceStr { from, to }]
                if p == "\u{2581}" && from == " " && to == "\u{2581}" =>
            {
                1
            }
            [NormStage::ReplaceStr { from, to }] if from == " " && to == "\u{2581}" => 2,
            _ => 0,
        }
    }

    /// Apply every stage in order, staying zero-copy (`Cow::Borrowed`) whenever
    /// no stage alters the input text.
    #[inline]
    pub fn apply<'a>(&self, text: &'a str) -> Cow<'a, str> {
        if self.stages.is_empty() {
            return Cow::Borrowed(text);
        }
        if self.fast_unicode != 0 {
            return normalize(self.fast_unicode, text);
        }
        let mut cur: Cow<'a, str> = Cow::Borrowed(text);
        for st in &self.stages {
            match st {
                NormStage::Unicode(code) => match normalize(*code, &cur) {
                    Cow::Borrowed(_) => {}
                    Cow::Owned(s) => cur = Cow::Owned(s),
                },
                NormStage::Lowercase => {
                    // Fast ASCII check: avoid allocating if already all lowercase ASCII.
                    if cur.bytes().all(|b| b < 0x80 && !b.is_ascii_uppercase()) {
                        continue;
                    }
                    cur = Cow::Owned(cur.to_lowercase());
                }
                NormStage::StripAccents => {
                    if cur.chars().any(|c| crate::unicode::is_mark_nonspacing(c as u32)) {
                        let s: String = cur
                            .chars()
                            .filter(|&c| !crate::unicode::is_mark_nonspacing(c as u32))
                            .collect();
                        cur = Cow::Owned(s);
                    }
                }
                NormStage::Strip { left, right } => {
                    let trimmed = match (*left, *right) {
                        (true, true) => cur.trim(),
                        (true, false) => cur.trim_start(),
                        (false, true) => cur.trim_end(),
                        (false, false) => cur.as_ref(),
                    };
                    if trimmed.len() != cur.len() {
                        cur = match cur {
                            Cow::Borrowed(orig) => {
                                let start = trimmed.as_ptr() as usize - orig.as_ptr() as usize;
                                Cow::Borrowed(&orig[start..start + trimmed.len()])
                            }
                            Cow::Owned(_) => Cow::Owned(trimmed.to_owned()),
                        };
                    }
                }
                NormStage::Prepend(p) => {
                    if !cur.is_empty() {
                        let mut s = String::with_capacity(p.len() + cur.len());
                        s.push_str(p);
                        s.push_str(&cur);
                        cur = Cow::Owned(s);
                    }
                }
                NormStage::ReplaceStr { from, to } => {
                    if cur.contains(from.as_str()) {
                        cur = Cow::Owned(cur.replace(from.as_str(), to.as_str()));
                    }
                }
                NormStage::ReplaceRe { re, to } => {
                    // HF Replace treats `content` literally (no $1 expansion).
                    match re.replace_all(&cur, fancy_regex::NoExpand(to.as_str())) {
                        Cow::Borrowed(_) => {}
                        Cow::Owned(s) => cur = Cow::Owned(s),
                    }
                }
            }
        }
        cur
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// Segmented normalization == whole-string normalization, all four forms, on
    /// random strings dense in combining marks, Hangul jamo, compatibility chars
    /// and ASCII starters that compose with following marks.
    #[test]
    fn segmented_equals_whole() {
        let pool: Vec<char> = "aeiouAEOnNcCsSkK 1.\n\u{0301}\u{0300}\u{0308}\u{0323}\u{0327}\u{030A}\u{0345}\u{05B0}\u{0F71}\u{1100}\u{1161}\u{11A8}\u{AC00}\u{FB01}\u{2163}\u{FF21}\u{00C5}\u{212B}\u{1E0A}\u{0344}\u{0F73}\u{3099}\u{304B}\u{00E9}é漢😀"
            .chars()
            .collect();
        let mut seed = 42u64;
        for _ in 0..20000 {
            let len = (lcg(&mut seed) % 24) as usize;
            let s: String = (0..len).map(|_| pool[(lcg(&mut seed) as usize) % pool.len()]).collect();
            for form in 1..=4u8 {
                let mut want = String::new();
                full(form, &s, &mut want);
                assert_eq!(normalize(form, &s).as_ref(), want, "form={form} s={s:?}");
            }
        }
    }

    #[test]
    fn composable_normalizer_stages() {
        // CLIP-style pipeline: NFC -> Replace(\s+ -> " ") -> Lowercase
        let n = Normalizer::from_json(&json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "NFC"},
                {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
                {"type": "Lowercase"}
            ]
        }))
        .unwrap();
        assert_eq!(n.apply("Hello   \t\n WORLD  Cafe\u{0301}"), "hello world café");

        // Prepend on empty stays empty (HF semantics)
        let sp = Normalizer::from_json(&json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "Prepend", "prepend": "\u{2581}"},
                {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"}
            ]
        }))
        .unwrap();
        assert_eq!(sp.apply(""), "");
        assert_eq!(sp.apply("hello world"), "\u{2581}hello\u{2581}world");
        assert_eq!(sp.apply(" hello"), "\u{2581}\u{2581}hello");

        // Strip + NFD + StripAccents
        let sa = Normalizer::from_json(&json!({
            "type": "Sequence",
            "normalizers": [
                {"type": "Strip", "strip_left": true, "strip_right": true},
                {"type": "NFD"},
                {"type": "StripAccents"}
            ]
        }))
        .unwrap();
        assert_eq!(sa.apply("  café  "), "cafe");

        // Fail closed on unknown normalizer types
        assert!(Normalizer::from_json(&json!({"type": "BertNormalizer"})).is_err());
        assert!(Normalizer::from_json(&json!({"type": "Precompiled"})).is_err());
    }
}
