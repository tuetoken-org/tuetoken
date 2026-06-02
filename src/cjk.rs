//! Hand-written fast pre-tokenizer for the DeepSeek-V3/R1/V4 grammar — a 3-stage
//! Isolated Split sequence that `detect_pretok` declines (multi-Split + a
//! mark-inclusive letter class), so it otherwise falls to the slow fancy-regex
//! path. We reproduce the SAME composition the regex fallback does (each Split is
//! Isolated: matches become pieces, gaps are kept), one byte scan per stage:
//!
//!   1. `\p{N}{1,3}`                                   digit groups of up to 3
//!   2. `[一-龥぀-ゟ゠-ヿ]+`   CJK/kana run isolation
//!   3. main split (tried in alternation order A..F):
//!      A `[ascii-punct][A-Za-z]+`                       (snake_case-style)
//!      B `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`         optional prefix + letter/mark run
//!      C ` ?[\p{P}\p{S}]+[\r\n]*`                       optional space + punct/symbol run
//!      D `\s*[\r\n]+`   E `\s+(?!\S)`   F `\s+`         whitespace (same as the
//!                                                       generic/camelcase machines)
//!
//! Each stage's regex is run by HF per-substring (on the piece), so the `(?!\S)`
//! lookahead is bounded by the piece — which is why the whitespace helpers stop at
//! `end` rather than the whole input.

use crate::unicode::{
    decode, is_letter, is_mark, is_number, is_punct_or_symbol, is_unicode_whitespace,
};

#[derive(Clone, Copy)]
enum Stage {
    Digits13,
    Cjk,
    Main,
}

/// `[一-龥぀-ゟ゠-ヿ]` — CJK unified ideographs + hiragana
/// + katakana (the exact ranges in the DeepSeek stage-2 Split).
#[inline]
fn is_cjk(cp: u32) -> bool {
    matches!(cp, 0x4E00..=0x9FA5 | 0x3040..=0x309F | 0x30A0..=0x30FF)
}

/// `[^\r\n\p{L}\p{P}\p{S}]` — the optional one-char prefix of alternative B.
#[inline]
fn is_main_prefix(cp: u32) -> bool {
    cp != '\r' as u32 && cp != '\n' as u32 && !is_letter(cp) && !is_punct_or_symbol(cp)
}

/// `\s*[\r\n]+` — whitespace run that contains a newline, ending at the last
/// newline (greedy `\s*` gives back to the trailing `[\r\n]+`). None if no newline.
#[inline]
fn ws_nl(b: &[u8], start: usize, end: usize) -> Option<usize> {
    let mut p = start;
    let mut last_nl = 0usize;
    let mut found = false;
    while p < end {
        let (cp, l) = decode(b, p);
        if !is_unicode_whitespace(cp) {
            break;
        }
        if cp == '\r' as u32 || cp == '\n' as u32 {
            last_nl = p + l;
            found = true;
        }
        p += l;
    }
    if found {
        Some(last_nl)
    } else {
        None
    }
}

/// `\s+(?!\S)|\s+` — consume the whitespace run; if it has >1 codepoint and does
/// not reach the piece end, leave the last one for the following ` ?X` token.
#[inline]
fn ws_trail(b: &[u8], start: usize, end: usize) -> usize {
    let mut p = start;
    let mut prev = start;
    let mut count = 0u32;
    while p < end {
        let (cp, l) = decode(b, p);
        if !is_unicode_whitespace(cp) {
            break;
        }
        prev = p;
        p += l;
        count += 1;
    }
    if p < end && count > 1 {
        prev
    } else {
        p
    }
}

/// Match alternative A..F of the stage-3 pattern starting at `i`. Returns the byte
/// index where the match ends, or None if `i` is an unmatched gap char.
#[inline]
fn main_match_at(b: &[u8], i: usize, end: usize) -> Option<usize> {
    let (c0, l0) = decode(b, i);
    // A: [ascii-punct][A-Za-z]+
    if c0 < 0x80 && (c0 as u8).is_ascii_punctuation() {
        let p = i + 1;
        if p < end && b[p].is_ascii_alphabetic() {
            let mut q = p;
            while q < end && b[q].is_ascii_alphabetic() {
                q += 1;
            }
            return Some(q);
        }
    }
    // B: [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    //   - c0 itself a letter/mark => run starts at i (prefix empty; a leading mark
    //     is equivalently "first of run" or "prefix", same end).
    //   - else c0 a valid prefix and next is a letter/mark => prefix + run.
    if is_letter(c0) || is_mark(c0) {
        let mut q = i + l0;
        while q < end {
            let (c, l) = decode(b, q);
            if !(is_letter(c) || is_mark(c)) {
                break;
            }
            q += l;
        }
        return Some(q);
    }
    if is_main_prefix(c0) {
        let p = i + l0;
        if p < end {
            let (c1, l1) = decode(b, p);
            if is_letter(c1) || is_mark(c1) {
                let mut q = p + l1;
                while q < end {
                    let (c, l) = decode(b, q);
                    if !(is_letter(c) || is_mark(c)) {
                        break;
                    }
                    q += l;
                }
                return Some(q);
            }
        }
    }
    // C: ` ?`[\p{P}\p{S}]+[\r\n]*   (the optional prefix is a literal space, 0x20)
    {
        let p = if c0 == ' ' as u32 { i + 1 } else { i };
        if p < end {
            let (c1, l1) = decode(b, p);
            if is_punct_or_symbol(c1) {
                let mut q = p + l1;
                while q < end {
                    let (c, l) = decode(b, q);
                    if !is_punct_or_symbol(c) {
                        break;
                    }
                    q += l;
                }
                while q < end && (b[q] == b'\r' || b[q] == b'\n') {
                    q += 1;
                }
                return Some(q);
            }
        }
    }
    // D/E/F: whitespace
    if is_unicode_whitespace(c0) {
        if let Some(e) = ws_nl(b, i, end) {
            return Some(e);
        }
        return Some(ws_trail(b, i, end));
    }
    None
}

/// Next match `(start, end)` of `stage` at or after `from`, within `[from, lim)`.
#[inline]
fn find_next(stage: Stage, b: &[u8], from: usize, lim: usize) -> Option<(usize, usize)> {
    let mut p = from;
    while p < lim {
        match stage {
            Stage::Digits13 => {
                let (c, l) = decode(b, p);
                if is_number(c) {
                    let start = p;
                    let (mut q, mut cnt) = (p, 0u32);
                    while q < lim && cnt < 3 {
                        let (c2, l2) = decode(b, q);
                        if !is_number(c2) {
                            break;
                        }
                        q += l2;
                        cnt += 1;
                    }
                    return Some((start, q));
                }
                p += l;
            }
            Stage::Cjk => {
                let (c, l) = decode(b, p);
                if is_cjk(c) {
                    let start = p;
                    let mut q = p;
                    while q < lim {
                        let (c2, l2) = decode(b, q);
                        if !is_cjk(c2) {
                            break;
                        }
                        q += l2;
                    }
                    return Some((start, q));
                }
                p += l;
            }
            Stage::Main => {
                if let Some(e) = main_match_at(b, p, lim) {
                    return Some((p, e));
                }
                let (_, l) = decode(b, p);
                p += l;
            }
        }
    }
    None
}

/// Apply one Isolated Split stage to every piece in `cur`, writing the resulting
/// pieces (matches + kept gaps) into `nxt`.
fn apply_stage(stage: Stage, b: &[u8], cur: &[(usize, usize)], nxt: &mut Vec<(usize, usize)>) {
    nxt.clear();
    for &(ps, pe) in cur {
        let mut last = ps;
        let mut from = ps;
        while let Some((ms, me)) = find_next(stage, b, from, pe) {
            if ms > last {
                nxt.push((last, ms));
            }
            nxt.push((ms, me));
            last = me;
            from = me;
        }
        if last < pe {
            nxt.push((last, pe));
        }
    }
}

/// Compose all three stages, returning the final pre-token byte ranges. `cur`/`nxt`
/// are reused scratch buffers (no allocation per call beyond growth).
pub fn chunk_ranges(text: &str, cur: &mut Vec<(usize, usize)>, nxt: &mut Vec<(usize, usize)>) {
    let b = text.as_bytes();
    cur.clear();
    if b.is_empty() {
        return;
    }
    cur.push((0, b.len()));
    for stage in [Stage::Digits13, Stage::Cjk, Stage::Main] {
        apply_stage(stage, b, cur, nxt);
        std::mem::swap(cur, nxt);
    }
}
