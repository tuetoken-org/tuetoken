//! Pre-tokenizer state machines (the correctness-validated BPE pre-tokenizers).
//! Each `*_next_chunk_end` returns the byte index where the next pre-token ends,
//! given a start position.

use crate::unicode::{
    classify, classify_at, classify_fine_at, decode, is_letter, is_mark, is_number,
    is_unicode_whitespace,
};

/// A ByteLevel pre-tokenizer grammar, detected from the model's config (never
/// from its name). The three shapes differ only in how letter runs, contractions,
/// and digit runs are split.
#[derive(Clone, Copy)]
pub enum Pretok {
    // Plain word grammar: case-sensitive contractions, `\p{L}+` letter runs (no
    // case split), unlimited digits. `true` = a leading `Digits{individual_digits}`
    // stage isolates every digit into its own pre-token (common in code models).
    Word(bool),
    // Like `Word` but case-insensitive contractions and a digit-run cap
    // (`max_digits`; 0 = unlimited). `marks`: when true the grammar uses
    // mark-inclusive letter runs (`[\p{L}\p{M}]+`) and excludes Marks from the
    // symbol class (`[^\s\p{L}\p{M}\p{N}]`) — combining marks join the preceding
    // letters instead of being lumped with symbols (e.g. the Qwen3.6 variant).
    Generic { max_digits: u32, marks: bool },
    // Case-aware (camelCase) letter runs. `contractions`: whether the `(?i:'s|'t|..)`
    // suffix is emitted (some variants drop it). `max_digits`: digit-run cap.
    CamelCase { contractions: bool, max_digits: u32 },
}

#[inline]
fn is_upper_like(c: u8) -> bool {
    c == b'U' || c == b'L' || c == b'm'
}
#[inline]
fn is_lower_like(c: u8) -> bool {
    c == b'l' || c == b'L' || c == b'm'
}
#[inline]
fn is_any_letter(c: u8) -> bool {
    c == b'U' || c == b'l' || c == b'L' || c == b'm'
}

/// `'(?:[sdmt]|ll|ve|re)` (ci=false) or `(?i:'s|'t|'re|'ve|'m|'ll|'d)` (ci=true).
#[inline]
fn contraction(b: &[u8], start: usize, ci: bool) -> Option<usize> {
    let n = b.len();
    if start >= n || b[start] != b'\'' || start + 1 >= n {
        return None;
    }
    let lower = |x: u8| if ci && x.is_ascii_uppercase() { x + 32 } else { x };
    let a = lower(b[start + 1]);
    if a == b's' || a == b't' || a == b'm' || a == b'd' {
        return Some(start + 2);
    }
    if start + 2 < n {
        let c2 = lower(b[start + 2]);
        if (a == b'l' && c2 == b'l') || (a == b'v' && c2 == b'e') || (a == b'r' && c2 == b'e') {
            return Some(start + 3);
        }
    }
    None
}

/// Is `b[pos]` a symbol-class char? Coarse class 'O' excludes whitespace/letter/
/// number and *includes* Marks. With `marks=true` (the `[^\s\p{L}\p{M}\p{N}]`
/// grammar) Marks are additionally excluded, so a combining mark ends the run.
#[inline]
fn is_symbol_at(b: &[u8], pos: usize, marks: bool) -> (bool, usize) {
    let (cls, l) = classify_at(b, pos);
    if cls != b'O' {
        return (false, l);
    }
    if marks {
        // Only class-'O' (non-ASCII) chars can be Marks; re-decode to test.
        let (cp, _) = decode(b, pos);
        if is_mark(cp) {
            return (false, l);
        }
    }
    (true, l)
}

/// ` ?[^\s\p{L}\p{N}]+` (or `[^\s\p{L}\p{M}\p{N}]+` when `marks`) + trailing
/// newlines. `trail`: 0 none, 1 `[\r\n]*`, 2 `[\r\n/]*`. Returns None on no match.
#[inline]
fn symbol_run(b: &[u8], start: usize, trail: u8, marks: bool) -> Option<usize> {
    let n = b.len();
    let mut pos = start;
    if b[start] == b' ' && start + 1 < n {
        pos += 1;
    }
    if pos >= n {
        return None;
    }
    let (ok, cl) = is_symbol_at(b, pos, marks);
    if !ok {
        return None;
    }
    pos += cl;
    while pos < n {
        let (ok, l) = is_symbol_at(b, pos, marks);
        if !ok {
            break;
        }
        pos += l;
    }
    if trail > 0 {
        while pos < n
            && (b[pos] == b'\r' || b[pos] == b'\n' || (trail == 2 && b[pos] == b'/'))
        {
            pos += 1;
        }
    }
    Some(pos)
}

/// `\s*[\r\n]+` greedy to the last newline in the whitespace run.
#[inline]
fn ws_nl(b: &[u8], start: usize) -> Option<usize> {
    let n = b.len();
    let mut p = start;
    let mut last_nl_end = 0usize;
    let mut found = false;
    while p < n {
        let (cp, cl) = decode(b, p);
        if !is_unicode_whitespace(cp) {
            break;
        }
        if cp == '\r' as u32 || cp == '\n' as u32 {
            last_nl_end = p + cl;
            found = true;
        }
        p += cl;
    }
    if found {
        Some(last_nl_end)
    } else {
        None
    }
}

/// `\s+(?!\S)|\s+` — back off the last codepoint if >1 and not at end-of-input.
#[inline]
fn ws_trail(b: &[u8], start: usize, indiv_digits: bool) -> usize {
    let n = b.len();
    let mut p = start;
    let mut prev = start;
    let mut count = 0u32;
    while p < n {
        let (cp, cl) = decode(b, p);
        if !is_unicode_whitespace(cp) {
            break;
        }
        prev = p;
        p += cl;
        count += 1;
    }
    if p < n && count > 1 {
        // `\s+(?!\S)`: normally leave the last ws char for the following ` ?X`.
        // With individual_digits the Digits pre-split makes a following digit a
        // hard piece boundary, so the whole whitespace run is consumed instead.
        if indiv_digits {
            let (cp, _) = decode(b, p);
            if is_number(cp) {
                return p;
            }
        }
        prev
    } else {
        p
    }
}

#[inline]
fn letter_run(b: &[u8], mut pos: usize, marks: bool) -> usize {
    let n = b.len();
    while pos < n {
        let (x, l) = decode(b, pos);
        if !(is_letter(x) || (marks && is_mark(x))) {
            break;
        }
        pos += l;
    }
    pos
}

fn word(b: &[u8], start: usize, indiv_digits: bool) -> usize {
    let n = b.len();
    if let Some(e) = contraction(b, start, false) {
        return e;
    }
    // ` ?\p{L}+`
    {
        let mut pos = start;
        if b[start] == b' ' && start + 1 < n {
            pos += 1;
        }
        if pos < n {
            let (cp, cl) = decode(b, pos);
            if is_letter(cp) {
                return letter_run(b, pos + cl, false);
            }
        }
    }
    // `\p{N}` (individual_digits: single digit, no leading space) or ` ?\p{N}+`.
    if indiv_digits {
        let (cp, cl) = decode(b, start);
        if is_number(cp) {
            return start + cl;
        }
    } else {
        let mut pos = start;
        if b[start] == b' ' && start + 1 < n {
            pos += 1;
        }
        if pos < n {
            let (cp, cl) = decode(b, pos);
            if is_number(cp) {
                pos += cl;
                while pos < n {
                    let (x, l) = decode(b, pos);
                    if !is_number(x) {
                        break;
                    }
                    pos += l;
                }
                return pos;
            }
        }
    }
    // ` ?[^\s\p{L}\p{N}]+`  (no trailing newlines for word)
    if let Some(e) = symbol_run(b, start, 0, false) {
        return e;
    }
    // `\s+(?!\S)|\s+`
    let (cp0, _) = decode(b, start);
    if is_unicode_whitespace(cp0) {
        return ws_trail(b, start, indiv_digits);
    }
    let (_, cl) = decode(b, start);
    start + cl
}

fn generic(b: &[u8], start: usize, max_digits: u32, marks: bool) -> usize {
    let n = b.len();
    if let Some(e) = contraction(b, start, true) {
        return e;
    }
    // `[^\r\n\p{L}\p{N}]?\p{L}+`  (or `[\p{L}\p{M}]+` when `marks`). A leading
    // Mark is class 'O', so it is taken as the optional prefix and then, being a
    // valid run char, the run continues — yielding the same boundary as the regex.
    {
        let (cp, cl) = decode(b, start);
        let cls = classify(cp);
        let mut prefix_end = start;
        if (cls == b'O' || cls == b'S') && cp != '\r' as u32 && cp != '\n' as u32 {
            prefix_end = start + cl;
        }
        if prefix_end < n {
            let (cp2, cl2) = decode(b, prefix_end);
            if is_letter(cp2) || (marks && is_mark(cp2)) {
                return letter_run(b, prefix_end + cl2, marks);
            }
        }
    }
    // `\p{N}{1,max}`
    {
        let (cp, cl) = decode(b, start);
        if is_number(cp) {
            let mut pos = start + cl;
            let mut count = 1u32;
            while pos < n && (max_digits == 0 || count < max_digits) {
                let (x, l) = decode(b, pos);
                if !is_number(x) {
                    break;
                }
                pos += l;
                count += 1;
            }
            return pos;
        }
    }
    // ` ?[^\s\p{L}\p{N}]+[\r\n]*`  (excludes Marks when `marks`)
    if let Some(e) = symbol_run(b, start, 1, marks) {
        return e;
    }
    // `\s*[\r\n]+`
    if let Some(e) = ws_nl(b, start) {
        return e;
    }
    // `\s+(?!\S)|\s+`
    let (cp0, _) = decode(b, start);
    if is_unicode_whitespace(cp0) {
        return ws_trail(b, start, false);
    }
    let (_, cl) = decode(b, start);
    start + cl
}

#[inline]
fn camelcase_contraction(b: &[u8], pos: usize) -> usize {
    contraction(b, pos, true).unwrap_or(pos)
}

fn camelcase(b: &[u8], start: usize, contractions: bool, max_digits: u32) -> usize {
    let n = b.len();
    let contr = |b: &[u8], pos: usize| if contractions { camelcase_contraction(b, pos) } else { pos };
    let c = b[start];
    // Pattern 1+2: [^\r\n\p{L}\p{N}]? (Upper-like* Lower-like* | Upper-like+ ...) contraction?
    {
        let (cls, cl) = classify_fine_at(b, start);
        let mut prefix_end = start;
        if c != b'\r' && c != b'\n' && !is_any_letter(cls) && cls != b'N' {
            prefix_end = start + cl;
        }
        if prefix_end < n {
            let (cls2, cl2) = classify_fine_at(b, prefix_end);
            if is_upper_like(cls2) {
                let mut pos = prefix_end + cl2;
                while pos < n {
                    let (k, l) = classify_fine_at(b, pos);
                    if !is_upper_like(k) {
                        break;
                    }
                    pos += l;
                }
                while pos < n {
                    let (k, l) = classify_fine_at(b, pos);
                    if !is_lower_like(k) {
                        break;
                    }
                    pos += l;
                }
                return contr(b, pos);
            } else if is_lower_like(cls2) {
                let mut pos = prefix_end + cl2;
                while pos < n {
                    let (k, l) = classify_fine_at(b, pos);
                    if !is_lower_like(k) {
                        break;
                    }
                    pos += l;
                }
                return contr(b, pos);
            }
        }
    }
    // Pattern 3: \p{N}{1,max_digits}
    {
        let (cp, cl) = decode(b, start);
        if is_number(cp) {
            let mut pos = start + cl;
            let mut count = 1u32;
            while pos < n && (max_digits == 0 || count < max_digits) {
                let (x, l) = decode(b, pos);
                if !is_number(x) {
                    break;
                }
                pos += l;
                count += 1;
            }
            return pos;
        }
    }
    // Pattern 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
    if let Some(e) = symbol_run(b, start, 2, false) {
        return e;
    }
    // Patterns 5/6: whitespace
    let (cp0, _) = decode(b, start);
    if is_unicode_whitespace(cp0) {
        if let Some(e) = ws_nl(b, start) {
            return e;
        }
        return ws_trail(b, start, false);
    }
    let (_, cl) = decode(b, start);
    start + cl
}

#[inline]
pub fn next_chunk_end(kind: Pretok, b: &[u8], start: usize) -> usize {
    match kind {
        Pretok::Word(indiv) => word(b, start, indiv),
        Pretok::Generic { max_digits, marks } => generic(b, start, max_digits, marks),
        Pretok::CamelCase { contractions, max_digits } => camelcase(b, start, contractions, max_digits),
    }
}
