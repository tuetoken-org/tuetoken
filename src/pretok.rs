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
#[derive(Clone, Copy, Debug)]
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

// ---------------------------------------------------------------------------
// ASCII fast path.
//
// Each `fast_*` mirrors its Unicode function above step for step, but only for
// decisions that depend on ASCII bytes. The moment a decision would need to
// classify a byte >= 0x80 (could be a Unicode letter, digit, mark, symbol or
// space), it returns None and the caller re-runs the exact Unicode function from
// the same start. So whenever a fast path returns Some(end), `end` is what the
// slow path returns — the fast path only ever saves work, never changes output.
// Letter/digit runs are scanned 8 bytes at a time with SWAR (no intrinsics).
// ---------------------------------------------------------------------------

use crate::unicode::{ASCII_COARSE as AC, ASCII_FINE as AF};

const HI: u64 = 0x8080_8080_8080_8080;
const ONES: u64 = 0x0101_0101_0101_0101;

/// Per-lane high bit set iff the lane is an ASCII byte in [lo, hi] (hi < 0x80).
#[inline(always)]
fn swar_range(x: u64, lo: u8, hi: u8) -> u64 {
    let y = x | HI;
    let ge_lo = y.wrapping_sub(ONES * lo as u64);
    let gt_hi = y.wrapping_sub(ONES * (hi as u64 + 1));
    ge_lo & !gt_hi & !x & HI
}
#[inline(always)]
fn swar_alpha(x: u64) -> u64 {
    // fold case: 'A'..'Z' | 0x20 == 'a'..'z'; no other ASCII byte folds into a-z
    swar_range(x | 0x2020_2020_2020_2020, b'a', b'z') & !x & HI
}
#[inline(always)]
fn swar_digit(x: u64) -> u64 {
    swar_range(x, b'0', b'9')
}
#[inline(always)]
fn swar_upper(x: u64) -> u64 {
    swar_range(x, b'A', b'Z')
}
#[inline(always)]
fn swar_lower(x: u64) -> u64 {
    swar_range(x, b'a', b'z')
}

/// Advance from `p` while bytes satisfy the predicate (SWAR 8 at a time).
#[inline(always)]
fn scan(b: &[u8], mut p: usize, swar: impl Fn(u64) -> u64, one: impl Fn(u8) -> bool) -> usize {
    let n = b.len();
    while p + 8 <= n {
        // SAFETY: p + 8 <= n.
        let x = u64::from_le(unsafe { (b.as_ptr().add(p) as *const u64).read_unaligned() });
        let miss = !swar(x) & HI;
        if miss != 0 {
            return p + (miss.trailing_zeros() / 8) as usize;
        }
        p += 8;
    }
    while p < n && one(b[p]) {
        p += 1;
    }
    p
}

/// A run ended at `e`: if the byte there is non-ASCII it might continue the run
/// (Unicode letter/digit/…), so the fast path must bail.
#[inline(always)]
fn ends_ascii(b: &[u8], e: usize) -> Option<usize> {
    if e < b.len() && b[e] >= 0x80 {
        None
    } else {
        Some(e)
    }
}

#[inline(always)]
fn f_letters(b: &[u8], p: usize) -> Option<usize> {
    ends_ascii(b, scan(b, p, swar_alpha, |c| c.is_ascii_alphabetic()))
}

/// ASCII symbol run (class 'O'); bails on non-ASCII.
#[inline(always)]
fn f_symbols(b: &[u8], mut p: usize) -> Option<usize> {
    let n = b.len();
    while p < n {
        let c = b[p];
        if c >= 0x80 {
            return None;
        }
        if AC[c as usize] != b'O' {
            break;
        }
        p += 1;
    }
    Some(p)
}

/// Scan an ASCII whitespace run from `start`. Returns (end, last_ws_pos, count,
/// last_newline_end) or None if the run touches a non-ASCII byte.
#[inline(always)]
fn f_ws(b: &[u8], start: usize) -> Option<(usize, usize, u32, usize)> {
    let n = b.len();
    let (mut p, mut prev, mut count, mut nl_end) = (start, start, 0u32, 0usize);
    while p < n {
        let c = b[p];
        if c >= 0x80 {
            return None;
        }
        if AC[c as usize] != b'S' {
            break;
        }
        if c == b'\r' || c == b'\n' {
            nl_end = p + 1;
        }
        prev = p;
        p += 1;
        count += 1;
    }
    Some((p, prev, count, nl_end))
}

/// `\s+(?!\S)|\s+` (ws_trail) on an ASCII run.
#[inline(always)]
fn f_ws_trail(b: &[u8], start: usize, indiv_digits: bool) -> Option<usize> {
    let (p, prev, count, _) = f_ws(b, start)?;
    if p < b.len() && count > 1 {
        if indiv_digits && b[p].is_ascii_digit() {
            return Some(p);
        }
        Some(prev)
    } else {
        Some(p)
    }
}

/// Digit run with a cap (0 = unlimited) starting at a digit at `start`.
#[inline(always)]
fn f_digits(b: &[u8], start: usize, max_digits: u32) -> Option<usize> {
    if max_digits == 0 {
        return ends_ascii(b, scan(b, start + 1, swar_digit, |c| c.is_ascii_digit()));
    }
    let n = b.len();
    let (mut p, mut count) = (start + 1, 1u32);
    while p < n && count < max_digits {
        let c = b[p];
        if c >= 0x80 {
            return None;
        }
        if !c.is_ascii_digit() {
            break;
        }
        p += 1;
        count += 1;
    }
    Some(p)
}

/// symbol_run (` ?[^\s\p{L}\p{N}]+` + trailing) on ASCII. Outer None = bail,
/// inner None = no match (fall through to the next alternative).
#[inline(always)]
fn f_symbol_run(b: &[u8], start: usize, trail: u8) -> Option<Option<usize>> {
    let n = b.len();
    let mut pos = start;
    if b[start] == b' ' && start + 1 < n {
        pos += 1;
    }
    let c = b[pos];
    if c >= 0x80 {
        return None;
    }
    if AC[c as usize] != b'O' {
        return Some(None);
    }
    let mut e = f_symbols(b, pos + 1)?;
    if trail > 0 {
        while e < n && (b[e] == b'\r' || b[e] == b'\n' || (trail == 2 && b[e] == b'/')) {
            e += 1;
        }
    }
    Some(Some(e))
}

#[inline(always)]
fn fast_word(b: &[u8], start: usize, indiv: bool) -> Option<usize> {
    let n = b.len();
    let c = b[start];
    if c >= 0x80 {
        return None;
    }
    match AC[c as usize] {
        b'L' => f_letters(b, start + 1),
        b'N' => {
            if indiv {
                Some(start + 1)
            } else {
                f_digits(b, start, 0)
            }
        }
        b'O' => {
            if c == b'\'' {
                if let Some(e) = contraction(b, start, false) {
                    return Some(e);
                }
            }
            f_symbols(b, start + 1)
        }
        _ => {
            // whitespace; ' ' may prefix a letter/digit/symbol run
            if c == b' ' && start + 1 < n {
                let d = b[start + 1];
                if d >= 0x80 {
                    return None;
                }
                match AC[d as usize] {
                    b'L' => return f_letters(b, start + 2),
                    b'N' if !indiv => return f_digits(b, start + 1, 0),
                    b'O' => return f_symbols(b, start + 2),
                    _ => {}
                }
            }
            f_ws_trail(b, start, indiv)
        }
    }
}

#[inline(always)]
fn fast_generic(b: &[u8], start: usize, max_digits: u32) -> Option<usize> {
    let n = b.len();
    let c = b[start];
    if c >= 0x80 {
        return None;
    }
    let cls = AC[c as usize];
    if c == b'\'' {
        if let Some(e) = contraction(b, start, true) {
            return Some(e);
        }
    }
    // `[^\r\n\p{L}\p{N}]?\p{L}+`
    let pe = if (cls == b'O' || cls == b'S') && c != b'\r' && c != b'\n' { start + 1 } else { start };
    if pe < n {
        let d = b[pe];
        if d >= 0x80 {
            return None;
        }
        if d.is_ascii_alphabetic() {
            return f_letters(b, pe + 1);
        }
    }
    if cls == b'N' {
        return f_digits(b, start, max_digits);
    }
    if let Some(e) = f_symbol_run(b, start, 1)? {
        return Some(e);
    }
    // `\s*[\r\n]+` then `\s+(?!\S)|\s+`
    if cls == b'S' {
        let (_, _, _, nl_end) = f_ws(b, start)?;
        if nl_end > 0 {
            return Some(nl_end);
        }
        return f_ws_trail(b, start, false);
    }
    Some(start + 1)
}

#[inline(always)]
fn fast_camel(b: &[u8], start: usize, contractions: bool, max_digits: u32) -> Option<usize> {
    let n = b.len();
    let c = b[start];
    if c >= 0x80 {
        return None;
    }
    let f = AF[c as usize];
    let contr = |p: usize| if contractions { camelcase_contraction(b, p) } else { p };
    let pe = if c != b'\r' && c != b'\n' && f != b'U' && f != b'l' && f != b'N' { start + 1 } else { start };
    if pe < n {
        let d = b[pe];
        if d >= 0x80 {
            return None;
        }
        if d.is_ascii_uppercase() {
            let p = ends_ascii(b, scan(b, pe + 1, swar_upper, |x| x.is_ascii_uppercase()))?;
            let p = ends_ascii(b, scan(b, p, swar_lower, |x| x.is_ascii_lowercase()))?;
            return Some(contr(p));
        }
        if d.is_ascii_lowercase() {
            let p = ends_ascii(b, scan(b, pe + 1, swar_lower, |x| x.is_ascii_lowercase()))?;
            return Some(contr(p));
        }
    }
    if f == b'N' {
        return f_digits(b, start, max_digits);
    }
    if let Some(e) = f_symbol_run(b, start, 2)? {
        return Some(e);
    }
    if f == b'S' {
        let (_, _, _, nl_end) = f_ws(b, start)?;
        if nl_end > 0 {
            return Some(nl_end);
        }
        return f_ws_trail(b, start, false);
    }
    Some(start + 1)
}

/// Fast path only (None = needs the Unicode path). Exposed for self-checks.
#[inline(always)]
pub fn fast_chunk_end(kind: Pretok, b: &[u8], start: usize) -> Option<usize> {
    match kind {
        Pretok::Word(indiv) => fast_word(b, start, indiv),
        Pretok::Generic { max_digits, .. } => fast_generic(b, start, max_digits),
        Pretok::CamelCase { contractions, max_digits } => fast_camel(b, start, contractions, max_digits),
    }
}

/// Unicode (reference) path only.
#[inline]
pub fn slow_chunk_end(kind: Pretok, b: &[u8], start: usize) -> usize {
    match kind {
        Pretok::Word(indiv) => word(b, start, indiv),
        Pretok::Generic { max_digits, marks } => generic(b, start, max_digits, marks),
        Pretok::CamelCase { contractions, max_digits } => camelcase(b, start, contractions, max_digits),
    }
}

#[inline]
pub fn next_chunk_end(kind: Pretok, b: &[u8], start: usize) -> usize {
    if let Some(e) = fast_chunk_end(kind, b, start) {
        return e;
    }
    slow_chunk_end(kind, b, start)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    const KINDS: [Pretok; 9] = [
        Pretok::Word(false),
        Pretok::Word(true),
        Pretok::Generic { max_digits: 0, marks: false },
        Pretok::Generic { max_digits: 1, marks: false },
        Pretok::Generic { max_digits: 3, marks: false },
        Pretok::Generic { max_digits: 3, marks: true },
        Pretok::CamelCase { contractions: true, max_digits: 3 },
        Pretok::CamelCase { contractions: false, max_digits: 3 },
        Pretok::CamelCase { contractions: true, max_digits: 0 },
    ];

    /// Random strings dense in every edge the grammars care about: case changes,
    /// contractions (any case), digit runs, all ASCII whitespace incl. CRLF and
    /// runs, punctuation/'/'/control bytes, and non-ASCII letters, marks, digits,
    /// spaces and symbols right next to ASCII runs.
    fn random_text(seed: &mut u64) -> String {
        const POOL: &[&str] = &[
            "a", "b", "Z", "Q", "hello", "World", "HTTPServer", "camelCase", "x", "'s", "'S",
            "'t", "'ll", "'LL", "'ve", "'Re", "'d", "'m", "'", "''", "0", "7", "123", "4567",
            " ", "  ", "   ", "\t", "\n", "\r\n", "\r", "\x0b", "\x0c", " \n", "\n\n", ".", ",",
            "!", "?", "/", "//", "-", "(", ")", "\"", "#", "\x01", "\x7f", "é", "É", "ß", "漢",
            "字", "\u{0301}", "\u{0308}", "\u{00a0}", "\u{3000}", "\u{2003}", "\u{0663}", "Ⅻ",
            "½", "€", "😀", "👋🏽", "\u{0085}", "\u{2028}", "ǅ", "ʰ",
        ];
        let n = (lcg(seed) % 40) as usize;
        (0..n).map(|_| POOL[(lcg(seed) as usize) % POOL.len()]).collect()
    }

    /// Wherever the ASCII fast path answers, it must equal the Unicode path.
    #[test]
    fn fast_path_equals_unicode_path() {
        let mut seed = 7u64;
        let mut answered = 0usize;
        for _ in 0..30000 {
            let t = random_text(&mut seed);
            let b = t.as_bytes();
            for &k in &KINDS {
                let mut pos = 0;
                while pos < b.len() {
                    let e = slow_chunk_end(k, b, pos);
                    if let Some(f) = fast_chunk_end(k, b, pos) {
                        answered += 1;
                        assert_eq!(f, e, "kind={:?} text={t:?} pos={pos}", k_name(k));
                    }
                    assert!(e > pos);
                    pos = e;
                }
            }
        }
        assert!(answered > 100_000, "fast path barely exercised: {answered}");
    }

    /// The document-splitting rule: every position with `b[p-1] == '\n'` and an
    /// ASCII letter at `b[p]` is a chunk boundary of the full scan, in every grammar.
    #[test]
    fn newline_letter_is_always_a_boundary() {
        let mut seed = 11u64;
        for _ in 0..30000 {
            let t = random_text(&mut seed);
            let b = t.as_bytes();
            for &k in &KINDS {
                let mut bounds = vec![0usize];
                let mut pos = 0;
                while pos < b.len() {
                    pos = next_chunk_end(k, b, pos);
                    bounds.push(pos);
                }
                for p in 1..b.len() {
                    if b[p - 1] == b'\n' && b[p].is_ascii_alphabetic() {
                        assert!(bounds.binary_search(&p).is_ok(), "kind={} text={t:?} p={p}", k_name(k));
                    }
                }
            }
        }
    }

    fn k_name(k: Pretok) -> String {
        match k {
            Pretok::Word(i) => format!("Word({i})"),
            Pretok::Generic { max_digits, marks } => format!("Generic({max_digits},{marks})"),
            Pretok::CamelCase { contractions, max_digits } => format!("Camel({contractions},{max_digits})"),
        }
    }
}
