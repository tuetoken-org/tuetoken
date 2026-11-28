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
//!      D `\s*[\r\n]+`   E `\s+(?!\S)`   F `\s+`         whitespace (generic/camelcase rules)
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

/// Apply only the DeepSeek stage-3 `Main` Isolated Split pattern to `sub`
/// (`base`-anchored), pushing resulting `(abs_start, abs_end)` pieces to `out`.
/// Reused by `PretokPipe` whenever any model (e.g. Dolphin-X1) embeds this exact
/// regex stage.
pub(crate) fn split_main_stage(sub: &str, base: usize, out: &mut Vec<(usize, usize)>) {
    let b = sub.as_bytes();
    let pe = b.len();
    let mut last = 0usize;
    let mut from = 0usize;
    while let Some((ms, me)) = find_next(Stage::Main, b, from, pe) {
        if ms > last {
            out.push((base + last, base + ms));
        }
        out.push((base + ms, base + me));
        last = me;
        from = me;
    }
    if last < pe {
        out.push((base + last, base + pe));
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

// ---------------------------------------------------------------------------
// Single-pass zero-allocation streaming CJK pre-tokenizer (`CjkScan`)
// ---------------------------------------------------------------------------
// Produces the EXACT same chunk boundaries as the 3-stage `chunk_ranges`
// (`Digits13 -> Cjk -> Main`) in ONE forward byte pass with an ASCII lookup
// table fast path and direct `ends[..]` emission for the 256-chunk BPE pipeline.

// ASCII byte classes:
//   0 = control (unmatched gap unless prefix before letter/mark)
//   1 = letter [A-Za-z]
//   2 = digit [0-9] (Stage-1 hard boundary)
//   3 = punctuation/symbol (u8::is_ascii_punctuation)
//   4 = space ' ' (0x20)
//   5 = CR/LF (\r, \n)
//   6 = other ASCII whitespace (\t, \x0B, \x0C)
const ASCII_CLS: [u8; 128] = {
    let mut t = [0u8; 128];
    let mut i = 0usize;
    while i < 128 {
        let b = i as u8;
        t[i] = if b.is_ascii_alphabetic() {
            1
        } else if b.is_ascii_digit() {
            2
        } else if b.is_ascii_punctuation() {
            3
        } else if b == b' ' {
            4
        } else if b == b'\r' || b == b'\n' {
            5
        } else if b == b'\t' || b == 0x0B || b == 0x0C {
            6
        } else {
            0
        };
        i += 1;
    }
    t
};

/// True if the codepoint at `p` starts a Stage-1 (`\p{N}`) or Stage-2 (CJK/kana)
/// isolated piece — which acts as a hard piece end for Stage-3 (`Main`).
#[inline(always)]
fn is_stage12_boundary(b: &[u8], p: usize) -> bool {
    let c0 = b[p];
    if c0 < 0x80 {
        return c0.is_ascii_digit();
    }
    let (cp, _) = decode(b, p);
    is_number(cp) || is_cjk(cp)
}

/// Fast-path scan of a Stage-3 letter/mark run starting at `q` inside a Stage-1+2
/// gap (cannot cross `stop` or any Stage-1+2 boundary).
#[inline(always)]
fn scan_lm_gap(b: &[u8], mut q: usize, stop: usize) -> usize {
    while q < stop {
        let c = b[q];
        if c < 0x80 {
            if c.is_ascii_alphabetic() {
                q += 1;
                continue;
            }
            break;
        }
        let (cp, l) = decode(b, q);
        // CJK/kana chars belong to Stage 2 (separate piece), so stop here!
        if is_cjk(cp) || !(is_letter(cp) || is_mark(cp)) {
            break;
        }
        q += l;
    }
    q
}

/// Scan Stage-3 Alt C (`[\p{P}\p{S}]+[\r\n]*`) starting after its first matched
/// punct/symbol at `q`, stopping at `stop` or any Stage-1+2 boundary.
#[inline(always)]
fn scan_ps_nl_gap(b: &[u8], mut q: usize, stop: usize) -> usize {
    while q < stop {
        let c = b[q];
        if c < 0x80 {
            if ASCII_CLS[c as usize] == 3 {
                q += 1;
                continue;
            }
            break;
        }
        let (cp, l) = decode(b, q);
        if is_cjk(cp) || !is_punct_or_symbol(cp) {
            break;
        }
        q += l;
    }
    while q < stop && (b[q] == b'\r' || b[q] == b'\n') {
        q += 1;
    }
    q
}

/// Scan Stage-3 whitespace (Alt D `\s*[\r\n]+`, E `\s+(?!\S)`, F `\s+`) starting
/// at `start` (known whitespace), bounded by `stop` AND any Stage-1+2 boundary
/// (which terminates the Stage-3 piece so `\s+(?!\S)` does NOT back off!).
#[inline(always)]
fn scan_ws_gap(b: &[u8], start: usize, stop: usize) -> usize {
    let mut p = start;
    let mut prev = start;
    let mut last_nl = 0usize;
    let mut found_nl = false;
    let mut count = 0u32;
    while p < stop {
        let c = b[p];
        if c < 0x80 {
            let cls = ASCII_CLS[c as usize];
            if cls < 4 {
                // Non-whitespace ASCII: if digit (cls == 2), Stage 1 split the piece here!
                if found_nl {
                    return last_nl;
                }
                if cls == 2 {
                    return p;
                }
                return if count > 1 { prev } else { p };
            }
            prev = p;
            p += 1;
            count += 1;
            if cls == 5 {
                last_nl = p;
                found_nl = true;
            }
            continue;
        }
        let (cp, l) = decode(b, p);
        if !is_unicode_whitespace(cp) {
            if found_nl {
                return last_nl;
            }
            if is_number(cp) || is_cjk(cp) {
                return p;
            }
            return if count > 1 { prev } else { p };
        }
        prev = p;
        p += l;
        count += 1;
    }
    if found_nl {
        last_nl
    } else {
        p
    }
}

/// Try matching Stage-3 (`Main`) at `i` inside a non-number, non-CJK gap.
/// Returns `Some(end)` on match or `None` if `i` is an unmatched gap char.
#[inline(always)]
fn main_match_gap(b: &[u8], i: usize, stop: usize) -> Option<usize> {
    let c0 = b[i];
    if c0 < 0x80 {
        match ASCII_CLS[c0 as usize] {
            // 1: ASCII letter -> Alt B
            1 => return Some(scan_lm_gap(b, i + 1, stop)),
            // 4: ASCII space -> Alt B (` [L/M]+`), Alt C (` [P/S]+[\r\n]*`), or Alt D/E/F (ws)
            4 => {
                let p = i + 1;
                if p < stop {
                    let c1 = b[p];
                    if c1 < 0x80 {
                        match ASCII_CLS[c1 as usize] {
                            1 => return Some(scan_lm_gap(b, p + 1, stop)),
                            3 => return Some(scan_ps_nl_gap(b, p + 1, stop)),
                            2 => return Some(p), // digit is Stage-1 boundary -> 1-byte space
                            4..=6 => return Some(scan_ws_gap(b, i, stop)),
                            _ => return Some(p), // control char -> single space (count==1)
                        }
                    } else {
                        let (cp1, l1) = decode(b, p);
                        if !is_cjk(cp1) {
                            if is_letter(cp1) || is_mark(cp1) {
                                return Some(scan_lm_gap(b, p + l1, stop));
                            }
                            if is_punct_or_symbol(cp1) {
                                return Some(scan_ps_nl_gap(b, p + l1, stop));
                            }
                        }
                        if is_unicode_whitespace(cp1) {
                            return Some(scan_ws_gap(b, i, stop));
                        }
                        return Some(p);
                    }
                }
                return Some(p);
            }
            // 3: ASCII punct -> Alt A (`[ascii-punct][A-Za-z]+`) or Alt C (`[\p{P}\p{S}]+[\r\n]*`)
            3 => {
                let p = i + 1;
                if p < stop && b[p].is_ascii_alphabetic() {
                    let mut q = p + 1;
                    while q < stop && b[q].is_ascii_alphabetic() {
                        q += 1;
                    }
                    return Some(q);
                }
                return Some(scan_ps_nl_gap(b, p, stop));
            }
            // 5: ASCII newline (\r, \n) -> excluded from Alt B prefix -> Alt D/E/F
            5 => return Some(scan_ws_gap(b, i, stop)),
            // 6: other ASCII whitespace (\t, \x0B, \x0C) -> valid Alt B prefix before
            // non-CJK letter/mark, otherwise Alt D/E/F whitespace!
            6 => {
                let p = i + 1;
                if p < stop {
                    let c1 = b[p];
                    if c1 < 0x80 {
                        if c1.is_ascii_alphabetic() {
                            return Some(scan_lm_gap(b, p + 1, stop));
                        }
                    } else {
                        let (cp1, l1) = decode(b, p);
                        if !is_cjk(cp1) && (is_letter(cp1) || is_mark(cp1)) {
                            return Some(scan_lm_gap(b, p + l1, stop));
                        }
                    }
                }
                return Some(scan_ws_gap(b, i, stop));
            }
            // 0: ASCII control char -> Alt B prefix if followed by non-CJK letter/mark, else gap
            _ => {
                let p = i + 1;
                if p < stop {
                    let c1 = b[p];
                    if c1 < 0x80 {
                        if c1.is_ascii_alphabetic() {
                            return Some(scan_lm_gap(b, p + 1, stop));
                        }
                    } else {
                        let (cp1, l1) = decode(b, p);
                        if !is_cjk(cp1) && (is_letter(cp1) || is_mark(cp1)) {
                            return Some(scan_lm_gap(b, p + l1, stop));
                        }
                    }
                }
                return None;
            }
        }
    }

    // Non-ASCII (known neither `\p{N}` nor CJK/kana)
    let (cp0, l0) = decode(b, i);
    if is_letter(cp0) || is_mark(cp0) {
        return Some(scan_lm_gap(b, i + l0, stop));
    }
    if is_punct_or_symbol(cp0) {
        return Some(scan_ps_nl_gap(b, i + l0, stop));
    }
    // `cp0` is neither `\r`/`\n` (since >= 0x80), nor letter, nor punct/symbol -> valid Alt B prefix!
    let p = i + l0;
    if p < stop {
        let c1 = b[p];
        if c1 < 0x80 {
            if c1.is_ascii_alphabetic() {
                return Some(scan_lm_gap(b, p + 1, stop));
            }
        } else {
            let (cp1, l1) = decode(b, p);
            if !is_cjk(cp1) && (is_letter(cp1) || is_mark(cp1)) {
                return Some(scan_lm_gap(b, p + l1, stop));
            }
        }
    }
    if is_unicode_whitespace(cp0) {
        return Some(scan_ws_gap(b, i, stop));
    }
    None
}

/// Stateful streaming DeepSeek-V3/R1 CJK pre-tokenizer.
/// Emits chunk end offsets into `ends[..]` in a single pass with zero allocations.
pub struct CjkScan {
    pub pos: usize,
    /// When a Stage-2 CJK/kana piece contains non-letter/mark punctuation (e.g.
    /// `゠` U+30A0 or `・` U+30FB), we sub-split `[pos..pending_cjk_end]` with `Stage::Main`.
    pending_cjk_end: usize,
}

impl CjkScan {
    #[inline(always)]
    pub fn new(start: usize) -> Self {
        Self { pos: start, pending_cjk_end: 0 }
    }

    /// Fill chunk end byte offsets into `ends[k..cap]` from `b[self.pos..stop]`.
    /// Returns the updated `k`. Requires `stop <= u32::MAX as usize`.
    #[inline]
    pub fn fill(&mut self, b: &[u8], stop: usize, ends: &mut [u32], mut k: usize, cap: usize) -> usize {
        while k < cap && self.pos < stop {
            // Drain any pending mixed-kana Stage-2 piece via exact `Stage::Main`.
            if self.pending_cjk_end != 0 {
                let pe = self.pending_cjk_end;
                let e = if let Some(me) = main_match_at(b, self.pos, pe) {
                    me
                } else {
                    let (_, l) = decode(b, self.pos);
                    let mut q = self.pos + l;
                    while q < pe && main_match_at(b, q, pe).is_none() {
                        let (_, l2) = decode(b, q);
                        q += l2;
                    }
                    q
                };
                ends[k] = e as u32;
                k += 1;
                self.pos = e;
                if self.pos >= pe {
                    self.pending_cjk_end = 0;
                }
                continue;
            }

            let i = self.pos;
            let c0 = b[i];
            if c0 < 0x80 {
                // Stage 1: ASCII digit run of up to 3 `\p{N}`
                if c0.is_ascii_digit() {
                    let mut q = i + 1;
                    let mut cnt = 1u32;
                    while q < stop && cnt < 3 {
                        let c = b[q];
                        if c < 0x80 {
                            if !c.is_ascii_digit() {
                                break;
                            }
                            q += 1;
                            cnt += 1;
                        } else {
                            let (cp, l) = decode(b, q);
                            if !is_number(cp) {
                                break;
                            }
                            q += l;
                            cnt += 1;
                        }
                    }
                    ends[k] = q as u32;
                    k += 1;
                    self.pos = q;
                    continue;
                }
                // Stage 3 (`Main`) inside non-digit, non-CJK gap
                let e = if let Some(me) = main_match_gap(b, i, stop) {
                    me
                } else {
                    let mut q = i + 1;
                    while q < stop && !is_stage12_boundary(b, q) && main_match_gap(b, q, stop).is_none() {
                        let c = b[q];
                        q += if c < 0x80 { 1 } else { decode(b, q).1 };
                    }
                    q
                };
                ends[k] = e as u32;
                k += 1;
                self.pos = e;
                continue;
            }

            // Fast path for pure CJK Unified Ideographs 0xE5..=0xE8 (U+5000..=U+8FFF, 100% Lo)
            let (cp0, l0) = decode(b, i);
            if is_number(cp0) {
                let mut q = i + l0;
                let mut cnt = 1u32;
                while q < stop && cnt < 3 {
                    let c = b[q];
                    if c < 0x80 {
                        if !c.is_ascii_digit() {
                            break;
                        }
                        q += 1;
                        cnt += 1;
                    } else {
                        let (cp, l) = decode(b, q);
                        if !is_number(cp) {
                            break;
                        }
                        q += l;
                        cnt += 1;
                    }
                }
                ends[k] = q as u32;
                k += 1;
                self.pos = q;
                continue;
            }
            if is_cjk(cp0) {
                let mut q = i + l0;
                let mut all_lm = (0x4E00..=0x9FA5).contains(&cp0) || is_letter(cp0) || is_mark(cp0);
                while q < stop {
                    let c = b[q];
                    if c < 0x80 {
                        break;
                    }
                    if (0xE5..=0xE8).contains(&c) {
                        // 0xE5..=0xE8 is strictly U+5000..=U+8FFF: 100% CJK Lo!
                        q += 3;
                        continue;
                    }
                    let (cp, l) = decode(b, q);
                    if !is_cjk(cp) {
                        break;
                    }
                    if all_lm && !(0x4E00..=0x9FA5).contains(&cp) && !(is_letter(cp) || is_mark(cp)) {
                        all_lm = false;
                    }
                    q += l;
                }
                if all_lm {
                    ends[k] = q as u32;
                    k += 1;
                    self.pos = q;
                } else {
                    self.pending_cjk_end = q;
                }
                continue;
            }

            // Non-ASCII, non-number, non-CJK gap char
            let e = if let Some(me) = main_match_gap(b, i, stop) {
                me
            } else {
                let mut q = i + l0;
                while q < stop && !is_stage12_boundary(b, q) && main_match_gap(b, q, stop).is_none() {
                    let c = b[q];
                    q += if c < 0x80 { 1 } else { decode(b, q).1 };
                }
                q
            };
            ends[k] = e as u32;
            k += 1;
            self.pos = e;
        }
        k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_all(s: &str) -> Vec<(usize, usize)> {
        let b = s.as_bytes();
        let mut scan = CjkScan::new(0);
        let mut ends = [0u32; 64];
        let mut out = Vec::new();
        let mut a = 0usize;
        loop {
            let k = scan.fill(b, b.len(), &mut ends, 0, 64);
            if k == 0 {
                break;
            }
            for &e in &ends[..k] {
                let eu = e as usize;
                out.push((a, eu));
                a = eu;
            }
        }
        assert_eq!(a, b.len());
        out
    }

    #[test]
    fn cjk_ranges_disjoint_and_ascii_punct_exact() {
        for b in 0u8..128 {
            assert_eq!(
                b.is_ascii_punctuation(),
                is_punct_or_symbol(b as u32),
                "ASCII byte {b:#x} punct mismatch"
            );
        }
        for cp in 0x4E00u32..=0x9FA5 {
            assert!(is_letter(cp), "U+{cp:04X} not letter");
            assert!(!is_number(cp), "U+{cp:04X} is number");
        }
        for cp in (0x3040u32..=0x309F).chain(0x30A0u32..=0x30FF) {
            assert!(!is_number(cp), "kana U+{cp:04X} is number");
        }
    }

    #[test]
    fn cjk_scan_matches_three_stage_reference_fuzz() {
        let pool: &[&str] = &[
            "a", "Z", "0", "9", " ", "  ", "\t", "\n", "\r", "\r\n", "_", ".", "!", "$", "(", ")",
            "/", "\x00", "\x01", "\x7f", "é", "e\u{0301}", "中", "国", "人", "あ", "ア",
            "\u{30A0}", "\u{30FB}", "\u{3099}", "\u{309B}", "\u{30FC}", "。", "，",
            "\u{0662}", "\u{00B2}", "\u{00A0}", "\u{200D}", "🚀", "—", "“",
        ];
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut cur = Vec::new();
        let mut nxt = Vec::new();
        for case in 0..20_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let len = ((seed >> 33) % 24) as usize;
            let mut s = String::new();
            for _ in 0..len {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s.push_str(pool[((seed >> 33) as usize) % pool.len()]);
            }
            chunk_ranges(&s, &mut cur, &mut nxt);
            let got = scan_all(&s);
            assert_eq!(got, cur, "mismatch on case {case}: {s:?}");

            // Also prove `\n` + (ASCII alnum or CJK 0xE4..=0xE9) split-point invariance!
            let b = s.as_bytes();
            for p in 1..b.len() {
                if b[p - 1] == b'\n' && (b[p].is_ascii_alphanumeric() || (0xE4..=0xE9).contains(&b[p])) {
                    let mut left = scan_all(&s[..p]);
                    let right = scan_all(&s[p..]);
                    left.extend(right.into_iter().map(|(x, y)| (x + p, y + p)));
                    assert_eq!(left, cur, "split mismatch at p={p} on {s:?}");
                }
            }
        }
    }
}
