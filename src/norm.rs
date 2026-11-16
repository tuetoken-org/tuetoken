//! Unicode normalization (NFC/NFD/NFKC/NFKD) that only touches non-ASCII text.
//!
//! Key fact: splitting a string immediately BEFORE any ASCII character is safe
//! for all four forms — `N(a + b) == N(a) + N(b)` whenever `b` starts with ASCII:
//!   * every ASCII char is a starter (ccc = 0), so canonical reordering never
//!     moves a mark across it;
//!   * no canonical composition has an ASCII char as its SECOND element, so an
//!     ASCII char never composes with what precedes it;
//!   * it is NF*_QC = Yes, i.e. unchanged by (de)composition itself.
//! (An ASCII char CAN be the first element of a composition — `e` + U+0301 ->
//! `é` — so each non-ASCII run is normalized together with the one ASCII char
//! in front of it.)
//!
//! So: skip ASCII 8 bytes at a time, quick-check each non-ASCII run (with its
//! leading ASCII char), and only run the real normalizer on the runs that need
//! it. Already-normalized input returns `Cow::Borrowed` with no allocation; a
//! 100 MB document with one decomposed accent no longer re-normalizes 100 MB.

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
