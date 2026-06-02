//! Unicode classification + UTF-8 decode, matching tuetoken's C engine exactly
//! (so pre-tokenization is byte-for-byte parity with HF/tiktoken).

use crate::unicode_tables::{
    LETTER_ASTRAL, LETTER_BMP, LOWER_ASTRAL, LOWER_BMP, MARK_ASTRAL, MARK_BMP, NUMBER_ASTRAL,
    NUMBER_BMP, PUNCTSYM_ASTRAL, PUNCTSYM_BMP, UPPER_ASTRAL, UPPER_BMP,
};

#[inline]
fn in_astral(table: &[(u32, u32)], cp: u32) -> bool {
    let mut lo = 0usize;
    let mut hi = table.len();
    while lo < hi {
        let mid = (lo + hi) >> 1;
        let (a, b) = table[mid];
        if cp < a {
            hi = mid;
        } else if cp > b {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

/// BMP O(1) bitmap lookup, astral binary-search. ASCII handled by callers.
#[inline]
fn has_prop(bmp: &[u64; 1024], astral: &[(u32, u32)], cp: u32) -> bool {
    if cp < 0x10000 {
        // SAFETY: cp < 0x10000 so (cp >> 6) < 1024.
        (unsafe { *bmp.get_unchecked((cp >> 6) as usize) } >> (cp & 63)) & 1 != 0
    } else {
        in_astral(astral, cp)
    }
}

/// Decode one UTF-8 codepoint at `bytes[pos]`. Returns (codepoint, byte_len).
/// Mirrors uc_decode_utf8 (lenient: malformed -> single byte).
#[inline]
pub fn decode(bytes: &[u8], pos: usize) -> (u32, usize) {
    let c = bytes[pos];
    if c < 0x80 {
        (c as u32, 1)
    } else if (c & 0xE0) == 0xC0 && pos + 1 < bytes.len() {
        (((c as u32 & 0x1F) << 6) | (bytes[pos + 1] as u32 & 0x3F), 2)
    } else if (c & 0xF0) == 0xE0 && pos + 2 < bytes.len() {
        (
            ((c as u32 & 0x0F) << 12)
                | ((bytes[pos + 1] as u32 & 0x3F) << 6)
                | (bytes[pos + 2] as u32 & 0x3F),
            3,
        )
    } else if (c & 0xF8) == 0xF0 && pos + 3 < bytes.len() {
        (
            ((c as u32 & 0x07) << 18)
                | ((bytes[pos + 1] as u32 & 0x3F) << 12)
                | ((bytes[pos + 2] as u32 & 0x3F) << 6)
                | (bytes[pos + 3] as u32 & 0x3F),
            4,
        )
    } else {
        (c as u32, 1)
    }
}

#[inline]
pub fn is_unicode_whitespace(cp: u32) -> bool {
    if cp < 0x80 {
        return cp == b' ' as u32
            || cp == b'\t' as u32
            || cp == b'\n' as u32
            || cp == b'\r' as u32
            || cp == 0x0b
            || cp == 0x0c;
    }
    matches!(
        cp,
        0x85 | 0xA0
            | 0x1680
            | 0x2000..=0x200A
            | 0x2028
            | 0x2029
            | 0x202F
            | 0x205F
            | 0x3000
    )
}

#[inline]
pub fn is_letter(cp: u32) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_alphabetic();
    }
    has_prop(&LETTER_BMP, LETTER_ASTRAL, cp)
}

#[inline]
pub fn is_number(cp: u32) -> bool {
    if cp < 0x80 {
        return cp >= '0' as u32 && cp <= '9' as u32;
    }
    has_prop(&NUMBER_BMP, NUMBER_ASTRAL, cp)
}

#[inline]
pub fn is_mark(cp: u32) -> bool {
    if cp < 0x80 {
        return false;
    }
    has_prop(&MARK_BMP, MARK_ASTRAL, cp)
}

/// `\p{P}` (punctuation) | `\p{S}` (symbol). ASCII punctuation/symbols are exactly
/// `is_ascii_punctuation` (`!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`).
#[inline]
pub fn is_punct_or_symbol(cp: u32) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_punctuation();
    }
    has_prop(&PUNCTSYM_BMP, PUNCTSYM_ASTRAL, cp)
}

#[inline]
pub fn is_upper(cp: u32) -> bool {
    if cp < 0x80 {
        return cp >= 'A' as u32 && cp <= 'Z' as u32;
    }
    has_prop(&UPPER_BMP, UPPER_ASTRAL, cp)
}

#[inline]
pub fn is_lower(cp: u32) -> bool {
    if cp < 0x80 {
        return cp >= 'a' as u32 && cp <= 'z' as u32;
    }
    has_prop(&LOWER_BMP, LOWER_ASTRAL, cp)
}

// Compile-time ASCII class tables (no runtime init / atomic on the hot path).
const fn build_ascii(fine: bool) -> [u8; 128] {
    let mut t = [b'O'; 128];
    let mut c = 0usize;
    while c < 128 {
        let b = c as u8;
        t[c] = if b.is_ascii_alphabetic() {
            if fine {
                if b.is_ascii_uppercase() {
                    b'U'
                } else {
                    b'l'
                }
            } else {
                b'L'
            }
        } else if b.is_ascii_digit() {
            b'N'
        } else if matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c) {
            b'S'
        } else {
            b'O'
        };
        c += 1;
    }
    t
}
static ASCII_COARSE: [u8; 128] = build_ascii(false);
static ASCII_FINE: [u8; 128] = build_ascii(true);

/// Coarse class: 'L' letter, 'N' number, 'S' whitespace, 'O' other.
#[inline]
pub fn classify(cp: u32) -> u8 {
    if cp < 0x80 {
        return ASCII_COARSE[cp as usize];
    }
    if is_letter(cp) {
        b'L'
    } else if is_number(cp) {
        b'N'
    } else if is_unicode_whitespace(cp) {
        b'S'
    } else {
        b'O'
    }
}

/// Fused decode+classify (coarse): one ASCII fast-path, matching C's
/// uc_classify. Returns (class, byte_len). Avoids the double ASCII check of
/// `classify(decode(..))`.
#[inline]
pub fn classify_at(b: &[u8], pos: usize) -> (u8, usize) {
    let c = b[pos];
    if c < 0x80 {
        return (ASCII_COARSE[c as usize], 1);
    }
    let (cp, len) = decode(b, pos);
    (classify(cp), len)
}

/// Fused decode+classify_fine.
#[inline]
pub fn classify_fine_at(b: &[u8], pos: usize) -> (u8, usize) {
    let c = b[pos];
    if c < 0x80 {
        return (ASCII_FINE[c as usize], 1);
    }
    let (cp, len) = decode(b, pos);
    (classify_fine(cp), len)
}

/// Fine class for CamelCase letter runs: 'U' upper, 'l' lower, 'L' other-letter,
/// 'm' mark, 'N' number, 'S' whitespace, 'O' other.
#[inline]
pub fn classify_fine(cp: u32) -> u8 {
    if cp < 0x80 {
        return ASCII_FINE[cp as usize];
    }
    if is_mark(cp) {
        b'm'
    } else if is_letter(cp) {
        if is_upper(cp) {
            b'U'
        } else if is_lower(cp) {
            b'l'
        } else {
            b'L'
        }
    } else if is_number(cp) {
        b'N'
    } else if is_unicode_whitespace(cp) {
        b'S'
    } else {
        b'O'
    }
}
