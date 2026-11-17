//! 64-byte block pre-tokenizer for all three ByteLevel grammars.
//!
//! The serial pre-tokenizer (`pretok.rs`) decides chunk ends one branchy step
//! at a time. Here a whole 64-byte block is classified at once into bitmasks,
//! and "a chunk starts at byte i" is derived with word-wide bit operations.
//! Every rule is a LOCAL FACT about the exact serial scan of that grammar,
//! derived from its code in `pretok.rs` for ASCII text. Notation: letters L
//! (upper U, lower W), digits D, apostrophe Q, other punctuation/control P,
//! O = P|Q (the symbol class), space SP, NL = \r|\n, WS = SP|\t|\n|\v|\f|\r.
//!
//! Whatever the rules don't decide is marked BAD: non-ASCII bytes and the four
//! bytes after them (whose rules read them), whitespace runs whose extent is
//! unknown (crossing a block edge or touching non-ASCII, which may be Unicode
//! whitespace), digit groups whose alignment is unknown. The driver takes a
//! mask boundary only when no BAD bit lies between the current position and
//! it; otherwise it runs the exact serial `next_chunk_end`. So output is always
//! identical to the serial scan — verified by the tests at the bottom on
//! hundreds of thousands of adversarial strings.
//!
//! Classification is the only CPU-specific step: NEON on aarch64 (always
//! present), SSE2 on x86-64 (always present), plain Rust everywhere else.
//!
//! ## Generic (Llama-3, Qwen, cl100k, GLM, OLMo, …)
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,k}|
//!  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//! * O starts ⟺ prev ∉ O∪{SP} (a space before O always starts a chunk and O
//!   joins it; after O it continues the symbol run).
//! * L starts ⟺ prev ∈ D∪NL, or prev ∈ O and prev did NOT start a chunk (end of
//!   a symbol run); after SP/T or a chunk-starting O, L joins as the prefix.
//! * A chunk-starting Q + s|t|m|d|ll|ve|re (any case) ends right after it.
//! * D starts ⟺ prev ∉ D, then every k digits.
//! * NL right after O is absorbed by the symbol run's `[\r\n]*` tail.
//! * A whitespace run [r,e): starts at r (or, after O, at its first non-NL), right
//!   after its last NL (`\s*[\r\n]+`), and at e-1 when the NL-free tail is ≥ 2
//!   long and text follows (`\s+(?!\S)` leaves one char to prefix the next).
//!
//! ## CamelCase (o200k, MiniMax, …)
//! Generic, except: inside a letter run a chunk starts only at a lower→upper
//! transition; a contraction is a SUFFIX of the preceding letter chunk (`it's`
//! is one chunk) and ends it; the symbol tail is `[\r\n/]*`, so a '/' right
//! after an absorbed NL is absorbed too.
//!
//! ## Word (GPT-2 family)
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//! Only a space prefixes a run (letters, digits, symbols); contractions are
//! case-sensitive and must start a chunk; whitespace has no NL rule (`\s+(?!\S)`
//! backs off one char), and with `individual_digits` every digit is a chunk and
//! a whitespace run before a digit is not backed off.

use crate::pretok::{next_chunk_end, Pretok};

/// Per-byte class bitmasks for one 64-byte block (bit i = byte base+i).
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Classes {
    pub l: u64,  // ASCII letter
    pub up: u64, // A-Z
    pub d: u64,  // ASCII digit
    pub sp: u64, // ' '
    pub nl: u64, // '\r' | '\n'
    pub ws: u64, // ' ' | \t \n \v \f \r
    pub q: u64,  // '\''
    pub sl: u64, // '/'
    pub na: u64, // >= 0x80
}

#[inline(always)]
fn is_ws(c: u8) -> bool {
    c == b' ' || (0x09..=0x0d).contains(&c)
}

/// Reference classification (also used for the tail block). Bytes at or past
/// `b.len()` get no class bits.
pub fn classify_scalar(b: &[u8], base: usize) -> Classes {
    let mut m = Classes::default();
    for i in 0..64 {
        let p = base + i;
        if p >= b.len() {
            break;
        }
        let c = b[p];
        let bit = 1u64 << i;
        if c.is_ascii_alphabetic() { m.l |= bit; }
        if c.is_ascii_uppercase() { m.up |= bit; }
        if c.is_ascii_digit() { m.d |= bit; }
        if c == b' ' { m.sp |= bit; }
        if c == b'\r' || c == b'\n' { m.nl |= bit; }
        if is_ws(c) { m.ws |= bit; }
        if c == b'\'' { m.q |= bit; }
        if c == b'/' { m.sl |= bit; }
        if c >= 0x80 { m.na |= bit; }
    }
    m
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn classify64(p: *const u8) -> Classes {
    use std::arch::aarch64::*;
    // De-interleaving load: x[j] lane i = byte 4i + j. Then a 0xFF/0x00 mask
    // per vector folds into a 64-bit mask in NATURAL byte order with three
    // shift-right-and-insert ops + one narrowing shift (5 instructions per
    // class instead of 4 AND + 4 pairwise-add + move).
    let tomask = |a: [uint8x16_t; 4]| -> u64 {
        // Pinned with asm: LLVM rewrites the vsriq_n_u8 chain into AND/ORR with
        // per-lane constants (8 instructions per mask instead of 4).
        let (mut t1, mut t3) = (a[1], a[3]);
        std::arch::asm!(
            "sri {t1:v}.16b, {a0:v}.16b, #1",
            "sri {t3:v}.16b, {a2:v}.16b, #1",
            "sri {t3:v}.16b, {t1:v}.16b, #2",
            "sri {t3:v}.16b, {t3:v}.16b, #4",
            t1 = inout(vreg) t1,
            t3 = inout(vreg) t3,
            a0 = in(vreg) a[0],
            a2 = in(vreg) a[2],
            options(pure, nomem, nostack, preserves_flags),
        );
        let _ = t1;
        let t4 = vshrn_n_u16::<4>(vreinterpretq_u16_u8(t3)); // nibble per lane, packed
        vget_lane_u64::<0>(vreinterpret_u64_u8(t4))
    };
    let v = vld4q_u8(p);
    let x = [v.0, v.1, v.2, v.3];
    let k = |v: u8| vdupq_n_u8(v);
    Classes {
        l: tomask(x.map(|v| vcltq_u8(vsubq_u8(vorrq_u8(v, k(0x20)), k(b'a')), k(26)))),
        up: tomask(x.map(|v| vcltq_u8(vsubq_u8(v, k(b'A')), k(26)))),
        d: tomask(x.map(|v| vcltq_u8(vsubq_u8(v, k(b'0')), k(10)))),
        sp: tomask(x.map(|v| vceqq_u8(v, k(b' ')))),
        nl: tomask(x.map(|v| vorrq_u8(vceqq_u8(v, k(b'\n')), vceqq_u8(v, k(b'\r'))))),
        ws: tomask(x.map(|v| vorrq_u8(vceqq_u8(v, k(b' ')), vcltq_u8(vsubq_u8(v, k(0x09)), k(5))))),
        q: tomask(x.map(|v| vceqq_u8(v, k(b'\'')))),
        sl: tomask(x.map(|v| vceqq_u8(v, k(b'/')))),
        na: tomask(x.map(|v| vcgeq_u8(v, k(0x80)))),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn classify64(p: *const u8) -> Classes {
    use std::arch::x86_64::*;
    let x = [
        _mm_loadu_si128(p as *const __m128i),
        _mm_loadu_si128(p.add(16) as *const __m128i),
        _mm_loadu_si128(p.add(32) as *const __m128i),
        _mm_loadu_si128(p.add(48) as *const __m128i),
    ];
    let k = |v: u8| _mm_set1_epi8(v as i8);
    // unsigned "lo <= v < lo+len" for every byte value: shift into signed range
    let range = |v: __m128i, lo: u8, len: u8| {
        let y = _mm_add_epi8(v, k(0x80u8.wrapping_sub(lo)));
        _mm_cmplt_epi8(y, k(0x80u8.wrapping_add(len)))
    };
    let tomask = |a: [__m128i; 4]| -> u64 {
        (_mm_movemask_epi8(a[0]) as u16 as u64)
            | ((_mm_movemask_epi8(a[1]) as u16 as u64) << 16)
            | ((_mm_movemask_epi8(a[2]) as u16 as u64) << 32)
            | ((_mm_movemask_epi8(a[3]) as u16 as u64) << 48)
    };
    Classes {
        l: tomask(x.map(|v| range(_mm_or_si128(v, k(0x20)), b'a', 26))),
        up: tomask(x.map(|v| range(v, b'A', 26))),
        d: tomask(x.map(|v| range(v, b'0', 10))),
        sp: tomask(x.map(|v| _mm_cmpeq_epi8(v, k(b' ')))),
        nl: tomask(x.map(|v| _mm_or_si128(_mm_cmpeq_epi8(v, k(b'\n')), _mm_cmpeq_epi8(v, k(b'\r'))))),
        ws: tomask(x.map(|v| _mm_or_si128(_mm_cmpeq_epi8(v, k(b' ')), range(v, 0x09, 5)))),
        q: tomask(x.map(|v| _mm_cmpeq_epi8(v, k(b'\'')))),
        sl: tomask(x.map(|v| _mm_cmpeq_epi8(v, k(b'/')))),
        na: tomask(x), // movemask of the raw bytes = high bit = >= 0x80
    }
}

/// Classes of the 64 bytes at `base` (fewer near the end of `b`).
#[inline(always)]
pub fn classify(b: &[u8], base: usize) -> Classes {
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    {
        if base + 64 <= b.len() {
            // SAFETY: 64 bytes are readable at base; NEON / SSE2 are baseline
            // features of these targets.
            return unsafe { classify64(b.as_ptr().add(base)) };
        }
        // Tail block (every document has one): SIMD on a zero-padded copy. A
        // zero byte is in none of the classes, so the padding sets no bits.
        let mut pad = [0u8; 64];
        let k = b.len() - base;
        pad[..k].copy_from_slice(&b[base..]);
        // SAFETY: `pad` is 64 readable bytes.
        return unsafe { classify64(pad.as_ptr()) };
    }
    #[allow(unreachable_code)]
    classify_scalar(b, base)
}

/// Contraction length at an apostrophe at `q` (0 = none). `ci`: case-insensitive.
#[inline]
fn contraction_len(b: &[u8], q: usize, ci: bool) -> usize {
    let get = |i: usize| b.get(i).map(|&c| if ci { c.to_ascii_lowercase() } else { c });
    match get(q + 1) {
        Some(b's' | b't' | b'm' | b'd') => 2,
        Some(b'l') if get(q + 2) == Some(b'l') => 3,
        Some(b'v' | b'r') if get(q + 2) == Some(b'e') => 3,
        _ => 0,
    }
}

/// State carried from one block to the next in a sequential scan.
#[derive(Clone, Copy, Default)]
pub struct Carry {
    o: bool,
    sp: bool,
    d: bool,
    nl: bool,
    ws: bool,
    l: bool,
    lw: bool, // ASCII lowercase
    na: bool,
    na4: u64,     // bit k-1 = byte base-k is non-ASCII, k = 1..4
    ostart: bool, // byte base-1 started a chunk as an O
    abs_nl: bool, // byte base-1 is an NL absorbed into a symbol chunk's tail
    abs_sl: bool, // byte base-1 is a '/' absorbed into a symbol chunk's tail (CamelCase)
    bad_in: u64,  // BAD bits carried into this block (contraction spans)
    bad3: u64,    // BAD flags of bytes base-3..base-1 (bit 2 = base-1)
    ws_unk: bool, // byte base-1 is whitespace in an uncertain run (CamelCase)
    sl_unk: bool, // byte base-1 is a '/' whose tail-absorption is uncertain (CamelCase)
    contr: bool,  // byte base-1 ends a contraction suffix (CamelCase)
    contr_mask: u64, // suffix-end bytes inside this block, from a suffix that began earlier
    forced: u64,  // chunk starts carried into this block
    suppress: u64, // non-starts carried into this block (inside a contraction)
}

/// Result of one block: (starts, bad, carry for the next block).
type Block = (u64, u64, Carry);

#[inline(always)]
fn p1(x: u64, c: bool) -> u64 {
    (x << 1) | c as u64
}
#[inline(always)]
fn n1(x: u64, c: bool) -> u64 {
    (x >> 1) | ((c as u64) << 63)
}
#[inline(always)]
fn run_mask(r: usize, e: usize) -> u64 {
    // bits [r, e) of a 64-bit block; empty when r >= 64 (a shift by 64 would
    // wrap to a shift by 0 in release builds)
    if r >= 64 || e <= r {
        return 0;
    }
    let hi = if e >= 64 { u64::MAX } else { (1u64 << e) - 1 };
    hi & (u64::MAX << r)
}

/// Common pieces shared by the three grammars.
struct Ctx {
    m: Classes,
    o: u64,
    valid: u64,
    next_ws: bool,
    next_na: bool,
    cont: u64, // continuation bytes of resolved non-ASCII characters
}

/// Resolve the commonest non-ASCII characters of web text to their regex class
/// so they don't make the block BAD (OpenWebText: 20% of blocks contain
/// non-ASCII, 2.5% once these are resolved). Only characters entirely inside
/// the block are resolved; everything else stays non-ASCII (BAD), as before.
///   * U+00C0..U+00FF (lead 0xC3): letters, except U+00D7 × and U+00F7 ÷ (Sm)
///   * U+2010..U+2027, U+2030..U+203E (E2 80 90..A7 / B0..BE): punctuation —
///     dashes, curly quotes, bullets, ellipsis, per-mille, primes (none is a
///     letter, digit or whitespace; U+2028/2029/202F whitespace are excluded)
/// The resolved letter bytes join `l`; resolved punctuation bytes become the
/// symbol class O (not L/D/WS/NA). Valid for grammars in which a multi-byte
/// character only ever extends a run of its class (Word).
/// Returns the continuation bytes of the resolved characters (all but their
/// lead byte), for grammars whose rules look at the previous CHARACTER.
#[inline]
fn resolve_na<const CONT: bool>(b: &[u8], base: usize, m: &mut Classes) -> u64 {
    let lim = (b.len() - base).min(64);
    let mut r = m.na;
    let (mut ul, mut up, mut cont) = (0u64, 0u64, 0u64);
    while r != 0 {
        let i = r.trailing_zeros() as usize;
        let c = b[base + i];
        if c == 0xC3 && i + 1 < lim {
            let d = b[base + i + 1];
            let w = 3u64 << i;
            if d == 0x97 || d == 0xB7 { up |= w } else { ul |= w }
            if CONT {
                cont |= 2u64 << i;
            }
            r &= !w;
            continue;
        }
        if c == 0xE2 && i + 2 < lim && b[base + i + 1] == 0x80 {
            let d = b[base + i + 2];
            if (0x90..=0xA7).contains(&d) || (0xB0..=0xBE).contains(&d) {
                let w = 7u64 << i;
                up |= w;
                if CONT {
                    cont |= 6u64 << i;
                }
                r &= !w;
                continue;
            }
        }
        r &= r - 1;
    }
    m.l |= ul;
    m.na &= !(ul | up);
    cont
}

#[inline(always)]
fn ctx(b: &[u8], base: usize) -> Ctx {
    ctx_r(b, base, false)
}

#[inline(always)]
fn ctx_r(b: &[u8], base: usize, resolve: bool) -> Ctx {
    if resolve { ctx_m::<true, true>(b, base, classify(b, base)) } else { ctx_m::<false, false>(b, base, classify(b, base)) }
}

/// RESOLVE: classify common non-ASCII in-block; CONT: also track the resolved
/// characters' continuation bytes (only the Generic rules need them).
#[inline(always)]
fn ctx_m<const RESOLVE: bool, const CONT: bool>(b: &[u8], base: usize, mut m: Classes) -> Ctx {
    let n = b.len();
    let mut cont = 0u64;
    if RESOLVE && m.na != 0 {
        cont = resolve_na::<CONT>(b, base, &mut m);
    }
    let valid = if base + 64 <= n { u64::MAX } else { (1u64 << (n - base)) - 1 };
    let next = b.get(base + 64).copied();
    Ctx {
        o: !(m.l | m.d | m.ws | m.na) & valid,
        m,
        valid,
        next_ws: next.map_or(false, is_ws),
        next_na: next.map_or(false, |c| c >= 0x80),
        cont,
    }
}

/// BAD bits common to every grammar: non-ASCII and the 4 bytes after it,
/// 1-byte whitespace next to non-ASCII, and whitespace continuing from the
/// previous block.
#[inline(always)]
fn common_bad(x: &Ctx, c: &Carry) -> u64 {
    let m = &x.m;
    let mut bad = m.na | (m.na << 1) | (m.na << 2) | (m.na << 3) | (m.na << 4);
    // NA at base-k (k = 1..4, bit k-1 of na4) taints bits 0..=4-k; the union is
    // decided by the nearest one: k = tz + 1 -> bits 0..=3-tz. Branch-free
    // (these branches were data-dependent mispredicts).
    bad |= (16u64 >> c.na4.trailing_zeros().min(63)).saturating_sub(1);
    bad |= m.ws & (p1(m.na, c.na) | n1(m.na, x.next_na));
    // whitespace continuing from the previous block: its leading run is BAD.
    // trailing ones of ws = ws & !(ws + 1); selected by c.ws without a branch.
    bad |= m.ws & !m.ws.wrapping_add(1) & (c.ws as u64).wrapping_neg();
    bad
}

/// Digit-group starts (`\p{N}{1,k}`, k = 0 unlimited). Runs after non-ASCII
/// (a Unicode digit might lead the group) or continuing from the previous
/// block are BAD for k >= 2. `prefix_sp`: Word's ` ?\p{N}+` (a space before a
/// digit run joins it, so the run does not start a chunk).
#[inline(always)]
fn digit_starts(m: &Classes, c: &Carry, k: u32, prefix_sp: bool) -> (u64, u64) {
    let runs = m.d & !p1(m.d, c.d);
    let joined = if prefix_sp { p1(m.sp, c.sp) } else { 0 };
    match k {
        0 => (runs & !joined, 0),
        1 => (m.d, 0),
        k => {
            let k = k as usize;
            let (mut st, mut bad) = (0u64, 0u64);
            let after_na = p1(m.na, c.na);
            let mut rr = runs;
            while rr != 0 {
                let r = rr.trailing_zeros() as usize;
                rr &= rr - 1;
                let e = (r + ((!m.d) >> r).trailing_zeros() as usize).min(64);
                if after_na >> r & 1 == 1 {
                    bad |= run_mask(r, e);
                    continue;
                }
                let mut j = r;
                while j < e {
                    st |= 1u64 << j;
                    j += k;
                }
            }
            if c.d {
                let lead = (!m.d).trailing_zeros();
                bad |= if lead >= 64 { u64::MAX } else { (1u64 << lead) - 1 };
            }
            (st, bad)
        }
    }
}

/// Whitespace starts for the Generic/CamelCase grammars (with `\s*[\r\n]+` and
/// NL absorption after O). Returns (starts, bad, absorbed NLs).
#[inline(always)]
fn ws_starts_nl(x: &Ctx, c: &Carry, base: usize, n: usize) -> (u64, u64, u64) {
    let m = &x.m;
    let long = m.ws & (p1(m.ws, c.ws) | n1(m.ws, x.next_ws));
    let o_before = p1(x.o, c.o);
    let single_abs = m.nl & !long & o_before;
    let mut starts = m.ws & !long & !single_abs;
    let (mut bad, mut abs) = (0u64, single_abs);
    let mut runs = long & !p1(long, c.ws);
    while runs != 0 {
        let r = runs.trailing_zeros() as usize;
        runs &= runs - 1;
        let e = r + ((!m.ws) >> r).trailing_zeros() as usize;
        let run = run_mask(r, e);
        let na_before = if r == 0 { c.na } else { m.na >> (r - 1) & 1 == 1 };
        if e >= 64 || na_before || m.na >> e & 1 == 1 {
            bad |= run;
            continue;
        }
        let mut a = r;
        if o_before >> r & 1 == 1 {
            a = r + ((!m.nl) >> r).trailing_zeros() as usize; // absorbed leading NLs
            abs |= run_mask(r, a.min(e));
            if a >= e {
                continue;
            }
        }
        starts |= 1u64 << a;
        let nls = m.nl & run & (u64::MAX << a);
        let t = if nls != 0 {
            let l1 = 64 - nls.leading_zeros() as usize; // one past the last NL
            if l1 < e {
                starts |= 1u64 << l1;
            }
            l1
        } else {
            a
        };
        if e > t + 1 && base + e < n {
            starts |= 1u64 << (e - 1);
        }
    }
    (starts, bad, abs)
}

/// A contraction at Q = bit `i` ending before bit `end`: if Q itself is BAD
/// (its chunk-start / suffix status is uncertain), everything derived from it —
/// its span and the forced boundary after it — is BAD too, here and in the next
/// block.
#[inline(always)]
fn span_bad(bad: &mut u64, bad_next: &mut u64, i: usize, end: usize) {
    if *bad >> i & 1 == 1 {
        *bad |= run_mask(i, (end + 1).min(64));
        if end + 1 > 64 {
            *bad_next |= (1u64 << (end + 1 - 64)) - 1;
        }
    }
}

#[inline(always)]
fn generic_block(b: &[u8], base: usize, max_digits: u32, c: Carry) -> Block {
    let n = b.len();
    // Resolved multi-byte characters (Latin-1 letters, general punctuation)
    // are classified in-block. A letter run may take ONE preceding symbol
    // CHARACTER as prefix, so "did the previous symbol start a chunk" is a
    // per-character question: smear each symbol start over its character's
    // continuation bytes (resolved characters are at most 3 bytes).
    let x = ctx_r(b, base, true);
    let (m, o) = (&x.m, x.o);
    let ostart = o & !p1(o, c.o) & !p1(m.sp, c.sp);
    let ostart_c = ostart | ((ostart << 1) & x.cont);
    let ostart_c = ostart_c | ((ostart_c << 1) & x.cont);
    let lstart = m.l & (p1(m.d, c.d) | p1(m.nl, c.nl) | (p1(o, c.o) & !p1(ostart_c, c.ostart)));
    let (dstart, dbad) = digit_starts(m, &c, max_digits, false);
    let (wsstart, wsbad, _) = ws_starts_nl(&x, &c, base, n);
    let mut starts = ostart | lstart | dstart | wsstart | c.forced;
    let mut forced = 0u64;
    let mut bad = common_bad(&x, &c) | dbad | wsbad | c.bad_in;
    let mut bad_next = 0u64;
    let mut qc = m.q & ostart;
    while qc != 0 {
        let i = qc.trailing_zeros() as usize;
        qc &= qc - 1;
        let len = contraction_len(b, base + i, true);
        if len != 0 {
            if i + len < 64 {
                starts |= 1u64 << (i + len);
            } else {
                forced |= 1u64 << (i + len - 64);
            }
            span_bad(&mut bad, &mut bad_next, i, i + len);
        }
    }
    // a letter after a symbol reads that symbol's start status: uncertain if any
    // byte of the symbol character is (smear over continuation bytes first)
    let bad_c = bad | ((bad << 1) & x.cont);
    let bad_c = bad_c | ((bad_c << 1) & x.cont);
    bad |= m.l & p1(o, c.o) & p1(bad_c, c.bad3 >> 2 & 1 == 1); // reads the O's start status
    let bad = bad | ((bad << 1) & x.cont) | ((bad << 2) & x.cont & (x.cont << 1));
    let mut out = carry_out(&x, ostart_c, 0, false, forced, 0);
    out.bad_in = bad_next;
    out.bad3 = bad >> 61;
    (starts & x.valid, bad & x.valid, out)
}

#[inline(always)]
fn camel_block(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: Carry) -> Block {
    let n = b.len();
    let x = ctx(b, base);
    let (m, o) = (&x.m, x.o);
    let lw = m.l & !m.up;
    let (wsstart, wsbad, abs_nl) = ws_starts_nl(&x, &c, base, n);
    // `[\r\n/]*` tail: a run of '/' right after an absorbed NL is absorbed too
    // (run-fill: adding the run's first bit carries through the whole run)
    let seed = m.sl & p1(abs_nl, c.abs_nl || c.abs_sl);
    let abs_sl = (m.sl.wrapping_add(seed) ^ m.sl) & m.sl;
    // after the tail, a symbol (other than '/') starts a new chunk
    let mut ostart = (o & !p1(o, c.o) & !p1(m.sp, c.sp) & !abs_sl) | (o & !m.sl & p1(abs_sl, c.abs_sl));
    // contraction suffixes: Q right after a letter chunk (not after another suffix)
    let (mut suppress, mut forced_in, mut forced, mut sup_next) = (c.suppress, 0u64, 0u64, 0u64);
    let (dstart, dbad) = digit_starts(m, &c, max_digits, false);
    let mut bad = common_bad(&x, &c) | dbad | wsbad | c.bad_in;
    // A whitespace run of uncertain extent may or may not have been absorbed
    // into a symbol tail, so a '/' run right after it may or may not continue
    // that tail: the '/' run and the byte after it are uncertain too.
    let ws_unk = m.ws & bad;
    let useed = m.sl & p1(ws_unk, c.ws_unk) | (m.sl & 1 & c.sl_unk as u64);
    let sl_unk = (m.sl.wrapping_add(useed) ^ m.sl) & m.sl;
    bad |= sl_unk | (p1(sl_unk, c.sl_unk) & !m.sl);
    let mut bad_next = 0u64;
    let mut contr_end = c.contr_mask; // last byte of each contraction suffix
    let mut contr_next = 0u64;
    if contractions {
        let mut qs = m.q & p1(m.l, c.l);
        while qs != 0 {
            let i = qs.trailing_zeros() as usize;
            qs &= qs - 1;
            // Whether this Q is a suffix depends on whether the letter before it
            // ended an earlier suffix (decided 2-3 bytes back). If anything in
            // bytes i-3..=i is uncertain, so is this Q: mark its whole possible
            // span (+ the boundary after it) BAD — which in turn makes the next
            // Q in a chain uncertain, so the uncertainty propagates correctly.
            let window = (bad << 3 | c.bad3) >> i & 0b1111;
            if window != 0 {
                span_bad(&mut bad, &mut bad_next, i, i + 3);
                bad |= 1u64 << i;
                span_bad(&mut bad, &mut bad_next, i, i + 3);
            }
            // the letter before must belong to a letter chunk, not end a suffix
            let prev_is_suffix_end = if i == 0 { c.contr } else { contr_end >> (i - 1) & 1 == 1 };
            if prev_is_suffix_end {
                continue;
            }
            let len = contraction_len(b, base + i, true);
            if len == 0 {
                continue;
            }
            let end = i + len; // first byte after the suffix
            suppress |= run_mask(i, end.min(64));
            if end > 64 {
                sup_next |= (1u64 << (end - 64)) - 1;
            }
            if end - 1 < 64 {
                contr_end |= 1u64 << (end - 1);
            } else {
                contr_next |= 1u64 << (end - 1 - 64);
            }
            if end < 64 {
                forced_in |= 1u64 << end;
            } else {
                forced |= 1u64 << (end - 64);
            }
            span_bad(&mut bad, &mut bad_next, i, end);
        }
    }
    ostart &= !suppress;
    let ostart_p1 = p1(ostart, c.ostart);
    let lstart = m.l
        & (p1(m.d, c.d) | p1(m.nl, c.nl) | (p1(o, c.o) & !ostart_p1) | (m.up & p1(lw, c.lw)));
    let starts = ((ostart | lstart | dstart | wsstart) & !suppress) | c.forced | forced_in;
    bad |= m.l & p1(o, c.o) & p1(bad, c.bad3 >> 2 & 1 == 1); // reads the O's start status
    let mut out = carry_out(&x, ostart, abs_nl, contr_end >> 63 & 1 == 1, forced, sup_next);
    out.contr_mask = contr_next;
    out.abs_sl = abs_sl >> 63 & 1 == 1;
    out.bad_in = bad_next;
    out.bad3 = bad >> 61;
    out.ws_unk = ws_unk >> 63 & 1 == 1;
    out.sl_unk = sl_unk >> 63 & 1 == 1;
    (starts & x.valid, bad & x.valid, out)
}

/// The GPT-2 rules for a pure-ASCII block under a clean carry: word-wide bit
/// operations; only whitespace runs touching the block edges are uncertain.
/// None if a contraction runs into the next block (the full rules handle it).
#[inline(always)]
fn word_fast(b: &[u8], base: usize, indiv: bool, c: &Carry, m: &Classes) -> Option<Block> {
    let o = !(m.l | m.d | m.ws);
    let sp_before = p1(m.sp, c.sp);
    let ostart = o & !p1(o, c.o) & !sp_before;
    let lstart = m.l & !p1(m.l, c.l) & !sp_before;
    let dstart = if indiv { m.d } else { m.d & !p1(m.d, c.d) & !sp_before };
    let mut starts = ostart | lstart | dstart;
    let mut qc = m.q & ostart;
    let mut suppress = 0u64;
    while qc != 0 {
        let i = qc.trailing_zeros() as usize;
        qc &= qc - 1;
        let len = contraction_len(b, base + i, false);
        if len != 0 {
            let end = i + len;
            if end >= 64 {
                return None; // the contraction runs into the next block
            }
            suppress |= run_mask(i + 1, end);
            starts |= 1u64 << end;
        }
    }
    let next = b[base + 64];
    let (next_ws, next_na) = (is_ws(next), next >= 0x80);
    let long = m.ws & (p1(m.ws, c.ws) | n1(m.ws, next_ws));
    let cont = m.ws & !m.ws.wrapping_add(1) & (c.ws as u64).wrapping_neg();
    let lr = long & !cont;
    let r = lr & !(lr << 1);
    let last = lr & !(lr >> 1);
    // only the run reaching bit 63 can be uncertain (its end is in the
    // next block): smear its top bit down over the run
    let top = last & (1u64 << 63);
    let wsbad = (lr.reverse_bits().wrapping_add(top.reverse_bits()) ^ lr.reverse_bits()).reverse_bits() & lr;
    let good = lr & !wsbad;
    let no_digit_rule = if indiv { !(m.d >> 1) } else { u64::MAX };
    starts |= (m.ws & !long) | (r & good) | (last & good & (u64::MAX >> 1) & no_digit_rule);
    starts &= !suppress;
    let bad = cont | wsbad | (m.ws & ((next_na as u64) << 63));
    let top1 = |v: u64| v >> 63 & 1 == 1;
    let out = Carry {
        o: top1(o),
        sp: top1(m.sp),
        d: top1(m.d),
        ws: top1(m.ws),
        l: top1(m.l),
        bad3: bad >> 61,
        ..Carry::default()
    };
    Some((starts, bad, out))
}

#[inline(always)]
fn word_block(b: &[u8], base: usize, indiv: bool, c: Carry) -> Block {
    let m = classify(b, base);
    if base + 64 < b.len() && !c.na && (c.na4 | c.bad_in | c.suppress | c.forced) == 0 && m.na == 0 {
        if let Some(r) = word_fast(b, base, indiv, &c, &m) {
            return r;
        }
    }
    word_slow(b, base, indiv, c, m)
}

#[inline(never)]
fn word_slow(b: &[u8], base: usize, indiv: bool, c: Carry, m: Classes) -> Block {
    let n = b.len();
    let x = ctx_m::<true, false>(b, base, m);
    let (m, o) = (&x.m, x.o);
    let sp_before = p1(m.sp, c.sp);
    let ostart = o & !p1(o, c.o) & !sp_before;
    let lstart = m.l & !p1(m.l, c.l) & !sp_before;
    let (dstart, dbad) = if indiv { (m.d, 0) } else { digit_starts(m, &c, 0, true) };
    // whitespace: `\s+(?!\S)|\s+` — run start, and e-1 when >= 2 long and text
    // follows (unless individual digits and a digit follows: no back-off)
    //
    // Branch-free over all runs at once (no per-run loop):
    //   lr    = bits of the long runs that START in this block (a run continuing
    //           from the previous block is left to common_bad, as before)
    //   r     = first byte of each such run, last = its last byte in the word
    //   a run is BAD if it reaches bit 63 (its end is in the next block), or a
    //   non-ASCII byte sits right before it or right after it (it might be
    //   Unicode whitespace). Flags at the start are smeared over the run with
    //   a carry: ((lr + flag) ^ lr) & lr; flags at the end the same way on the
    //   bit-reversed words.
    //   good runs start a chunk at r, and at last (= e-1) when text follows
    //   (e < n) — unless individual digits and a digit follows.
    let long = m.ws & (p1(m.ws, c.ws) | n1(m.ws, x.next_ws));
    let cont = m.ws & !m.ws.wrapping_add(1) & (c.ws as u64).wrapping_neg();
    let lr = long & !cont;
    let r = lr & !(lr << 1);
    let last = lr & !(lr >> 1);
    let flag_start = r & p1(m.na, c.na);
    let flag_end = (last & (m.na >> 1)) | (last & (1u64 << 63));
    let smear_up = |runs: u64, f: u64| (runs.wrapping_add(f) ^ runs) & runs;
    let wsbad = smear_up(lr, flag_start)
        | smear_up(lr.reverse_bits(), flag_end.reverse_bits()).reverse_bits();
    let good = lr & !wsbad;
    let text_follows = x.valid >> 1; // byte i+1 exists
    let no_digit_rule = if indiv { !(m.d >> 1) } else { u64::MAX };
    let wsstart = (m.ws & !long) | (r & good) | (last & good & text_follows & no_digit_rule);
    let _ = (base, n);
    let mut starts = ostart | lstart | dstart | wsstart | c.forced;
    let (mut suppress, mut forced, mut sup_next) = (c.suppress, 0u64, 0u64);
    let mut bad = common_bad(&x, &c) | dbad | wsbad | c.bad_in;
    let mut bad_next = 0u64;
    let mut qc = m.q & ostart;
    while qc != 0 {
        let i = qc.trailing_zeros() as usize;
        qc &= qc - 1;
        let len = contraction_len(b, base + i, false);
        if len != 0 {
            let end = i + len;
            suppress |= run_mask(i + 1, end.min(64));
            if end > 64 {
                sup_next |= (1u64 << (end - 64)) - 1;
            }
            if end < 64 {
                starts |= 1u64 << end;
            } else {
                forced |= 1u64 << (end - 64);
            }
            span_bad(&mut bad, &mut bad_next, i, end);
        }
    }
    starts &= !suppress;
    let _ = n;
    // Only the fields the Word rules read: building the shared carry here made
    // the classifier compute the NL and uppercase masks (for c.nl / c.lw, which
    // only the Generic/CamelCase rules use) on every block.
    let top = |v: u64| v >> 63 & 1 == 1;
    let out = Carry {
        o: top(x.o),
        sp: top(m.sp),
        d: top(m.d),
        ws: top(m.ws),
        l: top(m.l),
        na: top(m.na),
        na4: (m.na >> 60).reverse_bits() >> 60,
        bad_in: bad_next,
        bad3: bad >> 61,
        forced,
        suppress: sup_next,
        ..Carry::default()
    };
    (starts & x.valid, bad & x.valid, out)
}

#[inline(always)]
fn carry_out(x: &Ctx, ostart: u64, abs_nl: u64, contr: bool, forced: u64, suppress: u64) -> Carry {
    let m = &x.m;
    let top = |v: u64| v >> 63 & 1 == 1;
    Carry {
        o: top(x.o),
        sp: top(m.sp),
        d: top(m.d),
        nl: top(m.nl),
        ws: top(m.ws),
        l: top(m.l),
        lw: top(m.l & !m.up),
        na: top(m.na),
        na4: (m.na >> 60).reverse_bits() >> 60, // bit k-1 = byte 64-k
        ostart: top(ostart),
        abs_nl: top(abs_nl),
        abs_sl: false,
        bad_in: 0,
        bad3: 0,
        ws_unk: false,
        sl_unk: false,
        contr,
        contr_mask: 0,
        forced,
        suppress,
    }
}

#[inline(always)]
#[allow(dead_code)]
pub fn block(kind: Pretok, b: &[u8], base: usize, c: Carry) -> Block {
    let mut c = c;
    let (s, bad) = block_m(kind, b, base, &mut c);
    (s, bad, c)
}

/// One block with the carry updated IN PLACE, returning (starts, bad) in two
/// registers. Returning the ~72-byte carry by value made every driver unpack
/// it, repack it to the stack before the call and reload it after (~60
/// instructions per block, as much as the GPT-2 rules themselves).
#[inline(always)]
pub fn block_m(kind: Pretok, b: &[u8], base: usize, c: &mut Carry) -> (u64, u64) {
    match kind {
        Pretok::Generic { max_digits, .. } => generic_m(b, base, max_digits, c),
        Pretok::CamelCase { contractions, max_digits } => camel_m(b, base, contractions, max_digits, c),
        Pretok::Word(indiv) => word_m(b, base, indiv, c),
    }
}

#[inline(never)]
fn word_m(b: &[u8], base: usize, indiv: bool, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = word_block(b, base, indiv, *c);
    *c = out;
    (s, bad)
}

#[inline(never)]
fn generic_m(b: &[u8], base: usize, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = generic_block(b, base, max_digits, *c);
    *c = out;
    (s, bad)
}

#[inline(never)]
fn camel_m(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = camel_block(b, base, contractions, max_digits, *c);
    *c = out;
    (s, bad)
}

/// Iterate the chunks of [start, stop) (both proven chunk boundaries) for any
/// ByteLevel grammar, handing over each block's chunks at once: `f(first,
/// ends)` covers [first, ends[0]), [ends[0], ends[1]), … Exactly the
/// boundaries of repeated `next_chunk_end`.
#[inline(always)]
pub fn chunks_batched(kind: Pretok, b: &[u8], start: usize, stop: usize, mut fb: impl FnMut(usize, &[usize])) {
    let mut ends = [0usize; 72]; // 64 ends + 8 slots of flattening slack
    let mut pos = start;
    let mut base = start;
    // the scan start is a known boundary: no context, forced start at bit 0
    let mut carry = Carry { forced: 1, ..Carry::default() };
    while pos < stop {
        if base + 64 <= pos {
            // a serial step jumped past this block: restart the block scan at
            // pos, itself a proven chunk boundary (no context needed)
            base = pos;
            carry = Carry { forced: 1, ..Carry::default() };
        }
        let (starts, bad) = block_m(kind, b, base, &mut carry);
        let lim = (base + 64).min(stop);
        let lim_mask = if lim - base >= 64 { u64::MAX } else { (1u64 << (lim - base)) - 1 };
        let above0 = if pos < base { u64::MAX } else { u64::MAX.checked_shl((pos - base + 1) as u32).unwrap_or(0) };
        let mut trusted = starts & !bad & lim_mask & above0;
        let bad = bad & lim_mask;
        if bad & above0 == 0 {
            // common case: every boundary in this block is trusted. Flatten the
            // bitmask simdjson-style: 8 positions per round written
            // unconditionally (slots past the count are garbage, overwritten
            // or ignored), so the only branches are on the count, not per bit.
            let first = pos;
            let cnt = trusted.count_ones() as usize;
            let mut k = 0usize;
            loop {
                for j in 0..8 {
                    ends[k + j] = base + trusted.trailing_zeros() as usize;
                    trusted &= trusted.wrapping_sub(1);
                }
                k += 8;
                if k >= cnt {
                    break;
                }
            }
            k = cnt;
            if k > 0 {
                pos = ends[k - 1];
            }
            if lim == stop && pos < stop {
                ends[k] = stop;
                k += 1;
                pos = stop;
            }
            if k > 0 {
                fb(first, &ends[..k]);
            }
        } else {
            loop {
                if pos >= lim {
                    break;
                }
                let above = if pos < base {
                    u64::MAX
                } else {
                    u64::MAX.checked_shl((pos - base + 1) as u32).unwrap_or(0)
                };
                let cand = trusted & above;
                let badm = bad & above;
                if cand != 0 {
                    let c = cand.trailing_zeros();
                    if badm & ((1u64 << c) - 1) == 0 {
                        let end = base + c as usize;
                        fb(pos, &[end]);
                        pos = end;
                        continue;
                    }
                } else if badm == 0 {
                    if lim == stop {
                        fb(pos, &[stop]);
                        pos = stop;
                    }
                    break; // chunk continues into the next block
                }
                // uncertain zone before the next trusted boundary: exact serial step
                let end = next_chunk_end(kind, b, pos);
                fb(pos, &[end]);
                pos = end;
            }
        }
        base += 64;
    }
}

/// Set-bit positions of every byte value as 8 u32 lanes (unused lanes 0).
static POS32: [[u32; 8]; 256] = {
    let mut t = [[0u32; 8]; 256];
    let mut v = 0;
    while v < 256 {
        let (mut x, mut j, mut bit) = (v, 0usize, 0u32);
        while x != 0 {
            if x & 1 == 1 {
                t[v][j] = bit;
                j += 1;
            }
            x >>= 1;
            bit += 1;
        }
        v += 1;
    }
    t
};

/// Slack `Scan::fill` needs past `limit` in its output buffer.
pub const SCAN_SLACK: usize = 80;

/// Resumable batch scanner: appends chunk END positions (u32) of many blocks
/// to one buffer, so the consumer runs one long loop per batch. A block's ends
/// come from a table (8 positions per mask byte, stored whole at the byte's
/// exclusive prefix popcount; all prefix sums from one multiply), with no loop
/// over bits; the block's last end is computed from the mask, never read back
/// from those overlapping vector stores (that read stalls on forwarding).
pub struct Scan {
    pos: usize,
    base: usize,
    carry: Carry,
}

impl Scan {
    pub fn new(start: usize) -> Self {
        Scan { pos: start, base: start, carry: Carry { forced: 1, ..Carry::default() } }
    }

    /// Append ends to `out[k..]` until `k >= limit` (checked per block) or
    /// `stop`; returns the new length. `out.len() >= limit + SCAN_SLACK`,
    /// `b.len() <= u32::MAX`.
    #[inline]
    pub fn fill(&mut self, kind: Pretok, b: &[u8], stop: usize, out: &mut [u32], mut k: usize, limit: usize) -> usize {
        assert!(out.len() >= limit + SCAN_SLACK);
        let (mut pos, mut base, mut carry) = (self.pos, self.base, self.carry);
        let op = out.as_mut_ptr();
        while pos < stop && k < limit {
            if base + 64 <= pos {
                base = pos;
                carry = Carry { forced: 1, ..Carry::default() };
            }
            let (starts, bad) = block_m(kind, b, base, &mut carry);
            let lim = (base + 64).min(stop);
            let lim_mask = if lim - base >= 64 { u64::MAX } else { (1u64 << (lim - base)) - 1 };
            let above0 = if pos < base { u64::MAX } else { u64::MAX.checked_shl((pos - base + 1) as u32).unwrap_or(0) };
            let trusted = starts & !bad & lim_mask & above0;
            let bad = bad & lim_mask;
            if bad & above0 == 0 {
                let x = trusted - ((trusted >> 1) & 0x5555_5555_5555_5555);
                let x = (x & 0x3333_3333_3333_3333) + ((x >> 2) & 0x3333_3333_3333_3333);
                let cnts = (x + (x >> 4)) & 0x0F0F_0F0F_0F0F_0F0F;
                let incl = cnts.wrapping_mul(0x0101_0101_0101_0101);
                let excl = incl << 8;
                let b32 = base as u32;
                for j in 0..8 {
                    let row = &POS32[(trusted >> (8 * j)) as u8 as usize];
                    let add = b32 + 8 * j as u32;
                    let mut v = [0u32; 8];
                    for t in 0..8 {
                        v[t] = row[t] + add;
                    }
                    let at = k + ((excl >> (8 * j)) as u8 as usize);
                    // SAFETY: k < limit at block start; at + 8 <= k + 64 <= out.len().
                    unsafe { (op.add(at) as *mut [u32; 8]).write_unaligned(v) };
                }
                k += (incl >> 56) as usize;
                if trusted != 0 {
                    pos = base + 63 - trusted.leading_zeros() as usize;
                }
                if lim == stop && pos < stop {
                    out[k] = stop as u32;
                    k += 1;
                    pos = stop;
                }
            } else {
                loop {
                    if pos >= lim {
                        break;
                    }
                    let above = if pos < base { u64::MAX } else { u64::MAX.checked_shl((pos - base + 1) as u32).unwrap_or(0) };
                    let cand = trusted & above;
                    let badm = bad & above;
                    if cand != 0 {
                        let c = cand.trailing_zeros();
                        if badm & ((1u64 << c) - 1) == 0 {
                            let end = base + c as usize;
                            out[k] = end as u32;
                            k += 1;
                            pos = end;
                            continue;
                        }
                    } else if badm == 0 {
                        if lim == stop {
                            out[k] = stop as u32;
                            k += 1;
                            pos = stop;
                        }
                        break;
                    }
                    let end = next_chunk_end(kind, b, pos);
                    out[k] = end as u32;
                    k += 1;
                    pos = end;
                    if k >= limit + SCAN_SLACK - 1 {
                        // stop mid-block: resume by restarting the block scan
                        // at pos, a proven chunk boundary (the carry has
                        // already advanced past this block)
                        self.pos = pos;
                        self.base = pos;
                        self.carry = Carry { forced: 1, ..Carry::default() };
                        return k;
                    }
                }
            }
            base += 64;
        }
        self.pos = pos;
        self.base = base;
        self.carry = carry;
        k
    }
}

/// Per-chunk form of `chunks_batched` (tests / simple callers).
#[cfg(test)]
pub fn chunks(kind: Pretok, b: &[u8], start: usize, stop: usize, mut f: impl FnMut(usize, usize)) {
    chunks_batched(kind, b, start, stop, |first, ends| {
        let mut a = first;
        for &e in ends {
            f(a, e);
            a = e;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretok::slow_chunk_end;

    fn lcg(s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s >> 33
    }

    fn iters() -> usize {
        std::env::var("BLOCKSCAN_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20000)
    }

    const KINDS: [Pretok; 11] = [
        Pretok::Generic { max_digits: 0, marks: false },
        Pretok::Generic { max_digits: 1, marks: false },
        Pretok::Generic { max_digits: 3, marks: false },
        Pretok::Generic { max_digits: 3, marks: true },
        Pretok::CamelCase { contractions: true, max_digits: 3 },
        Pretok::CamelCase { contractions: false, max_digits: 3 },
        Pretok::CamelCase { contractions: true, max_digits: 0 },
        Pretok::CamelCase { contractions: true, max_digits: 1 },
        Pretok::Word(false),
        Pretok::Word(true),
        Pretok::Generic { max_digits: 2, marks: false },
    ];

    #[test]
    fn simd_classify_matches_scalar() {
        let mut seed = 3u64;
        for _ in 0..20000 {
            let v: Vec<u8> = (0..80).map(|_| (lcg(&mut seed) & 0xFF) as u8).collect();
            assert_eq!(classify(&v, 0), classify_scalar(&v, 0));
            assert_eq!(classify(&v, 16), classify_scalar(&v, 16));
            assert_eq!(classify(&v, 40), classify_scalar(&v, 40)); // tail block
        }
    }

    fn first_diff(got: &[(usize, usize)], want: &[(usize, usize)], b: &[u8]) -> String {
        let i = got.iter().zip(want).position(|(g, w)| g != w).unwrap_or(got.len().min(want.len()));
        let a = want.get(i).or(got.get(i)).map(|x| x.0).unwrap_or(0);
        let lo = a.saturating_sub(12);
        let hi = (a + 12).min(b.len());
        format!(
            "first diff #{i}: got {:?} want {:?}; bytes[{lo}..{hi}] = {:?} (pos {a}, block offset {})",
            got.get(i), want.get(i), String::from_utf8_lossy(&b[lo..hi]), a % 64
        )
    }

    fn random_text(seed: &mut u64) -> String {
        const POOL: &[&str] = &[
            "a", "b", "Z", "Q", "hello", "World", "HTTPServer", "camelCase", "aB", "x", "'s",
            "'S", "'t", "'ll", "'LL", "'lL", "'ve", "'Re", "'d", "'m", "'M", "'", "''", "'x",
            "0", "7", "123", "4567", "89012", " ", "  ", "   ", "\t", "\n", "\r\n", "\r", "\x0b",
            "\x0c", " \n", "\n\n", ".", ",", "!", "?", "/", "//", "-", "(", ")", "\"", "#",
            "\x01", "\x7f", "é", "É", "漢", "\u{0301}", "\u{00a0}", "\u{3000}", "\u{0663}",
            "½", "€", "😀", "\u{0085}", "\u{2028}", "ǅ", "ʰ",
            "’", "‘", "“", "”", "—", "–", "…", "•", "‰", "‾", "‿", "×", "÷", "ß", "ÿ", "À",
            "Ö", "ø", "\u{2000}", "\u{200a}", "\u{200b}", "\u{2029}", "\u{202f}",
            "\u{205f}", "\u{2070}", "\u{2071}", "don’t", " “Hi”", "café", " é", "’s", "a’",
        ];
        let n = (lcg(seed) % 200) as usize;
        (0..n).map(|_| POOL[(lcg(seed) as usize) % POOL.len()]).collect()
    }

    /// Mostly-ASCII prose/code-like text: exercises the mask rules themselves
    /// (few BAD zones), long enough to cross many block edges.
    fn ascii_text(seed: &mut u64) -> String {
        const POOL: &[&str] = &[
            " the", " quick", " Brown", " fox", "jumps", " over", ",", ".", "!", "?", " (", ")",
            "\"", " \"", "'s", "'t", "'re", "'LL", "'lL", "'d", "'m", " '", "''", "'x", "don",
            "it", " 1", "2", " 2024", "123456", "7", "\n", "\n\n", " \n", "\t", "  ", "   ",
            "\r\n", "#", "//", "/", ";\n//", "->", "==", " = ", "x_y", "(a)", "[0]", "{", "}",
            ";", ":", "--", "...", "a", "B", "Z", "'", " .", "\x0b", "\x0c", "HTTPServer",
            "camelCase", "getX", "ABc", "é", "\u{a0}", "½", "’", " “", "”", "—", " –",
            "…", "×", "÷", "café", "Ö", "\u{2028}", "\u{202f}", "\u{200b}", "’s", "‘",
        ];
        let n = (lcg(seed) % 600) as usize;
        (0..n)
            .map(|_| {
                let i = (lcg(seed) as usize) % POOL.len();
                if POOL[i].is_ascii() || lcg(seed) % 8 == 0 { POOL[i] } else { " " }
            })
            .collect()
    }

    fn check(t: &str) {
        let b = t.as_bytes();
        for &kind in &KINDS {
            let mut want = Vec::new();
            let mut pos = 0;
            while pos < b.len() {
                let e = slow_chunk_end(kind, b, pos);
                want.push((pos, e));
                pos = e;
            }
            let mut got = Vec::new();
            chunks(kind, b, 0, b.len(), |a, e| got.push((a, e)));
            assert!(got == want, "{kind:?} {}", first_diff(&got, &want, b));
            for lim in [7usize, 512] {
                let mut sc = Scan::new(0);
                let mut out = vec![0u32; lim + SCAN_SLACK];
                let mut got3 = Vec::new();
                let mut a = 0usize;
                loop {
                    let k = sc.fill(kind, b, b.len(), &mut out, 0, lim);
                    if k == 0 { break; }
                    for &e in &out[..k] { got3.push((a, e as usize)); a = e as usize; }
                }
                assert!(got3 == want, "Scan[{lim}] {kind:?} {}", first_diff(&got3, &want, b));
            }
            for frac in [5usize, 2] {
                if want.len() > 10 {
                    let s0 = want[want.len() / frac].0;
                    let s1 = want[want.len() - 2].0;
                    let mut got2 = Vec::new();
                    chunks(kind, b, s0, s1, |a, e| got2.push((a, e)));
                    let want2: Vec<_> = want.iter().copied().filter(|&(a, _)| a >= s0 && a < s1).collect();
                    assert!(got2 == want2, "[{s0},{s1}) {kind:?} {}", first_diff(&got2, &want2, b));
                }
            }
        }
    }

    /// Block scanner boundaries == serial scan boundaries for every grammar
    /// variant, from arbitrary proven starts, across block edges.
    #[test]
    fn block_scan_equals_serial() {
        let mut seed = 99u64;
        for _ in 0..iters() {
            check(&random_text(&mut seed));
        }
    }

    /// Dense single-byte soup of exactly the characters the subtle rules hinge
    /// on: '/', NLs, apostrophes, contraction letters in both cases, digits,
    /// spaces, tabs.
    fn soup_text(seed: &mut u64) -> String {
        const C: &[u8] = b"/ /\n\r''sStTlLeErRvVdDmMaAxX19 \t.,;\x0b\x0c";
        let n = (lcg(seed) % 400) as usize;
        (0..n).map(|_| C[(lcg(seed) as usize) % C.len()] as char).collect()
    }

    #[test]
    fn block_scan_equals_serial_soup() {
        let mut seed = 777u64;
        for _ in 0..iters() {
            check(&soup_text(&mut seed));
        }
    }

    /// Dense soup mixing the resolved non-ASCII characters (Latin-1 letters,
    /// general punctuation) with the ASCII rule characters and the unresolved
    /// whitespace / format characters right next to them.
    #[test]
    fn block_scan_equals_serial_unisoup() {
        const C: &[char] = &['’', '“', '—', '…', 'é', 'Ö', '×', '÷', ' ', ' ', 'a', 's', 't', 'Z', '1', '9',
            '.', '\'', '\n', '\r', '\t', '/', '\u{a0}', '\u{2028}', '\u{202f}', '\u{200b}', '\u{2000}', '漢'];
        let mut seed = 4242u64;
        for _ in 0..iters() {
            let n = (lcg(&mut seed) % 300) as usize;
            let t: String = (0..n).map(|_| C[(lcg(&mut seed) as usize) % C.len()]).collect();
            check(&t);
        }
    }

    #[test]
    fn block_scan_equals_serial_ascii() {
        let mut seed = 1234u64;
        for _ in 0..iters() {
            check(&ascii_text(&mut seed));
        }
    }
}

#[cfg(test)]
mod perf {
    use super::*;
    /// `BLOCKSCAN_CORPUS=path cargo test --release blockscan_perf -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn blockscan_perf() {
        let path = std::env::var("BLOCKSCAN_CORPUS").expect("set BLOCKSCAN_CORPUS");
        let data = std::fs::read(path).unwrap();
        let b = &data[..data.len().min(30_000_000)];
        for kind in [
            Pretok::Generic { max_digits: 3, marks: false },
            Pretok::CamelCase { contractions: true, max_digits: 3 },
            Pretok::Word(false),
        ] {
            let mut best = (f64::MAX, f64::MAX);
            let (mut n1, mut n2) = (0usize, 0usize);
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let mut pos = 0usize;
                n1 = 0;
                while pos < b.len() {
                    pos = next_chunk_end(kind, b, pos);
                    n1 += 1;
                }
                best.0 = best.0.min(t.elapsed().as_secs_f64());
                let t = std::time::Instant::now();
                n2 = 0;
                chunks(kind, b, 0, b.len(), |_, _| n2 += 1);
                best.1 = best.1.min(t.elapsed().as_secs_f64());
            }
            let (mut badn, mut base, mut c) = (0u64, 0usize, Carry::default());
            while base < b.len() {
                let (_, bd, nc) = block(kind, b, base, c);
                badn += bd.count_ones() as u64;
                c = nc;
                base += 64;
            }
            println!(
                "{kind:?}: serial {:.0} MB/s | block {:.0} MB/s ({n1} vs {n2} chunks) | bad bytes {:.2}%",
                b.len() as f64 / best.0 / 1e6,
                b.len() as f64 / best.1 / 1e6,
                badn as f64 / b.len() as f64 * 100.0
            );
        }
    }
}
