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
//! SIMD tiers:
//! * `aarch64`: NEON (`vld4q_u8` + `sri` bitmask fold + SIMD Latin-1 resolver).
//! * `x86_64`: runtime-dispatched AVX-512BW+VL (`vpcompressd` extraction, ~18
//!   zmm ops/block) -> AVX2+BMI1/2+POPCNT -> portable SSE2 baseline.
//! * Everything else: plain Rust scalar.
//!
//! ## Generic (Llama-3, Qwen, cl100k, GLM, OLMo, …)
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,k}|
//!  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! ## CamelCase (o200k, MiniMax, …)
//! Generic, except: inside a letter run a chunk starts only at a lower->upper
//! transition; a contraction is a SUFFIX of the preceding letter chunk (`it's`
//! is one chunk) and ends it; the symbol tail is `[\r\n/]*`, so a '/' right
//! after an absorbed NL is absorbed too.
//!
//! ## Word (GPT-2 family)
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

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

/// Reference classification (also used for the tail block on non-SIMD targets).
/// Bytes at or past `b.len()` get no class bits.
#[allow(dead_code)]
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

// ---------------------------------------------------------------------------
// SIMD classification + branch-free in-block Latin-1 (0xC3) resolution
// ---------------------------------------------------------------------------

/// Internal SIMD backend trait. Every method is `#[inline(always)]` so calling
/// generic block/driver functions from a `#[target_feature]` entry point compiles
/// the entire loop with those CPU features enabled (zero per-block dispatch).
trait Simd: Copy {
    /// Classify 64 readable bytes at `p`.
    unsafe fn classify64(p: *const u8) -> Classes;

    /// Branch-free SIMD resolver for blocks whose non-ASCII bytes are purely
    /// in-block Latin-1 `0xC3 xx` pairs (U+00C0..U+00FF — the vast majority of
    /// non-ASCII blocks in European prose).
    ///
    /// If every set bit in `m.na` belongs to a disjoint `0xC3 xx` pair inside
    /// `[0, 64)`, updates `m.l`, `m.up`, sets `m.na = 0`, and returns
    /// `(cont, punct_pair)` where `cont` is the follower-byte mask (`c3 << 1`)
    /// and `punct_pair` is the mask of `×`/`÷` pairs. Otherwise leaves `m`
    /// untouched and returns `(0, 0)` (so `m.na` stays non-zero).
    unsafe fn resolve_latin1_64(p: *const u8, m: &mut Classes) -> (u64, u64);

    /// Extract set-bit positions of `trusted` (adding `base`) into `out[k..]`,
    /// returning the updated `k`.
    unsafe fn extract_ends(trusted: u64, base: usize, op: *mut u32, k: usize) -> usize {
        extract_ends_pos32(trusted, base, op, k)
    }
}

#[inline(always)]
unsafe fn extract_ends_pos32(trusted: u64, base: usize, op: *mut u32, k: usize) -> usize {
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
        (op.add(at) as *mut [u32; 8]).write_unaligned(v);
    }
    k + (incl >> 56) as usize
}

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct Neon;

#[cfg(target_arch = "aarch64")]
impl Neon {
    #[inline(always)]
    unsafe fn tomask(a: [std::arch::aarch64::uint8x16_t; 4]) -> u64 {
        use std::arch::aarch64::*;
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
        let t4 = vshrn_n_u16::<4>(vreinterpretq_u16_u8(t3));
        vget_lane_u64::<0>(vreinterpret_u64_u8(t4))
    }
}

#[cfg(target_arch = "aarch64")]
impl Simd for Neon {
    #[inline(always)]
    unsafe fn classify64(p: *const u8) -> Classes {
        use std::arch::aarch64::*;
        let v = vld4q_u8(p);
        let x = [v.0, v.1, v.2, v.3];
        let k = |v: u8| vdupq_n_u8(v);
        Classes {
            l: Self::tomask(x.map(|v| vcltq_u8(vsubq_u8(vorrq_u8(v, k(0x20)), k(b'a')), k(26)))),
            up: Self::tomask(x.map(|v| vcltq_u8(vsubq_u8(v, k(b'A')), k(26)))),
            d: Self::tomask(x.map(|v| vcltq_u8(vsubq_u8(v, k(b'0')), k(10)))),
            sp: Self::tomask(x.map(|v| vceqq_u8(v, k(b' ')))),
            nl: Self::tomask(x.map(|v| vorrq_u8(vceqq_u8(v, k(b'\n')), vceqq_u8(v, k(b'\r'))))),
            ws: Self::tomask(x.map(|v| vorrq_u8(vceqq_u8(v, k(b' ')), vcltq_u8(vsubq_u8(v, k(0x09)), k(5))))),
            q: Self::tomask(x.map(|v| vceqq_u8(v, k(b'\'')))),
            sl: Self::tomask(x.map(|v| vceqq_u8(v, k(b'/')))),
            na: Self::tomask(x.map(|v| vcgeq_u8(v, k(0x80)))),
        }
    }

    #[inline(always)]
    unsafe fn resolve_latin1_64(p: *const u8, m: &mut Classes) -> (u64, u64) {
        use std::arch::aarch64::*;
        let v = vld4q_u8(p);
        let x = [v.0, v.1, v.2, v.3];
        let k = |c: u8| vdupq_n_u8(c);
        let c3 = Self::tomask(x.map(|v| vceqq_u8(v, k(0xC3)))) & !(1u64 << 63);
        let c3_pair = c3 | (c3 << 1);
        if m.na != c3_pair || (c3 & (c3 << 1)) != 0 {
            return (0, 0);
        }
        let foll = c3 << 1;
        let md = Self::tomask(x.map(|v| vorrq_u8(vceqq_u8(v, k(0x97)), vceqq_u8(v, k(0xB7))))) & foll;
        let punct = md | (md >> 1);
        let letters = c3_pair & !punct;
        let up_f = Self::tomask(x.map(|v| vcleq_u8(v, k(0x9E)))) & foll & !md;
        let upper = up_f | (up_f >> 1);
        m.l |= letters;
        m.up |= upper;
        m.na = 0;
        (foll, punct)
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Sse2;

#[cfg(target_arch = "x86_64")]
impl Simd for Sse2 {
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
            na: tomask(x),
        }
    }

    #[inline(always)]
    unsafe fn resolve_latin1_64(p: *const u8, m: &mut Classes) -> (u64, u64) {
        use std::arch::x86_64::*;
        let x = [
            _mm_loadu_si128(p as *const __m128i),
            _mm_loadu_si128(p.add(16) as *const __m128i),
            _mm_loadu_si128(p.add(32) as *const __m128i),
            _mm_loadu_si128(p.add(48) as *const __m128i),
        ];
        let k = |v: u8| _mm_set1_epi8(v as i8);
        let tomask = |a: [__m128i; 4]| -> u64 {
            (_mm_movemask_epi8(a[0]) as u16 as u64)
                | ((_mm_movemask_epi8(a[1]) as u16 as u64) << 16)
                | ((_mm_movemask_epi8(a[2]) as u16 as u64) << 32)
                | ((_mm_movemask_epi8(a[3]) as u16 as u64) << 48)
        };
        let c3 = tomask(x.map(|v| _mm_cmpeq_epi8(v, k(0xC3)))) & !(1u64 << 63);
        let c3_pair = c3 | (c3 << 1);
        if m.na != c3_pair || (c3 & (c3 << 1)) != 0 {
            return (0, 0);
        }
        let foll = c3 << 1;
        let md = tomask(x.map(|v| _mm_or_si128(_mm_cmpeq_epi8(v, k(0x97)), _mm_cmpeq_epi8(v, k(0xB7))))) & foll;
        let punct = md | (md >> 1);
        let letters = c3_pair & !punct;
        // Signed i8 comparison: 0x80..=0x9E is -128..=-98, strictly < -97 (0x9F).
        let up_f = tomask(x.map(|v| _mm_cmplt_epi8(v, k(0x9F)))) & foll & !md;
        let upper = up_f | (up_f >> 1);
        m.l |= letters;
        m.up |= upper;
        m.na = 0;
        (foll, punct)
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Avx2;

#[cfg(target_arch = "x86_64")]
impl Simd for Avx2 {
    #[inline(always)]
    unsafe fn classify64(p: *const u8) -> Classes {
        use std::arch::x86_64::*;
        let x0 = _mm256_loadu_si256(p as *const __m256i);
        let x1 = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let k = |v: u8| _mm256_set1_epi8(v as i8);
        let range = |v: __m256i, lo: u8, len: u8| {
            let y = _mm256_add_epi8(v, k(0x80u8.wrapping_sub(lo)));
            _mm256_cmpgt_epi8(k(0x80u8.wrapping_add(len)), y)
        };
        let tomask = |a: __m256i, b: __m256i| -> u64 {
            (_mm256_movemask_epi8(a) as u32 as u64)
                | ((_mm256_movemask_epi8(b) as u32 as u64) << 32)
        };
        let sp0 = _mm256_cmpeq_epi8(x0, k(b' '));
        let sp1 = _mm256_cmpeq_epi8(x1, k(b' '));
        Classes {
            l: tomask(
                range(_mm256_or_si256(x0, k(0x20)), b'a', 26),
                range(_mm256_or_si256(x1, k(0x20)), b'a', 26),
            ),
            up: tomask(range(x0, b'A', 26), range(x1, b'A', 26)),
            d: tomask(range(x0, b'0', 10), range(x1, b'0', 10)),
            sp: tomask(sp0, sp1),
            nl: tomask(
                _mm256_or_si256(_mm256_cmpeq_epi8(x0, k(b'\n')), _mm256_cmpeq_epi8(x0, k(b'\r'))),
                _mm256_or_si256(_mm256_cmpeq_epi8(x1, k(b'\n')), _mm256_cmpeq_epi8(x1, k(b'\r'))),
            ),
            ws: tomask(
                _mm256_or_si256(sp0, range(x0, 0x09, 5)),
                _mm256_or_si256(sp1, range(x1, 0x09, 5)),
            ),
            q: tomask(_mm256_cmpeq_epi8(x0, k(b'\'')), _mm256_cmpeq_epi8(x1, k(b'\''))),
            sl: tomask(_mm256_cmpeq_epi8(x0, k(b'/')), _mm256_cmpeq_epi8(x1, k(b'/'))),
            na: tomask(x0, x1),
        }
    }

    #[inline(always)]
    unsafe fn resolve_latin1_64(p: *const u8, m: &mut Classes) -> (u64, u64) {
        use std::arch::x86_64::*;
        let x0 = _mm256_loadu_si256(p as *const __m256i);
        let x1 = _mm256_loadu_si256(p.add(32) as *const __m256i);
        let k = |v: u8| _mm256_set1_epi8(v as i8);
        let tomask = |a: __m256i, b: __m256i| -> u64 {
            (_mm256_movemask_epi8(a) as u32 as u64)
                | ((_mm256_movemask_epi8(b) as u32 as u64) << 32)
        };
        let c3 = tomask(_mm256_cmpeq_epi8(x0, k(0xC3)), _mm256_cmpeq_epi8(x1, k(0xC3))) & !(1u64 << 63);
        let c3_pair = c3 | (c3 << 1);
        if m.na != c3_pair || (c3 & (c3 << 1)) != 0 {
            return (0, 0);
        }
        let foll = c3 << 1;
        let md = tomask(
            _mm256_or_si256(_mm256_cmpeq_epi8(x0, k(0x97)), _mm256_cmpeq_epi8(x0, k(0xB7))),
            _mm256_or_si256(_mm256_cmpeq_epi8(x1, k(0x97)), _mm256_cmpeq_epi8(x1, k(0xB7))),
        ) & foll;
        let punct = md | (md >> 1);
        let letters = c3_pair & !punct;
        let k9f = k(0x9F);
        let up_f = tomask(_mm256_cmpgt_epi8(k9f, x0), _mm256_cmpgt_epi8(k9f, x1)) & foll & !md;
        let upper = up_f | (up_f >> 1);
        m.l |= letters;
        m.up |= upper;
        m.na = 0;
        (foll, punct)
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Avx512;

#[cfg(target_arch = "x86_64")]
impl Simd for Avx512 {
    #[inline(always)]
    unsafe fn classify64(p: *const u8) -> Classes {
        use std::arch::x86_64::*;
        let v = _mm512_loadu_si512(p as *const _);
        let k = |c: u8| _mm512_set1_epi8(c as i8);
        let sp = _mm512_cmpeq_epi8_mask(v, k(b' '));
        let l = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(_mm512_or_si512(v, k(0x20)), k(b'a')), k(26));
        Classes {
            l,
            up: _mm512_mask_testn_epi8_mask(l, v, k(0x20)),
            d: _mm512_cmplt_epu8_mask(_mm512_sub_epi8(v, k(b'0')), k(10)),
            sp,
            nl: _mm512_cmpeq_epi8_mask(v, k(b'\n')) | _mm512_cmpeq_epi8_mask(v, k(b'\r')),
            ws: sp | _mm512_cmplt_epu8_mask(_mm512_sub_epi8(v, k(0x09)), k(5)),
            q: _mm512_cmpeq_epi8_mask(v, k(b'\'')),
            sl: _mm512_cmpeq_epi8_mask(v, k(b'/')),
            na: _mm512_movepi8_mask(v),
        }
    }

    #[inline(always)]
    unsafe fn resolve_latin1_64(p: *const u8, m: &mut Classes) -> (u64, u64) {
        use std::arch::x86_64::*;
        let v = _mm512_loadu_si512(p as *const _);
        let k = |c: u8| _mm512_set1_epi8(c as i8);
        let c3 = _mm512_cmpeq_epi8_mask(v, k(0xC3)) & !(1u64 << 63);
        let c3_pair = c3 | (c3 << 1);
        if m.na != c3_pair || (c3 & (c3 << 1)) != 0 {
            return (0, 0);
        }
        let foll = c3 << 1;
        let md = (_mm512_cmpeq_epi8_mask(v, k(0x97)) | _mm512_cmpeq_epi8_mask(v, k(0xB7))) & foll;
        let punct = md | (md >> 1);
        let letters = c3_pair & !punct;
        let up_f = _mm512_cmple_epu8_mask(v, k(0x9E)) & foll & !md;
        let upper = up_f | (up_f >> 1);
        m.l |= letters;
        m.up |= upper;
        m.na = 0;
        (foll, punct)
    }

    #[inline(always)]
    unsafe fn extract_ends(trusted: u64, base: usize, op: *mut u32, mut k: usize) -> usize {
        use std::arch::x86_64::*;
        // Register-to-register `vpcompressd` (1-2 uops, single-cycle throughput
        // on Zen 4 & Ice Lake, unlike memory-dest compressstore which is ~25c
        // microcode!) + unconditional 64B store (fits well inside SCAN_SLACK=80).
        // Zero 8 KiB POS32 L1d lookup table traffic!
        let ramp = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let c16 = _mm512_set1_epi32(16);
        let v0 = _mm512_add_epi32(_mm512_set1_epi32(base as i32), ramp);
        let m0 = trusted as u16;
        _mm512_storeu_si512(op.add(k) as *mut _, _mm512_maskz_compress_epi32(m0, v0));
        k += m0.count_ones() as usize;
        let v1 = _mm512_add_epi32(v0, c16);
        let m1 = (trusted >> 16) as u16;
        _mm512_storeu_si512(op.add(k) as *mut _, _mm512_maskz_compress_epi32(m1, v1));
        k += m1.count_ones() as usize;
        let v2 = _mm512_add_epi32(v1, c16);
        let m2 = (trusted >> 32) as u16;
        _mm512_storeu_si512(op.add(k) as *mut _, _mm512_maskz_compress_epi32(m2, v2));
        k += m2.count_ones() as usize;
        let v3 = _mm512_add_epi32(v2, c16);
        let m3 = (trusted >> 48) as u16;
        _mm512_storeu_si512(op.add(k) as *mut _, _mm512_maskz_compress_epi32(m3, v3));
        k + m3.count_ones() as usize
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[derive(Clone, Copy)]
struct Scalar;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
impl Simd for Scalar {
    #[inline(always)]
    unsafe fn classify64(p: *const u8) -> Classes {
        classify_scalar(std::slice::from_raw_parts(p, 64), 0)
    }

    #[inline(always)]
    unsafe fn resolve_latin1_64(_p: *const u8, _m: &mut Classes) -> (u64, u64) {
        (0, 0)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn x86_tier() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static TIER: AtomicU8 = AtomicU8::new(0);
    let t = TIER.load(Ordering::Relaxed);
    if t != 0 {
        return t;
    }
    let detected = if std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vl")
        && std::is_x86_feature_detected!("bmi1")
        && std::is_x86_feature_detected!("bmi2")
        && std::is_x86_feature_detected!("lzcnt")
        && std::is_x86_feature_detected!("popcnt")
    {
        3
    } else if std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("bmi1")
        && std::is_x86_feature_detected!("bmi2")
        && std::is_x86_feature_detected!("lzcnt")
        && std::is_x86_feature_detected!("popcnt")
    {
        2
    } else {
        1
    };
    TIER.store(detected, Ordering::Relaxed);
    detected
}

#[inline(always)]
unsafe fn classify_with<S: Simd>(b: &[u8], base: usize) -> Classes {
    if base + 64 <= b.len() {
        return S::classify64(b.as_ptr().add(base));
    }
    let mut pad = [0u8; 64];
    let k = b.len() - base;
    pad[..k].copy_from_slice(&b[base..]);
    S::classify64(pad.as_ptr())
}

/// Classes of the 64 bytes at `base` (fewer near the end of `b`).
#[inline(always)]
#[allow(dead_code)]
pub fn classify(b: &[u8], base: usize) -> Classes {
    #[cfg(target_arch = "aarch64")]
    return unsafe { classify_with::<Neon>(b, base) };
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        unsafe fn c512(b: &[u8], base: usize) -> Classes {
            classify_with::<Avx512>(b, base)
        }
        #[target_feature(enable = "avx2")]
        unsafe fn c256(b: &[u8], base: usize) -> Classes {
            classify_with::<Avx2>(b, base)
        }
        return match x86_tier() {
            3 => unsafe { c512(b, base) },
            2 => unsafe { c256(b, base) },
            _ => unsafe { classify_with::<Sse2>(b, base) },
        };
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
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
    na4: u64,        // bit k-1 = byte base-k is non-ASCII, k = 1..4
    ostart: bool,    // byte base-1 started a chunk as an O
    abs_nl: bool,    // byte base-1 is an NL absorbed into a symbol chunk's tail
    abs_sl: bool,    // byte base-1 is a '/' absorbed into a symbol chunk's tail (CamelCase)
    bad_in: u64,     // BAD bits carried into this block (contraction spans)
    bad3: u64,       // BAD flags of bytes base-3..base-1 (bit 2 = base-1)
    ws_unk: bool,    // byte base-1 is whitespace in an uncertain run (CamelCase)
    sl_unk: bool,    // byte base-1 is a '/' whose tail-absorption is uncertain (CamelCase)
    contr: bool,     // byte base-1 ends a contraction suffix (CamelCase)
    contr_mask: u64, // suffix-end bytes inside this block, from a suffix that began earlier
    forced: u64,     // chunk starts carried into this block
    suppress: u64,   // non-starts carried into this block (inside a contraction)
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
    if r >= 64 || e <= r {
        return 0;
    }
    let hi = if e >= 64 { u64::MAX } else { (1u64 << e) - 1 };
    hi & (u64::MAX << r)
}

/// Common pieces shared by the three grammars (slow / full path).
struct Ctx {
    m: Classes,
    o: u64,
    valid: u64,
    next_ws: bool,
    next_na: bool,
    cont: u64, // continuation bytes of resolved non-ASCII characters
}

/// Scalar fallback resolver for non-ASCII characters when a block contains
/// non-Latin-1 bytes (curly quotes, dashes `E2 80 xx`, CJK, or tail blocks).
#[inline]
fn resolve_na<const CONT: bool>(b: &[u8], base: usize, m: &mut Classes) -> u64 {
    let lim = (b.len() - base).min(64);
    let mut r = m.na;
    let (mut ul, mut up, mut cont, mut upper) = (0u64, 0u64, 0u64, 0u64);
    while r != 0 {
        let i = r.trailing_zeros() as usize;
        let c = b[base + i];
        if c == 0xC3 && i + 1 < lim {
            let d = b[base + i + 1];
            let w = 3u64 << i;
            if d == 0x97 || d == 0xB7 { up |= w } else { ul |= w }
            if d <= 0x9E && d != 0x97 {
                upper |= w;
            }
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
    m.up |= upper;
    m.na &= !(ul | up);
    cont
}

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

#[inline(always)]
fn common_bad(x: &Ctx, c: &Carry) -> u64 {
    let m = &x.m;
    let mut bad = m.na | (m.na << 1) | (m.na << 2) | (m.na << 3) | (m.na << 4);
    bad |= (16u64 >> c.na4.trailing_zeros().min(63)).saturating_sub(1);
    bad |= m.ws & (p1(m.na, c.na) | n1(m.na, x.next_na));
    bad |= m.ws & !m.ws.wrapping_add(1) & (c.ws as u64).wrapping_neg();
    bad
}

#[inline(always)]
fn digit_starts(m: &Classes, c: &Carry, k: u32, prefix_sp: bool) -> (u64, u64) {
    let runs = m.d & !p1(m.d, c.d);
    let joined = if prefix_sp { p1(m.sp, c.sp) } else { 0 };
    match k {
        0 => (runs & !joined, 0),
        1 => (m.d, 0),
        k => {
            let mut dm = m.d;
            for t in 1..k {
                dm &= m.d << t;
            }
            let flag = runs & p1(m.na, c.na);
            let mut bad = (m.d.wrapping_add(flag) ^ m.d) & m.d;
            let mut st = runs & !flag;
            let mut sh = k;
            while sh < 64 {
                st |= (st << sh) & dm;
                dm &= dm << sh;
                sh <<= 1;
            }
            st &= !bad;
            bad |= m.d & !m.d.wrapping_add(1) & (c.d as u64).wrapping_neg();
            (st, bad)
        }
    }
}

/// Pure-ASCII digit-group starts (no `m.na` / `c.na`, `prefix_sp == false`).
#[inline(always)]
fn digit_starts_ascii(d: u64, cd: bool, k: u32) -> (u64, u64) {
    let runs = d & !p1(d, cd);
    match k {
        0 => (runs, 0),
        1 => (d, 0),
        k => {
            let bad = d & !d.wrapping_add(1) & (cd as u64).wrapping_neg();
            let mut dm = d;
            for t in 1..k {
                dm &= d << t;
            }
            let mut st = runs;
            let first_extra = (runs << k) & dm;
            if first_extra != 0 {
                st |= first_extra;
                dm &= dm << k;
                let mut sh = k << 1;
                while sh < 64 {
                    st |= (st << sh) & dm;
                    dm &= dm << sh;
                    sh <<= 1;
                }
            }
            (st & !bad, bad)
        }
    }
}

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
            a = r + ((!m.nl) >> r).trailing_zeros() as usize;
            abs |= run_mask(r, a.min(e));
            if a >= e {
                continue;
            }
        }
        starts |= 1u64 << a;
        let nls = m.nl & run & (u64::MAX << a);
        let t = if nls != 0 {
            let l1 = 64 - nls.leading_zeros() as usize;
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

/// 100% branch-free, loop-free SWAR whitespace + newline rules for pure-ASCII /
/// resolved Latin-1 blocks (`m.na == 0`, `!c.na`, `base + 64 < b.len()`).
///
/// Uses the Exact Segmented Right-Smear identity:
///   `right_smear(M, S) = (((!M).wrapping_sub(S)) | S) & M`
/// to smear any arbitrary subset `S ⊆ M` to the end of every run in `M` in
/// 4 instructions with zero loops — handling `\n\n`, `\n    `, and `o`-absorbed
/// leading newlines (69.3% of real English/code blocks!) completely word-wide.
#[inline(always)]
fn ws_starts_nl_fast(
    b: &[u8],
    base: usize,
    m: &Classes,
    o_before: u64,
    c_ws: bool,
) -> (u64, u64, u64, u64, bool) {
    let next = unsafe { *b.get_unchecked(base + 64) };
    let (next_ws, next_na) = (is_ws(next), next >= 0x80);
    let long = m.ws & (p1(m.ws, c_ws) | n1(m.ws, next_ws));
    let cont_ws = m.ws & !m.ws.wrapping_add(1) & (c_ws as u64).wrapping_neg();
    let lr = long & !cont_ws;
    let single_abs = m.nl & !long & o_before;
    let wsbad = if lr >> 63 == 0 {
        0
    } else {
        let inv = !lr;
        if inv == 0 { u64::MAX } else { !(u64::MAX >> inv.leading_zeros()) }
    };
    let good = lr & !wsbad;
    let nl_g = m.nl & good;
    // Leading \r/\n prefix of any run in `good` immediately after `o` is absorbed.
    // Computed branch-free (eliminating the ~30%-mispredicted `nl_g == 0` branch):
    // when `nl_g == 0`, `abs_lr = 0`, `rem = good`, `non_nl = good`, `cand_l1 = 0`.
    let r_starts = good & !(good << 1);
    let abs_seed = r_starts & o_before & nl_g;
    let abs_lr = (nl_g.wrapping_add(abs_seed) ^ nl_g) & nl_g;
    let rem = good & !abs_lr;
    let nl_rem = nl_g & rem;
    let non_nl = rem & !nl_rem;
    let cand_l1 = non_nl & (nl_rem << 1);
    // Forward carry check: adding `cand_l1` to `non_nl` carries each post-NL
    // non-NL run to its end bit `v`. `v` hits `nl_rem` iff there is a rare
    // `\n <spaces> \n` inside the same whitespace run; otherwise EVERY bit in
    // `cand_l1` is exact `l1` with zero `reverse_bits()` calls.
    let l1_starts = if (non_nl.wrapping_add(cand_l1) & nl_rem) == 0 {
        cand_l1
    } else {
        let r_rem = rem.reverse_bits();
        let r_nl = nl_rem.reverse_bits();
        let at_or_before = (((!r_rem).wrapping_sub(r_nl) | r_nl) & r_rem).reverse_bits();
        let after_nl = rem & !at_or_before;
        after_nl & !(after_nl << 1)
    };
    // Starts: run start `a`, post-last-newline start `l1` (if l1 < e), and
    // final trailing ws split `e - 1` whenever `after_nl` has length >= 2
    // (identically `non_nl & !(rem >> 1) & (non_nl << 1)`).
    let wsstart = (m.ws & !long & !single_abs)
        | (rem & !(rem << 1))
        | l1_starts
        | (non_nl & !(rem >> 1) & (non_nl << 1));
    (wsstart, wsbad, single_abs | abs_lr, cont_ws, next_na)
}

#[inline(always)]
fn span_bad(bad: &mut u64, bad_next: &mut u64, i: usize, end: usize) {
    if *bad >> i & 1 == 1 {
        *bad |= run_mask(i, (end + 1).min(64));
        if end + 1 > 64 {
            *bad_next |= (1u64 << (end + 1 - 64)) - 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Generic grammar (Llama-3, Qwen, cl100k, ...)
// ---------------------------------------------------------------------------

#[inline(always)]
fn generic_fast(b: &[u8], base: usize, max_digits: u32, c: &Carry, m: &Classes, cont: u64) -> Option<Block> {
    let o = !(m.l | m.d | m.ws);
    let o_before = p1(o, c.o);
    let ostart = o & !o_before & !p1(m.sp, c.sp);
    let lstart = m.l & (p1(m.d, c.d) | p1(m.nl, c.nl) | (o_before & !p1(ostart, c.ostart)));
    let (dstart, dbad) = digit_starts_ascii(m.d, c.d, max_digits);
    let (wsstart, wsbad, _, cont_ws, next_na) = ws_starts_nl_fast(b, base, m, o_before, c.ws);
    let mut starts = ostart | lstart | dstart | wsstart | c.forced;
    let mut bad = cont_ws | (m.ws & ((next_na as u64) << 63)) | dbad | wsbad;
    let mut qc = m.q & ostart;
    while qc != 0 {
        let i = qc.trailing_zeros() as usize;
        qc &= qc - 1;
        let len = contraction_len(b, base + i, true);
        if len != 0 {
            let end = i + len;
            if end >= 64 {
                return None;
            }
            starts |= 1u64 << end;
            if bad >> i & 1 == 1 {
                bad |= run_mask(i, end + 1);
            }
        }
    }
    bad |= m.l & o_before & p1(bad, c.bad3 >> 2 & 1 == 1);
    bad |= (bad << 1) & cont;
    // Tailored Carry: Generic never reads c.l, c.lw, c.abs_nl, c.abs_sl, etc.,
    // allowing LLVM to dead-code-eliminate `m.up` and `m.sl` in classify64!
    let top = |v: u64| v >> 63 & 1 == 1;
    let out = Carry {
        o: top(o),
        sp: top(m.sp),
        d: top(m.d),
        nl: top(m.nl),
        ws: top(m.ws),
        ostart: top(ostart),
        bad3: bad >> 61,
        ..Carry::default()
    };
    Some((starts, bad, out))
}

#[cold]
#[inline(never)]
fn generic_slow(b: &[u8], base: usize, max_digits: u32, c: Carry, m: Classes) -> Block {
    let n = b.len();
    let x = ctx_m::<true, true>(b, base, m);
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
    let bad_c = bad | ((bad << 1) & x.cont);
    let bad_c = bad_c | ((bad_c << 1) & x.cont);
    bad |= m.l & p1(o, c.o) & p1(bad_c, c.bad3 >> 2 & 1 == 1);
    let bad = bad | ((bad << 1) & x.cont) | ((bad << 2) & x.cont & (x.cont << 1));
    let top = |v: u64| v >> 63 & 1 == 1;
    let out = Carry {
        o: top(x.o),
        sp: top(m.sp),
        d: top(m.d),
        nl: top(m.nl),
        ws: top(m.ws),
        na: top(m.na),
        na4: (m.na >> 60).reverse_bits() >> 60,
        ostart: top(ostart_c),
        bad_in: bad_next,
        bad3: bad >> 61,
        forced,
        ..Carry::default()
    };
    (starts & x.valid, bad & x.valid, out)
}

#[inline(always)]
unsafe fn generic_block<S: Simd>(b: &[u8], base: usize, max_digits: u32, c: Carry) -> Block {
    let m = classify_with::<S>(b, base);
    if base + 64 < b.len() && !c.na && (c.na4 | c.bad_in | (c.forced & !1)) == 0 {
        if m.na == 0 {
            if let Some(r) = generic_fast(b, base, max_digits, &c, &m, 0) {
                return r;
            }
        } else {
            let mut rm = m;
            let (cont, punct) = S::resolve_latin1_64(b.as_ptr().add(base), &mut rm);
            if rm.na == 0 && punct == 0 {
                if let Some(r) = generic_fast(b, base, max_digits, &c, &rm, cont) {
                    return r;
                }
            }
        }
    }
    generic_slow(b, base, max_digits, c, m)
}

// ---------------------------------------------------------------------------
// CamelCase grammar (o200k, MiniMax, ...)
// ---------------------------------------------------------------------------

#[inline(always)]
fn camel_fast(
    b: &[u8],
    base: usize,
    contractions: bool,
    max_digits: u32,
    c: &Carry,
    m: &Classes,
    cont: u64,
) -> Option<Block> {
    let o = !(m.l | m.d | m.ws);
    let lw = m.l & !m.up;
    let o_before = p1(o, c.o);
    let (wsstart, wsbad, abs_nl, cont_ws, next_na) = ws_starts_nl_fast(b, base, m, o_before, c.ws);
    let seed = m.sl & p1(abs_nl, c.abs_nl || c.abs_sl);
    let abs_sl = (m.sl.wrapping_add(seed) ^ m.sl) & m.sl;
    let mut ostart = (o & !o_before & !p1(m.sp, c.sp) & !abs_sl) | (o & !m.sl & p1(abs_sl, c.abs_sl));
    let (dstart, dbad) = digit_starts_ascii(m.d, c.d, max_digits);
    let mut bad = cont_ws | (m.ws & ((next_na as u64) << 63)) | dbad | wsbad;
    let ws_unk = m.ws & bad;
    let useed = m.sl & p1(ws_unk, c.ws_unk) | (m.sl & 1 & c.sl_unk as u64);
    let sl_unk = (m.sl.wrapping_add(useed) ^ m.sl) & m.sl;
    bad |= sl_unk | (p1(sl_unk, c.sl_unk) & !m.sl);
    let (mut suppress, mut forced_in, mut contr_end) = (0u64, 0u64, 0u64);
    if contractions {
        let mut qs = m.q & p1(m.l, c.l);
        while qs != 0 {
            let i = qs.trailing_zeros() as usize;
            qs &= qs - 1;
            let window = ((bad << 3) | c.bad3) >> i & 0b1111;
            if window != 0 {
                if i >= 60 {
                    return None;
                }
                bad |= run_mask(i, i + 4);
            }
            let prev_is_suffix_end = if i == 0 { c.contr } else { contr_end >> (i - 1) & 1 == 1 };
            if prev_is_suffix_end {
                continue;
            }
            let len = contraction_len(b, base + i, true);
            if len == 0 {
                continue;
            }
            let end = i + len;
            if end >= 64 {
                return None;
            }
            suppress |= run_mask(i, end);
            contr_end |= 1u64 << (end - 1);
            forced_in |= 1u64 << end;
            if bad >> i & 1 == 1 {
                bad |= run_mask(i, end + 1);
            }
        }
    }
    ostart &= !suppress;
    let lstart = m.l
        & (p1(m.d, c.d) | p1(m.nl, c.nl) | (o_before & !p1(ostart, c.ostart)) | (m.up & p1(lw, c.lw)));
    let starts = ((ostart | lstart | dstart | wsstart) & !suppress) | c.forced | forced_in;
    bad |= m.l & o_before & p1(bad, c.bad3 >> 2 & 1 == 1);
    bad |= (bad << 1) & cont;
    let top = |v: u64| v >> 63 & 1 == 1;
    let out = Carry {
        o: top(o),
        sp: top(m.sp),
        d: top(m.d),
        nl: top(m.nl),
        ws: top(m.ws),
        l: top(m.l),
        lw: top(lw),
        ostart: top(ostart),
        abs_nl: top(abs_nl),
        abs_sl: top(abs_sl),
        bad3: bad >> 61,
        ws_unk: top(ws_unk),
        sl_unk: top(sl_unk),
        contr: top(contr_end),
        ..Carry::default()
    };
    Some((starts, bad, out))
}

#[cold]
#[inline(never)]
fn camel_slow(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: Carry, m: Classes) -> Block {
    let n = b.len();
    let x = ctx_m::<true, true>(b, base, m);
    let (m, o) = (&x.m, x.o);
    let lw = m.l & !m.up;
    let (wsstart, wsbad, abs_nl) = ws_starts_nl(&x, &c, base, n);
    let seed = m.sl & p1(abs_nl, c.abs_nl || c.abs_sl);
    let abs_sl = (m.sl.wrapping_add(seed) ^ m.sl) & m.sl;
    let mut ostart = (o & !p1(o, c.o) & !p1(m.sp, c.sp) & !abs_sl) | (o & !m.sl & p1(abs_sl, c.abs_sl));
    let (mut suppress, mut forced_in, mut forced, mut sup_next) = (c.suppress, 0u64, 0u64, 0u64);
    let (dstart, dbad) = digit_starts(m, &c, max_digits, false);
    let mut bad = common_bad(&x, &c) | dbad | wsbad | c.bad_in;
    let ws_unk = m.ws & bad;
    let useed = m.sl & p1(ws_unk, c.ws_unk) | (m.sl & 1 & c.sl_unk as u64);
    let sl_unk = (m.sl.wrapping_add(useed) ^ m.sl) & m.sl;
    bad |= sl_unk | (p1(sl_unk, c.sl_unk) & !m.sl);
    let mut bad_next = 0u64;
    let mut contr_end = c.contr_mask;
    let mut contr_next = 0u64;
    if contractions {
        let mut qs = m.q & p1(m.l, c.l);
        while qs != 0 {
            let i = qs.trailing_zeros() as usize;
            qs &= qs - 1;
            let window = (bad << 3 | c.bad3) >> i & 0b1111;
            if window != 0 {
                span_bad(&mut bad, &mut bad_next, i, i + 3);
                bad |= 1u64 << i;
                span_bad(&mut bad, &mut bad_next, i, i + 3);
            }
            let prev_is_suffix_end = if i == 0 { c.contr } else { contr_end >> (i - 1) & 1 == 1 };
            if prev_is_suffix_end {
                continue;
            }
            let len = contraction_len(b, base + i, true);
            if len == 0 {
                continue;
            }
            let end = i + len;
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
    let ostart_c = ostart | ((ostart << 1) & x.cont);
    let ostart_c = ostart_c | ((ostart_c << 1) & x.cont);
    let ostart_p1 = p1(ostart_c, c.ostart);
    let lstart = m.l
        & (p1(m.d, c.d) | p1(m.nl, c.nl) | (p1(o, c.o) & !ostart_p1) | (m.up & p1(lw, c.lw)));
    let starts = ((ostart | lstart | dstart | wsstart) & !suppress) | c.forced | forced_in;
    let bad_c = bad | ((bad << 1) & x.cont);
    let bad_c = bad_c | ((bad_c << 1) & x.cont);
    bad |= m.l & p1(o, c.o) & p1(bad_c, c.bad3 >> 2 & 1 == 1);
    let bad = bad | ((bad << 1) & x.cont) | ((bad << 2) & x.cont & (x.cont << 1));
    let mut out = carry_out(&x, ostart_c, abs_nl, contr_end >> 63 & 1 == 1, forced, sup_next);
    out.contr_mask = contr_next;
    out.abs_sl = abs_sl >> 63 & 1 == 1;
    out.bad_in = bad_next;
    out.bad3 = bad >> 61;
    out.ws_unk = ws_unk >> 63 & 1 == 1;
    out.sl_unk = sl_unk >> 63 & 1 == 1;
    (starts & x.valid, bad & x.valid, out)
}

#[inline(always)]
unsafe fn camel_block<S: Simd>(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: Carry) -> Block {
    let m = classify_with::<S>(b, base);
    if base + 64 < b.len()
        && !c.na
        && (c.na4 | c.bad_in | c.suppress | (c.forced & !1) | c.contr_mask) == 0
    {
        if m.na == 0 {
            if let Some(r) = camel_fast(b, base, contractions, max_digits, &c, &m, 0) {
                return r;
            }
        } else {
            let mut rm = m;
            let (cont, punct) = S::resolve_latin1_64(b.as_ptr().add(base), &mut rm);
            if rm.na == 0 && punct == 0 {
                if let Some(r) = camel_fast(b, base, contractions, max_digits, &c, &rm, cont) {
                    return r;
                }
            }
        }
    }
    camel_slow(b, base, contractions, max_digits, c, m)
}

// ---------------------------------------------------------------------------
// Word grammar (GPT-2 family)
// ---------------------------------------------------------------------------

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
                return None;
            }
            suppress |= run_mask(i + 1, end);
            starts |= 1u64 << end;
        }
    }
    let next = unsafe { *b.get_unchecked(base + 64) };
    let (next_ws, next_na) = (is_ws(next), next >= 0x80);
    let long = m.ws & (p1(m.ws, c.ws) | n1(m.ws, next_ws));
    let cont = m.ws & !m.ws.wrapping_add(1) & (c.ws as u64).wrapping_neg();
    let lr = long & !cont;
    let r = lr & !(lr << 1);
    let last = lr & !(lr >> 1);
    let wsbad = if lr >> 63 == 0 {
        0
    } else {
        let inv = !lr;
        if inv == 0 { u64::MAX } else { !(u64::MAX >> inv.leading_zeros()) }
    };
    let good = lr & !wsbad;
    let no_digit_rule = if indiv { !(m.d >> 1) } else { u64::MAX };
    starts |= (m.ws & !long) | (r & good) | (last & good & (u64::MAX >> 1) & no_digit_rule) | c.forced;
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
unsafe fn word_block<S: Simd>(b: &[u8], base: usize, indiv: bool, c: Carry) -> Block {
    let m = classify_with::<S>(b, base);
    if base + 64 < b.len() && !c.na && (c.na4 | c.bad_in | c.suppress | (c.forced & !1)) == 0 {
        if m.na == 0 {
            if let Some(r) = word_fast(b, base, indiv, &c, &m) {
                return r;
            }
        } else {
            let mut rm = m;
            let _ = S::resolve_latin1_64(b.as_ptr().add(base), &mut rm);
            if rm.na == 0 {
                if let Some(r) = word_fast(b, base, indiv, &c, &rm) {
                    return r;
                }
            }
        }
    }
    word_slow(b, base, indiv, c, m)
}

#[cold]
#[inline(never)]
fn word_slow(b: &[u8], base: usize, indiv: bool, c: Carry, m: Classes) -> Block {
    let n = b.len();
    if m.na != 0 && base + 64 < n && !c.na && (c.na4 | c.bad_in | c.suppress | c.forced) == 0 {
        let mut r = m;
        resolve_na::<false>(b, base, &mut r);
        if r.na == 0 {
            if let Some(out) = word_fast(b, base, indiv, &c, &r) {
                return out;
            }
        }
    }
    let x = ctx_m::<true, false>(b, base, m);
    let (m, o) = (&x.m, x.o);
    let sp_before = p1(m.sp, c.sp);
    let ostart = o & !p1(o, c.o) & !sp_before;
    let lstart = m.l & !p1(m.l, c.l) & !sp_before;
    let (dstart, dbad) = if indiv { (m.d, 0) } else { digit_starts(m, &c, 0, true) };
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
    let text_follows = x.valid >> 1;
    let no_digit_rule = if indiv { !(m.d >> 1) } else { u64::MAX };
    let wsstart = (m.ws & !long) | (r & good) | (last & good & text_follows & no_digit_rule);
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
        na4: (m.na >> 60).reverse_bits() >> 60,
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

#[inline(always)]
unsafe fn block_with<S: Simd>(kind: Pretok, b: &[u8], base: usize, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = match kind {
        Pretok::Generic { max_digits, .. } => generic_block::<S>(b, base, max_digits, *c),
        Pretok::CamelCase { contractions, max_digits } => camel_block::<S>(b, base, contractions, max_digits, *c),
        Pretok::Word(indiv) => word_block::<S>(b, base, indiv, *c),
    };
    *c = out;
    (s, bad)
}

/// One block with the carry updated IN PLACE, returning (starts, bad) in two
/// registers.
#[inline(always)]
pub fn block_m(kind: Pretok, b: &[u8], base: usize, c: &mut Carry) -> (u64, u64) {
    #[cfg(target_arch = "aarch64")]
    {
        match kind {
            Pretok::Generic { max_digits, .. } => generic_m_neon(b, base, max_digits, c),
            Pretok::CamelCase { contractions, max_digits } => camel_m_neon(b, base, contractions, max_digits, c),
            Pretok::Word(indiv) => word_m_neon(b, base, indiv, c),
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        match x86_tier() {
            3 => match kind {
                Pretok::Generic { max_digits, .. } => unsafe { generic_m_avx512(b, base, max_digits, c) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { camel_m_avx512(b, base, contractions, max_digits, c) },
                Pretok::Word(indiv) => unsafe { word_m_avx512(b, base, indiv, c) },
            },
            2 => match kind {
                Pretok::Generic { max_digits, .. } => unsafe { generic_m_avx2(b, base, max_digits, c) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { camel_m_avx2(b, base, contractions, max_digits, c) },
                Pretok::Word(indiv) => unsafe { word_m_avx2(b, base, indiv, c) },
            },
            _ => match kind {
                Pretok::Generic { max_digits, .. } => unsafe { generic_m_sse2(b, base, max_digits, c) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { camel_m_sse2(b, base, contractions, max_digits, c) },
                Pretok::Word(indiv) => unsafe { word_m_sse2(b, base, indiv, c) },
            },
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    unsafe { block_with::<Scalar>(kind, b, base, c) }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn word_m_neon(b: &[u8], base: usize, indiv: bool, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = unsafe { word_block::<Neon>(b, base, indiv, *c) };
    *c = out;
    (s, bad)
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn generic_m_neon(b: &[u8], base: usize, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = unsafe { generic_block::<Neon>(b, base, max_digits, *c) };
    *c = out;
    (s, bad)
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn camel_m_neon(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = unsafe { camel_block::<Neon>(b, base, contractions, max_digits, *c) };
    *c = out;
    (s, bad)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn word_m_avx512(b: &[u8], base: usize, indiv: bool, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = word_block::<Avx512>(b, base, indiv, *c);
    *c = out;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn generic_m_avx512(b: &[u8], base: usize, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = generic_block::<Avx512>(b, base, max_digits, *c);
    *c = out;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn camel_m_avx512(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = camel_block::<Avx512>(b, base, contractions, max_digits, *c);
    *c = out;
    (s, bad)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn word_m_avx2(b: &[u8], base: usize, indiv: bool, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = word_block::<Avx2>(b, base, indiv, *c);
    *c = out;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn generic_m_avx2(b: &[u8], base: usize, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = generic_block::<Avx2>(b, base, max_digits, *c);
    *c = out;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn camel_m_avx2(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = camel_block::<Avx2>(b, base, contractions, max_digits, *c);
    *c = out;
    (s, bad)
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn word_m_sse2(b: &[u8], base: usize, indiv: bool, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = word_block::<Sse2>(b, base, indiv, *c);
    *c = out;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn generic_m_sse2(b: &[u8], base: usize, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = generic_block::<Sse2>(b, base, max_digits, *c);
    *c = out;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn camel_m_sse2(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, out) = camel_block::<Sse2>(b, base, contractions, max_digits, *c);
    *c = out;
    (s, bad)
}

// ---------------------------------------------------------------------------
// Drivers: `chunks_batched` and resumable `Scan::fill`
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn chunks_batched_with<S: Simd>(
    kind: Pretok,
    b: &[u8],
    start: usize,
    stop: usize,
    mut fb: impl FnMut(usize, &[usize]),
) {
    let mut ends = [0usize; 72];
    let mut pos = start;
    let mut base = start;
    let mut carry = Carry { forced: 1, ..Carry::default() };
    while pos < stop {
        if base + 64 <= pos {
            base = pos;
            carry = Carry { forced: 1, ..Carry::default() };
        }
        let (starts, bad) = block_with::<S>(kind, b, base, &mut carry);
        let lim = (base + 64).min(stop);
        let lim_mask = if lim - base >= 64 { u64::MAX } else { (1u64 << (lim - base)) - 1 };
        let above0 = if pos < base { u64::MAX } else { u64::MAX.checked_shl((pos - base + 1) as u32).unwrap_or(0) };
        let mut trusted = starts & !bad & lim_mask & above0;
        let bad = bad & lim_mask;
        if bad & above0 == 0 {
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
                    break;
                }
                let end = next_chunk_end(kind, b, pos);
                fb(pos, &[end]);
                pos = end;
            }
        }
        base += 64;
    }
}

/// Iterate the chunks of [start, stop) (both proven chunk boundaries) for any
/// ByteLevel grammar, handing over each block's chunks at once: `f(first,
/// ends)` covers [first, ends[0]), [ends[0], ends[1]), … Exactly the
/// boundaries of repeated `next_chunk_end`.
#[inline(always)]
pub fn chunks_batched(kind: Pretok, b: &[u8], start: usize, stop: usize, fb: impl FnMut(usize, &[usize])) {
    #[cfg(target_arch = "aarch64")]
    unsafe { chunks_batched_with::<Neon>(kind, b, start, stop, fb) }
    #[cfg(target_arch = "x86_64")]
    {
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
        unsafe fn run512(kind: Pretok, b: &[u8], start: usize, stop: usize, fb: impl FnMut(usize, &[usize])) {
            chunks_batched_with::<Avx512>(kind, b, start, stop, fb)
        }
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
        unsafe fn run256(kind: Pretok, b: &[u8], start: usize, stop: usize, fb: impl FnMut(usize, &[usize])) {
            chunks_batched_with::<Avx2>(kind, b, start, stop, fb)
        }
        match x86_tier() {
            3 => unsafe { run512(kind, b, start, stop, fb) },
            2 => unsafe { run256(kind, b, start, stop, fb) },
            _ => unsafe { chunks_batched_with::<Sse2>(kind, b, start, stop, fb) },
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    unsafe { chunks_batched_with::<Scalar>(kind, b, start, stop, fb) }
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
/// to one buffer, so the consumer runs one long loop per batch.
pub struct Scan {
    pos: usize,
    base: usize,
    carry: Carry,
}

/// Grammar-specialized inner loop for `Scan::fill`. Because `block_fn` is
/// monomorphized per `(Simd, Pretok)`, the fast rules + SIMD classifier inline
/// directly into the loop with `Carry` kept in registers and zero per-block
/// dispatch or stack spill.
#[inline(always)]
unsafe fn fill_loop<S: Simd>(
    scan: &mut Scan,
    kind: Pretok,
    b: &[u8],
    stop: usize,
    out: &mut [u32],
    mut k: usize,
    limit: usize,
    mut step: impl FnMut(&[u8], usize, Carry) -> Block,
) -> usize {
    let (mut pos, mut base, mut carry) = (scan.pos, scan.base, scan.carry);
    let op = out.as_mut_ptr();
    while pos < stop && k < limit {
        if base + 64 <= pos {
            base = pos;
            carry = Carry { forced: 1, ..Carry::default() };
        }
        let (starts, bad, next_carry) = step(b, base, carry);
        carry = next_carry;
        let lim = (base + 64).min(stop);
        let lim_mask = if lim - base >= 64 { u64::MAX } else { (1u64 << (lim - base)) - 1 };
        let above0 = if pos < base { u64::MAX } else { u64::MAX.checked_shl((pos - base + 1) as u32).unwrap_or(0) };
        let trusted = starts & !bad & lim_mask & above0;
        let bad = bad & lim_mask;
        if bad & above0 == 0 {
            k = S::extract_ends(trusted, base, op, k);
            if trusted != 0 {
                pos = base + 63 - trusted.leading_zeros() as usize;
            }
            if lim == stop && pos < stop {
                *op.add(k) = stop as u32;
                k += 1;
                pos = stop;
            }
        } else {
            let mut realigned = false;
            loop {
                if pos >= lim {
                    base = pos;
                    carry = Carry { forced: 1, ..Carry::default() };
                    realigned = true;
                    break;
                }
                let above = if pos < base { u64::MAX } else { u64::MAX.checked_shl((pos - base + 1) as u32).unwrap_or(0) };
                let cand = trusted & above;
                let badm = bad & above;
                if badm == 0 {
                    if cand != 0 {
                        k = S::extract_ends(cand, base, op, k);
                        pos = base + 63 - cand.leading_zeros() as usize;
                    }
                    if lim == stop && pos < stop {
                        *op.add(k) = stop as u32;
                        k += 1;
                        pos = stop;
                    }
                    break;
                }
                // Extract all clean chunks strictly before the first bad bit via SIMD.
                let fb = badm.trailing_zeros();
                let clean_pre = cand & ((1u64 << fb) - 1);
                if clean_pre != 0 {
                    k = S::extract_ends(clean_pre, base, op, k);
                    pos = base + 63 - clean_pre.leading_zeros() as usize;
                }
                let end = next_chunk_end(kind, b, pos);
                *op.add(k) = end as u32;
                k += 1;
                pos = end;
                if k >= limit + SCAN_SLACK - 1 {
                    scan.pos = pos;
                    scan.base = pos;
                    scan.carry = Carry { forced: 1, ..Carry::default() };
                    return k;
                }
            }
            if realigned {
                continue;
            }
        }
        base += 64;
    }
    scan.pos = pos;
    scan.base = base;
    scan.carry = carry;
    k
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn fill_word_neon(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, indiv: bool, out: &mut [u32], k: usize, limit: usize) -> usize {
    unsafe { fill_loop::<Neon>(scan, kind, b, stop, out, k, limit, |b, base, c| word_block::<Neon>(b, base, indiv, c)) }
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn fill_generic_neon(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    unsafe { fill_loop::<Neon>(scan, kind, b, stop, out, k, limit, |b, base, c| generic_block::<Neon>(b, base, max_digits, c)) }
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn fill_camel_neon(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, contractions: bool, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    unsafe { fill_loop::<Neon>(scan, kind, b, stop, out, k, limit, |b, base, c| camel_block::<Neon>(b, base, contractions, max_digits, c)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn fill_word_avx512(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, indiv: bool, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx512>(scan, kind, b, stop, out, k, limit, |b, base, c| word_block::<Avx512>(b, base, indiv, c))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn fill_generic_avx512(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx512>(scan, kind, b, stop, out, k, limit, |b, base, c| generic_block::<Avx512>(b, base, max_digits, c))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn fill_camel_avx512(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, contractions: bool, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx512>(scan, kind, b, stop, out, k, limit, |b, base, c| camel_block::<Avx512>(b, base, contractions, max_digits, c))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn fill_word_avx2(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, indiv: bool, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx2>(scan, kind, b, stop, out, k, limit, |b, base, c| word_block::<Avx2>(b, base, indiv, c))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn fill_generic_avx2(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx2>(scan, kind, b, stop, out, k, limit, |b, base, c| generic_block::<Avx2>(b, base, max_digits, c))
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn fill_camel_avx2(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, contractions: bool, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx2>(scan, kind, b, stop, out, k, limit, |b, base, c| camel_block::<Avx2>(b, base, contractions, max_digits, c))
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn fill_word_sse2(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, indiv: bool, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Sse2>(scan, kind, b, stop, out, k, limit, |b, base, c| word_block::<Sse2>(b, base, indiv, c))
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn fill_generic_sse2(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Sse2>(scan, kind, b, stop, out, k, limit, |b, base, c| generic_block::<Sse2>(b, base, max_digits, c))
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn fill_camel_sse2(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, contractions: bool, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Sse2>(scan, kind, b, stop, out, k, limit, |b, base, c| camel_block::<Sse2>(b, base, contractions, max_digits, c))
}

impl Scan {
    pub fn new(start: usize) -> Self {
        Scan { pos: start, base: start, carry: Carry { forced: 1, ..Carry::default() } }
    }

    /// Append ends to `out[k..]` until `k >= limit` (checked per block) or
    /// `stop`; returns the new length. `out.len() >= limit + SCAN_SLACK`,
    /// `b.len() <= u32::MAX`.
    #[inline]
    pub fn fill(&mut self, kind: Pretok, b: &[u8], stop: usize, out: &mut [u32], k: usize, limit: usize) -> usize {
        assert!(out.len() >= limit + SCAN_SLACK);
        #[cfg(target_arch = "aarch64")]
        return match kind {
            Pretok::Word(indiv) => fill_word_neon(self, kind, b, stop, indiv, out, k, limit),
            Pretok::Generic { max_digits, .. } => fill_generic_neon(self, kind, b, stop, max_digits, out, k, limit),
            Pretok::CamelCase { contractions, max_digits } => fill_camel_neon(self, kind, b, stop, contractions, max_digits, out, k, limit),
        };
        #[cfg(target_arch = "x86_64")]
        return match x86_tier() {
            3 => match kind {
                Pretok::Word(indiv) => unsafe { fill_word_avx512(self, kind, b, stop, indiv, out, k, limit) },
                Pretok::Generic { max_digits, .. } => unsafe { fill_generic_avx512(self, kind, b, stop, max_digits, out, k, limit) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { fill_camel_avx512(self, kind, b, stop, contractions, max_digits, out, k, limit) },
            },
            2 => match kind {
                Pretok::Word(indiv) => unsafe { fill_word_avx2(self, kind, b, stop, indiv, out, k, limit) },
                Pretok::Generic { max_digits, .. } => unsafe { fill_generic_avx2(self, kind, b, stop, max_digits, out, k, limit) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { fill_camel_avx2(self, kind, b, stop, contractions, max_digits, out, k, limit) },
            },
            _ => match kind {
                Pretok::Word(indiv) => unsafe { fill_word_sse2(self, kind, b, stop, indiv, out, k, limit) },
                Pretok::Generic { max_digits, .. } => unsafe { fill_generic_sse2(self, kind, b, stop, max_digits, out, k, limit) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { fill_camel_sse2(self, kind, b, stop, contractions, max_digits, out, k, limit) },
            },
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        match kind {
            Pretok::Word(indiv) => unsafe { fill_loop::<Scalar>(self, kind, b, stop, out, k, limit, |b, base, c| word_block::<Scalar>(b, base, indiv, c)) },
            Pretok::Generic { max_digits, .. } => unsafe { fill_loop::<Scalar>(self, kind, b, stop, out, k, limit, |b, base, c| generic_block::<Scalar>(b, base, max_digits, c)) },
            Pretok::CamelCase { contractions, max_digits } => unsafe { fill_loop::<Scalar>(self, kind, b, stop, out, k, limit, |b, base, c| camel_block::<Scalar>(b, base, contractions, max_digits, c)) },
        }
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

    #[test]
    fn block_scan_equals_serial() {
        let mut seed = 99u64;
        for _ in 0..iters() {
            check(&random_text(&mut seed));
        }
    }

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
    fn block_scan_equals_serial_casesoup() {
        const C: &[char] = &['a', 'Z', 's', 'T', 'é', 'É', 'Ö', 'ø', 'ß', 'ÿ', 'À', 'Þ', '×', '÷', '\'', '’',
            '—', ' ', ' ', '/', '\n', '1', '.', 'l', 'L', 'v', 'e', 'R', 'd', 'M'];
        let mut seed = 31337u64;
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
            let mut best = (f64::MAX, f64::MAX, f64::MAX);
            let (mut n1, mut n2, mut n3) = (0usize, 0usize, 0usize);
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
                let t = std::time::Instant::now();
                n3 = 0;
                let mut sc = Scan::new(0);
                let mut out = vec![0u32; 256 + SCAN_SLACK];
                loop {
                    let k = sc.fill(kind, b, b.len(), &mut out, 0, 256);
                    if k == 0 { break; }
                    n3 += k;
                }
                best.2 = best.2.min(t.elapsed().as_secs_f64());
            }
            let (mut badn, mut base, mut c) = (0u64, 0usize, Carry::default());
            while base < b.len() {
                let (_, bd, nc) = block(kind, b, base, c);
                badn += bd.count_ones() as u64;
                c = nc;
                base += 64;
            }
            println!(
                "{kind:?}: serial {:.0} MB/s | block {:.0} MB/s | Scan::fill {:.0} MB/s ({n1} vs {n2} vs {n3} chunks) | bad bytes {:.2}%",
                b.len() as f64 / best.0 / 1e6,
                b.len() as f64 / best.1 / 1e6,
                b.len() as f64 / best.2 / 1e6,
                badn as f64 / b.len() as f64 * 100.0
            );
        }
    }
}
