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

impl Classes {
    #[inline(always)]
    fn mask(self, valid: u64) -> Self {
        Classes {
            l: self.l & valid,
            up: self.up & valid,
            d: self.d & valid,
            sp: self.sp & valid,
            nl: self.nl & valid,
            ws: self.ws & valid,
            q: self.q & valid,
            sl: self.sl & valid,
            na: self.na & valid,
        }
    }
}

/// Inline register copy into a zero-initialized 64-byte stack buffer for
/// non-AVX-512 tail blocks (avoids calling `libc::memcpy` + `vzeroupper`).
#[inline(always)]
unsafe fn copy_tail_pad(p: *const u8, k: usize) -> [u8; 64] {
    let mut pad = [0u8; 64];
    let d = pad.as_mut_ptr();
    let mut i = 0usize;
    while i + 16 <= k {
        (d.add(i) as *mut u128).write_unaligned((p.add(i) as *const u128).read_unaligned());
        i += 16;
    }
    if i + 8 <= k {
        (d.add(i) as *mut u64).write_unaligned((p.add(i) as *const u64).read_unaligned());
        i += 8;
    }
    if i + 4 <= k {
        (d.add(i) as *mut u32).write_unaligned((p.add(i) as *const u32).read_unaligned());
        i += 4;
    }
    if i + 2 <= k {
        (d.add(i) as *mut u16).write_unaligned((p.add(i) as *const u16).read_unaligned());
        i += 2;
    }
    if i < k {
        *d.add(i) = *p.add(i);
    }
    pad
}

/// Internal SIMD backend trait. Every method is `#[inline(always)]` so calling
/// generic block/driver functions from a `#[target_feature]` entry point compiles
/// the entire loop with those CPU features enabled (zero per-block dispatch).
trait Simd: Copy {
    type Vec64: Copy;

    /// Load 64 readable bytes at `p`.
    unsafe fn load64(p: *const u8) -> Self::Vec64;

    /// Load `k` readable bytes (`0 <= k < 64`) zero-padded to 64 bytes.
    unsafe fn load_tail(p: *const u8, k: usize) -> Self::Vec64;

    /// Classify a loaded 64-byte block (all 64 lanes).
    unsafe fn classify_vec(v: Self::Vec64) -> Classes;

    /// Branch-free SIMD resolver for blocks whose non-ASCII bytes are purely
    /// in-block valid UTF-8 Latin-1 `0xC3 (0x80..=0xBF)` pairs (U+00C0..U+00FF).
    ///
    /// Reuses the already-loaded `v` (zero second memory load). If every set bit
    /// in `m.na` belongs to a disjoint valid `0xC3 (0x80..=0xBF)` pair inside
    /// `valid`, updates `m.l`, `m.up`, sets `m.na = 0`, and returns
    /// `(cont, punct_pair, true)`. Otherwise leaves `m` untouched and returns
    /// `(0, 0, false)`.
    unsafe fn resolve_latin1_vec(v: Self::Vec64, valid: u64, m: &mut Classes) -> (u64, u64, bool);

    /// Extract set-bit positions of `trusted` (adding `base`) into `out[k..]`,
    /// returning the updated `k`.
    unsafe fn extract_ends(trusted: u64, base: usize, op: *mut u32, k: usize) -> usize;
}

#[cfg(not(target_arch = "aarch64"))]
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
    type Vec64 = std::arch::aarch64::uint8x16x4_t;

    #[inline(always)]
    unsafe fn load64(p: *const u8) -> Self::Vec64 {
        std::arch::aarch64::vld4q_u8(p)
    }

    #[inline(always)]
    unsafe fn load_tail(p: *const u8, k: usize) -> Self::Vec64 {
        let pad = copy_tail_pad(p, k);
        std::arch::aarch64::vld4q_u8(pad.as_ptr())
    }

    #[inline(always)]
    unsafe fn classify_vec(v: Self::Vec64) -> Classes {
        use std::arch::aarch64::*;
        let x = [v.0, v.1, v.2, v.3];
        let k = |v: u8| vdupq_n_u8(v);
        let l = Self::tomask(x.map(|v| vcltq_u8(vsubq_u8(vorrq_u8(v, k(0x20)), k(b'a')), k(26))));
        // `tomask` reads only bit 7 of each byte: shift bit 5 (`0x20`) to bit 7
        // for `up`, reuse `sp` for `ws`, and pass `x` directly for `na` (-16 ops/64B).
        let sp = Self::tomask(x.map(|v| vceqq_u8(v, k(b' '))));
        Classes {
            l,
            up: !Self::tomask(x.map(|v| vshlq_n_u8::<2>(v))) & l,
            d: Self::tomask(x.map(|v| vcltq_u8(vsubq_u8(v, k(b'0')), k(10)))),
            sp,
            nl: Self::tomask(x.map(|v| vorrq_u8(vceqq_u8(v, k(b'\n')), vceqq_u8(v, k(b'\r'))))),
            ws: sp | Self::tomask(x.map(|v| vcltq_u8(vsubq_u8(v, k(0x09)), k(5)))),
            q: Self::tomask(x.map(|v| vceqq_u8(v, k(b'\'')))),
            sl: Self::tomask(x.map(|v| vceqq_u8(v, k(b'/')))),
            na: Self::tomask(x),
        }
    }

    #[inline(always)]
    unsafe fn extract_ends(trusted: u64, base: usize, op: *mut u32, k: usize) -> usize {
        use std::arch::aarch64::*;
        // Hardware per-byte `cnt` + 2 KiB `POS8` table (4x smaller in L1d than 8 KiB POS32).
        let cnt_v = vcnt_u8(vreinterpret_u8_u64(vdup_n_u64(trusted)));
        let cnts = vget_lane_u64::<0>(vreinterpret_u64_u8(cnt_v));
        let incl = cnts.wrapping_mul(0x0101_0101_0101_0101);
        let excl = incl << 8;
        let mut add0 = vdupq_n_u32(base as u32);
        let mut add1 = vdupq_n_u32((base + 8) as u32);
        let c16 = vdupq_n_u32(16);
        let p8 = POS8.as_ptr() as *const u8;
        for pair in 0..4usize {
            let j0 = pair * 2;
            let b0 = (trusted >> (8 * j0)) as u8 as usize;
            let r16_0 = vmovl_u8(vld1_u8(p8.add(b0 * 8)));
            let at0 = k + ((excl >> (8 * j0)) as u8 as usize);
            vst1q_u32(op.add(at0), vaddq_u32(vmovl_u16(vget_low_u16(r16_0)), add0));
            vst1q_u32(op.add(at0 + 4), vaddq_u32(vmovl_high_u16(r16_0), add0));
            add0 = vaddq_u32(add0, c16);

            let j1 = j0 + 1;
            let b1 = (trusted >> (8 * j1)) as u8 as usize;
            let r16_1 = vmovl_u8(vld1_u8(p8.add(b1 * 8)));
            let at1 = k + ((excl >> (8 * j1)) as u8 as usize);
            vst1q_u32(op.add(at1), vaddq_u32(vmovl_u16(vget_low_u16(r16_1)), add1));
            vst1q_u32(op.add(at1 + 4), vaddq_u32(vmovl_high_u16(r16_1), add1));
            add1 = vaddq_u32(add1, c16);
        }
        k + (incl >> 56) as usize
    }

    #[inline(always)]
    unsafe fn resolve_latin1_vec(v: Self::Vec64, valid: u64, m: &mut Classes) -> (u64, u64, bool) {
        use std::arch::aarch64::*;
        let x = [v.0, v.1, v.2, v.3];
        let k = |c: u8| vdupq_n_u8(c);
        let c3 = Self::tomask(x.map(|v| vceqq_u8(v, k(0xC3)))) & (valid >> 1);
        let c3_pair = c3 | (c3 << 1);
        if m.na != c3_pair || (c3 & (c3 << 1)) != 0 {
            return (0, 0, false);
        }
        let foll = c3 << 1;
        // Every follower byte is already >= 0x80 (since m.na == c3_pair); require <= 0xBF for valid UTF-8.
        if (Self::tomask(x.map(|v| vcgtq_u8(v, k(0xBF)))) & foll) != 0 {
            return (0, 0, false);
        }
        let md = Self::tomask(x.map(|v| vorrq_u8(vceqq_u8(v, k(0x97)), vceqq_u8(v, k(0xB7))))) & foll;
        let punct = md | (md >> 1);
        let letters = c3_pair & !punct;
        let up_f = Self::tomask(x.map(|v| vcleq_u8(v, k(0x9E)))) & foll & !md;
        let upper = up_f | (up_f >> 1);
        m.l |= letters;
        m.up |= upper;
        m.na = 0;
        (foll, punct, true)
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Sse2;

#[cfg(target_arch = "x86_64")]
impl Simd for Sse2 {
    type Vec64 = [std::arch::x86_64::__m128i; 4];

    #[inline(always)]
    unsafe fn load64(p: *const u8) -> Self::Vec64 {
        use std::arch::x86_64::*;
        [
            _mm_loadu_si128(p as *const __m128i),
            _mm_loadu_si128(p.add(16) as *const __m128i),
            _mm_loadu_si128(p.add(32) as *const __m128i),
            _mm_loadu_si128(p.add(48) as *const __m128i),
        ]
    }

    #[inline(always)]
    unsafe fn load_tail(p: *const u8, k: usize) -> Self::Vec64 {
        let pad = copy_tail_pad(p, k);
        Self::load64(pad.as_ptr())
    }

    #[inline(always)]
    unsafe fn classify_vec(x: Self::Vec64) -> Classes {
        use std::arch::x86_64::*;
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
    unsafe fn resolve_latin1_vec(x: Self::Vec64, valid: u64, m: &mut Classes) -> (u64, u64, bool) {
        use std::arch::x86_64::*;
        let k = |v: u8| _mm_set1_epi8(v as i8);
        let tomask = |a: [__m128i; 4]| -> u64 {
            (_mm_movemask_epi8(a[0]) as u16 as u64)
                | ((_mm_movemask_epi8(a[1]) as u16 as u64) << 16)
                | ((_mm_movemask_epi8(a[2]) as u16 as u64) << 32)
                | ((_mm_movemask_epi8(a[3]) as u16 as u64) << 48)
        };
        let c3 = tomask(x.map(|v| _mm_cmpeq_epi8(v, k(0xC3)))) & (valid >> 1);
        let c3_pair = c3 | (c3 << 1);
        if m.na != c3_pair || (c3 & (c3 << 1)) != 0 {
            return (0, 0, false);
        }
        let foll = c3 << 1;
        // Signed i8: 0x80..=0xBF is -128..=-65; 0xC0..=0xFF is -64..=-1 > -65 (0xBF).
        if (tomask(x.map(|v| _mm_cmpgt_epi8(v, k(0xBF)))) & foll) != 0 {
            return (0, 0, false);
        }
        let md = tomask(x.map(|v| _mm_or_si128(_mm_cmpeq_epi8(v, k(0x97)), _mm_cmpeq_epi8(v, k(0xB7))))) & foll;
        let punct = md | (md >> 1);
        let letters = c3_pair & !punct;
        // Signed i8 comparison: 0x80..=0x9E is -128..=-98, strictly < -97 (0x9F).
        let up_f = tomask(x.map(|v| _mm_cmplt_epi8(v, k(0x9F)))) & foll & !md;
        let upper = up_f | (up_f >> 1);
        m.l |= letters;
        m.up |= upper;
        m.na = 0;
        (foll, punct, true)
    }

    #[inline(always)]
    unsafe fn extract_ends(trusted: u64, base: usize, op: *mut u32, k: usize) -> usize {
        extract_ends_pos32(trusted, base, op, k)
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Avx2;

#[cfg(target_arch = "x86_64")]
impl Simd for Avx2 {
    type Vec64 = (std::arch::x86_64::__m256i, std::arch::x86_64::__m256i);

    #[inline(always)]
    unsafe fn load64(p: *const u8) -> Self::Vec64 {
        use std::arch::x86_64::*;
        (
            _mm256_loadu_si256(p as *const __m256i),
            _mm256_loadu_si256(p.add(32) as *const __m256i),
        )
    }

    #[inline(always)]
    unsafe fn load_tail(p: *const u8, k: usize) -> Self::Vec64 {
        let pad = copy_tail_pad(p, k);
        Self::load64(pad.as_ptr())
    }

    #[inline(always)]
    unsafe fn classify_vec((x0, x1): Self::Vec64) -> Classes {
        use std::arch::x86_64::*;
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
    unsafe fn resolve_latin1_vec((x0, x1): Self::Vec64, valid: u64, m: &mut Classes) -> (u64, u64, bool) {
        use std::arch::x86_64::*;
        let k = |v: u8| _mm256_set1_epi8(v as i8);
        let tomask = |a: __m256i, b: __m256i| -> u64 {
            (_mm256_movemask_epi8(a) as u32 as u64)
                | ((_mm256_movemask_epi8(b) as u32 as u64) << 32)
        };
        let c3 = tomask(_mm256_cmpeq_epi8(x0, k(0xC3)), _mm256_cmpeq_epi8(x1, k(0xC3))) & (valid >> 1);
        let c3_pair = c3 | (c3 << 1);
        if m.na != c3_pair || (c3 & (c3 << 1)) != 0 {
            return (0, 0, false);
        }
        let foll = c3 << 1;
        let kbf = k(0xBF);
        if (tomask(_mm256_cmpgt_epi8(x0, kbf), _mm256_cmpgt_epi8(x1, kbf)) & foll) != 0 {
            return (0, 0, false);
        }
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
        (foll, punct, true)
    }

    #[inline(always)]
    unsafe fn extract_ends(trusted: u64, base: usize, op: *mut u32, k: usize) -> usize {
        extract_ends_pos32(trusted, base, op, k)
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Avx512;

#[cfg(target_arch = "x86_64")]
impl Simd for Avx512 {
    type Vec64 = std::arch::x86_64::__m512i;

    #[inline(always)]
    unsafe fn load64(p: *const u8) -> Self::Vec64 {
        std::arch::x86_64::_mm512_loadu_si512(p as *const _)
    }

    #[inline(always)]
    unsafe fn load_tail(p: *const u8, k: usize) -> Self::Vec64 {
        let mask = if k == 0 { 0u64 } else { (1u64 << k) - 1 };
        std::arch::x86_64::_mm512_maskz_loadu_epi8(mask, p as *const i8)
    }

    #[inline(always)]
    unsafe fn classify_vec(v: Self::Vec64) -> Classes {
        use std::arch::x86_64::*;
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
    unsafe fn resolve_latin1_vec(v: Self::Vec64, valid: u64, m: &mut Classes) -> (u64, u64, bool) {
        use std::arch::x86_64::*;
        let k = |c: u8| _mm512_set1_epi8(c as i8);
        let c3 = _mm512_cmpeq_epi8_mask(v, k(0xC3)) & (valid >> 1);
        let c3_pair = c3 | (c3 << 1);
        if m.na != c3_pair || (c3 & (c3 << 1)) != 0 {
            return (0, 0, false);
        }
        let foll = c3 << 1;
        if (_mm512_cmpgt_epu8_mask(v, k(0xBF)) & foll) != 0 {
            return (0, 0, false);
        }
        let md = (_mm512_cmpeq_epi8_mask(v, k(0x97)) | _mm512_cmpeq_epi8_mask(v, k(0xB7))) & foll;
        let punct = md | (md >> 1);
        let letters = c3_pair & !punct;
        let up_f = _mm512_cmple_epu8_mask(v, k(0x9E)) & foll & !md;
        let upper = up_f | (up_f >> 1);
        m.l |= letters;
        m.up |= upper;
        m.na = 0;
        (foll, punct, true)
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

#[cfg(target_arch = "x86_64")]
static RAMP64: [u8; 64] = {
    let mut a = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        a[i] = i as u8;
        i += 1;
    }
    a
};

/// AVX-512BW+VL+VBMI2 (Zen 4 / Ice Lake+): compresses all 64 chunk-end byte
/// positions in ONE `vpcompressb` instruction + ONE `popcnt`, then widens only
/// the actual `cnt` positions with disjoint 64-byte stores (zero overlapping
/// store-to-load forwarding stalls, 2-4x fewer stores per block).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Avx512Vbmi2;

#[cfg(target_arch = "x86_64")]
impl Simd for Avx512Vbmi2 {
    type Vec64 = std::arch::x86_64::__m512i;

    #[inline(always)]
    unsafe fn load64(p: *const u8) -> Self::Vec64 {
        Avx512::load64(p)
    }

    #[inline(always)]
    unsafe fn load_tail(p: *const u8, k: usize) -> Self::Vec64 {
        Avx512::load_tail(p, k)
    }

    #[inline(always)]
    unsafe fn classify_vec(v: Self::Vec64) -> Classes {
        Avx512::classify_vec(v)
    }

    #[inline(always)]
    unsafe fn resolve_latin1_vec(v: Self::Vec64, valid: u64, m: &mut Classes) -> (u64, u64, bool) {
        Avx512::resolve_latin1_vec(v, valid, m)
    }

    #[inline(always)]
    unsafe fn extract_ends(trusted: u64, base: usize, op: *mut u32, k: usize) -> usize {
        use std::arch::x86_64::*;
        let ramp = _mm512_loadu_si512(RAMP64.as_ptr() as *const _);
        let comp = _mm512_maskz_compress_epi8(trusted, ramp);
        let cnt = trusted.count_ones() as usize;
        let vbase = _mm512_set1_epi32(base as i32);
        let dst = op.add(k);
        // Unconditional 32-lane disjoint store covers ~99.5% of blocks (both
        // English/code and German) with zero 50/50 `cnt > 16` misprediction,
        // and fits safely inside SCAN_SLACK=80.
        let lo0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(comp));
        let lo1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<1>(comp));
        _mm512_storeu_si512(dst as *mut _, _mm512_add_epi32(lo0, vbase));
        _mm512_storeu_si512(dst.add(16) as *mut _, _mm512_add_epi32(lo1, vbase));
        if cnt > 32 {
            let lo2 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<2>(comp));
            _mm512_storeu_si512(dst.add(32) as *mut _, _mm512_add_epi32(lo2, vbase));
            if cnt > 48 {
                let lo3 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<3>(comp));
                _mm512_storeu_si512(dst.add(48) as *mut _, _mm512_add_epi32(lo3, vbase));
            }
        }
        k + cnt
    }
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[derive(Clone, Copy)]
struct Scalar;

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
impl Simd for Scalar {
    type Vec64 = [u8; 64];

    #[inline(always)]
    unsafe fn load64(p: *const u8) -> Self::Vec64 {
        let mut a = [0u8; 64];
        std::ptr::copy_nonoverlapping(p, a.as_mut_ptr(), 64);
        a
    }

    #[inline(always)]
    unsafe fn load_tail(p: *const u8, k: usize) -> Self::Vec64 {
        copy_tail_pad(p, k)
    }

    #[inline(always)]
    unsafe fn classify_vec(v: Self::Vec64) -> Classes {
        classify_scalar(&v, 0)
    }

    #[inline(always)]
    unsafe fn resolve_latin1_vec(_v: Self::Vec64, _valid: u64, _m: &mut Classes) -> (u64, u64, bool) {
        (0, 0, false)
    }

    #[inline(always)]
    unsafe fn extract_ends(trusted: u64, base: usize, op: *mut u32, k: usize) -> usize {
        extract_ends_pos32(trusted, base, op, k)
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
        if std::is_x86_feature_detected!("avx512vbmi2") { 4 } else { 3 }
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
unsafe fn load_and_classify_tail<S: Simd>(avail: &[u8], doc_len: usize, base: usize) -> (S::Vec64, Classes, u64) {
    let rem_doc = doc_len.saturating_sub(base);
    let valid = if rem_doc >= 64 {
        u64::MAX
    } else if rem_doc == 0 {
        0
    } else {
        (1u64 << rem_doc) - 1
    };
    let v = if base + 64 <= avail.len() {
        S::load64(avail.as_ptr().add(base))
    } else {
        S::load_tail(avail.as_ptr().add(base), avail.len().saturating_sub(base))
    };
    let m = S::classify_vec(v).mask(valid);
    (v, m, valid)
}

#[inline(always)]
#[allow(dead_code)]
unsafe fn classify_with<S: Simd>(b: &[u8], base: usize) -> Classes {
    if base + 64 <= b.len() {
        return S::classify_vec(S::load64(b.as_ptr().add(base)));
    }
    load_and_classify_tail::<S>(b, b.len(), base).1
}

/// Classes of the 64 bytes at `base` (fewer near the end of `b`).
#[inline(always)]
#[allow(dead_code)]
pub fn classify(b: &[u8], base: usize) -> Classes {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { classify_with::<Neon>(b, base) }
    }
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
        match x86_tier() {
            3 | 4 => unsafe { c512(b, base) },
            2 => unsafe { c256(b, base) },
            _ => unsafe { classify_with::<Sse2>(b, base) },
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        classify_scalar(b, base)
    }
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

/// Bit layout for the single-register `Carry::flags` word.
/// Every common-case cross-block bit + `forced0` + `bad3` + `na4` lives in one
/// `u32` register; `SlowCarry` is ONLY read/written when `(flags & F_SLOW) != 0`.
// Bit positions: 0=o, 1=sp, 2=d, 3=nl, 4=ws, 5=l, 6=lw, 7=na, 8=ostart,
// 9=abs_nl, 10=abs_sl, 11=ws_unk, 12=sl_unk, 13=contr, 14=forced0, 15=slow.
const F_D: u32 = 1 << 2;
const F_WS: u32 = 1 << 4;
const F_NA: u32 = 1 << 7;
const F_ABS_NL: u32 = 1 << 9;
const F_ABS_SL: u32 = 1 << 10;
const F_CONTR: u32 = 1 << 13;
/// Chunk start carried into byte 0 of the next block (set on start & realign).
const F_FORCED0: u32 = 1 << 14;
/// Set iff any slow-path carry is live (`na`, `na4`, `bad_in`, `contr_mask`,
/// `forced & !1`, `suppress`). When clear, `SlowCarry` is never touched.
const F_SLOW: u32 = 1 << 15;
const BAD3_SHIFT: u32 = 16; // bits 16..=18: BAD flags of base-3..base-1 (bit 18 = base-1)
const NA4_SHIFT: u32 = 20;  // bits 20..=23: bit k-1 = byte base-k is non-ASCII, k = 1..4

/// Slow-only cross-block carry masks (contraction spans crossing a 64B edge).
/// Valid iff `(flags & F_SLOW) != 0`; never touched on the fast path.
#[derive(Clone, Copy, Default)]
struct SlowCarry {
    bad_in: u64,
    contr_mask: u64,
    forced: u64,
    suppress: u64,
}

/// State carried from one block to the next in a sequential scan.
#[derive(Clone, Copy, Default)]
pub struct Carry {
    flags: u32,
    slow: SlowCarry,
}

impl Carry {
    #[inline(always)]
    const fn forced_start() -> Self {
        Carry {
            flags: F_FORCED0,
            slow: SlowCarry { bad_in: 0, contr_mask: 0, forced: 0, suppress: 0 },
        }
    }
}

/// Result of one block: (starts, bad, carry for the next block).
type Block = (u64, u64, Carry);

#[inline(always)]
fn fbit(flags: u32, mask: u32) -> bool {
    (flags & mask) != 0
}
#[inline(always)]
fn p1(x: u64, c: bool) -> u64 {
    (x << 1) | c as u64
}
#[inline(always)]
fn p1f(x: u64, flags: u32, bit: u32) -> u64 {
    (x << 1) | (((flags >> bit) & 1) as u64)
}
#[inline(always)]
fn top_bit(v: u64, bit: u32) -> u32 {
    ((v >> 63) as u32) << bit
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
            if (d & 0xC0) == 0x80 {
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
        next_ws: next.is_some_and(is_ws),
        next_na: next.is_some_and(|c| c >= 0x80),
        cont,
    }
}

#[inline(always)]
fn common_bad(x: &Ctx, flags: u32) -> u64 {
    let m = &x.m;
    let na4 = ((flags >> NA4_SHIFT) & 0xF) as u64;
    let mut bad = m.na | (m.na << 1) | (m.na << 2) | (m.na << 3) | (m.na << 4);
    bad |= (16u64 >> na4.trailing_zeros().min(63)).saturating_sub(1);
    bad |= m.ws & (p1f(m.na, flags, 7) | n1(m.na, x.next_na));
    bad |= m.ws & !m.ws.wrapping_add(1) & (((flags >> 4) & 1) as u64).wrapping_neg();
    bad
}

#[inline(always)]
fn digit_starts(m: &Classes, flags: u32, k: u32, prefix_sp: bool) -> (u64, u64) {
    let runs = m.d & !p1f(m.d, flags, 2);
    let joined = if prefix_sp { p1f(m.sp, flags, 1) } else { 0 };
    match k {
        0 => (runs & !joined, 0),
        1 => (m.d, 0),
        k => {
            let mut dm = m.d;
            for t in 1..k {
                dm &= m.d << t;
            }
            let flag = runs & p1f(m.na, flags, 7);
            let mut bad = (m.d.wrapping_add(flag) ^ m.d) & m.d;
            let mut st = runs & !flag;
            let mut sh = k;
            while sh < 64 {
                st |= (st << sh) & dm;
                dm &= dm << sh;
                sh <<= 1;
            }
            st &= !bad;
            bad |= m.d & !m.d.wrapping_add(1) & (((flags >> 2) & 1) as u64).wrapping_neg();
            (st, bad)
        }
    }
}

#[cold]
#[inline(never)]
fn digit_starts_ascii_general(d: u64, runs: u64, cd: bool, k: u32) -> (u64, u64) {
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

/// Pure-ASCII digit-group starts (no `m.na` / `c.na`, `prefix_sp == false`).
/// Unrolls `k == 3` (o200k, cl100k, Llama-3, DeepSeek, MiniMax) and moves
/// rare general `k` out-of-line so LLVM never emits vectorized `1..k` loop
/// bloat inside the hot 64-byte scanner.
#[inline(always)]
fn digit_starts_ascii<const K: u32>(d: u64, cd: bool, runtime_k: u32) -> (u64, u64) {
    let k = if K == u32::MAX { runtime_k } else { K };
    let runs = d & !p1(d, cd);
    match k {
        0 => (runs, 0),
        1 => (d, 0),
        3 => {
            let bad = d & !d.wrapping_add(1) & (cd as u64).wrapping_neg();
            let dm = d & (d << 1) & (d << 2);
            let mut st = runs;
            let first_extra = (runs << 3) & dm;
            if first_extra != 0 {
                st |= first_extra;
                let dm2 = dm & (dm << 3);
                st |= (st << 6) & dm2;
                let dm4 = dm2 & (dm2 << 6);
                st |= (st << 12) & dm4;
                let dm8 = dm4 & (dm4 << 12);
                st |= (st << 24) & dm8;
                st |= (st << 48) & (dm8 & (dm8 << 24));
            }
            (st & !bad, bad)
        }
        k => digit_starts_ascii_general(d, runs, cd, k),
    }
}

#[inline(always)]
fn ws_starts_nl(x: &Ctx, flags: u32, base: usize, n: usize) -> (u64, u64, u64) {
    let m = &x.m;
    let c_ws = fbit(flags, F_WS);
    let c_na = fbit(flags, F_NA);
    let long = m.ws & (p1(m.ws, c_ws) | n1(m.ws, x.next_ws));
    let o_before = p1f(x.o, flags, 0);
    let single_abs = m.nl & !long & o_before;
    let mut starts = m.ws & !long & !single_abs;
    let (mut bad, mut abs) = (0u64, single_abs);
    let mut runs = long & !p1(long, c_ws);
    while runs != 0 {
        let r = runs.trailing_zeros() as usize;
        runs &= runs - 1;
        let e = r + ((!m.ws) >> r).trailing_zeros() as usize;
        let run = run_mask(r, e);
        let na_before = if r == 0 { c_na } else { m.na >> (r - 1) & 1 == 1 };
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
/// resolved Latin-1 blocks (including final `<64B` document tails via `valid`!).
///
/// Uses the Exact Segmented Right-Smear identity:
///   `right_smear(M, S) = (((!M).wrapping_sub(S)) | S) & M`
/// to smear any arbitrary subset `S ⊆ M` to the end of every run in `M` in
/// 4 instructions with zero loops — handling `\n\n`, `\n    `, and `o`-absorbed
/// leading newlines completely word-wide.
#[inline(always)]
fn ws_starts_nl_fast<const FULL: bool>(
    b: &[u8],
    base: usize,
    valid: u64,
    m: &Classes,
    o_before: u64,
    c_ws: bool,
) -> (u64, u64, u64, u64, bool) {
    let (next_ws, next_na) = if FULL {
        let next = unsafe { *b.get_unchecked(base + 64) };
        (is_ws(next), next >= 0x80)
    } else {
        (false, false)
    };
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
    let r_starts = good & !(good << 1);
    let abs_seed = r_starts & o_before & nl_g;
    let abs_lr = (nl_g.wrapping_add(abs_seed) ^ nl_g) & nl_g;
    let rem = good & !abs_lr;
    let nl_rem = nl_g & rem;
    let non_nl = rem & !nl_rem;
    let cand_l1 = non_nl & (nl_rem << 1);
    let l1_starts = if (non_nl.wrapping_add(cand_l1) & nl_rem) == 0 {
        cand_l1
    } else {
        let r_rem = rem.reverse_bits();
        let r_nl = nl_rem.reverse_bits();
        let at_or_before = (((!r_rem).wrapping_sub(r_nl) | r_nl) & r_rem).reverse_bits();
        let after_nl = rem & !at_or_before;
        after_nl & !(after_nl << 1)
    };
    let end_split = if FULL {
        non_nl & !(rem >> 1) & (non_nl << 1)
    } else {
        non_nl & !(rem >> 1) & (non_nl << 1) & (valid >> 1)
    };
    let wsstart = (m.ws & !long & !single_abs)
        | (rem & !(rem << 1))
        | l1_starts
        | end_split;
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
fn generic_fast<const FULL: bool, const K: u32>(
    b: &[u8],
    base: usize,
    max_digits: u32,
    flags: u32,
    m: &Classes,
    valid: u64,
    cont: u64,
) -> Option<(u64, u64, u32)> {
    let o = if FULL { !(m.l | m.d | m.ws) } else { !(m.l | m.d | m.ws) & valid };
    let o_before = p1f(o, flags, 0);
    let ostart = o & !o_before & !p1f(m.sp, flags, 1);
    let lstart = m.l & (p1f(m.d, flags, 2) | p1f(m.nl, flags, 3) | (o_before & !p1f(ostart, flags, 8)));
    let (dstart, dbad) = digit_starts_ascii::<K>(m.d, fbit(flags, F_D), max_digits);
    let (wsstart, wsbad, _, cont_ws, next_na) = ws_starts_nl_fast::<FULL>(b, base, valid, m, o_before, fbit(flags, F_WS));
    let mut starts = ostart | lstart | dstart | wsstart | (((flags >> 14) & 1) as u64);
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
    bad |= m.l & o_before & p1f(bad, flags, 18);
    bad |= (bad << 1) & cont;
    if !FULL {
        starts &= valid;
        bad &= valid;
    }
    let out_flags = top_bit(o, 0)
        | top_bit(m.sp, 1)
        | top_bit(m.d, 2)
        | top_bit(m.nl, 3)
        | top_bit(m.ws, 4)
        | top_bit(ostart, 8)
        | (((bad >> 61) as u32) << BAD3_SHIFT);
    Some((starts, bad, out_flags))
}

#[cold]
#[inline(never)]
fn generic_slow(b: &[u8], base: usize, max_digits: u32, flags: u32, slow: &mut SlowCarry, m: Classes) -> (u64, u64, u32) {
    let (c_bad_in, c_forced) = if fbit(flags, F_SLOW) {
        (slow.bad_in, slow.forced)
    } else {
        (0, ((flags >> 14) & 1) as u64)
    };
    let n = b.len();
    let x = ctx_m::<true, true>(b, base, m);
    let (m, o) = (&x.m, x.o);
    let ostart = o & !p1f(o, flags, 0) & !p1f(m.sp, flags, 1);
    let ostart_c = ostart | ((ostart << 1) & x.cont);
    let ostart_c = ostart_c | ((ostart_c << 1) & x.cont);
    let lstart = m.l & (p1f(m.d, flags, 2) | p1f(m.nl, flags, 3) | (p1f(o, flags, 0) & !p1f(ostart_c, flags, 8)));
    let (dstart, dbad) = digit_starts(m, flags, max_digits, false);
    let (wsstart, wsbad, _) = ws_starts_nl(&x, flags, base, n);
    let mut starts = ostart | lstart | dstart | wsstart | c_forced;
    let mut forced = 0u64;
    let mut bad = common_bad(&x, flags) | dbad | wsbad | c_bad_in;
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
    bad |= m.l & p1f(o, flags, 0) & p1f(bad_c, flags, 18);
    let bad = bad | ((bad << 1) & x.cont) | ((bad << 2) & x.cont & (x.cont << 1));
    let na4 = ((m.na >> 60).reverse_bits() >> 60) as u32;
    let has_slow = (m.na >> 63 != 0) || na4 != 0 || (bad_next | (forced & !1)) != 0;
    if has_slow {
        *slow = SlowCarry { bad_in: bad_next, contr_mask: 0, forced, suppress: 0 };
    }
    let out_flags = top_bit(x.o, 0)
        | top_bit(m.sp, 1)
        | top_bit(m.d, 2)
        | top_bit(m.nl, 3)
        | top_bit(m.ws, 4)
        | top_bit(m.na, 7)
        | top_bit(ostart_c, 8)
        | (((forced & 1) as u32) << 14)
        | if has_slow { F_SLOW } else { 0 }
        | (((bad >> 61) as u32) << BAD3_SHIFT)
        | (na4 << NA4_SHIFT);
    (starts & x.valid, bad & x.valid, out_flags)
}

#[inline(always)]
unsafe fn generic_block<S: Simd, const K: u32>(avail: &[u8], doc_len: usize, base: usize, max_digits: u32, flags: u32, slow: &mut SlowCarry) -> (u64, u64, u32, bool) {
    let doc = &avail[..doc_len];
    if base + 64 < doc_len {
        let v = S::load64(avail.as_ptr().add(base));
        let m = S::classify_vec(v);
        if (flags & F_SLOW) == 0 {
            let mut rm = m;
            let mut cont = 0u64;
            let mut ok = true;
            if rm.na != 0 {
                let (c, punct, utf8_ok) = S::resolve_latin1_vec(v, u64::MAX, &mut rm);
                ok = utf8_ok && rm.na == 0 && punct == 0;
                cont = c;
            }
            if ok {
                if let Some((s, bad, nf)) = generic_fast::<true, K>(doc, base, max_digits, flags, &rm, u64::MAX, cont) {
                    return (s, bad, nf, true);
                }
            }
        }
        let (s, bad, nf) = generic_slow(doc, base, max_digits, flags, slow, m);
        return (s, bad, nf, false);
    }
    let (v, m, valid) = load_and_classify_tail::<S>(avail, doc_len, base);
    if (flags & F_SLOW) == 0 {
        let mut rm = m;
        let mut cont = 0u64;
        let mut ok = true;
        if rm.na != 0 {
            let (c, punct, utf8_ok) = S::resolve_latin1_vec(v, valid, &mut rm);
            ok = utf8_ok && rm.na == 0 && punct == 0;
            cont = c;
        }
        if ok {
            if let Some((s, bad, nf)) = generic_fast::<false, K>(doc, base, max_digits, flags, &rm, valid, cont) {
                return (s, bad, nf, true);
            }
        }
    }
    let (s, bad, nf) = generic_slow(doc, base, max_digits, flags, slow, m);
    (s, bad, nf, false)
}

// ---------------------------------------------------------------------------
// CamelCase grammar (o200k, MiniMax, ...)
// ---------------------------------------------------------------------------

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn camel_fast<const FULL: bool, const C_MODE: u8, const K: u32>(
    b: &[u8],
    base: usize,
    contractions: bool,
    max_digits: u32,
    flags: u32,
    m: &Classes,
    valid: u64,
    cont: u64,
) -> Option<(u64, u64, u32)> {
    let contr = match C_MODE {
        1 => true,
        0 => false,
        _ => contractions,
    };
    let o = if FULL { !(m.l | m.d | m.ws) } else { !(m.l | m.d | m.ws) & valid };
    let lw = m.l & !m.up;
    let o_before = p1f(o, flags, 0);
    let (wsstart, wsbad, abs_nl, cont_ws, next_na) = ws_starts_nl_fast::<FULL>(b, base, valid, m, o_before, fbit(flags, F_WS));
    let seed = m.sl & p1(abs_nl, (flags & (F_ABS_NL | F_ABS_SL)) != 0);
    let abs_sl = (m.sl.wrapping_add(seed) ^ m.sl) & m.sl;
    let mut ostart = (o & !o_before & !p1f(m.sp, flags, 1) & !abs_sl) | (o & !m.sl & p1f(abs_sl, flags, 10));
    let (dstart, dbad) = digit_starts_ascii::<K>(m.d, fbit(flags, F_D), max_digits);
    let mut bad = cont_ws | (m.ws & ((next_na as u64) << 63)) | dbad | wsbad;
    let ws_unk = m.ws & bad;
    let useed = (m.sl & p1f(ws_unk, flags, 11)) | (m.sl & (((flags >> 12) & 1) as u64));
    let sl_unk = (m.sl.wrapping_add(useed) ^ m.sl) & m.sl;
    bad |= sl_unk | (p1f(sl_unk, flags, 12) & !m.sl);
    let (mut suppress, mut forced_in, mut contr_end) = (0u64, 0u64, 0u64);
    if contr {
        let mut qs = m.q & p1f(m.l, flags, 5);
        let c_bad3 = ((flags >> BAD3_SHIFT) & 7) as u64;
        while qs != 0 {
            let i = qs.trailing_zeros() as usize;
            qs &= qs - 1;
            let window = ((bad << 3) | c_bad3) >> i & 0b1111;
            if window != 0 {
                if i >= 60 {
                    return None;
                }
                bad |= run_mask(i, i + 4);
            }
            let prev_is_suffix_end = if i == 0 { fbit(flags, F_CONTR) } else { contr_end >> (i - 1) & 1 == 1 };
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
        & (p1f(m.d, flags, 2) | p1f(m.nl, flags, 3) | (o_before & !p1f(ostart, flags, 8)) | (m.up & p1f(lw, flags, 6)));
    let mut starts = ((ostart | lstart | dstart | wsstart) & !suppress) | (((flags >> 14) & 1) as u64) | forced_in;
    bad |= m.l & o_before & p1f(bad, flags, 18);
    bad |= (bad << 1) & cont;
    if !FULL {
        starts &= valid;
        bad &= valid;
    }
    let out_flags = top_bit(o, 0)
        | top_bit(m.sp, 1)
        | top_bit(m.d, 2)
        | top_bit(m.nl, 3)
        | top_bit(m.ws, 4)
        | top_bit(m.l, 5)
        | top_bit(lw, 6)
        | top_bit(ostart, 8)
        | top_bit(abs_nl, 9)
        | top_bit(abs_sl, 10)
        | top_bit(ws_unk, 11)
        | top_bit(sl_unk, 12)
        | top_bit(contr_end, 13)
        | (((bad >> 61) as u32) << BAD3_SHIFT);
    Some((starts, bad, out_flags))
}

#[cold]
#[inline(never)]
fn camel_slow(b: &[u8], base: usize, contractions: bool, max_digits: u32, flags: u32, slow: &mut SlowCarry, m: Classes) -> (u64, u64, u32) {
    let (c_bad_in, c_contr_mask, c_forced, c_suppress) = if fbit(flags, F_SLOW) {
        (slow.bad_in, slow.contr_mask, slow.forced, slow.suppress)
    } else {
        (0, 0, ((flags >> 14) & 1) as u64, 0)
    };
    let n = b.len();
    let x = ctx_m::<true, true>(b, base, m);
    let (m, o) = (&x.m, x.o);
    let lw = m.l & !m.up;
    let (wsstart, wsbad, abs_nl) = ws_starts_nl(&x, flags, base, n);
    let seed = m.sl & p1(abs_nl, (flags & (F_ABS_NL | F_ABS_SL)) != 0);
    let abs_sl = (m.sl.wrapping_add(seed) ^ m.sl) & m.sl;
    let mut ostart = (o & !p1f(o, flags, 0) & !p1f(m.sp, flags, 1) & !abs_sl) | (o & !m.sl & p1f(abs_sl, flags, 10));
    let (mut suppress, mut forced_in, mut forced, mut sup_next) = (c_suppress, 0u64, 0u64, 0u64);
    let (dstart, dbad) = digit_starts(m, flags, max_digits, false);
    let mut bad = common_bad(&x, flags) | dbad | wsbad | c_bad_in;
    let ws_unk = m.ws & bad;
    let useed = (m.sl & p1f(ws_unk, flags, 11)) | (m.sl & (((flags >> 12) & 1) as u64));
    let sl_unk = (m.sl.wrapping_add(useed) ^ m.sl) & m.sl;
    bad |= sl_unk | (p1f(sl_unk, flags, 12) & !m.sl);
    let mut bad_next = 0u64;
    let mut contr_end = c_contr_mask;
    let mut contr_next = 0u64;
    if contractions {
        let mut qs = m.q & p1f(m.l, flags, 5);
        let c_bad3 = ((flags >> BAD3_SHIFT) & 7) as u64;
        while qs != 0 {
            let i = qs.trailing_zeros() as usize;
            qs &= qs - 1;
            let window = ((bad << 3) | c_bad3) >> i & 0b1111;
            if window != 0 {
                span_bad(&mut bad, &mut bad_next, i, i + 3);
                bad |= 1u64 << i;
                span_bad(&mut bad, &mut bad_next, i, i + 3);
            }
            let prev_is_suffix_end = if i == 0 { fbit(flags, F_CONTR) } else { contr_end >> (i - 1) & 1 == 1 };
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
    let ostart_p1 = p1f(ostart_c, flags, 8);
    let lstart = m.l
        & (p1f(m.d, flags, 2) | p1f(m.nl, flags, 3) | (p1f(o, flags, 0) & !ostart_p1) | (m.up & p1f(lw, flags, 6)));
    let starts = ((ostart | lstart | dstart | wsstart) & !suppress) | c_forced | forced_in;
    let bad_c = bad | ((bad << 1) & x.cont);
    let bad_c = bad_c | ((bad_c << 1) & x.cont);
    bad |= m.l & p1f(o, flags, 0) & p1f(bad_c, flags, 18);
    let bad = bad | ((bad << 1) & x.cont) | ((bad << 2) & x.cont & (x.cont << 1));
    let na4 = ((m.na >> 60).reverse_bits() >> 60) as u32;
    let has_slow = (m.na >> 63 != 0)
        || na4 != 0
        || (bad_next | contr_next | (forced & !1) | sup_next) != 0;
    if has_slow {
        *slow = SlowCarry { bad_in: bad_next, contr_mask: contr_next, forced, suppress: sup_next };
    }
    let out_flags = top_bit(x.o, 0)
        | top_bit(m.sp, 1)
        | top_bit(m.d, 2)
        | top_bit(m.nl, 3)
        | top_bit(m.ws, 4)
        | top_bit(m.l, 5)
        | top_bit(lw, 6)
        | top_bit(m.na, 7)
        | top_bit(ostart_c, 8)
        | top_bit(abs_nl, 9)
        | top_bit(abs_sl, 10)
        | top_bit(ws_unk, 11)
        | top_bit(sl_unk, 12)
        | top_bit(contr_end, 13)
        | (((forced & 1) as u32) << 14)
        | if has_slow { F_SLOW } else { 0 }
        | (((bad >> 61) as u32) << BAD3_SHIFT)
        | (na4 << NA4_SHIFT);
    (starts & x.valid, bad & x.valid, out_flags)
}

#[inline(always)]
unsafe fn camel_block<S: Simd, const C_MODE: u8, const K: u32>(avail: &[u8], doc_len: usize, base: usize, contractions: bool, max_digits: u32, flags: u32, slow: &mut SlowCarry) -> (u64, u64, u32, bool) {
    let doc = &avail[..doc_len];
    if base + 64 < doc_len {
        let v = S::load64(avail.as_ptr().add(base));
        let m = S::classify_vec(v);
        if (flags & F_SLOW) == 0 {
            let mut rm = m;
            let mut cont = 0u64;
            let mut ok = true;
            if rm.na != 0 {
                let (c, punct, utf8_ok) = S::resolve_latin1_vec(v, u64::MAX, &mut rm);
                ok = utf8_ok && rm.na == 0 && punct == 0;
                cont = c;
            }
            if ok {
                if let Some((s, bad, nf)) = camel_fast::<true, C_MODE, K>(doc, base, contractions, max_digits, flags, &rm, u64::MAX, cont) {
                    return (s, bad, nf, true);
                }
            }
        }
        let (s, bad, nf) = camel_slow(doc, base, contractions, max_digits, flags, slow, m);
        return (s, bad, nf, false);
    }
    let (v, m, valid) = load_and_classify_tail::<S>(avail, doc_len, base);
    if (flags & F_SLOW) == 0 {
        let mut rm = m;
        let mut cont = 0u64;
        let mut ok = true;
        if rm.na != 0 {
            let (c, punct, utf8_ok) = S::resolve_latin1_vec(v, valid, &mut rm);
            ok = utf8_ok && rm.na == 0 && punct == 0;
            cont = c;
        }
        if ok {
            if let Some((s, bad, nf)) = camel_fast::<false, C_MODE, K>(doc, base, contractions, max_digits, flags, &rm, valid, cont) {
                return (s, bad, nf, true);
            }
        }
    }
    let (s, bad, nf) = camel_slow(doc, base, contractions, max_digits, flags, slow, m);
    (s, bad, nf, false)
}

// ---------------------------------------------------------------------------
// Word grammar (GPT-2 family)
// ---------------------------------------------------------------------------

#[inline(always)]
fn word_fast<const FULL: bool, const I_MODE: u8>(
    b: &[u8],
    base: usize,
    indiv: bool,
    flags: u32,
    m: &Classes,
    valid: u64,
) -> Option<(u64, u64, u32)> {
    let ind = match I_MODE {
        1 => true,
        0 => false,
        _ => indiv,
    };
    let o = if FULL { !(m.l | m.d | m.ws) } else { !(m.l | m.d | m.ws) & valid };
    let sp_before = p1f(m.sp, flags, 1);
    let ostart = o & !p1f(o, flags, 0) & !sp_before;
    let lstart = m.l & !p1f(m.l, flags, 5) & !sp_before;
    let dstart = if ind { m.d } else { m.d & !p1f(m.d, flags, 2) & !sp_before };
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
    let (next_ws, next_na) = if FULL {
        let next = unsafe { *b.get_unchecked(base + 64) };
        (is_ws(next), next >= 0x80)
    } else {
        (false, false)
    };
    let long = m.ws & (p1f(m.ws, flags, 4) | n1(m.ws, next_ws));
    let cont = m.ws & !m.ws.wrapping_add(1) & (((flags >> 4) & 1) as u64).wrapping_neg();
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
    let no_digit_rule = if ind { !(m.d >> 1) } else { u64::MAX };
    let lim_shift = if FULL { u64::MAX >> 1 } else { valid >> 1 };
    starts |= (m.ws & !long) | (r & good) | (last & good & lim_shift & no_digit_rule) | (((flags >> 14) & 1) as u64);
    starts &= !suppress;
    let mut bad = cont | wsbad | (m.ws & ((next_na as u64) << 63));
    if !FULL {
        starts &= valid;
        bad &= valid;
    }
    let out_flags = top_bit(o, 0)
        | top_bit(m.sp, 1)
        | top_bit(m.d, 2)
        | top_bit(m.ws, 4)
        | top_bit(m.l, 5)
        | (((bad >> 61) as u32) << BAD3_SHIFT);
    Some((starts, bad, out_flags))
}

#[inline(always)]
unsafe fn word_block<S: Simd, const I_MODE: u8>(avail: &[u8], doc_len: usize, base: usize, indiv: bool, flags: u32, slow: &mut SlowCarry) -> (u64, u64, u32, bool) {
    let doc = &avail[..doc_len];
    if base + 64 < doc_len {
        let v = S::load64(avail.as_ptr().add(base));
        let m = S::classify_vec(v);
        if (flags & F_SLOW) == 0 {
            let mut rm = m;
            let mut ok = true;
            if rm.na != 0 {
                let (_, _, utf8_ok) = S::resolve_latin1_vec(v, u64::MAX, &mut rm);
                ok = utf8_ok && rm.na == 0;
            }
            if ok {
                if let Some((s, bad, nf)) = word_fast::<true, I_MODE>(doc, base, indiv, flags, &rm, u64::MAX) {
                    return (s, bad, nf, true);
                }
            }
        }
        let (s, bad, nf) = word_slow(doc, base, indiv, flags, slow, m);
        return (s, bad, nf, false);
    }
    let (v, m, valid) = load_and_classify_tail::<S>(avail, doc_len, base);
    if (flags & F_SLOW) == 0 {
        let mut rm = m;
        let mut ok = true;
        if rm.na != 0 {
            let (_, _, utf8_ok) = S::resolve_latin1_vec(v, valid, &mut rm);
            ok = utf8_ok && rm.na == 0;
        }
        if ok {
            if let Some((s, bad, nf)) = word_fast::<false, I_MODE>(doc, base, indiv, flags, &rm, valid) {
                return (s, bad, nf, true);
            }
        }
    }
    let (s, bad, nf) = word_slow(doc, base, indiv, flags, slow, m);
    (s, bad, nf, false)
}

#[cold]
#[inline(never)]
fn word_slow(b: &[u8], base: usize, indiv: bool, flags: u32, slow: &mut SlowCarry, m: Classes) -> (u64, u64, u32) {
    let n = b.len();
    if m.na != 0 && base + 64 < n && (flags & F_SLOW) == 0 {
        let mut r = m;
        resolve_na::<false>(b, base, &mut r);
        if r.na == 0 {
            if let Some(out) = word_fast::<true, 2>(b, base, indiv, flags, &r, u64::MAX) {
                return out;
            }
        }
    }
    let (c_bad_in, c_forced, c_suppress) = if fbit(flags, F_SLOW) {
        (slow.bad_in, slow.forced, slow.suppress)
    } else {
        (0, ((flags >> 14) & 1) as u64, 0)
    };
    let x = ctx_m::<true, false>(b, base, m);
    let (m, o) = (&x.m, x.o);
    let sp_before = p1f(m.sp, flags, 1);
    let ostart = o & !p1f(o, flags, 0) & !sp_before;
    let lstart = m.l & !p1f(m.l, flags, 5) & !sp_before;
    let (dstart, dbad) = if indiv { (m.d, 0) } else { digit_starts(m, flags, 0, true) };
    let long = m.ws & (p1f(m.ws, flags, 4) | n1(m.ws, x.next_ws));
    let cont = m.ws & !m.ws.wrapping_add(1) & (((flags >> 4) & 1) as u64).wrapping_neg();
    let lr = long & !cont;
    let r = lr & !(lr << 1);
    let last = lr & !(lr >> 1);
    let flag_start = r & p1f(m.na, flags, 7);
    let flag_end = (last & (m.na >> 1)) | (last & (1u64 << 63));
    let smear_up = |runs: u64, f: u64| (runs.wrapping_add(f) ^ runs) & runs;
    let wsbad = smear_up(lr, flag_start)
        | smear_up(lr.reverse_bits(), flag_end.reverse_bits()).reverse_bits();
    let good = lr & !wsbad;
    let text_follows = x.valid >> 1;
    let no_digit_rule = if indiv { !(m.d >> 1) } else { u64::MAX };
    let wsstart = (m.ws & !long) | (r & good) | (last & good & text_follows & no_digit_rule);
    let mut starts = ostart | lstart | dstart | wsstart | c_forced;
    let (mut suppress, mut forced, mut sup_next) = (c_suppress, 0u64, 0u64);
    let mut bad = common_bad(&x, flags) | dbad | wsbad | c_bad_in;
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
    let na4 = ((m.na >> 60).reverse_bits() >> 60) as u32;
    let has_slow = (m.na >> 63 != 0) || na4 != 0 || (bad_next | (forced & !1) | sup_next) != 0;
    if has_slow {
        *slow = SlowCarry { bad_in: bad_next, contr_mask: 0, forced, suppress: sup_next };
    }
    let out_flags = top_bit(x.o, 0)
        | top_bit(m.sp, 1)
        | top_bit(m.d, 2)
        | top_bit(m.ws, 4)
        | top_bit(m.l, 5)
        | top_bit(m.na, 7)
        | (((forced & 1) as u32) << 14)
        | if has_slow { F_SLOW } else { 0 }
        | (((bad >> 61) as u32) << BAD3_SHIFT)
        | (na4 << NA4_SHIFT);
    (starts & x.valid, bad & x.valid, out_flags)
}

#[inline(always)]
#[allow(dead_code)]
pub fn block(kind: Pretok, b: &[u8], base: usize, c: Carry) -> Block {
    let mut c = c;
    let (s, bad) = block_m(kind, b, base, &mut c);
    (s, bad, c)
}

#[inline(always)]
unsafe fn block_with<S: Simd>(kind: Pretok, b: &[u8], base: usize, flags: &mut u32, slow: &mut SlowCarry) -> (u64, u64) {
    let (s, bad, out_flags, _) = match kind {
        Pretok::Generic { max_digits, .. } => generic_block::<S, { u32::MAX }>(b, b.len(), base, max_digits, *flags, slow),
        Pretok::CamelCase { contractions, max_digits } => camel_block::<S, 2, { u32::MAX }>(b, b.len(), base, contractions, max_digits, *flags, slow),
        Pretok::Word(indiv) => word_block::<S, 2>(b, b.len(), base, indiv, *flags, slow),
    };
    *flags = out_flags;
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
            3 | 4 => match kind {
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
    unsafe { block_with::<Scalar>(kind, b, base, &mut c.flags, &mut c.slow) }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn word_m_neon(b: &[u8], base: usize, indiv: bool, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = unsafe { word_block::<Neon, 2>(b, b.len(), base, indiv, c.flags, &mut c.slow) };
    c.flags = nf;
    (s, bad)
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn generic_m_neon(b: &[u8], base: usize, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = unsafe { generic_block::<Neon, { u32::MAX }>(b, b.len(), base, max_digits, c.flags, &mut c.slow) };
    c.flags = nf;
    (s, bad)
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn camel_m_neon(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = unsafe { camel_block::<Neon, 2, { u32::MAX }>(b, b.len(), base, contractions, max_digits, c.flags, &mut c.slow) };
    c.flags = nf;
    (s, bad)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn word_m_avx512(b: &[u8], base: usize, indiv: bool, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = word_block::<Avx512, 2>(b, b.len(), base, indiv, c.flags, &mut c.slow);
    c.flags = nf;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn generic_m_avx512(b: &[u8], base: usize, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = generic_block::<Avx512, { u32::MAX }>(b, b.len(), base, max_digits, c.flags, &mut c.slow);
    c.flags = nf;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn camel_m_avx512(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = camel_block::<Avx512, 2, { u32::MAX }>(b, b.len(), base, contractions, max_digits, c.flags, &mut c.slow);
    c.flags = nf;
    (s, bad)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn word_m_avx2(b: &[u8], base: usize, indiv: bool, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = word_block::<Avx2, 2>(b, b.len(), base, indiv, c.flags, &mut c.slow);
    c.flags = nf;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn generic_m_avx2(b: &[u8], base: usize, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = generic_block::<Avx2, { u32::MAX }>(b, b.len(), base, max_digits, c.flags, &mut c.slow);
    c.flags = nf;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn camel_m_avx2(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = camel_block::<Avx2, 2, { u32::MAX }>(b, b.len(), base, contractions, max_digits, c.flags, &mut c.slow);
    c.flags = nf;
    (s, bad)
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn word_m_sse2(b: &[u8], base: usize, indiv: bool, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = word_block::<Sse2, 2>(b, b.len(), base, indiv, c.flags, &mut c.slow);
    c.flags = nf;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn generic_m_sse2(b: &[u8], base: usize, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = generic_block::<Sse2, { u32::MAX }>(b, b.len(), base, max_digits, c.flags, &mut c.slow);
    c.flags = nf;
    (s, bad)
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
unsafe fn camel_m_sse2(b: &[u8], base: usize, contractions: bool, max_digits: u32, c: &mut Carry) -> (u64, u64) {
    let (s, bad, nf, _) = camel_block::<Sse2, 2, { u32::MAX }>(b, b.len(), base, contractions, max_digits, c.flags, &mut c.slow);
    c.flags = nf;
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
    let mut flags = F_FORCED0;
    let mut slow = SlowCarry::default();
    while pos < stop {
        if base + 64 <= pos {
            base = pos;
            flags = F_FORCED0;
        }
        let (starts, bad) = block_with::<S>(kind, b, base, &mut flags, &mut slow);
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
            3 | 4 => unsafe { run512(kind, b, start, stop, fb) },
            2 => unsafe { run256(kind, b, start, stop, fb) },
            _ => unsafe { chunks_batched_with::<Sse2>(kind, b, start, stop, fb) },
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    unsafe { chunks_batched_with::<Scalar>(kind, b, start, stop, fb) }
}

/// Set-bit positions of every byte value as 8 u32 lanes (unused lanes 0).
#[cfg(not(target_arch = "aarch64"))]
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

/// Compact 2 KiB set-bit positions table for NEON widening extraction.
#[cfg(target_arch = "aarch64")]
static POS8: [[u8; 8]; 256] = {
    let mut t = [[0u8; 8]; 256];
    let mut v = 0;
    while v < 256 {
        let (mut x, mut j, mut bit) = (v, 0usize, 0u8);
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
/// to one buffer, so the consumer runs one long loop per batch. Also tracks
/// fused UTF-8 validation coverage across scanned blocks (pure ASCII and valid
/// `0xC3 (0x80..=0xBF)` Latin-1 blocks are verified for free during scanning).
pub struct Scan {
    pos: usize,
    base: usize,
    carry: Carry,
    /// `0` when clean; `lo + 1` when a contiguous unvalidated span starting at
    /// byte offset `lo` is open. Stored in `Scan` (zero extra registers in
    /// `fill_loop`) so clean blocks add only a single zero-check.
    dirty_lo_p1: usize,
    check_utf8: bool,
    bad_utf8: bool,
}

#[cold]
#[inline(never)]
fn validate_dirty_span(b: &[u8], lo: usize, hi: usize) -> bool {
    simdutf8::basic::from_utf8(&b[lo..hi]).is_ok()
}

/// Result of the outlined `bad != 0` block handler.
enum BadOutcome {
    Advance64,
    Realigned,
    Full,
}

/// Out-of-line handler for blocks with `bad != 0` (non-ASCII runs, multi-block
/// whitespace indentation, or digit-alignment boundaries). Kept out of `fill_loop`
/// so LLVM allocates the entire hot clean-block path in registers with minimal
/// stack spilling.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_bad_body<S: Simd>(
    scan: &mut Scan,
    kind: Pretok,
    b: &[u8],
    stop: usize,
    lim: usize,
    op: *mut u32,
    limit: usize,
    pos: &mut usize,
    base: &mut usize,
    flags: &mut u32,
    k: &mut usize,
    trusted: u64,
    bad: u64,
) -> BadOutcome {
    loop {
        if *pos >= lim {
            if *pos > lim && scan.dirty_lo_p1 == 0 {
                scan.dirty_lo_p1 = lim + 1;
            }
            *base = *pos;
            *flags = F_FORCED0;
            return BadOutcome::Realigned;
        }
        let above = if *pos < *base {
            u64::MAX
        } else {
            u64::MAX.checked_shl((*pos - *base + 1) as u32).unwrap_or(0)
        };
        let cand = trusted & above;
        let badm = bad & above;
        if badm == 0 {
            if cand != 0 {
                *k = S::extract_ends(cand, *base, op, *k);
                *pos = *base + 63 - cand.leading_zeros() as usize;
            }
            if lim == stop && *pos < stop {
                *op.add(*k) = stop as u32;
                *k += 1;
                *pos = stop;
            }
            return BadOutcome::Advance64;
        }
        // Extract all clean chunks strictly before the first bad bit via SIMD.
        let fb = badm.trailing_zeros();
        let clean_pre = cand & ((1u64 << fb) - 1);
        if clean_pre != 0 {
            *k = S::extract_ends(clean_pre, *base, op, *k);
            *pos = *base + 63 - clean_pre.leading_zeros() as usize;
        }
        let end = next_chunk_end(kind, b, *pos);
        *op.add(*k) = end as u32;
        *k += 1;
        *pos = end;
        if *k >= limit + SCAN_SLACK - 1 {
            if *pos > lim && scan.dirty_lo_p1 == 0 {
                scan.dirty_lo_p1 = lim + 1;
            }
            scan.pos = *pos;
            scan.base = *pos;
            scan.carry.flags = F_FORCED0;
            return BadOutcome::Full;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn handle_bad_neon(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, lim: usize, op: *mut u32, limit: usize, pos: &mut usize, base: &mut usize, flags: &mut u32, k: &mut usize, trusted: u64, bad: u64) -> BadOutcome {
    fill_bad_body::<Neon>(scan, kind, b, stop, lim, op, limit, pos, base, flags, k, trusted, bad)
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn handle_bad_avx512_vbmi2(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, lim: usize, op: *mut u32, limit: usize, pos: &mut usize, base: &mut usize, flags: &mut u32, k: &mut usize, trusted: u64, bad: u64) -> BadOutcome {
    fill_bad_body::<Avx512Vbmi2>(scan, kind, b, stop, lim, op, limit, pos, base, flags, k, trusted, bad)
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn handle_bad_avx512(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, lim: usize, op: *mut u32, limit: usize, pos: &mut usize, base: &mut usize, flags: &mut u32, k: &mut usize, trusted: u64, bad: u64) -> BadOutcome {
    fill_bad_body::<Avx512>(scan, kind, b, stop, lim, op, limit, pos, base, flags, k, trusted, bad)
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn handle_bad_avx2(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, lim: usize, op: *mut u32, limit: usize, pos: &mut usize, base: &mut usize, flags: &mut u32, k: &mut usize, trusted: u64, bad: u64) -> BadOutcome {
    fill_bad_body::<Avx2>(scan, kind, b, stop, lim, op, limit, pos, base, flags, k, trusted, bad)
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn handle_bad_sse2(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, lim: usize, op: *mut u32, limit: usize, pos: &mut usize, base: &mut usize, flags: &mut u32, k: &mut usize, trusted: u64, bad: u64) -> BadOutcome {
    fill_bad_body::<Sse2>(scan, kind, b, stop, lim, op, limit, pos, base, flags, k, trusted, bad)
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
#[cold]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn handle_bad_scalar(scan: &mut Scan, kind: Pretok, b: &[u8], stop: usize, lim: usize, op: *mut u32, limit: usize, pos: &mut usize, base: &mut usize, flags: &mut u32, k: &mut usize, trusted: u64, bad: u64) -> BadOutcome {
    fill_bad_body::<Scalar>(scan, kind, b, stop, lim, op, limit, pos, base, flags, k, trusted, bad)
}

/// Grammar-specialized inner loop for `Scan::fill` / `Scan::fill_in`. Because
/// `step` is monomorphized per `(Simd, Pretok)`, the fast rules + SIMD
/// classifier inline directly into the loop with `Carry` kept in registers.
///
/// `avail` is the full readable buffer (may extend past `doc_len`, allowing
/// unmasked 64-byte SIMD loads on document tails in `encode_sep_macro`), while
/// all grammar rules, look-ahead, and `next_chunk_end` stay strictly within
/// `avail[..doc_len]` and stop at `stop <= doc_len`.
///
/// Exact UTF-8 Boundary Theorem: every `utf8_ok == true` block consists solely
/// of ASCII bytes and complete in-block `0xC3 (0x80..=0xBF)` pairs, so it
/// starts and ends on exact UTF-8 character boundaries. Therefore every
/// contiguous run of unvalidated blocks `[dirty_lo..base)` also starts and ends
/// on exact UTF-8 boundaries and can be validated surgically (typically 64B)
/// without re-scanning the surrounding clean document!
type BadHandler = unsafe fn(
    &mut Scan,
    Pretok,
    &[u8],
    usize,
    usize,
    *mut u32,
    usize,
    &mut usize,
    &mut usize,
    &mut u32,
    &mut usize,
    u64,
    u64,
) -> BadOutcome;

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_loop<S: Simd>(
    scan: &mut Scan,
    kind: Pretok,
    avail: &[u8],
    doc_len: usize,
    stop: usize,
    out: &mut [u32],
    mut k: usize,
    limit: usize,
    mut step: impl FnMut(&[u8], usize, usize, u32, &mut SlowCarry) -> (u64, u64, u32, bool),
    handle_bad: BadHandler,
) -> usize {
    let b = &avail[..doc_len];
    let (mut pos, mut base, mut flags) = (scan.pos, scan.base, scan.carry.flags);
    let op = out.as_mut_ptr();
    while pos < stop && k < limit {
        if base + 64 <= pos {
            if pos > base + 64 && scan.dirty_lo_p1 == 0 {
                scan.dirty_lo_p1 = base + 64 + 1;
            }
            base = pos;
            flags = F_FORCED0;
        }
        let (starts, bad, next_flags, utf8_ok) = step(avail, doc_len, base, flags, &mut scan.carry.slow);
        if !utf8_ok {
            if scan.dirty_lo_p1 == 0 {
                scan.dirty_lo_p1 = base + 1;
            }
        } else if scan.dirty_lo_p1 != 0 {
            if scan.check_utf8 && !scan.bad_utf8 && !validate_dirty_span(b, scan.dirty_lo_p1 - 1, base) {
                scan.bad_utf8 = true;
            }
            scan.dirty_lo_p1 = 0;
        }
        flags = next_flags;
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
            match handle_bad(scan, kind, b, stop, lim, op, limit, &mut pos, &mut base, &mut flags, &mut k, trusted, bad) {
                BadOutcome::Advance64 => {}
                BadOutcome::Realigned => continue,
                BadOutcome::Full => return k,
            }
        }
        base += 64;
    }
    scan.pos = pos;
    scan.base = base;
    scan.carry.flags = flags;
    k
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn fill_word_neon<const I_MODE: u8>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, indiv: bool, out: &mut [u32], k: usize, limit: usize) -> usize {
    unsafe { fill_loop::<Neon>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| word_block::<Neon, I_MODE>(a, dl, base, indiv, f, s), handle_bad_neon) }
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn fill_generic_neon<const K: u32>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    unsafe { fill_loop::<Neon>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| generic_block::<Neon, K>(a, dl, base, max_digits, f, s), handle_bad_neon) }
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn fill_camel_neon<const C_MODE: u8, const K: u32>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, contractions: bool, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    unsafe { fill_loop::<Neon>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| camel_block::<Neon, C_MODE, K>(a, dl, base, contractions, max_digits, f, s), handle_bad_neon) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_word_avx512_vbmi2<const I_MODE: u8>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, indiv: bool, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx512Vbmi2>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| word_block::<Avx512Vbmi2, I_MODE>(a, dl, base, indiv, f, s), handle_bad_avx512_vbmi2)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_generic_avx512_vbmi2<const K: u32>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx512Vbmi2>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| generic_block::<Avx512Vbmi2, K>(a, dl, base, max_digits, f, s), handle_bad_avx512_vbmi2)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi2,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_camel_avx512_vbmi2<const C_MODE: u8, const K: u32>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, contractions: bool, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx512Vbmi2>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| camel_block::<Avx512Vbmi2, C_MODE, K>(a, dl, base, contractions, max_digits, f, s), handle_bad_avx512_vbmi2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_word_avx512<const I_MODE: u8>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, indiv: bool, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx512>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| word_block::<Avx512, I_MODE>(a, dl, base, indiv, f, s), handle_bad_avx512)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_generic_avx512<const K: u32>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx512>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| generic_block::<Avx512, K>(a, dl, base, max_digits, f, s), handle_bad_avx512)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_camel_avx512<const C_MODE: u8, const K: u32>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, contractions: bool, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx512>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| camel_block::<Avx512, C_MODE, K>(a, dl, base, contractions, max_digits, f, s), handle_bad_avx512)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_word_avx2<const I_MODE: u8>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, indiv: bool, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx2>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| word_block::<Avx2, I_MODE>(a, dl, base, indiv, f, s), handle_bad_avx2)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_generic_avx2<const K: u32>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx2>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| generic_block::<Avx2, K>(a, dl, base, max_digits, f, s), handle_bad_avx2)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_camel_avx2<const C_MODE: u8, const K: u32>(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, contractions: bool, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Avx2>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| camel_block::<Avx2, C_MODE, K>(a, dl, base, contractions, max_digits, f, s), handle_bad_avx2)
}

#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_word_sse2(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, indiv: bool, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Sse2>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| word_block::<Sse2, 2>(a, dl, base, indiv, f, s), handle_bad_sse2)
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_generic_sse2(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Sse2>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| generic_block::<Sse2, { u32::MAX }>(a, dl, base, max_digits, f, s), handle_bad_sse2)
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn fill_camel_sse2(scan: &mut Scan, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, contractions: bool, max_digits: u32, out: &mut [u32], k: usize, limit: usize) -> usize {
    fill_loop::<Sse2>(scan, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| camel_block::<Sse2, 2, { u32::MAX }>(a, dl, base, contractions, max_digits, f, s), handle_bad_sse2)
}

impl Scan {
    pub fn new(start: usize) -> Self {
        Self::new_checked(start, false)
    }

    #[inline(always)]
    pub fn new_checked(start: usize, check_utf8: bool) -> Self {
        Scan {
            pos: start,
            base: start,
            carry: Carry::forced_start(),
            dirty_lo_p1: 0,
            check_utf8,
            bad_utf8: false,
        }
    }

    /// Finalize surgical fused UTF-8 validation after scanning to `stop`:
    /// returns `true` iff `[start..stop)` is valid UTF-8. Clean blocks (pure ASCII
    /// or valid `0xC3 (0x80..=0xBF)` Latin-1) cost zero extra validation; only
    /// contiguous unvalidated runs (typically 64B around a curly quote or emoji)
    /// are checked via `simdutf8`.
    #[inline(always)]
    pub fn finish_utf8(&mut self, doc: &[u8], stop: usize) -> bool {
        if !self.check_utf8 {
            return true;
        }
        if self.bad_utf8 {
            return false;
        }
        if self.dirty_lo_p1 != 0 {
            let lo = self.dirty_lo_p1 - 1;
            self.dirty_lo_p1 = 0;
            if !validate_dirty_span(doc, lo, stop) {
                self.bad_utf8 = true;
                return false;
            }
        }
        true
    }

    /// Append ends to `out[k..]` until `k >= limit` (checked per block) or
    /// `stop`; returns the new length. `out.len() >= limit + SCAN_SLACK`,
    /// `b.len() <= u32::MAX`.
    #[inline]
    pub fn fill(&mut self, kind: Pretok, b: &[u8], stop: usize, out: &mut [u32], k: usize, limit: usize) -> usize {
        self.fill_in(kind, b, b.len(), stop, out, k, limit)
    }

    /// Like `fill`, where `avail` is a larger readable buffer containing
    /// `doc = &avail[..doc_len]` (`stop <= doc_len <= avail.len()`). Enables
    /// zero-copy 64-byte SIMD loads even on document tails.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn fill_in(&mut self, kind: Pretok, avail: &[u8], doc_len: usize, stop: usize, out: &mut [u32], k: usize, limit: usize) -> usize {
        assert!(out.len() >= limit + SCAN_SLACK);
        debug_assert!(stop <= doc_len && doc_len <= avail.len());
        #[cfg(target_arch = "aarch64")]
        return match kind {
            Pretok::Word(false) => fill_word_neon::<0>(self, kind, avail, doc_len, stop, false, out, k, limit),
            Pretok::Word(true) => fill_word_neon::<1>(self, kind, avail, doc_len, stop, true, out, k, limit),
            Pretok::Generic { max_digits: 3, .. } => fill_generic_neon::<3>(self, kind, avail, doc_len, stop, 3, out, k, limit),
            Pretok::Generic { max_digits: 1, .. } => fill_generic_neon::<1>(self, kind, avail, doc_len, stop, 1, out, k, limit),
            Pretok::Generic { max_digits, .. } => fill_generic_neon::<{ u32::MAX }>(self, kind, avail, doc_len, stop, max_digits, out, k, limit),
            Pretok::CamelCase { contractions: true, max_digits: 3 } => fill_camel_neon::<1, 3>(self, kind, avail, doc_len, stop, true, 3, out, k, limit),
            Pretok::CamelCase { contractions, max_digits } => fill_camel_neon::<2, { u32::MAX }>(self, kind, avail, doc_len, stop, contractions, max_digits, out, k, limit),
        };
        #[cfg(target_arch = "x86_64")]
        return match x86_tier() {
            4 => match kind {
                Pretok::Word(false) => unsafe { fill_word_avx512_vbmi2::<0>(self, kind, avail, doc_len, stop, false, out, k, limit) },
                Pretok::Word(true) => unsafe { fill_word_avx512_vbmi2::<1>(self, kind, avail, doc_len, stop, true, out, k, limit) },
                Pretok::Generic { max_digits: 3, .. } => unsafe { fill_generic_avx512_vbmi2::<3>(self, kind, avail, doc_len, stop, 3, out, k, limit) },
                Pretok::Generic { max_digits: 1, .. } => unsafe { fill_generic_avx512_vbmi2::<1>(self, kind, avail, doc_len, stop, 1, out, k, limit) },
                Pretok::Generic { max_digits, .. } => unsafe { fill_generic_avx512_vbmi2::<{ u32::MAX }>(self, kind, avail, doc_len, stop, max_digits, out, k, limit) },
                Pretok::CamelCase { contractions: true, max_digits: 3 } => unsafe { fill_camel_avx512_vbmi2::<1, 3>(self, kind, avail, doc_len, stop, true, 3, out, k, limit) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { fill_camel_avx512_vbmi2::<2, { u32::MAX }>(self, kind, avail, doc_len, stop, contractions, max_digits, out, k, limit) },
            },
            3 => match kind {
                Pretok::Word(false) => unsafe { fill_word_avx512::<0>(self, kind, avail, doc_len, stop, false, out, k, limit) },
                Pretok::Word(true) => unsafe { fill_word_avx512::<1>(self, kind, avail, doc_len, stop, true, out, k, limit) },
                Pretok::Generic { max_digits: 3, .. } => unsafe { fill_generic_avx512::<3>(self, kind, avail, doc_len, stop, 3, out, k, limit) },
                Pretok::Generic { max_digits: 1, .. } => unsafe { fill_generic_avx512::<1>(self, kind, avail, doc_len, stop, 1, out, k, limit) },
                Pretok::Generic { max_digits, .. } => unsafe { fill_generic_avx512::<{ u32::MAX }>(self, kind, avail, doc_len, stop, max_digits, out, k, limit) },
                Pretok::CamelCase { contractions: true, max_digits: 3 } => unsafe { fill_camel_avx512::<1, 3>(self, kind, avail, doc_len, stop, true, 3, out, k, limit) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { fill_camel_avx512::<2, { u32::MAX }>(self, kind, avail, doc_len, stop, contractions, max_digits, out, k, limit) },
            },
            2 => match kind {
                Pretok::Word(false) => unsafe { fill_word_avx2::<0>(self, kind, avail, doc_len, stop, false, out, k, limit) },
                Pretok::Word(true) => unsafe { fill_word_avx2::<1>(self, kind, avail, doc_len, stop, true, out, k, limit) },
                Pretok::Generic { max_digits: 3, .. } => unsafe { fill_generic_avx2::<3>(self, kind, avail, doc_len, stop, 3, out, k, limit) },
                Pretok::Generic { max_digits: 1, .. } => unsafe { fill_generic_avx2::<1>(self, kind, avail, doc_len, stop, 1, out, k, limit) },
                Pretok::Generic { max_digits, .. } => unsafe { fill_generic_avx2::<{ u32::MAX }>(self, kind, avail, doc_len, stop, max_digits, out, k, limit) },
                Pretok::CamelCase { contractions: true, max_digits: 3 } => unsafe { fill_camel_avx2::<1, 3>(self, kind, avail, doc_len, stop, true, 3, out, k, limit) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { fill_camel_avx2::<2, { u32::MAX }>(self, kind, avail, doc_len, stop, contractions, max_digits, out, k, limit) },
            },
            _ => match kind {
                Pretok::Word(indiv) => unsafe { fill_word_sse2(self, kind, avail, doc_len, stop, indiv, out, k, limit) },
                Pretok::Generic { max_digits, .. } => unsafe { fill_generic_sse2(self, kind, avail, doc_len, stop, max_digits, out, k, limit) },
                Pretok::CamelCase { contractions, max_digits } => unsafe { fill_camel_sse2(self, kind, avail, doc_len, stop, contractions, max_digits, out, k, limit) },
            },
        };
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        match kind {
            Pretok::Word(indiv) => unsafe { fill_loop::<Scalar>(self, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| word_block::<Scalar, 2>(a, dl, base, indiv, f, s), handle_bad_scalar) },
            Pretok::Generic { max_digits, .. } => unsafe { fill_loop::<Scalar>(self, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| generic_block::<Scalar, { u32::MAX }>(a, dl, base, max_digits, f, s), handle_bad_scalar) },
            Pretok::CamelCase { contractions, max_digits } => unsafe { fill_loop::<Scalar>(self, kind, avail, doc_len, stop, out, k, limit, |a, dl, base, f, s| camel_block::<Scalar, 2, { u32::MAX }>(a, dl, base, contractions, max_digits, f, s), handle_bad_scalar) },
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

    #[test]
    fn fill_in_isolates_trailing_bytes_and_validates_utf8() {
        let mut seed = 5555u64;
        // 1. Trailing garbage after doc_len in avail must never change boundaries.
        let tails: &[&[u8]] = &[
            b"\nhello's 12345   ///\r\n",
            b"\xC3\xA4\xFF\x80\xC3\xC0'll   aB",
            b"   \t\n\n\r're999",
        ];
        for _ in 0..2000 {
            let t = random_text(&mut seed);
            let b = t.as_bytes();
            let tail = tails[(lcg(&mut seed) as usize) % tails.len()];
            let mut avail = Vec::with_capacity(b.len() + tail.len() + 64);
            avail.extend_from_slice(b);
            avail.extend_from_slice(tail);
            avail.extend_from_slice(&[b'Z'; 64]);
            for &kind in &[KINDS[2], KINDS[4], KINDS[8]] {
                let mut want = Vec::new();
                let mut pos = 0;
                while pos < b.len() {
                    let e = slow_chunk_end(kind, b, pos);
                    want.push(e as u32);
                    pos = e;
                }
                let mut sc = Scan::new(0);
                let mut out = vec![0u32; 256 + SCAN_SLACK];
                let mut got = Vec::new();
                loop {
                    let k = sc.fill_in(kind, &avail, b.len(), b.len(), &mut out, 0, 256);
                    if k == 0 {
                        break;
                    }
                    got.extend_from_slice(&out[..k]);
                }
                assert_eq!(got, want, "fill_in isolation failed for {kind:?}");
            }
        }
        // 2. Adversarial invalid UTF-8 must ALWAYS fail `finish_utf8` and valid UTF-8
        //    (including straddling 2/3/4-byte sequences across 64B edges) must ALWAYS pass.
        let cases: &[&[u8]] = &[
            b"\xFF",
            b"\x80",
            b"\xC3",
            b"hello \xC3",
            b"hello \xC3\x7F world",
            b"hello \xC3\xC0 world",
            b"hello \xC3\xC3 world",
            b"hello \xC3\xFF world",
            b"\xC0\x80",
            b"\xED\xA0\x80", // UTF-16 surrogate
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\xC3\xA4", // valid split 0xC3 at 63
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\xE2\x80\x99 tail text here", // valid split 3B at 63
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\xE2\x80\x7F tail text here", // invalid split 3B at 63
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\xF0\x9F\x98\x80 after emoji ascii", // valid split 4B at 62
            b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\xF0\x9F\x98\xFF after bad emoji", // invalid split 4B at 62
        ];
        for &raw in cases {
            let is_valid = std::str::from_utf8(raw).is_ok();
            let mut avail = raw.to_vec();
            avail.extend_from_slice(b" padding padding padding padding padding padding padding padding");
            for &kind in &[KINDS[2], KINDS[4], KINDS[8]] {
                let mut sc = Scan::new_checked(0, true);
                let mut out = vec![0u32; 256 + SCAN_SLACK];
                loop {
                    let k = sc.fill_in(kind, &avail, raw.len(), raw.len(), &mut out, 0, 256);
                    if k == 0 {
                        break;
                    }
                }
                assert_eq!(sc.finish_utf8(raw, raw.len()), is_valid, "utf8 mismatch for {raw:?} on {kind:?}");
            }
        }
        // 3. Random valid & invalid byte strings (len 1..200) must match std::str::from_utf8 100%.
        for _ in 0..8000 {
            let len = 1 + (lcg(&mut seed) as usize % 200);
            let mut buf = vec![b'a'; len];
            let nmut = 1 + (lcg(&mut seed) as usize % 4);
            for _ in 0..nmut {
                let pos = (lcg(&mut seed) as usize) % len;
                let mode = lcg(&mut seed) % 4;
                if mode == 0 && pos + 2 < len {
                    buf[pos] = 0xE2;
                    buf[pos + 1] = 0x80;
                    buf[pos + 2] = 0x99;
                } else if mode == 1 && pos + 1 < len {
                    buf[pos] = 0xC3;
                    buf[pos + 1] = 0xA4;
                } else {
                    buf[pos] = 0x80 | ((lcg(&mut seed) & 0x7F) as u8);
                }
            }
            let want_ok = std::str::from_utf8(&buf).is_ok();
            let mut sc = Scan::new_checked(0, true);
            let mut out = vec![0u32; 256 + SCAN_SLACK];
            loop {
                let k = sc.fill_in(KINDS[4], &buf, buf.len(), buf.len(), &mut out, 0, 256);
                if k == 0 {
                    break;
                }
            }
            assert_eq!(sc.finish_utf8(&buf, buf.len()), want_ok, "random utf8 mismatch len={len}");
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
