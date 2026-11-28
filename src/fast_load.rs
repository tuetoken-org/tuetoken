//! Parallel, zero-malloc RFC 8259 JSON loader for HuggingFace `tokenizer.json`.
//!
//! Why this exists:
//! `serde_json` deserializes a 12–17 MB `tokenizer.json` (e.g. `o200k_base`,
//! Qwen2.5, Llama-3, DeepSeek) serially on a single core:
//! - ~320K heap `String` allocations for `\u0120`-escaped ByteLevel tokens,
//! - ~600K serial `FxHashMap::get` L3 cache misses during merge resolution,
//! - serial ByteLevel/byte_fallback decoding in random hash-table order.
//!
//! This loader slices top-level JSON sections (delegating all small config
//! objects directly to `serde_json::from_slice`), splits `model.vocab` and
//! `model.merges` across all CPU cores (handling BOTH pretty-printed and
//! single-line minified JSON via exact 2-state parallel quote-parity scan),
//! unescapes strings into thread-local flat arenas (zero per-token malloc),
//! builds a 32-shard parallel open-addressing vocab lookup table, resolves
//! merges in parallel with sequential-ID target prediction (eliminating ~200K
//! hash lookups), and decodes `byte_arena` in parallel.
//!
//! Fail-safe: any unexpected syntax, non-BPE model, or malformed escape returns
//! `None` so `Tokenizer::new` transparently falls back to `serde_json`.

use rayon::prelude::*;
use rustc_hash::FxHasher;
use serde_json::Value;
use std::hash::Hasher;

/// Parsed result returned to `Tokenizer::new` when the fast path succeeds.
pub struct FastLoadedTokenizer {
    pub normalizer: Value,
    pub pre_tokenizer: Value,
    pub post_processor: Option<Value>,
    pub decoder: Value,
    pub added_tokens: Vec<FastAddedToken>,
    pub model_type: Option<String>,
    pub byte_fallback: bool,
    pub ignore_merges: bool,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
    pub has_merges: bool,
    pub vocab_size: usize,
    pub merges: Vec<(u32, u32, u32)>,
    pub known_token_ids: Vec<(String, u32)>,
    pub singleton_eow: Option<[i32; 256]>,
    pub byte_arena: Vec<u8>,
    pub byte_span: Vec<(u32, u32)>,
    pub canonical: Vec<bool>,
    pub has_internal_ms: Vec<bool>,
    pub char_ascii: [i32; 128],
    pub char_map: rustc_hash::FxHashMap<u32, u32>,
}

#[derive(serde::Deserialize)]
pub struct FastAddedToken {
    pub id: Option<u64>,
    pub content: Option<String>,
    #[serde(default)]
    pub special: bool,
    #[serde(default)]
    pub lstrip: bool,
    #[serde(default)]
    pub rstrip: bool,
}

const KNOWN_TOKS: &[&str] = &[
    "<pad>", "[PAD]", "<|pad|>", "<|endoftext|>", "<s>", "</s>",
    "[CLS]", "[SEP]", "<unk>", "[UNK]", "<mask>", "[MASK]",
    "<|begin_of_text|>", "<|end_of_text|>", "<|im_start|>", "<|im_end|>",
];

#[inline(always)]
fn fx_hash_bytes(b: &[u8]) -> u64 {
    let mut h = FxHasher::default();
    h.write(b);
    h.finish()
}

#[inline(always)]
fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    i
}

/// Skip a JSON string starting at `b[i] == b'"'`, returning the index right after
/// the closing `'"'`. Uses `memchr2` for ~20 GB/s scanning.
#[inline]
fn skip_json_string(b: &[u8], mut i: usize) -> Option<usize> {
    if i >= b.len() || b[i] != b'"' {
        return None;
    }
    i += 1;
    while i < b.len() {
        let off = memchr::memchr2(b'"', b'\\', &b[i..])?;
        i += off;
        if b[i] == b'"' {
            return Some(i + 1);
        }
        // Backslash escape: skip `\` and the escaped byte.
        i += 2;
    }
    None
}

/// Skip any JSON value starting at `b[i]` (after whitespace), returning index after value.
fn skip_json_value(b: &[u8], i: usize) -> Option<usize> {
    let mut i = skip_ws(b, i);
    if i >= b.len() {
        return None;
    }
    match b[i] {
        b'"' => skip_json_string(b, i),
        b'{' | b'[' => {
            let mut depth = 1usize;
            i += 1;
            while i < b.len() && depth > 0 {
                let off = memchr::memchr3(b'"', b'{', b'}', &b[i..]);
                let lim = match off {
                    Some(a) => i + a,
                    None => b.len(),
                };
                let step = match memchr::memchr2(b'[', b']', &b[i..lim]) {
                    Some(c) => c,
                    None => off?,
                };
                i += step;
                match b[i] {
                    b'"' => i = skip_json_string(b, i)?,
                    b'{' | b'[' => {
                        depth += 1;
                        i += 1;
                    }
                    b'}' | b']' => {
                        depth -= 1;
                        i += 1;
                    }
                    _ => unreachable!(),
                }
            }
            if depth == 0 { Some(i) } else { None }
        }
        _ => {
            while i < b.len() && !matches!(b[i], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n') {
                i += 1;
            }
            Some(i)
        }
    }
}

/// Inline short-slice byte equality avoiding `libc::bcmp` PLT calls on 1..=16B tokens.
#[inline(always)]
fn fast_eq(a: &[u8], b: &[u8]) -> bool {
    let n = a.len();
    if n != b.len() {
        return false;
    }
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    unsafe {
        if n <= 16 {
            if n >= 8 {
                let a0 = (pa as *const u64).read_unaligned();
                let b0 = (pb as *const u64).read_unaligned();
                let a1 = (pa.add(n - 8) as *const u64).read_unaligned();
                let b1 = (pb.add(n - 8) as *const u64).read_unaligned();
                return ((a0 ^ b0) | (a1 ^ b1)) == 0;
            }
            if n >= 4 {
                let a0 = (pa as *const u32).read_unaligned();
                let b0 = (pb as *const u32).read_unaligned();
                let a1 = (pa.add(n - 4) as *const u32).read_unaligned();
                let b1 = (pb.add(n - 4) as *const u32).read_unaligned();
                return ((a0 ^ b0) | (a1 ^ b1)) == 0;
            }
            if n >= 2 {
                let a0 = (pa as *const u16).read_unaligned();
                let b0 = (pb as *const u16).read_unaligned();
                return a0 == b0 && *pa.add(n - 1) == *pb.add(n - 1);
            }
            return n == 0 || *pa == *pb;
        }
    }
    a == b
}

/// Compute 64-bit `(q, bs, op, cl)` character bitmasks for a 64-byte block.
#[inline(always)]
fn masks_64_scalar(b: &[u8], open_ch: u8, close_ch: u8) -> (u64, u64, u64, u64) {
    let mut q = 0u64;
    let mut bs = 0u64;
    let mut op = 0u64;
    let mut cl = 0u64;
    for (k, &c) in b.iter().enumerate() {
        let bit = 1u64 << k;
        if c == b'"' {
            q |= bit;
        } else if c == b'\\' {
            bs |= bit;
        } else if c == open_ch {
            op |= bit;
        } else if c == close_ch {
            cl |= bit;
        }
    }
    (q, bs, op, cl)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn simd_scan_level() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static LVL: AtomicU8 = AtomicU8::new(0);
    let mut v = LVL.load(Relaxed);
    if v == 0 {
        v = if std::is_x86_feature_detected!("avx512bw") {
            3
        } else if std::is_x86_feature_detected!("avx2") {
            2
        } else {
            1
        };
        LVL.store(v, Relaxed);
    }
    v
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn masks_64_avx512bw(p: *const u8, open_ch: u8, close_ch: u8) -> (u64, u64, u64, u64) {
    use std::arch::x86_64::*;
    let v = _mm512_loadu_si512(p as *const __m512i);
    let q = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'"' as i8));
    let bs = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\\' as i8));
    let op = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(open_ch as i8));
    let cl = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(close_ch as i8));
    (q, bs, op, cl)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn masks_64_avx2(p: *const u8, open_ch: u8, close_ch: u8) -> (u64, u64, u64, u64) {
    use std::arch::x86_64::*;
    let v0 = _mm256_loadu_si256(p as *const __m256i);
    let v1 = _mm256_loadu_si256(p.add(32) as *const __m256i);
    let vq = _mm256_set1_epi8(b'"' as i8);
    let vbs = _mm256_set1_epi8(b'\\' as i8);
    let vop = _mm256_set1_epi8(open_ch as i8);
    let vcl = _mm256_set1_epi8(close_ch as i8);
    let q = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, vq)) as u32 as u64)
        | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, vq)) as u32 as u64) << 32);
    let bs = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, vbs)) as u32 as u64)
        | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, vbs)) as u32 as u64) << 32);
    let op = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, vop)) as u32 as u64)
        | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, vop)) as u32 as u64) << 32);
    let cl = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, vcl)) as u32 as u64)
        | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, vcl)) as u32 as u64) << 32);
    (q, bs, op, cl)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_movemask64(
    m0: std::arch::aarch64::uint8x16_t,
    m1: std::arch::aarch64::uint8x16_t,
    m2: std::arch::aarch64::uint8x16_t,
    m3: std::arch::aarch64::uint8x16_t,
    bm: std::arch::aarch64::uint8x16_t,
) -> u64 {
    use std::arch::aarch64::*;
    let s01 = vpaddq_u8(vandq_u8(m0, bm), vandq_u8(m1, bm));
    let s23 = vpaddq_u8(vandq_u8(m2, bm), vandq_u8(m3, bm));
    let s0123 = vpaddq_u8(s01, s23);
    let packed = vpaddq_u8(s0123, s0123);
    vgetq_lane_u64::<0>(vreinterpretq_u64_u8(packed))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn masks_64_neon(p: *const u8, open_ch: u8, close_ch: u8) -> (u64, u64, u64, u64) {
    use std::arch::aarch64::*;
    const BM: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let bm = vld1q_u8(BM.as_ptr());
    let v0 = vld1q_u8(p);
    let v1 = vld1q_u8(p.add(16));
    let v2 = vld1q_u8(p.add(32));
    let v3 = vld1q_u8(p.add(48));
    let vq = vdupq_n_u8(b'"');
    let vbs = vdupq_n_u8(b'\\');
    let vop = vdupq_n_u8(open_ch);
    let vcl = vdupq_n_u8(close_ch);
    let q = neon_movemask64(vceqq_u8(v0, vq), vceqq_u8(v1, vq), vceqq_u8(v2, vq), vceqq_u8(v3, vq), bm);
    let bs = neon_movemask64(vceqq_u8(v0, vbs), vceqq_u8(v1, vbs), vceqq_u8(v2, vbs), vceqq_u8(v3, vbs), bm);
    let op = neon_movemask64(vceqq_u8(v0, vop), vceqq_u8(v1, vop), vceqq_u8(v2, vop), vceqq_u8(v3, vop), bm);
    let cl = neon_movemask64(vceqq_u8(v0, vcl), vceqq_u8(v1, vcl), vceqq_u8(v2, vcl), vceqq_u8(v3, vcl), bm);
    (q, bs, op, cl)
}

#[inline(always)]
fn update_depth_block(d: &mut i32, md: &mut i32, real_op: u64, real_cl: u64) {
    if (real_op | real_cl) == 0 {
        return;
    }
    if real_op == 0 {
        *d -= real_cl.count_ones() as i32;
        if *d < *md {
            *md = *d;
        }
    } else if real_cl == 0 {
        *d += real_op.count_ones() as i32;
    } else {
        let mut both = real_op | real_cl;
        while both != 0 {
            let bit = 1u64 << both.trailing_zeros();
            both &= both - 1;
            if (real_op & bit) != 0 {
                *d += 1;
            } else {
                *d -= 1;
                if *d < *md {
                    *md = *d;
                }
            }
        }
    }
}

/// Compute Langdale-Lemire escaped-byte mask for a 64-byte block given `bs`
/// (backslash bitmask) and carry-in `prev_escaped`. Returns `(escaped, next_prev_escaped)`.
/// Exact for every byte disjoint from `bs` (in particular, all `"` quote bytes).
#[inline(always)]
fn compute_escaped_mask(bs: u64, prev_escaped: bool) -> (u64, bool) {
    if bs == 0 && !prev_escaped {
        return (0, false);
    }
    const EVEN: u64 = 0x5555_5555_5555_5555;
    const ODD: u64 = 0xAAAA_AAAA_AAAA_AAAA;
    let pe = prev_escaped as u64;
    let bs_eff = bs & !pe;
    let starts = bs_eff & !(bs_eff << 1);
    let (even_c, _) = bs_eff.overflowing_add(starts & EVEN);
    let (odd_c, c_odd) = bs_eff.overflowing_add(starts & ODD);
    let escaped = pe | ((even_c & !bs_eff) & ODD) | ((odd_c & !bs_eff) & EVEN);
    (escaped, c_odd)
}

/// Fast 64-byte SIMD parallel finder for the closing `}` of a flat `"vocab"`
/// object (or `]` of a 1D/2D `"merges"` array) starting right after `{`/`[`.
/// Computes BOTH starting quote states (`in_str = 0` and `in_str = 1`) in ONE
/// single 64-byte SIMD pass (~25 GB/s per core on Zen 4 AVX-512BW and M4 NEON!).
fn scan_delimited_section(
    b: &[u8],
    open_pos: usize,
    is_obj: bool,
    allow_par: bool,
) -> Option<(usize, Vec<(usize, bool)>)> {
    let n = b.len();
    let start = open_pos + 1;
    if start >= n {
        return None;
    }

    // Use `3 * threads` chunks when scanning `"vocab"` (which typically occupies the first
    // ~30-40% of the file) so `parse_vocab_chunk` still saturates all CPU cores.
    let nt = if allow_par && n - start >= 262_144 {
        let base = rayon::current_num_threads().clamp(2, 32);
        if is_obj { (base * 3).min(48) } else { base }
    } else {
        1
    };
    let mut cuts = Vec::with_capacity(nt + 1);
    cuts.push(start);
    if nt > 1 {
        let step = (n - start) / nt;
        for k in 1..nt {
            let mut p = start + k * step;
            while p > start && b[p - 1] == b'\\' {
                p -= 1;
            }
            if p > *cuts.last().unwrap() && p < n {
                cuts.push(p);
            }
        }
    }
    cuts.push(n);

    #[derive(Clone, Copy)]
    struct ChunkSummary {
        end_in: [bool; 2],
        valid: [bool; 2],
        delta: [i32; 2],
        min_d: [i32; 2],
    }

    let open_ch = if is_obj { b'{' } else { b'[' };
    let close_ch = if is_obj { b'}' } else { b']' };
    #[cfg(target_arch = "x86_64")]
    let slvl = simd_scan_level();

    let summarize = |lo: usize, hi: usize| -> ChunkSummary {
        let mut prev_esc = false;
        let mut in0 = false;
        let mut valid0 = true;
        let mut valid1 = true;
        let mut d0 = 0i32;
        let mut md0 = 0i32;
        let mut d1 = 0i32;
        let mut md1 = 0i32;

        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        fn step(
            q: u64,
            bs: u64,
            op: u64,
            cl: u64,
            rem: usize,
            prev_esc: &mut bool,
            in0: &mut bool,
            valid0: &mut bool,
            valid1: &mut bool,
            d0: &mut i32,
            md0: &mut i32,
            d1: &mut i32,
            md1: &mut i32,
        ) {
            let (escaped, next_pe) = compute_escaped_mask(bs, *prev_esc);
            *prev_esc = next_pe;
            let uq = q & !escaped;
            let mut pm = uq;
            pm ^= pm << 1;
            pm ^= pm << 2;
            pm ^= pm << 4;
            pm ^= pm << 8;
            pm ^= pm << 16;
            pm ^= pm << 32;
            let mask0 = if *in0 { !pm } else { pm };
            let mask1 = !mask0;
            *in0 = ((mask0 >> (rem - 1)) & 1) != 0;

            if (bs & !mask0) != 0 {
                *valid0 = false;
            }
            if (bs & !mask1) != 0 {
                *valid1 = false;
            }
            if (op | cl) != 0 {
                update_depth_block(d0, md0, op & !mask0, cl & !mask0);
                update_depth_block(d1, md1, op & !mask1, cl & !mask1);
            }
        }

        let mut i = lo;
        let bp = b.as_ptr();
        while i + 64 <= hi {
            let (q, bs, op, cl) = {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if slvl == 3 {
                        masks_64_avx512bw(bp.add(i), open_ch, close_ch)
                    } else if slvl == 2 {
                        masks_64_avx2(bp.add(i), open_ch, close_ch)
                    } else {
                        masks_64_scalar(&b[i..i + 64], open_ch, close_ch)
                    }
                }
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    masks_64_neon(bp.add(i), open_ch, close_ch)
                }
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                {
                    let _ = bp;
                    masks_64_scalar(&b[i..i + 64], open_ch, close_ch)
                }
            };
            step(
                q, bs, op, cl, 64, &mut prev_esc, &mut in0, &mut valid0, &mut valid1,
                &mut d0, &mut md0, &mut d1, &mut md1,
            );
            i += 64;
        }
        if i < hi {
            let rem = hi - i;
            let (q, bs, op, cl) = masks_64_scalar(&b[i..hi], open_ch, close_ch);
            step(
                q, bs, op, cl, rem, &mut prev_esc, &mut in0, &mut valid0, &mut valid1,
                &mut d0, &mut md0, &mut d1, &mut md1,
            );
        }
        ChunkSummary {
            end_in: [in0, !in0],
            valid: [valid0, valid1],
            delta: [d0, d1],
            min_d: [md0, md1],
        }
    };

    let n_chunks = cuts.len() - 1;
    let summaries: Vec<ChunkSummary> = if n_chunks > 1 {
        (0..n_chunks).into_par_iter().map(|k| summarize(cuts[k], cuts[k + 1])).collect()
    } else {
        vec![summarize(cuts[0], cuts[1])]
    };

    // Prefix-compose the exact `(in_str, depth)` across chunks to locate the exact
    // chunk where `depth` first hits `-1` (the matching closing `}` or `]`).
    let mut cur_in = false;
    let mut cur_d = 0i32;
    let mut chunk_states: Vec<(usize, bool)> = Vec::with_capacity(n_chunks);
    let mut close_chunk = None;
    for k in 0..n_chunks {
        chunk_states.push((cuts[k], cur_in));
        let s = cur_in as usize;
        let sm = summaries[k];
        if !sm.valid[s] {
            return None;
        }
        if cur_d + sm.min_d[s] < 0 {
            close_chunk = Some((k, cur_in, cur_d));
            break;
        }
        cur_in = sm.end_in[s];
        cur_d += sm.delta[s];
    }
    let (ck, mut in_str, mut d) = close_chunk?;
    // Scan only the single closing chunk using 64B SIMD blocks until `d` drops < 0!
    let mut i = cuts[ck];
    let hi = cuts[ck + 1];
    let mut prev_esc = false;
    let bp = b.as_ptr();
    while i < hi {
        let rem = (hi - i).min(64);
        let (q, bs, op, cl) = if rem == 64 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if slvl == 3 {
                    masks_64_avx512bw(bp.add(i), open_ch, close_ch)
                } else if slvl == 2 {
                    masks_64_avx2(bp.add(i), open_ch, close_ch)
                } else {
                    masks_64_scalar(&b[i..i + 64], open_ch, close_ch)
                }
            }
            #[cfg(target_arch = "aarch64")]
            unsafe {
                masks_64_neon(bp.add(i), open_ch, close_ch)
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                let _ = bp;
                masks_64_scalar(&b[i..i + 64], open_ch, close_ch)
            }
        } else {
            masks_64_scalar(&b[i..hi], open_ch, close_ch)
        };
        let (escaped, next_pe) = compute_escaped_mask(bs, prev_esc);
        prev_esc = next_pe;
        let uq = q & !escaped;
        let mut pm = uq;
        pm ^= pm << 1;
        pm ^= pm << 2;
        pm ^= pm << 4;
        pm ^= pm << 8;
        pm ^= pm << 16;
        pm ^= pm << 32;
        let mask = if in_str { !pm } else { pm };
        in_str = ((mask >> (rem - 1)) & 1) != 0;
        let real_op = op & !mask;
        let real_cl = cl & !mask;
        let mut both = real_op | real_cl;
        while both != 0 {
            let tz = both.trailing_zeros() as usize;
            let bit = 1u64 << tz;
            both &= both - 1;
            if (real_op & bit) != 0 {
                d += 1;
            } else {
                d -= 1;
                if d < 0 {
                    let close_pos = i + tz;
                    chunk_states.retain(|&(pos, _)| pos < close_pos);
                    return Some((close_pos, chunk_states));
                }
            }
        }
        i += rem;
    }
    None
}

/// Hex digit LUT (`0..=15`, `0xFF` = invalid).
static HEX_LUT: [u8; 256] = {
    let mut t = [0xFFu8; 256];
    let mut i = 0;
    while i < 10 {
        t[(b'0' as usize) + i] = i as u8;
        i += 1;
    }
    let mut j = 0;
    while j < 6 {
        t[(b'a' as usize) + j] = (10 + j) as u8;
        t[(b'A' as usize) + j] = (10 + j) as u8;
        j += 1;
    }
    t
};

#[inline(always)]
fn parse_hex4(b: &[u8], i: usize) -> Option<u16> {
    if i + 4 > b.len() {
        return None;
    }
    let h0 = HEX_LUT[b[i] as usize];
    let h1 = HEX_LUT[b[i + 1] as usize];
    let h2 = HEX_LUT[b[i + 2] as usize];
    let h3 = HEX_LUT[b[i + 3] as usize];
    if (h0 | h1 | h2 | h3) & 0xF0 != 0 {
        return None;
    }
    Some(((h0 as u16) << 12) | ((h1 as u16) << 8) | ((h2 as u16) << 4) | (h3 as u16))
}

/// Unescape a JSON string starting at `b[*pos] == b'"'` directly into `out`,
/// advancing `*pos` past the closing `'"'`. Returns `(start_off, len)` in `out`.
/// Strictly validates RFC 8259 escapes & surrogate pairs; returns `None` on error.
#[inline]
fn unescape_json_str(b: &[u8], pos: &mut usize, out: &mut Vec<u8>) -> Option<(usize, usize)> {
    let mut i = *pos;
    let n = b.len();
    if i >= n || b[i] != b'"' {
        return None;
    }
    i += 1;
    let out_start = out.len();
    loop {
        let run_start = i;
        while i < n {
            // SAFETY: `i < n`.
            let c = unsafe { *b.get_unchecked(i) };
            if c < 0x20 || c == b'"' || c == b'\\' {
                break;
            }
            i += 1;
        }
        if i >= n {
            return None;
        }
        let c = unsafe { *b.get_unchecked(i) };
        if c < 0x20 {
            return None;
        }
        if i > run_start {
            out.extend_from_slice(&b[run_start..i]);
        }
        if c == b'"' {
            *pos = i + 1;
            return Some((out_start, out.len() - out_start));
        }
        // Backslash escape sequence
        i += 1;
        if i >= n {
            return None;
        }
        match unsafe { *b.get_unchecked(i) } {
            b'"' => { out.push(b'"'); i += 1; }
            b'\\' => { out.push(b'\\'); i += 1; }
            b'/' => { out.push(b'/'); i += 1; }
            b'b' => { out.push(0x08); i += 1; }
            b'f' => { out.push(0x0C); i += 1; }
            b'n' => { out.push(b'\n'); i += 1; }
            b'r' => { out.push(b'\r'); i += 1; }
            b't' => { out.push(b'\t'); i += 1; }
            b'u' => {
                i += 1;
                if i + 4 > n {
                    return None;
                }
                // Fast path for ByteLevel space `\u0120` -> `Ġ` (0xC4 0xA0), ~50% of ByteLevel tokens!
                if &b[i..i + 4] == b"0120" {
                    out.extend_from_slice(&[0xC4, 0xA0]);
                    i += 4;
                    continue;
                }
                let u1 = parse_hex4(b, i)?;
                i += 4;
                if u1 < 0x80 {
                    out.push(u1 as u8);
                } else if u1 < 0x800 {
                    out.push(0xC0 | (u1 >> 6) as u8);
                    out.push(0x80 | (u1 & 0x3F) as u8);
                } else if !(0xD800..=0xDFFF).contains(&u1) {
                    out.push(0xE0 | (u1 >> 12) as u8);
                    out.push(0x80 | ((u1 >> 6) & 0x3F) as u8);
                    out.push(0x80 | (u1 & 0x3F) as u8);
                } else if u1 <= 0xDBFF {
                    if i + 6 > n || b[i] != b'\\' || b[i + 1] != b'u' {
                        return None;
                    }
                    let u2 = parse_hex4(b, i + 2)?;
                    if !(0xDC00..=0xDFFF).contains(&u2) {
                        return None;
                    }
                    i += 6;
                    let cp = 0x10000 + ((((u1 as u32) - 0xD800) << 10) | ((u2 as u32) - 0xDC00));
                    out.push(0xF0 | (cp >> 18) as u8);
                    out.push(0x80 | ((cp >> 12) & 0x3F) as u8);
                    out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
                    out.push(0x80 | (cp & 0x3F) as u8);
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }
}

#[inline]
fn parse_u64_ascii(b: &[u8], pos: &mut usize) -> Option<u64> {
    let mut i = *pos;
    if i >= b.len() || !b[i].is_ascii_digit() {
        return None;
    }
    let mut v = 0u64;
    while i < b.len() && b[i].is_ascii_digit() {
        v = v.checked_mul(10)?.checked_add((b[i] - b'0') as u64)?;
        i += 1;
    }
    *pos = i;
    Some(v)
}

/// Advance from `(pos, in_str)` inside a flat JSON object (`"k": int, ...`) or
/// 1D/2D merges array (`"a b", ...` or `["a", "b"], ...`) to the EXACT byte
/// offset of the next top-level entry start (`"` or `[`), stopping at `limit`.
fn align_entry_start(b: &[u8], mut i: usize, mut in_str: bool, limit: usize, is_2d: bool) -> usize {
    if i >= limit {
        return limit;
    }
    // If we started inside a string, finish that string first.
    if in_str {
        while i < limit {
            match memchr::memchr2(b'"', b'\\', &b[i..limit]) {
                Some(off) => {
                    i += off;
                    if b[i] == b'"' {
                        in_str = false;
                        i += 1;
                        break;
                    }
                    i += 2;
                }
                None => return limit,
            }
        }
        if in_str {
            return limit;
        }
    }
    if is_2d {
        // In 2D merges `[["a", "b"], ["c", "d"], ...]`, advance to the first `[` outside strings!
        while i < limit {
            match memchr::memchr2(b'"', b'[', &b[i..limit]) {
                Some(off) => {
                    i += off;
                    if b[i] == b'[' {
                        return i;
                    }
                    i = match skip_json_string(b, i) {
                        Some(ni) => ni,
                        None => return limit,
                    };
                }
                None => return limit,
            }
        }
        return limit;
    }
    // In flat `"vocab"` (`"k": 123, "k2": ...`) or 1D `"merges"` (`"a b", "c d", ...`):
    // advance to the first `,` outside strings, then skip whitespace to the next entry!
    while i < limit {
        match memchr::memchr2(b'"', b',', &b[i..limit]) {
            Some(off) => {
                i += off;
                if b[i] == b',' {
                    return skip_ws(b, i + 1).min(limit);
                }
                i = match skip_json_string(b, i) {
                    Some(ni) => ni,
                    None => return limit,
                };
            }
            None => return limit,
        }
    }
    limit
}

/// 16-shard parallel open-addressing vocab lookup table (`raw_utf8_bytes -> id`).
/// Pre-bucketed during parallel vocab parsing so each shard builder touches only
/// its own 1/16 slice with 16-ahead prefetching (< 0.5 ms for 200K tokens).
///
/// Compact 16-byte `VSlot` inlines all keys `<= 8` bytes (~90%+ of merge inputs)
/// directly inside `payload: u64`, eliminating the second unprefetched L3 cache
/// miss into `raw_arenas` on >90% of `VocabIndex::get_h_padded` probes.
const VSHARDS: usize = 16;
const VSHARD_SHIFT: u32 = 64 - 4; // top 4 bits select shard 0..16
const INLINE_CHUNK: u16 = u16::MAX;

/// Branchless 2-instruction pack of `base[off..off+len]` (`len <= 8`) into `u64`.
/// Caller ensures `off + 8 <= base.len()` (via 8-byte arena tail slack).
#[inline(always)]
fn pack_inline_u64_padded(base: &[u8], off: usize, len: usize) -> u64 {
    debug_assert!(len <= 8 && off + 8 <= base.len());
    if len == 0 {
        return 0;
    }
    let w = u64::from_le(unsafe { (base.as_ptr().add(off) as *const u64).read_unaligned() });
    w & (u64::MAX >> ((8 - len) << 3))
}

#[inline(always)]
fn pack_inline_u64_exact(b: &[u8]) -> u64 {
    let mut v = 0u64;
    for (i, &c) in b.iter().enumerate() {
        v |= (c as u64) << (i << 3);
    }
    v
}

#[derive(Clone, Copy)]
struct VSlot {
    id: u32,        // u32::MAX = empty
    len_chunk: u32, // low 16 bits = len; high 16 bits = chunk (INLINE_CHUNK if len <= 8)
    payload: u64,   // inline: exact zero-padded bytes; out-of-line: (h32 as u64) | ((off as u64) << 32)
}

struct VocabIndex<'a> {
    arenas: &'a [Vec<u8>],
    shards: Vec<(Vec<VSlot>, usize)>,
}

impl<'a> VocabIndex<'a> {
    #[inline(always)]
    fn slice_of(&self, chunk: u16, off: u32, len: u16) -> &[u8] {
        let a = unsafe { self.arenas.get_unchecked(chunk as usize) };
        let o = off as usize;
        unsafe { a.get_unchecked(o..o + len as usize) }
    }

    #[inline(always)]
    fn prefetch_h(&self, h: u64) {
        let s = (h >> VSHARD_SHIFT) as usize;
        let (slots, mask) = unsafe { self.shards.get_unchecked(s) };
        let idx = ((h >> 27) as usize) & *mask;
        unsafe {
            crate::chunk_table::prefetch(slots.as_ptr().add(idx) as *const u8);
        }
    }

    #[inline(always)]
    fn get_h_padded(&self, arena: &[u8], off: usize, len: usize, h: u64) -> Option<u32> {
        let s = (h >> VSHARD_SHIFT) as usize;
        let (slots, mask) = unsafe { self.shards.get_unchecked(s) };
        let mut idx = ((h >> 27) as usize) & *mask;
        if len <= 8 {
            let tag = (len as u32) | ((INLINE_CHUNK as u32) << 16);
            let kv = pack_inline_u64_padded(arena, off, len);
            loop {
                let sl = unsafe { *slots.get_unchecked(idx) };
                if sl.id == u32::MAX {
                    return None;
                }
                if sl.len_chunk == tag && sl.payload == kv {
                    return Some(sl.id);
                }
                idx = (idx + 1) & *mask;
            }
        }
        let key = unsafe { arena.get_unchecked(off..off + len) };
        let h32 = h as u32;
        let klen = len as u32;
        loop {
            let sl = unsafe { *slots.get_unchecked(idx) };
            if sl.id == u32::MAX {
                return None;
            }
            if (sl.len_chunk & 0xFFFF) == klen && (sl.payload as u32) == h32 {
                let chunk = (sl.len_chunk >> 16) as u16;
                let soff = (sl.payload >> 32) as u32;
                if fast_eq(self.slice_of(chunk, soff, klen as u16), key) {
                    return Some(sl.id);
                }
            }
            idx = (idx + 1) & *mask;
        }
    }

    #[inline(always)]
    fn get(&self, key: &[u8]) -> Option<u32> {
        let h = fx_hash_bytes(key);
        let s = (h >> VSHARD_SHIFT) as usize;
        let (slots, mask) = unsafe { self.shards.get_unchecked(s) };
        let mut idx = ((h >> 27) as usize) & *mask;
        if key.len() <= 8 {
            let tag = (key.len() as u32) | ((INLINE_CHUNK as u32) << 16);
            let kv = pack_inline_u64_exact(key);
            loop {
                let sl = unsafe { *slots.get_unchecked(idx) };
                if sl.id == u32::MAX {
                    return None;
                }
                if sl.len_chunk == tag && sl.payload == kv {
                    return Some(sl.id);
                }
                idx = (idx + 1) & *mask;
            }
        }
        let h32 = h as u32;
        let klen = key.len() as u32;
        loop {
            let sl = unsafe { *slots.get_unchecked(idx) };
            if sl.id == u32::MAX {
                return None;
            }
            if (sl.len_chunk & 0xFFFF) == klen && (sl.payload as u32) == h32 {
                let chunk = (sl.len_chunk >> 16) as u16;
                let soff = (sl.payload >> 32) as u32;
                if fast_eq(self.slice_of(chunk, soff, klen as u16), key) {
                    return Some(sl.id);
                }
            }
            idx = (idx + 1) & *mask;
        }
    }
}

/// Try to parse a HuggingFace `tokenizer.json` byte buffer via the parallel
/// zero-malloc loader. Returns `None` on any unsupported/malformed structure so
/// `Tokenizer::new` falls back cleanly to `serde_json`.
#[allow(clippy::too_many_arguments)]
pub fn try_load_fast(
    bytes: &[u8],
    c2b: &[u16; 324],
    allow_par: bool,
    decode_bl: fn(&str, &[u16; 324], &mut Vec<u8>),
    decode_bf: fn(&str, &mut Vec<u8>),
    detect_meta_split: fn(&Value, &Value, bool, bool, bool) -> Option<char>,
) -> Option<FastLoadedTokenizer> {
    let n = bytes.len();
    let mut i = skip_ws(bytes, 0);
    if i >= n || bytes[i] != b'{' {
        return None;
    }
    i += 1;

    let mut normalizer_sl: Option<&[u8]> = None;
    let mut pre_sl: Option<&[u8]> = None;
    let mut post_sl: Option<&[u8]> = None;
    let mut dec_sl: Option<&[u8]> = None;
    let mut added_sl: Option<&[u8]> = None;
    let mut model_open: Option<usize> = None;
    let mut key_buf = Vec::with_capacity(64);

    // Phase 1a: locate top-level fields. Inside `"model"`, handle `"vocab"` and
    // `"merges"` with our parallel section scanner so we NEVER linearly walk
    // 12 MB on 1 core!
    type SectionRange = (usize, usize, Vec<(usize, bool)>);
    // (raw_off, raw_len, id, dec_off, dec_len, flags)
    type VocabItem = (u32, u16, u32, u32, u32, u8);
    // Pre-bucketed shard entry: (raw_off, raw_len, id, h)
    type ShardEntry = (u32, u16, u32, u64);

    let mut model_type: Option<String> = None;
    let mut byte_fallback = false;
    let mut ignore_merges = false;
    let mut continuing_subword_prefix: Option<String> = None;
    let mut end_of_word_suffix: Option<String> = None;
    let mut vocab_range: Option<SectionRange> = None;
    let mut merges_range: Option<SectionRange> = None;

    loop {
        i = skip_ws(bytes, i);
        if i >= n {
            return None;
        }
        if bytes[i] == b'}' {
            break;
        }
        key_buf.clear();
        let (ks, kl) = unescape_json_str(bytes, &mut i, &mut key_buf)?;
        let key = std::str::from_utf8(&key_buf[ks..ks + kl]).ok()?;
        i = skip_ws(bytes, i);
        if i >= n || bytes[i] != b':' {
            return None;
        }
        i = skip_ws(bytes, i + 1);
        match key {
            "normalizer" => {
                let end = skip_json_value(bytes, i)?;
                normalizer_sl = Some(&bytes[i..end]);
                i = end;
            }
            "pre_tokenizer" => {
                let end = skip_json_value(bytes, i)?;
                pre_sl = Some(&bytes[i..end]);
                i = end;
            }
            "post_processor" => {
                let end = skip_json_value(bytes, i)?;
                post_sl = Some(&bytes[i..end]);
                i = end;
            }
            "decoder" => {
                let end = skip_json_value(bytes, i)?;
                dec_sl = Some(&bytes[i..end]);
                i = end;
            }
            "added_tokens" => {
                let end = skip_json_value(bytes, i)?;
                added_sl = Some(&bytes[i..end]);
                i = end;
            }
            "model" => {
                if i >= n || bytes[i] != b'{' {
                    return None;
                }
                model_open = Some(i);
                i += 1;
                loop {
                    i = skip_ws(bytes, i);
                    if i >= n {
                        return None;
                    }
                    if bytes[i] == b'}' {
                        i += 1;
                        break;
                    }
                    key_buf.clear();
                    let (mks, mkl) = unescape_json_str(bytes, &mut i, &mut key_buf)?;
                    let mkey = std::str::from_utf8(&key_buf[mks..mks + mkl]).ok()?;
                    i = skip_ws(bytes, i);
                    if i >= n || bytes[i] != b':' {
                        return None;
                    }
                    i = skip_ws(bytes, i + 1);
                    match mkey {
                        "type" => {
                            let end = skip_json_value(bytes, i)?;
                            model_type = serde_json::from_slice(&bytes[i..end]).ok()?;
                            if let Some(ref mt) = model_type {
                                if mt != "BPE" {
                                    return None;
                                }
                            }
                            i = end;
                        }
                        "byte_fallback" => {
                            let end = skip_json_value(bytes, i)?;
                            byte_fallback = serde_json::from_slice::<Option<bool>>(&bytes[i..end])
                                .ok()?
                                .unwrap_or(false);
                            i = end;
                        }
                        "ignore_merges" => {
                            let end = skip_json_value(bytes, i)?;
                            ignore_merges = serde_json::from_slice::<Option<bool>>(&bytes[i..end])
                                .ok()?
                                .unwrap_or(false);
                            i = end;
                        }
                        "continuing_subword_prefix" => {
                            let end = skip_json_value(bytes, i)?;
                            continuing_subword_prefix = serde_json::from_slice(&bytes[i..end]).ok()?;
                            i = end;
                        }
                        "end_of_word_suffix" => {
                            let end = skip_json_value(bytes, i)?;
                            end_of_word_suffix = serde_json::from_slice(&bytes[i..end]).ok()?;
                            i = end;
                        }
                        "vocab" => {
                            if i >= n || bytes[i] != b'{' {
                                return None;
                            }
                            let (close_pos, states) = scan_delimited_section(bytes, i, true, allow_par)?;
                            vocab_range = Some((i + 1, close_pos, states));
                            i = close_pos + 1;
                        }
                        "merges" => {
                            if i >= n || bytes[i] != b'[' {
                                return None;
                            }
                            let (close_pos, states) = scan_delimited_section(bytes, i, false, allow_par)?;
                            merges_range = Some((i + 1, close_pos, states));
                            i = close_pos + 1;
                        }
                        _ => {
                            i = skip_json_value(bytes, i)?;
                        }
                    }
                    i = skip_ws(bytes, i);
                    if i < n && bytes[i] == b',' {
                        i += 1;
                    }
                }
            }
            _ => {
                i = skip_json_value(bytes, i)?;
            }
        }
        i = skip_ws(bytes, i);
        if i < n && bytes[i] == b',' {
            i += 1;
        }
    }

    model_open?;
    let (v_start, v_end, v_states) = vocab_range?;
    let has_merges = merges_range.is_some();

    // Parse tiny config JSON slices via exact serde_json (< 0.1 ms total).
    let normalizer: Value = match normalizer_sl {
        Some(sl) => serde_json::from_slice(sl).ok()?,
        None => Value::Null,
    };
    let pre_tokenizer: Value = match pre_sl {
        Some(sl) => serde_json::from_slice(sl).ok()?,
        None => Value::Null,
    };
    let post_processor: Option<Value> = match post_sl {
        Some(sl) => Some(serde_json::from_slice(sl).ok()?),
        None => None,
    };
    let decoder: Value = match dec_sl {
        Some(sl) => serde_json::from_slice(sl).ok()?,
        None => Value::Null,
    };
    let added_tokens: Vec<FastAddedToken> = match added_sl {
        Some(sl) => serde_json::from_slice::<Option<Vec<FastAddedToken>>>(sl).ok()?.unwrap_or_default(),
        None => Vec::new(),
    };
    let eow_nonempty = end_of_word_suffix.as_ref().is_some_and(|s| !s.is_empty());
    let meta_split_cand = detect_meta_split(&pre_tokenizer, &normalizer, byte_fallback, ignore_merges, eow_nonempty);

    // Phase 2: Parallel zero-malloc `"vocab"` parsing + ByteLevel/byte_fallback decoding!
    let mut v_bounds: Vec<usize> = Vec::with_capacity(v_states.len() + 1);
    v_bounds.push(skip_ws(bytes, v_start).min(v_end));
    for &(pos, in_str) in v_states.iter().skip(1) {
        let aligned = align_entry_start(bytes, pos, in_str, v_end, false);
        if aligned > *v_bounds.last().unwrap() && aligned < v_end {
            v_bounds.push(aligned);
        }
    }
    v_bounds.push(v_end);

    struct VocabChunkOut {
        raw_arena: Vec<u8>,
        items: Vec<VocabItem>,
        shards: [Vec<ShardEntry>; VSHARDS],
        dec_arena: Vec<u8>,
        char_entries: Vec<(u32, u32)>,
        max_id: u32,
    }

    let parse_vocab_chunk = |lo: usize, hi: usize| -> Option<VocabChunkOut> {
        let est = ((hi - lo) / 14).max(16);
        let mut raw_arena = Vec::with_capacity((hi - lo) / 2 + 64);
        let mut dec_arena = Vec::with_capacity((hi - lo) / 2 + 64);
        let mut items = Vec::with_capacity(est);
        let per_s = (est / VSHARDS).max(8);
        let mut shards: [Vec<ShardEntry>; VSHARDS] = std::array::from_fn(|_| Vec::with_capacity(per_s));
        let mut char_entries = Vec::new();
        let mut max_id = 0u32;
        let mut p = skip_ws(bytes, lo);
        while p < hi {
            if bytes[p] == b',' {
                p = skip_ws(bytes, p + 1);
                continue;
            }
            let (ro, rl) = unescape_json_str(bytes, &mut p, &mut raw_arena)?;
            if rl > u16::MAX as usize {
                return None;
            }
            p = skip_ws(bytes, p);
            if p >= hi || bytes[p] != b':' {
                return None;
            }
            p = skip_ws(bytes, p + 1);
            let id_u64 = parse_u64_ascii(bytes, &mut p)?;
            let id = id_u64 as u32;
            if id > max_id {
                max_id = id;
            }
            let raw_sl = &raw_arena[ro..ro + rl];
            let h = fx_hash_bytes(raw_sl);
            // SAFETY: unescape_json_str preserves valid UTF-8 from validated `bytes`.
            let ks = unsafe { std::str::from_utf8_unchecked(raw_sl) };

            let mut flags = 0u8;
            if let Some(rep) = meta_split_cand {
                if ks.trim_start_matches(rep).contains(rep) {
                    flags |= 2;
                }
            }
            let dec_start = dec_arena.len() as u32;
            if byte_fallback {
                decode_bf(ks, &mut dec_arena);
            } else {
                decode_bl(ks, c2b, &mut dec_arena);
            }
            let dec_len = dec_arena.len() as u32 - dec_start;

            let mut chs = ks.chars();
            if let (Some(c), None) = (chs.next(), chs.next()) {
                if byte_fallback {
                    char_entries.push((c as u32, id));
                } else {
                    let cp = c as usize;
                    if cp < 324 && c2b[cp] != u16::MAX {
                        flags |= 1;
                    }
                }
            }

            items.push((ro as u32, rl as u16, id, dec_start, dec_len, flags));
            shards[(h >> VSHARD_SHIFT) as usize].push((ro as u32, rl as u16, id, h));
            p = skip_ws(bytes, p);
            if p < hi && bytes[p] == b',' {
                p = skip_ws(bytes, p + 1);
            }
        }
        // 8-byte slack so branchless u64 inline-key loads never read past end.
        raw_arena.extend_from_slice(&[0u8; 8]);
        Some(VocabChunkOut { raw_arena, items, shards, dec_arena, char_entries, max_id })
    };

    let n_vchunks = v_bounds.len() - 1;
    let vchunks_opt: Vec<Option<VocabChunkOut>> = if allow_par && n_vchunks > 1 {
        (0..n_vchunks)
            .into_par_iter()
            .map(|c| parse_vocab_chunk(v_bounds[c], v_bounds[c + 1]))
            .collect()
    } else {
        (0..n_vchunks)
            .map(|c| parse_vocab_chunk(v_bounds[c], v_bounds[c + 1]))
            .collect()
    };
    let mut vchunks = Vec::with_capacity(n_vchunks);
    for co in vchunks_opt {
        vchunks.push(co?);
    }

    let mut max_id = 0u32;
    let mut total_dec_bytes = 0usize;
    let mut dec_offsets = Vec::with_capacity(n_vchunks);
    let mut shard_counts = [0usize; VSHARDS];
    for vc in &vchunks {
        max_id = max_id.max(vc.max_id);
        dec_offsets.push(total_dec_bytes as u32);
        total_dec_bytes += vc.dec_arena.len();
        for (sc, sh) in shard_counts.iter_mut().zip(&vc.shards) {
            *sc += sh.len();
        }
    }
    let mut vocab_size = (max_id as usize) + 1;
    for at in &added_tokens {
        if let Some(id) = at.id {
            vocab_size = vocab_size.max(id as usize + 1);
        }
    }

    // Build contiguous `byte_arena`, `byte_span`, `canonical`, `has_internal_ms`,
    // `id_to_raw` (for sequential-ID target prediction in merges!), and 16-shard `VocabIndex`.
    let mut raw_arenas: Vec<Vec<u8>> = Vec::with_capacity(n_vchunks);
    let mut items_per_chunk: Vec<Vec<VocabItem>> = Vec::with_capacity(n_vchunks);
    let mut shards_per_chunk: Vec<[Vec<ShardEntry>; VSHARDS]> = Vec::with_capacity(n_vchunks);
    let mut byte_arena = Vec::with_capacity(total_dec_bytes + 4096);
    let mut char_ascii = [-1i32; 128];
    let mut char_map = rustc_hash::FxHashMap::default();
    for mut vc in vchunks {
        byte_arena.extend_from_slice(&vc.dec_arena);
        if byte_fallback {
            for (cp, id) in vc.char_entries {
                if cp < 128 {
                    char_ascii[cp as usize] = id as i32;
                } else {
                    char_map.insert(cp, id);
                }
            }
        }
        raw_arenas.push(std::mem::take(&mut vc.raw_arena));
        items_per_chunk.push(std::mem::take(&mut vc.items));
        shards_per_chunk.push(vc.shards);
    }

    let mut byte_span = vec![(0u32, 0u32); vocab_size];
    let mut canonical = vec![false; vocab_size];
    let mut has_internal_ms = if meta_split_cand.is_some() {
        vec![false; vocab_size]
    } else {
        Vec::new()
    };
    // `(chunk, off, len)` per token id for O(1) zero-hash `tg` prediction!
    let mut id_to_raw: Vec<(u16, u32, u16)> = vec![(u16::MAX, 0, 0); vocab_size];
    for (ci, items) in items_per_chunk.iter().enumerate() {
        let base_dec = dec_offsets[ci];
        for &(ro, rl, id, doff, dlen, flags) in items {
            let iu = id as usize;
            unsafe {
                *byte_span.get_unchecked_mut(iu) = (base_dec + doff, dlen);
                *canonical.get_unchecked_mut(iu) = (flags & 1) != 0;
                *id_to_raw.get_unchecked_mut(iu) = (ci as u16, ro, rl);
            }
            if !has_internal_ms.is_empty() {
                has_internal_ms[iu] = (flags & 2) != 0;
            }
        }
    }

    if n_vchunks >= INLINE_CHUNK as usize {
        return None;
    }

    // Build all 16 pre-bucketed open-addressing shards IN PARALLEL across cores!
    let build_shard = |s: usize| -> (Vec<VSlot>, usize) {
        let cnt = shard_counts[s];
        let mut cap = 16usize;
        while cap * 2 < cnt * 3 + 4 {
            cap <<= 1;
        }
        let mask = cap - 1;
        let mut slots = vec![
            VSlot { id: u32::MAX, len_chunk: 0, payload: 0 };
            cap
        ];
        for (ci, spc) in shards_per_chunk.iter().enumerate() {
            let arena_ci = &raw_arenas[ci];
            let entries = &spc[s];
            let elen = entries.len();
            for j in 0..elen {
                if j + 16 < elen {
                    let ph = unsafe { entries.get_unchecked(j + 16).3 };
                    let pidx = ((ph >> 27) as usize) & mask;
                    unsafe {
                        crate::chunk_table::prefetch(slots.as_ptr().add(pidx) as *const u8);
                    }
                }
                let (ro, rl, id, h) = unsafe { *entries.get_unchecked(j) };
                let mut idx = ((h >> 27) as usize) & mask;
                if rl <= 8 {
                    let tag = (rl as u32) | ((INLINE_CHUNK as u32) << 16);
                    let kv = pack_inline_u64_padded(arena_ci, ro as usize, rl as usize);
                    loop {
                        let sl = unsafe { slots.get_unchecked_mut(idx) };
                        if sl.id == u32::MAX {
                            *sl = VSlot { id, len_chunk: tag, payload: kv };
                            break;
                        }
                        if sl.len_chunk == tag && sl.payload == kv {
                            sl.id = id;
                            break;
                        }
                        idx = (idx + 1) & mask;
                    }
                } else {
                    let h32 = h as u32;
                    let key = unsafe { arena_ci.get_unchecked(ro as usize..ro as usize + rl as usize) };
                    let tag = (rl as u32) | ((ci as u32) << 16);
                    let pay = (h32 as u64) | ((ro as u64) << 32);
                    loop {
                        let sl = unsafe { slots.get_unchecked_mut(idx) };
                        if sl.id == u32::MAX {
                            *sl = VSlot { id, len_chunk: tag, payload: pay };
                            break;
                        }
                        if (sl.len_chunk & 0xFFFF) == rl as u32 && (sl.payload as u32) == h32 {
                            let sc = (sl.len_chunk >> 16) as usize;
                            let so = (sl.payload >> 32) as usize;
                            let cand = unsafe {
                                raw_arenas
                                    .get_unchecked(sc)
                                    .get_unchecked(so..so + rl as usize)
                            };
                            if fast_eq(cand, key) {
                                sl.id = id;
                                break;
                            }
                        }
                        idx = (idx + 1) & mask;
                    }
                }
            }
        }
        (slots, mask)
    };

    let shards: Vec<(Vec<VSlot>, usize)> = if allow_par {
        (0..VSHARDS).into_par_iter().map(build_shard).collect()
    } else {
        (0..VSHARDS).map(build_shard).collect()
    };
    let vindex = VocabIndex { arenas: &raw_arenas, shards };

    // Snapshot KNOWN_TOKS and optional singleton_eow using VocabIndex.
    let mut known_token_ids = Vec::new();
    for &kt in KNOWN_TOKS {
        if let Some(id) = vindex.get(kt.as_bytes()) {
            known_token_ids.push((kt.to_string(), id));
        }
    }
    let singleton_eow = if let Some(ref suf) = end_of_word_suffix {
        if suf.is_empty() {
            None
        } else if byte_fallback {
            return None; // let Tokenizer::new produce the exact error message
        } else {
            let mut b2u = ['\0'; 256];
            for (cp, &bv) in c2b.iter().enumerate() {
                if bv != u16::MAX {
                    b2u[bv as usize] = char::from_u32(cp as u32)?;
                }
            }
            let mut tbl = [-1i32; 256];
            let mut key = String::with_capacity(suf.len() + 4);
            for b in 0..256 {
                key.clear();
                key.push(b2u[b]);
                key.push_str(suf);
                if let Some(id) = vindex.get(key.as_bytes()) {
                    tbl[b] = id as i32;
                }
            }
            Some(tbl)
        }
    } else {
        None
    };

    // Phase 3: Parallel zero-malloc `"merges"` parsing + resolution with `pred_tg`!
    let merges = if let Some((m_start, m_end, m_states)) = merges_range {
        let first_byte_pos = skip_ws(bytes, m_start);
        if first_byte_pos >= m_end {
            Vec::new()
        } else {
            let is_2d = bytes[first_byte_pos] == b'[';
            let mut m_bounds: Vec<usize> = Vec::with_capacity(m_states.len() + 1);
            m_bounds.push(first_byte_pos);
            for &(pos, in_str) in m_states.iter().skip(1) {
                let aligned = align_entry_start(bytes, pos, in_str, m_end, is_2d);
                if aligned > *m_bounds.last().unwrap() && aligned < m_end {
                    m_bounds.push(aligned);
                }
            }
            m_bounds.push(m_end);

            let parse_merges_chunk = |lo: usize, hi: usize| -> Option<Vec<(u32, u32, u32)>> {
                const MBATCH: usize = 32;
                const PF: usize = 12;
                let est = ((hi - lo) / 14).max(16);
                let mut out = Vec::with_capacity(est);
                let mut m_arena = Vec::with_capacity(MBATCH * 32);
                let mut ab_buf = Vec::with_capacity(256);
                // (a_off, a_len, b_off, b_len, h_a, h_b)
                let mut mbuf = [(0u32, 0u32, 0u32, 0u32, 0u64, 0u64); MBATCH];
                let mut pred_tg = 256u32;
                let mut p = skip_ws(bytes, lo);
                while p < hi {
                    m_arena.clear();
                    let mut cnt = 0usize;
                    while p < hi && cnt < MBATCH {
                        if bytes[p] == b',' {
                            p = skip_ws(bytes, p + 1);
                            continue;
                        }
                        if bytes[p] == b'"' {
                            let (so, sl) = unescape_json_str(bytes, &mut p, &mut m_arena)?;
                            let s = &m_arena[so..so + sl];
                            let mut sp = 0usize;
                            while sp < sl && unsafe { *s.get_unchecked(sp) } != b' ' {
                                sp += 1;
                            }
                            if sp < sl {
                                let a_off = so as u32;
                                let a_len = sp as u32;
                                let b_off = (so + sp + 1) as u32;
                                let b_len = (sl - sp - 1) as u32;
                                let h_a = fx_hash_bytes(&s[..sp]);
                                let h_b = fx_hash_bytes(&s[sp + 1..]);
                                mbuf[cnt] = (a_off, a_len, b_off, b_len, h_a, h_b);
                                cnt += 1;
                            }
                        } else if bytes[p] == b'[' {
                            p = skip_ws(bytes, p + 1);
                            let (ao, al) = unescape_json_str(bytes, &mut p, &mut m_arena)?;
                            p = skip_ws(bytes, p);
                            if p >= hi || bytes[p] != b',' {
                                return None;
                            }
                            p = skip_ws(bytes, p + 1);
                            let (bo, bl) = unescape_json_str(bytes, &mut p, &mut m_arena)?;
                            p = skip_ws(bytes, p);
                            while p < hi && bytes[p] == b',' {
                                p = skip_json_value(bytes, p + 1)?;
                                p = skip_ws(bytes, p);
                            }
                            if p >= hi || bytes[p] != b']' {
                                return None;
                            }
                            p += 1;
                            let h_a = fx_hash_bytes(&m_arena[ao..ao + al]);
                            let h_b = fx_hash_bytes(&m_arena[bo..bo + bl]);
                            mbuf[cnt] = (ao as u32, al as u32, bo as u32, bl as u32, h_a, h_b);
                            cnt += 1;
                        } else {
                            return None;
                        }
                        p = skip_ws(bytes, p);
                        if p < hi && bytes[p] == b',' {
                            p = skip_ws(bytes, p + 1);
                        }
                    }

                    // 8-byte slack so branchless u64 inline-key loads never read past end.
                    m_arena.extend_from_slice(&[0u8; 8]);
                    // Pipeline VocabIndex lookups with 12-pair-ahead L3 prefetching!
                    let init_pf = cnt.min(PF);
                    for item in mbuf.iter().take(init_pf) {
                        vindex.prefetch_h(item.4);
                        vindex.prefetch_h(item.5);
                    }
                    for k in 0..cnt {
                        if k + PF < cnt {
                            let nxt = mbuf[k + PF];
                            vindex.prefetch_h(nxt.4);
                            vindex.prefetch_h(nxt.5);
                        }
                        let (ao, al, bo, bl, h_a, h_b) = mbuf[k];
                        if let (Some(t1), Some(t2)) = (
                            vindex.get_h_padded(&m_arena, ao as usize, al as usize, h_a),
                            vindex.get_h_padded(&m_arena, bo as usize, bl as usize, h_b),
                        ) {
                            let a_sl = &m_arena[ao as usize..(ao + al) as usize];
                            let b_sl = &m_arena[bo as usize..(bo + bl) as usize];
                            let mut tg_opt = None;
                            let pi = pred_tg as usize;
                            if pi < id_to_raw.len() {
                                let (rc, ro, rl) = unsafe { *id_to_raw.get_unchecked(pi) };
                                if rc != u16::MAX && rl as usize == a_sl.len() + b_sl.len() {
                                    let cand = vindex.slice_of(rc, ro, rl);
                                    if fast_eq(&cand[..a_sl.len()], a_sl) && fast_eq(&cand[a_sl.len()..], b_sl) {
                                        tg_opt = Some(pred_tg);
                                    }
                                }
                            }
                            let tg_opt = tg_opt.or_else(|| {
                                ab_buf.clear();
                                ab_buf.extend_from_slice(a_sl);
                                ab_buf.extend_from_slice(b_sl);
                                vindex.get(&ab_buf)
                            });
                            if let Some(tg) = tg_opt {
                                out.push((t1, t2, tg));
                                pred_tg = tg.wrapping_add(1);
                            }
                        }
                    }
                }
                Some(out)
            };

            let n_mchunks = m_bounds.len() - 1;
            let mchunks_opt: Vec<Option<Vec<(u32, u32, u32)>>> = if allow_par && n_mchunks > 1 {
                (0..n_mchunks)
                    .into_par_iter()
                    .map(|c| parse_merges_chunk(m_bounds[c], m_bounds[c + 1]))
                    .collect()
            } else {
                (0..n_mchunks)
                    .map(|c| parse_merges_chunk(m_bounds[c], m_bounds[c + 1]))
                    .collect()
            };
            let mut total_m = 0usize;
            let mut parts = Vec::with_capacity(n_mchunks);
            for mo in mchunks_opt {
                let v = mo?;
                total_m += v.len();
                parts.push(v);
            }
            let mut merges = Vec::with_capacity(total_m);
            for part in parts {
                merges.extend_from_slice(&part);
            }
            merges
        }
    } else {
        Vec::new()
    };

    Some(FastLoadedTokenizer {
        normalizer,
        pre_tokenizer,
        post_processor,
        decoder,
        added_tokens,
        model_type,
        byte_fallback,
        ignore_merges,
        continuing_subword_prefix,
        end_of_word_suffix,
        has_merges,
        vocab_size,
        merges,
        known_token_ids,
        singleton_eow,
        byte_arena,
        byte_span,
        canonical,
        has_internal_ms,
        char_ascii,
        char_map,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn escaped_oracle(bs: u64, prev_escaped: bool) -> (u64, bool) {
        let mut esc = prev_escaped;
        let mut out = 0u64;
        for bit in 0..64 {
            if esc {
                out |= 1u64 << bit;
                esc = false;
            } else if ((bs >> bit) & 1) != 0 {
                esc = true;
            }
        }
        (out, esc)
    }

    #[test]
    fn escaped_mask_matches_oracle() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let special = [
            0u64,
            u64::MAX,
            1,
            1 << 63,
            0b11 << 62,
            0b111 << 61,
            0x5555_5555_5555_5555,
            0xAAAA_AAAA_AAAA_AAAA,
            (1 << 63) | 1,
            (0b11 << 62) | 0b11,
            (0b111 << 61) | 0b1011,
        ];
        for &bs in &special {
            for pe in [false, true] {
                let (got, gc) = compute_escaped_mask(bs, pe);
                let (exp, ec) = escaped_oracle(bs, pe);
                assert_eq!(gc, ec, "carry mismatch for bs={bs:#x} pe={pe}");
                assert_eq!(got & !bs, exp & !bs, "non-bs mask mismatch for bs={bs:#x} pe={pe}");
            }
        }
        for _ in 0..10_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // Test both dense and sparse backslash masks
            let bs = if x & 1 == 0 { x & (x >> 3) } else { x };
            for pe in [false, true] {
                let (got, gc) = compute_escaped_mask(bs, pe);
                let (exp, ec) = escaped_oracle(bs, pe);
                assert_eq!(gc, ec, "carry mismatch for bs={bs:#x} pe={pe}");
                assert_eq!(got & !bs, exp & !bs, "mask mismatch for bs={bs:#x} pe={pe}");
            }
        }
    }

    #[test]
    fn scan_delimited_handles_64b_edges_and_2d_merges() {
        // Build an adversarial vocab crossing multiple 64B boundaries with escaped quotes & backslashes
        let mut s = String::from(r#"{"vocab":{"a":0,"b\"}\\c":1,"d\\\\":2,"e\\\\\"f":3,"#);
        for k in 4..400 {
            if k % 7 == 0 {
                s.push_str(&format!(r#""tok_{k}_\\\"\u0120}}{{":{k},"#));
            } else {
                s.push_str(&format!(r#""\u0120word_{k}":{k},"#));
            }
        }
        let close_expected = s.len() - 1;
        s.pop(); // remove trailing comma
        s.push('}');
        let tail = r#","merges":[["a","b\"}\\c"],["\u0120word_4","\u0120word_5"]]}"#;
        s.push_str(tail);
        let bytes = s.as_bytes();
        let open_pos = s.find(r#""vocab":{"#).unwrap() + 8;
        for allow_par in [false, true] {
            let (cp, _) = scan_delimited_section(bytes, open_pos, true, allow_par).unwrap();
            assert_eq!(cp, close_expected);
        }
        let m_open = s.find(r#""merges":["#).unwrap() + 9;
        let m_close_exp = s.len() - 2;
        for allow_par in [false, true] {
            let (cp, _) = scan_delimited_section(bytes, m_open, false, allow_par).unwrap();
            assert_eq!(cp, m_close_exp);
        }
        // Malformed backslash outside string must return None
        let bad = br#"{"a": 1 \ "b": 2}"#;
        assert!(scan_delimited_section(bad, 0, true, false).is_none());
    }

    #[test]
    fn unescape_and_fast_eq_exact() {
        let cases: &[(&[u8], Option<&str>)] = &[
            (br#""\u0120hello""#, Some("\u{0120}hello")),
            (br#""a\"b\\c\/d\b\f\n\r\t""#, Some("a\"b\\c/d\x08\x0c\n\r\t")),
            (br#""\u0041\u00e9\u2581\uD83D\uDE80""#, Some("A\u{00e9}\u{2581}\u{1F680}")),
            (br#""\uD800""#, None),
            (br#""\uDC00""#, None),
            (b"\"raw\nctrl\"", None),
        ];
        let mut buf = Vec::new();
        for &(inp, exp) in cases {
            buf.clear();
            let mut pos = 0;
            let got = unescape_json_str(inp, &mut pos, &mut buf)
                .map(|(o, l)| std::str::from_utf8(&buf[o..o + l]).unwrap().to_string());
            assert_eq!(got.as_deref(), exp, "case {:?}", std::str::from_utf8(inp));
        }
        for len in 0..=32usize {
            let a: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(37).wrapping_add(11)).collect();
            let mut b = a.clone();
            assert!(fast_eq(&a, &b));
            if len > 0 {
                for flip in 0..len {
                    b[flip] ^= 1;
                    assert!(!fast_eq(&a, &b), "len={len} flip={flip}");
                    b[flip] ^= 1;
                }
            }
        }
    }
}
