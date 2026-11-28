//! SIMD-accelerated UTF-8 transcoding for CPython compact/wide `str` buffers
//! (Latin-1 `u8`, UCS-2 `u16`, and UCS-4 `u32`).
//!
//! All functions use a store-and-skip strategy: pack 16–32 ASCII code units in
//! one vector store; when a non-ASCII code unit appears at lane `tz`, keep the
//! already-written ASCII prefix `0..tz`, emit the 2/3/4-byte UTF-8 sequence,
//! and resume vector scanning at `tz + 1`. Lone UTF-16 surrogates (`0xD800..0xE000`)
//! are rejected with `Err(code_unit_index)` to match CPython `str.encode('utf-8')`.

/// Exact UTF-8 length of `units` (a surrogate counts 3, like the 3 bytes it
/// would take; `units_to_utf8` rejects it). Branch-free, vectorizes.
pub(crate) fn utf8_len<T: Copy + Into<u32>>(units: &[T]) -> usize {
    units
        .iter()
        .map(|&u| {
            let c: u32 = u.into();
            1 + (c >= 0x80) as usize + (c >= 0x800) as usize + (c >= 0x10000) as usize
        })
        .sum()
}

/// Write the UTF-8 of `units` to `dst`; Ok(bytes written) — exactly
/// `utf8_len(units)` — or Err(i) at the first surrogate (which UTF-8 cannot
/// encode, exactly as CPython's `str.encode` refuses). Runs of ASCII are
/// copied a word at a time.
/// SAFETY: `dst` must be valid for `utf8_len(units)` writes.
pub(crate) unsafe fn units_to_utf8<T: Copy + Into<u32>>(units: &[T], dst: *mut u8) -> Result<usize, usize> {
    let w = std::mem::size_of::<T>();
    let n = units.len();
    let mut i = 0usize;
    let mut o = dst;
    let src = units.as_ptr() as *const u8;
    // ASCII fast path: 8 bytes of input per step
    let per = 8 / w;
    let hi: u64 = match w {
        1 => 0x8080_8080_8080_8080,
        2 => 0xFF80_FF80_FF80_FF80,
        _ => 0xFFFF_FF80_FFFF_FF80,
    };
    while i < n {
        if i + per <= n {
            let x = std::ptr::read_unaligned(src.add(i * w) as *const u64);
            if x & hi == 0 {
                for k in 0..per {
                    *o.add(k) = (x >> (8 * w * k)) as u8;
                }
                o = o.add(per);
                i += per;
                continue;
            }
        }
        let c: u32 = units[i].into();
        if c < 0x80 {
            *o = c as u8;
            o = o.add(1);
        } else if c < 0x800 {
            *o = 0xC0 | (c >> 6) as u8;
            *o.add(1) = 0x80 | (c & 0x3F) as u8;
            o = o.add(2);
        } else if c < 0x10000 {
            if (0xD800..0xE000).contains(&c) {
                return Err(i);
            }
            *o = 0xE0 | (c >> 12) as u8;
            *o.add(1) = 0x80 | ((c >> 6) & 0x3F) as u8;
            *o.add(2) = 0x80 | (c & 0x3F) as u8;
            o = o.add(3);
        } else {
            *o = 0xF0 | (c >> 18) as u8;
            *o.add(1) = 0x80 | ((c >> 12) & 0x3F) as u8;
            *o.add(2) = 0x80 | ((c >> 6) & 0x3F) as u8;
            *o.add(3) = 0x80 | (c & 0x3F) as u8;
            o = o.add(4);
        }
        i += 1;
    }
    Ok(o.offset_from(dst) as usize)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn has_avx2_bmi1() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static T: AtomicU8 = AtomicU8::new(0);
    let mut v = T.load(Relaxed);
    if v == 0 {
        v = if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("bmi1")
            && std::is_x86_feature_detected!("bmi2")
        {
            2
        } else {
            1
        };
        T.store(v, Relaxed);
    }
    v == 2
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn has_avx512f() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static T: AtomicU8 = AtomicU8::new(0);
    let mut v = T.load(Relaxed);
    if v == 0 {
        v = if std::is_x86_feature_detected!("avx512f") { 2 } else { 1 };
        T.store(v, Relaxed);
    }
    v == 2
}

#[inline(always)]
unsafe fn write_non_ascii_u32(c: u32, idx: usize, dst: *mut u8) -> Result<usize, usize> {
    if c < 0x800 {
        *dst = 0xC0 | (c >> 6) as u8;
        *dst.add(1) = 0x80 | (c & 0x3F) as u8;
        Ok(2)
    } else if c < 0x10000 {
        if (0xD800..0xE000).contains(&c) {
            return Err(idx);
        }
        *dst = 0xE0 | (c >> 12) as u8;
        *dst.add(1) = 0x80 | ((c >> 6) & 0x3F) as u8;
        *dst.add(2) = 0x80 | (c & 0x3F) as u8;
        Ok(3)
    } else {
        *dst = 0xF0 | (c >> 18) as u8;
        *dst.add(1) = 0x80 | ((c >> 12) & 0x3F) as u8;
        *dst.add(2) = 0x80 | ((c >> 6) & 0x3F) as u8;
        *dst.add(3) = 0x80 | (c & 0x3F) as u8;
        Ok(4)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1")]
unsafe fn latin1_to_utf8_avx2(sp: *const u8, n: usize, o: &mut *mut u8, i: &mut usize) {
    use std::arch::x86_64::*;
    let mut op = *o;
    let mut idx = *i;
    while idx + 32 <= n {
        let v = _mm256_loadu_si256(sp.add(idx) as *const __m256i);
        _mm256_storeu_si256(op as *mut __m256i, v);
        let m = _mm256_movemask_epi8(v) as u32;
        if m == 0 {
            op = op.add(32);
            idx += 32;
        } else {
            let tz = m.trailing_zeros() as usize;
            let c = *sp.add(idx + tz);
            *op.add(tz) = 0xC0 | (c >> 6);
            *op.add(tz + 1) = 0x80 | (c & 0x3F);
            op = op.add(tz + 2);
            idx += tz + 1;
        }
    }
    *o = op;
    *i = idx;
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1")]
unsafe fn ucs2_to_utf8_avx2(sp: *const u16, n: usize, o: &mut *mut u8, i: &mut usize) -> Result<(), usize> {
    use std::arch::x86_64::*;
    let k_hi = _mm256_set1_epi16(0xFF80u16 as i16);
    let zero = _mm256_setzero_si256();
    let mut op = *o;
    let mut idx = *i;
    while idx + 32 <= n {
        let v0 = _mm256_loadu_si256(sp.add(idx) as *const __m256i);
        let v1 = _mm256_loadu_si256(sp.add(idx + 16) as *const __m256i);
        let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(v0, v1));
        _mm256_storeu_si256(op as *mut __m256i, packed);
        if _mm256_testz_si256(_mm256_or_si256(v0, v1), k_hi) != 0 {
            op = op.add(32);
            idx += 32;
        } else {
            let c0 = _mm256_cmpeq_epi16(_mm256_and_si256(v0, k_hi), zero);
            let m0 = _mm256_movemask_epi8(c0) as u32;
            let tz = if m0 != u32::MAX {
                ((!m0).trailing_zeros() >> 1) as usize
            } else {
                let c1 = _mm256_cmpeq_epi16(_mm256_and_si256(v1, k_hi), zero);
                let m1 = _mm256_movemask_epi8(c1) as u32;
                16 + ((!m1).trailing_zeros() >> 1) as usize
            };
            let c = *sp.add(idx + tz) as u32;
            if c < 0x800 {
                *op.add(tz) = 0xC0 | (c >> 6) as u8;
                *op.add(tz + 1) = 0x80 | (c & 0x3F) as u8;
                op = op.add(tz + 2);
            } else {
                if (0xD800..0xE000).contains(&c) {
                    return Err(idx + tz);
                }
                *op.add(tz) = 0xE0 | (c >> 12) as u8;
                *op.add(tz + 1) = 0x80 | ((c >> 6) & 0x3F) as u8;
                *op.add(tz + 2) = 0x80 | (c & 0x3F) as u8;
                op = op.add(tz + 3);
            }
            idx += tz + 1;
        }
    }
    *o = op;
    *i = idx;
    Ok(())
}

/// AVX-512F store-and-skip UCS-4 (`u32`) -> UTF-8: packs 16 `u32` (64B) into
/// 16 `u8` via one `VPMOVDB` (`_mm512_cvtepi32_epi8`) + one `VPTESTMD` mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn ucs4_to_utf8_avx512f(sp: *const u32, n: usize, o: &mut *mut u8, i: &mut usize) -> Result<(), usize> {
    use std::arch::x86_64::*;
    let k_hi = _mm512_set1_epi32(!0x7F_u32 as i32);
    let mut op = *o;
    let mut idx = *i;
    while idx + 16 <= n {
        let v = _mm512_loadu_si512(sp.add(idx) as *const __m512i);
        _mm_storeu_si128(op as *mut __m128i, _mm512_cvtepi32_epi8(v));
        let m = _mm512_test_epi32_mask(v, k_hi);
        if m == 0 {
            op = op.add(16);
            idx += 16;
        } else {
            let tz = m.trailing_zeros() as usize;
            let c = *sp.add(idx + tz);
            let w = write_non_ascii_u32(c, idx + tz, op.add(tz))?;
            op = op.add(tz + w);
            idx += tz + 1;
        }
    }
    *o = op;
    *i = idx;
    Ok(())
}

/// AVX2 store-and-skip UCS-4 (`u32`) -> UTF-8: packs 16 `u32` (64B) per step.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1")]
unsafe fn ucs4_to_utf8_avx2(sp: *const u32, n: usize, o: &mut *mut u8, i: &mut usize) -> Result<(), usize> {
    use std::arch::x86_64::*;
    let k_hi = _mm256_set1_epi32(!0x7F_u32 as i32);
    let zero = _mm256_setzero_si256();
    let mut op = *o;
    let mut idx = *i;
    while idx + 16 <= n {
        let v0 = _mm256_loadu_si256(sp.add(idx) as *const __m256i);
        let v1 = _mm256_loadu_si256(sp.add(idx + 8) as *const __m256i);
        let p16 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi32(v0, v1));
        let p8 = _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(p16, p16));
        _mm_storeu_si128(op as *mut __m128i, _mm256_castsi256_si128(p8));
        if _mm256_testz_si256(_mm256_or_si256(v0, v1), k_hi) != 0 {
            op = op.add(16);
            idx += 16;
        } else {
            let c0 = _mm256_cmpeq_epi32(_mm256_and_si256(v0, k_hi), zero);
            let m0 = _mm256_movemask_ps(_mm256_castsi256_ps(c0)) as u32;
            let tz = if m0 != 0xFF {
                ((!m0) & 0xFF).trailing_zeros() as usize
            } else {
                let c1 = _mm256_cmpeq_epi32(_mm256_and_si256(v1, k_hi), zero);
                let m1 = _mm256_movemask_ps(_mm256_castsi256_ps(c1)) as u32;
                8 + ((!m1) & 0xFF).trailing_zeros() as usize
            };
            let c = *sp.add(idx + tz);
            let w = write_non_ascii_u32(c, idx + tz, op.add(tz))?;
            op = op.add(tz + w);
            idx += tz + 1;
        }
    }
    *o = op;
    *i = idx;
    Ok(())
}

/// SIMD-accelerated branch-free store-and-skip Latin-1 (`u8`) -> UTF-8 append
/// into a reusable L1/L2 Vec. Pure-ASCII 32B (AVX2) / 16B (NEON/SSE2) spans
/// copy in one vector store; spans with a non-ASCII byte keep the ASCII prefix
/// already written, emit the 2-byte UTF-8 sequence at `tz`, and resume at `tz + 1`.
#[inline]
pub(crate) fn latin1_to_utf8_append(src: &[u8], dst: &mut Vec<u8>) {
    let n = src.len();
    dst.reserve(n * 2 + 32);
    let base = dst.len();
    let sp = src.as_ptr();
    let mut o = unsafe { dst.as_mut_ptr().add(base) };
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let k80 = vdupq_n_u8(0x80);
        while i + 16 <= n {
            let v = vld1q_u8(sp.add(i));
            vst1q_u8(o, v);
            let msb = vandq_u8(v, k80);
            let lo = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(msb));
            let hi = vgetq_lane_u64::<1>(vreinterpretq_u64_u8(msb));
            if (lo | hi) == 0 {
                o = o.add(16);
                i += 16;
            } else {
                let tz = if lo != 0 {
                    (lo.trailing_zeros() >> 3) as usize
                } else {
                    8 + (hi.trailing_zeros() >> 3) as usize
                };
                let c = *sp.add(i + tz);
                *o.add(tz) = 0xC0 | (c >> 6);
                *o.add(tz + 1) = 0x80 | (c & 0x3F);
                o = o.add(tz + 2);
                i += tz + 1;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if n >= 32 && has_avx2_bmi1() {
            latin1_to_utf8_avx2(sp, n, &mut o, &mut i);
        }
        use std::arch::x86_64::*;
        while i + 16 <= n {
            let v = _mm_loadu_si128(sp.add(i) as *const __m128i);
            _mm_storeu_si128(o as *mut __m128i, v);
            let m = _mm_movemask_epi8(v) as u32;
            if m == 0 {
                o = o.add(16);
                i += 16;
            } else {
                let tz = m.trailing_zeros() as usize;
                let c = *sp.add(i + tz);
                *o.add(tz) = 0xC0 | (c >> 6);
                *o.add(tz + 1) = 0x80 | (c & 0x3F);
                o = o.add(tz + 2);
                i += tz + 1;
            }
        }
    }
    unsafe {
        while i < n {
            let c = *sp.add(i);
            if c < 0x80 {
                *o = c;
                o = o.add(1);
            } else {
                *o = 0xC0 | (c >> 6);
                *o.add(1) = 0x80 | (c & 0x3F);
                o = o.add(2);
            }
            i += 1;
        }
        let written = o.offset_from(dst.as_ptr().add(base)) as usize;
        dst.set_len(base + written);
    }
}

/// SIMD-accelerated store-and-skip UCS-2 (`u16`) -> UTF-8 append into a
/// reusable L1/L2 Vec. Packs 32 (AVX2) / 16 (NEON/SSE2) code units per step;
/// rejects lone surrogates (`0xD800..0xE000`) with `Err(code_unit_index)`.
#[inline]
pub(crate) fn ucs2_to_utf8_append(src: &[u16], dst: &mut Vec<u8>) -> Result<(), usize> {
    let n = src.len();
    dst.reserve(n * 3 + 32);
    let base = dst.len();
    let sp = src.as_ptr();
    let mut o = unsafe { dst.as_mut_ptr().add(base) };
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let k_hi = vdupq_n_u16(0xFF80);
        while i + 16 <= n {
            let v0 = vld1q_u16(sp.add(i));
            let v1 = vld1q_u16(sp.add(i + 8));
            let packed = vcombine_u8(vmovn_u16(v0), vmovn_u16(v1));
            vst1q_u8(o, packed);
            let m0 = vtstq_u16(v0, k_hi);
            let m1 = vtstq_u16(v1, k_hi);
            let mm = vcombine_u8(vmovn_u16(m0), vmovn_u16(m1));
            let lo = vgetq_lane_u64::<0>(vreinterpretq_u64_u8(mm));
            let hi = vgetq_lane_u64::<1>(vreinterpretq_u64_u8(mm));
            if (lo | hi) == 0 {
                o = o.add(16);
                i += 16;
            } else {
                let tz = if lo != 0 {
                    (lo.trailing_zeros() >> 3) as usize
                } else {
                    8 + (hi.trailing_zeros() >> 3) as usize
                };
                let c = *sp.add(i + tz) as u32;
                if c < 0x800 {
                    *o.add(tz) = 0xC0 | (c >> 6) as u8;
                    *o.add(tz + 1) = 0x80 | (c & 0x3F) as u8;
                    o = o.add(tz + 2);
                } else {
                    if (0xD800..0xE000).contains(&c) {
                        return Err(i + tz);
                    }
                    *o.add(tz) = 0xE0 | (c >> 12) as u8;
                    *o.add(tz + 1) = 0x80 | ((c >> 6) & 0x3F) as u8;
                    *o.add(tz + 2) = 0x80 | (c & 0x3F) as u8;
                    o = o.add(tz + 3);
                }
                i += tz + 1;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if n >= 32 && has_avx2_bmi1() {
            ucs2_to_utf8_avx2(sp, n, &mut o, &mut i)?;
        }
        use std::arch::x86_64::*;
        let k_hi = _mm_set1_epi16(0xFF80u16 as i16);
        let zero = _mm_setzero_si128();
        while i + 16 <= n {
            let v0 = _mm_loadu_si128(sp.add(i) as *const __m128i);
            let v1 = _mm_loadu_si128(sp.add(i + 8) as *const __m128i);
            _mm_storeu_si128(o as *mut __m128i, _mm_packus_epi16(v0, v1));
            let m0 = _mm_movemask_epi8(_mm_cmpeq_epi16(_mm_and_si128(v0, k_hi), zero)) as u32;
            let m1 = _mm_movemask_epi8(_mm_cmpeq_epi16(_mm_and_si128(v1, k_hi), zero)) as u32;
            let bad = !(m0 | (m1 << 16));
            if bad == 0 {
                o = o.add(16);
                i += 16;
            } else {
                let tz = (bad.trailing_zeros() >> 1) as usize;
                let c = *sp.add(i + tz) as u32;
                if c < 0x800 {
                    *o.add(tz) = 0xC0 | (c >> 6) as u8;
                    *o.add(tz + 1) = 0x80 | (c & 0x3F) as u8;
                    o = o.add(tz + 2);
                } else {
                    if (0xD800..0xE000).contains(&c) {
                        return Err(i + tz);
                    }
                    *o.add(tz) = 0xE0 | (c >> 12) as u8;
                    *o.add(tz + 1) = 0x80 | ((c >> 6) & 0x3F) as u8;
                    *o.add(tz + 2) = 0x80 | (c & 0x3F) as u8;
                    o = o.add(tz + 3);
                }
                i += tz + 1;
            }
        }
    }
    unsafe {
        while i < n {
            let c = *sp.add(i) as u32;
            if c < 0x80 {
                *o = c as u8;
                o = o.add(1);
            } else if c < 0x800 {
                *o = 0xC0 | (c >> 6) as u8;
                *o.add(1) = 0x80 | (c & 0x3F) as u8;
                o = o.add(2);
            } else {
                if (0xD800..0xE000).contains(&c) {
                    return Err(i);
                }
                *o = 0xE0 | (c >> 12) as u8;
                *o.add(1) = 0x80 | ((c >> 6) & 0x3F) as u8;
                *o.add(2) = 0x80 | (c & 0x3F) as u8;
                o = o.add(3);
            }
            i += 1;
        }
        let written = o.offset_from(dst.as_ptr().add(base)) as usize;
        dst.set_len(base + written);
    }
    Ok(())
}

/// SIMD-accelerated store-and-skip UCS-4 (`u32`) -> UTF-8 writer into `dst`.
/// Packs 16 code units (64 input bytes) per step on AVX-512F (`VPMOVDB`), AVX2,
/// and ARM64 NEON (`vmovn_u32` + `vmovn_u16`).
/// SAFETY: `dst` must be valid for `utf8_len(src)` writes.
#[inline]
pub(crate) unsafe fn ucs4_to_utf8_ptr(src: &[u32], dst: *mut u8) -> Result<usize, usize> {
    let n = src.len();
    let sp = src.as_ptr();
    let mut o = dst;
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        let k_hi = vdupq_n_u32(!0x7F_u32);
        while i + 16 <= n {
            let v0 = vld1q_u32(sp.add(i));
            let v1 = vld1q_u32(sp.add(i + 4));
            let v2 = vld1q_u32(sp.add(i + 8));
            let v3 = vld1q_u32(sp.add(i + 12));
            let u01 = vcombine_u16(vmovn_u32(v0), vmovn_u32(v1));
            let u23 = vcombine_u16(vmovn_u32(v2), vmovn_u32(v3));
            vst1q_u8(o, vcombine_u8(vmovn_u16(u01), vmovn_u16(u23)));
            let or_all = vorrq_u32(vorrq_u32(v0, v1), vorrq_u32(v2, v3));
            if vmaxvq_u32(vandq_u32(or_all, k_hi)) == 0 {
                o = o.add(16);
                i += 16;
            } else {
                let mut tz = 0usize;
                while tz < 16 && *sp.add(i + tz) < 0x80 {
                    tz += 1;
                }
                let c = *sp.add(i + tz);
                let w = write_non_ascii_u32(c, i + tz, o.add(tz))?;
                o = o.add(tz + w);
                i += tz + 1;
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if n >= 16 {
            if has_avx512f() {
                ucs4_to_utf8_avx512f(sp, n, &mut o, &mut i)?;
            } else if has_avx2_bmi1() {
                ucs4_to_utf8_avx2(sp, n, &mut o, &mut i)?;
            }
        }
    }
    while i < n {
        let c = *sp.add(i);
        if c < 0x80 {
            *o = c as u8;
            o = o.add(1);
        } else {
            let w = write_non_ascii_u32(c, i, o)?;
            o = o.add(w);
        }
        i += 1;
    }
    Ok(o.offset_from(dst) as usize)
}

/// SIMD-accelerated store-and-skip UCS-4 (`u32`) -> UTF-8 append into a Vec.
#[inline]
pub(crate) fn ucs4_to_utf8_append(src: &[u32], dst: &mut Vec<u8>) -> Result<(), usize> {
    let n = src.len();
    if n == 0 {
        return Ok(());
    }
    dst.reserve(n * 4 + 32);
    let base = dst.len();
    unsafe {
        let w = ucs4_to_utf8_ptr(src, dst.as_mut_ptr().add(base))?;
        dst.set_len(base + w);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucs4_simd_matches_scalar_all_cases() {
        // Pure ASCII across every alignment/tail length 0..65
        for len in 0..=65 {
            let src: Vec<u32> = (0..len).map(|i| 0x20 + (i as u32 % 95)).collect();
            let mut got = Vec::new();
            ucs4_to_utf8_append(&src, &mut got).unwrap();
            let mut exp = vec![0u8; utf8_len(&src)];
            let w = unsafe { units_to_utf8(&src, exp.as_mut_ptr()) }.unwrap();
            assert_eq!(w, exp.len());
            assert_eq!(got, exp, "ascii len={len}");
            // Also verify exact-sized ucs4_to_utf8_ptr (no spare padding)
            let mut exact = vec![0u8; exp.len()];
            let w2 = unsafe { ucs4_to_utf8_ptr(&src, exact.as_mut_ptr()) }.unwrap();
            assert_eq!(w2, exp.len());
            assert_eq!(exact, exp, "exact ptr len={len}");
        }
        // Mixed German + CJK + 4-byte emoji across SIMD lanes
        let sample = "The quick brown fox jumps over Über-straße — café 日本語 🚀🎉! ".repeat(17);
        let src: Vec<u32> = sample.chars().map(|c| c as u32).collect();
        let mut got = Vec::new();
        ucs4_to_utf8_append(&src, &mut got).unwrap();
        assert_eq!(got, sample.as_bytes());
        let mut exact = vec![0u8; utf8_len(&src)];
        let w = unsafe { ucs4_to_utf8_ptr(&src, exact.as_mut_ptr()) }.unwrap();
        assert_eq!(&exact[..w], sample.as_bytes());
        // Surrogate rejection at every lane position 0..33
        for pos in 0..=33 {
            let mut bad = vec![b'a' as u32; 40];
            bad[pos] = 0xD800 + (pos as u32 % 0x800);
            let mut dst = Vec::new();
            assert_eq!(ucs4_to_utf8_append(&bad, &mut dst), Err(pos), "surrogate at {pos}");
            let mut exact_buf = vec![0u8; utf8_len(&bad)];
            assert_eq!(unsafe { ucs4_to_utf8_ptr(&bad, exact_buf.as_mut_ptr()) }, Err(pos));
        }
    }
}
