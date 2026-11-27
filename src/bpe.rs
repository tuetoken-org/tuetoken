//! BPE merge: open-addressing hash map keyed (t1<<32|t2), with the same
//! scan-all-pairs / apply-all-lowest-rank-in-one-pass loop as the C engine.

/// One hash-map slot for rare >2M-vocab tokenizers that don't fit the packed u64 layout.
#[derive(Clone, Copy)]
struct Entry {
    key: u64, // 0 = empty slot
    rank: u32,
}

const PACK_RANK_BITS: u32 = 22;
const PACK_RANK_MASK: u32 = (1u32 << PACK_RANK_BITS) - 1;
const PACK_KEY_MASK: u64 = !(PACK_RANK_MASK as u64);
const PACK_MAX_TOK: usize = 1 << 21;
const PACK_MAX_RANK: usize = (1 << PACK_RANK_BITS) - 2; // +1 stored in 1..=PACK_RANK_MASK
const DENSE_NONE: u16 = u16::MAX;
const DENSE_OVERFLOW: u16 = u16::MAX - 1;
/// Direct-indexed 2D merge-rank matrix for `(t1 | t2) < DENSE_N`:
/// `512 × 512 × u16 = 512 KiB` (fits in L2 on Zen 4 & Apple M4) resolves
/// 54-66% of all post-`can_lr` BPE probes and 69-88% of all merge hits in a
/// single `movzwl` load with zero hashing, zero Bloom, and zero SIMD probing!
const DENSE_BITS: u32 = 9;
const DENSE_N: u32 = 1 << DENSE_BITS;

/// One 64-byte cache-line bucket holding 8 packed `(t1 << 43) | (t2 << 22) | (rank + 1)`
/// entries filled left-to-right in ascending rank order. Probed 8-wide in parallel
/// via AVX-512F / AVX2 / NEON with a single branch for all 8 slots and exact
/// overflow termination via `slot[7] == 0`.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct MergeBucket([u64; 8]);
const _: () = assert!(std::mem::size_of::<MergeBucket>() == 64);

pub struct MergeMap {
    /// 64B cache-line SIMD-bucketed table partitioned disjointly by
    /// `(t1 | t2) < hot_t`: hot common-token merges pack into `[0..cold_off)`
    /// (~256–512 KiB, L2-resident, capturing 85–97% of hits with an L1-resident
    /// Bloom prefix), while cold long-tail merges live in `[cold_off..)`.
    /// Each partition ends with an empty sentinel bucket so probing never wraps.
    buckets: crate::chunk_table::HugeVec<MergeBucket>,
    hot_t: u32,
    hot_shift: u32,
    cold_shift: u32,
    cold_off: usize,
    /// Fallback 16-byte table (empty for all normal vocabularies < 2M tokens).
    entries: Vec<Entry>,
    mask: usize,
    shift: u32,
    /// Dense 16-bit rank table for byte-token pairs (t1 < 256 && t2 < 256):
    /// 128 KiB (4x smaller than baseline 512 KiB tuple table — fits M4 L1d!).
    /// `DENSE_NONE` = no merge; `DENSE_OVERFLOW` = rare rank >= 65534 (probe table).
    dense: Vec<u16>,
    /// True iff any byte-byte merge had rank >= DENSE_OVERFLOW (false for gpt2).
    has_dense_overflow: bool,
    /// Raw-byte indexed initial merge-rank table `dense_byte[(b0 << 8) | b1]`
    /// (128 KiB, populated for ByteLevel models): breaks the `singleton[b] -> dense`
    /// serial dependency chain and works even when byte token IDs >= 256.
    dense_byte: Vec<u16>,
    has_dense_byte_overflow: bool,
    /// Precomputed 2-bit `can_lr` flags per raw byte (256 B in L1).
    byte_lr: [u8; 256],
    /// 2-bit-per-token L1 terminal bitset (50 KiB for o200k, 12.5 KiB for gpt2):
    /// bit 0 = token ever appears as t1; bit 1 = token ever appears as t2.
    /// In real BPE vocabularies >82% of tokens never appear on the left and >85%
    /// never appear on the right, filtering most negative MergeMap probes in L1.
    can_lr: Vec<u8>,
    /// Pure-GPR k=3 Blocked Bloom filter 1-to-1 indexed with `buckets` via
    /// `bucket_of(t1, t2, h)` (hot prefix is only 32–64 KiB, fitting L1d!).
    bloom: Vec<u64>,
    /// Fast rank -> target resolution: if every merge satisfies `target == rank + off`
    /// (true for 100% of gpt2 / cl100k / o200k), `linear_off != u32::MAX` and
    /// `target(rank)` is one ADD with zero memory loads; otherwise `targets[rank]`.
    linear_off: u32,
    /// True iff every merge target has token ID >= 256 (true for all standard
    /// ByteLevel BPEs), letting the inner merge loop skip the `merged >= 256` check.
    all_targets_ge_256: bool,
    /// Number of token IDs covered by `can_lr` (`can_lr.len() * 4`).
    vocab_n: usize,
    targets: Vec<u32>,
}

impl MergeMap {
    pub fn new(merges: &[(u32, u32, u32)]) -> Self {
        let max_tok = merges
            .iter()
            .fold(255usize, |acc, &(t1, t2, tg)| acc.max(t1 as usize).max(t2 as usize).max(tg as usize));
        let vocab_n = max_tok + 1;
        let use_packed = vocab_n <= PACK_MAX_TOK && merges.len() <= PACK_MAX_RANK;

        // Count merges that actually need the hash table (pairs with
        // `(t1 | t2) < DENSE_N` and `rank < DENSE_OVERFLOW` live solely in `dense`).
        let n_table = if use_packed {
            merges
                .iter()
                .enumerate()
                .filter(|&(i, &(t1, t2, _))| (t1 | t2) >= DENSE_N || i >= DENSE_OVERFLOW as usize)
                .count()
        } else {
            merges.len()
        };
        let mut cap = 16usize;
        while cap.saturating_mul(4) < n_table.saturating_mul(5).max(1) {
            cap <<= 1;
        }
        let bits = cap.trailing_zeros();

        // Detect linear `target == rank + off` mapping (100% of OpenAI BPEs).
        let linear_off = if let Some(&(_, _, tg0)) = merges.first() {
            if merges.iter().enumerate().all(|(i, &(_, _, tg))| tg == (i as u32).wrapping_add(tg0)) {
                tg0
            } else {
                u32::MAX
            }
        } else {
            u32::MAX
        };
        let all_targets_ge_256 = merges.iter().all(|&(_, _, tg)| tg >= 256);
        let targets = if linear_off == u32::MAX {
            merges.iter().map(|&(_, _, tg)| tg).collect()
        } else {
            Vec::new()
        };

        let mut can_lr = vec![0u8; (vocab_n + 3) >> 2];
        let mut dense = vec![DENSE_NONE; 1usize << (DENSE_BITS * 2)];
        let mut has_dense_overflow = false;
        let total_nb = (cap >> 3).max(16);
        let mask = cap - 1;
        let shift = 64 - bits;

        // Disjoint `(t1 | t2) < hot_t` cache stratification: pack the ~20–45%
        // of merges involving low-ID (high-frequency) tokens into a dense
        // 256–512 KiB hot prefix (`<= 80%` load) so remaining hot BPE hits and
        // their 32–64 KiB Bloom words stay resident in L1/L2 without pollution
        // from long-tail merges. Zero extra lookups: `(t1 | t2) < hot_t` is an
        // exact partition of the lookup key space.
        let (hot_t, hot_nb, cold_nb) = if use_packed && total_nb >= 8192 {
            let hnb = (total_nb >> 2).max(4096).min(total_nb >> 1);
            let hmax_entries = (hnb * 8 * 4) / 5; // <= 80% load factor
            let mut best_t = 0u32;
            let mut best_nhot = 0usize;
            let mut t = DENSE_N << 1;
            while (t as usize) < vocab_n {
                let nh = merges
                    .iter()
                    .enumerate()
                    .filter(|&(i, &(t1, t2, _))| {
                        ((t1 | t2) >= DENSE_N || i >= DENSE_OVERFLOW as usize) && (t1 | t2) < t
                    })
                    .count();
                if nh <= hmax_entries {
                    best_t = t;
                    best_nhot = nh;
                    t <<= 1;
                } else {
                    break;
                }
            }
            let ncold = n_table.saturating_sub(best_nhot);
            if best_t != 0 && best_nhot != 0 && ncold != 0 {
                let mut ccap = 128usize;
                while ccap.saturating_mul(4) < ncold.saturating_mul(5).max(1) {
                    ccap <<= 1;
                }
                (best_t, hnb, (ccap >> 3).max(16))
            } else {
                (u32::MAX, total_nb, 0)
            }
        } else {
            (u32::MAX, total_nb, 0)
        };
        let hot_shift = 64 - hot_nb.trailing_zeros();
        let cold_shift = if cold_nb != 0 { 64 - cold_nb.trailing_zeros() } else { hot_shift };

        let mut hot_vec = if use_packed {
            vec![MergeBucket([0u64; 8]); hot_nb + 1]
        } else {
            Vec::new()
        };
        let mut cold_vec = if use_packed && cold_nb != 0 {
            vec![MergeBucket([0u64; 8]); cold_nb + 1]
        } else {
            Vec::new()
        };
        let mut hot_bloom = if use_packed { vec![0u64; hot_nb] } else { vec![0u64; total_nb] };
        let mut cold_bloom = if use_packed && cold_nb != 0 { vec![0u64; cold_nb] } else { Vec::new() };
        let mut entries = if use_packed {
            Vec::new()
        } else {
            vec![Entry { key: 0, rank: 0 }; cap]
        };

        let insert_no_wrap = |vec: &mut Vec<MergeBucket>, mut b: usize, kh: u64, r: u32| {
            'ins: loop {
                // Keep the final bucket strictly all-zero as an exact probe sentinel.
                if b + 1 >= vec.len() {
                    vec.push(MergeBucket([0u64; 8]));
                }
                let buck = &mut vec[b].0;
                for slot in buck.iter_mut() {
                    let e = *slot;
                    if e == 0 {
                        *slot = kh | (r as u64 + 1);
                        break 'ins;
                    }
                    if (e & PACK_KEY_MASK) == kh {
                        break 'ins;
                    }
                }
                b += 1;
            }
        };

        for (i, &(t1, t2, _tg)) in merges.iter().enumerate() {
            let r = i as u32;
            // Record left/right participation in the 2-bit-per-token L1 filter.
            can_lr[(t1 as usize) >> 2] |= 1u8 << (((t1 as usize) & 3) * 2);
            can_lr[(t2 as usize) >> 2] |= 2u8 << (((t2 as usize) & 3) * 2);

            if (t1 | t2) < DENSE_N {
                let d = &mut dense[((t1 as usize) << DENSE_BITS) | (t2 as usize)];
                if *d == DENSE_NONE {
                    if r < DENSE_OVERFLOW as u32 {
                        *d = r as u16;
                        if use_packed {
                            continue;
                        }
                    } else {
                        *d = DENSE_OVERFLOW;
                        has_dense_overflow = true;
                    }
                } else if use_packed {
                    // Already recorded a lower-rank duplicate for this pair.
                    continue;
                }
            }
            if use_packed {
                let kh = Self::pack_key(t1, t2);
                let h = Self::hash_packed(kh);
                if (t1 | t2) < hot_t {
                    let b = (h >> hot_shift) as usize;
                    hot_bloom[b] |= Self::bloom_mask(h);
                    insert_no_wrap(&mut hot_vec, b, kh, r);
                } else {
                    let b = (h >> cold_shift) as usize;
                    cold_bloom[b] |= Self::bloom_mask(h);
                    insert_no_wrap(&mut cold_vec, b, kh, r);
                }
            } else {
                let key = Self::key(t1, t2);
                let h = Self::hash_key(key);
                hot_bloom[(h >> hot_shift) as usize] |= Self::bloom_mask(h);
                let mut idx = (h >> shift) as usize;
                while entries[idx].key != 0 {
                    if entries[idx].key == key {
                        break;
                    }
                    idx = (idx + 1) & mask;
                }
                if entries[idx].key == 0 {
                    entries[idx] = Entry { key, rank: r };
                }
            }
        }
        let cold_off = if use_packed && cold_nb != 0 { hot_vec.len() } else { 0 };
        let buckets = if use_packed {
            let mut bv = crate::chunk_table::HugeVec::<MergeBucket>::zeroed(hot_vec.len() + cold_vec.len());
            bv[..hot_vec.len()].copy_from_slice(&hot_vec);
            if !cold_vec.is_empty() {
                bv[cold_off..].copy_from_slice(&cold_vec);
            }
            bv
        } else {
            crate::chunk_table::HugeVec::empty()
        };
        let bloom = if use_packed && cold_nb != 0 {
            let mut bl = vec![0u64; cold_off + cold_nb];
            bl[..hot_nb].copy_from_slice(&hot_bloom);
            bl[cold_off..cold_off + cold_nb].copy_from_slice(&cold_bloom);
            bl
        } else {
            hot_bloom
        };
        MergeMap {
            buckets,
            hot_t,
            hot_shift,
            cold_shift,
            cold_off,
            entries,
            mask,
            shift,
            dense,
            has_dense_overflow,
            dense_byte: Vec::new(),
            has_dense_byte_overflow: false,
            byte_lr: [0u8; 256],
            can_lr,
            bloom,
            linear_off,
            all_targets_ge_256,
            vocab_n,
            targets,
        }
    }

    /// Precompute direct raw-byte pair ranks (`dense_byte`) and per-byte `can_lr`
    /// flags for ByteLevel models so cold BPE reads straight from `&[u8]`.
    pub fn init_byte_tables(&mut self, singleton: &[i32; 256]) -> bool {
        if self.buckets.is_empty() || singleton.iter().any(|&t| t < 0 || (t as usize) >= self.vocab_n) {
            return false;
        }
        let mut db = vec![DENSE_NONE; 1 << 16];
        let mut ov = false;
        for b0 in 0..256usize {
            let t0 = singleton[b0] as u32;
            self.byte_lr[b0] = self.lr_bits(t0);
            for b1 in 0..256usize {
                let t1 = singleton[b1] as u32;
                let r = self.lookup_rank(t0, t1);
                if r != u32::MAX {
                    if r < DENSE_OVERFLOW as u32 {
                        db[(b0 << 8) | b1] = r as u16;
                    } else {
                        db[(b0 << 8) | b1] = DENSE_OVERFLOW;
                        ov = true;
                    }
                }
            }
        }
        self.dense_byte = db;
        self.has_dense_byte_overflow = ov;
        true
    }

    #[inline(always)]
    fn pack_key(t1: u32, t2: u32) -> u64 {
        ((t1 as u64) << 43) | ((t2 as u64) << PACK_RANK_BITS)
    }

    /// Full 64-bit hash directly from packed `kh` (zero `key(t1, t2)` branch!).
    #[inline(always)]
    fn hash_packed(kh: u64) -> u64 {
        (kh ^ (kh >> 21)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Pure-GPR k=3 Blocked Bloom mask from high-entropy avalanche bits 26..43
    /// (disjoint from `b = h >> bshift`). Rejects ~99.1% of negative probes in
    /// one L2 load while keeping 3 scalar shifts out of LLVM's 4-lane AVX2 SLP
    /// vectorizer (zero GPR<->XMM crossings, zero Bloom load sinking).
    #[inline(always)]
    fn bloom_mask(h: u64) -> u64 {
        let m0 = 1u64.wrapping_shl((h >> 26) as u32);
        let m1 = 1u64.wrapping_shl((h >> 32) as u32);
        let m2 = 1u64.wrapping_shl((h >> 38) as u32);
        m0 | m1 | m2
    }

    /// (0,0) packs to 0 (the empty sentinel); remap it to u64::MAX so it does
    /// not alias the common pair (0,1) — the same fix as the C version.
    #[inline(always)]
    fn key(t1: u32, t2: u32) -> u64 {
        let k = ((t1 as u64) << 32) | (t2 as u64);
        if k == 0 { u64::MAX } else { k }
    }

    /// Full 64-bit hash for an unpacked pair key (fallback >2M vocab).
    #[inline(always)]
    fn hash_key(key: u64) -> u64 {
        (key ^ (key >> 29)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
    }

    /// Branchless home-bucket address in the disjoint hot/cold layout.
    #[inline(always)]
    fn bucket_of(&self, t1: u32, t2: u32, h: u64) -> usize {
        if (t1 | t2) < self.hot_t {
            (h >> self.hot_shift) as usize
        } else {
            self.cold_off + ((h >> self.cold_shift) as usize)
        }
    }

    /// Direct-indexed O(1) lookup in the `DENSE_N × DENSE_N` L2 merge matrix
    /// for `(t1 | t2) < DENSE_N` (zero hashing, zero Bloom, zero SIMD probing).
    ///
    /// SAFETY: caller guarantees `t1 < DENSE_N && t2 < DENSE_N`.
    #[inline(always)]
    unsafe fn lookup_dense(&self, t1: u32, t2: u32) -> u32 {
        let d = *self.dense.get_unchecked(((t1 as usize) << DENSE_BITS) | (t2 as usize));
        if d < DENSE_OVERFLOW {
            d as u32
        } else if d == DENSE_NONE {
            u32::MAX
        } else {
            self.probe_dense_overflow(t1, t2)
        }
    }

    /// Pure-GPR k=3 Bloom hit test via right-shifts (40% fewer ops than `1 << s`
    /// and prevents LLVM SLP-vectorization into XMM on dual left+right probes).
    #[inline(always)]
    fn bloom_hit(w: u64, h: u64) -> bool {
        let b0 = w.wrapping_shr((h >> 26) as u32);
        let b1 = w.wrapping_shr((h >> 32) as u32);
        let b2 = w.wrapping_shr((h >> 38) as u32);
        ((b0 & b1 & b2) & 1) != 0
    }

    #[inline(always)]
    fn bloom_may_contain_b(&self, h: u64, b: usize) -> bool {
        let w = unsafe { *self.bloom.get_unchecked(b) };
        Self::bloom_hit(w, h)
    }

    #[inline(always)]
    fn bloom_may_contain(&self, h: u64) -> bool {
        self.bloom_may_contain_b(h, (h >> self.hot_shift) as usize)
    }

    #[inline(always)]
    fn lr_bits(&self, t: u32) -> u8 {
        let b = (t as usize) >> 2;
        if b >= self.can_lr.len() {
            return 0;
        }
        // SAFETY: bounds checked just above.
        (unsafe { *self.can_lr.get_unchecked(b) } >> (((t as usize) & 3) * 2)) & 3
    }

    /// Bounds-check-free `lr_bits` for tokens guaranteed `< self.vocab_n`
    /// (all merge targets and validated ByteLevel singletons).
    #[inline(always)]
    unsafe fn lr_bits_unchecked(&self, t: u32) -> u8 {
        let b = (t as usize) >> 2;
        (*self.can_lr.get_unchecked(b) >> (((t as usize) & 3) * 2)) & 3
    }

    #[inline(always)]
    fn can_merge(&self, t1: u32, t2: u32) -> bool {
        let b1 = (t1 as usize) >> 2;
        let b2 = (t2 as usize) >> 2;
        if b1 >= self.can_lr.len() || b2 >= self.can_lr.len() {
            return false;
        }
        // SAFETY: bounds checked just above.
        let f1 = unsafe { *self.can_lr.get_unchecked(b1) } >> (((t1 as usize) & 3) * 2);
        let f2 = unsafe { *self.can_lr.get_unchecked(b2) } >> (((t2 as usize) & 3) * 2);
        (f1 & 1) & (f2 >> 1) != 0
    }

    /// Resolve a valid merge `rank` to its target token ID (no hash lookup).
    #[inline(always)]
    pub fn target(&self, rank: u32) -> u32 {
        if self.linear_off != u32::MAX {
            rank.wrapping_add(self.linear_off)
        } else {
            // SAFETY: rank came from a valid entry in this MergeMap.
            unsafe { *self.targets.get_unchecked(rank as usize) }
        }
    }

    /// Portable scalar 64B-bucket probe (zero-wrap via partition sentinel).
    #[cfg(not(target_arch = "aarch64"))]
    #[inline(always)]
    fn probe_bucket_scalar(&self, kh: u64, b: usize) -> u32 {
        let mut bp = unsafe { self.buckets.as_ptr().add(b) };
        loop {
            let buck = unsafe { &(*bp).0 };
            for &e in buck {
                let diff = e ^ kh;
                if diff.wrapping_sub(1) < (1u64 << PACK_RANK_BITS) - 1 {
                    return (diff as u32) - 1;
                }
                if e == 0 {
                    return u32::MAX;
                }
            }
            bp = unsafe { bp.add(1) };
        }
    }

    /// ARM64 NEON 8-wide branchless 64B-bucket probe: loads one cache line
    /// (`2 x ldp q`), compares all 8 keys in parallel, and OR-reduces the
    /// matching slot's value directly in registers (zero wrap, zero `tzcnt`,
    /// single branch for all 8 slots).
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn probe_bucket_neon(&self, kh: u64, b: usize) -> u32 {
        use std::arch::aarch64::*;
        let mut bp = self.buckets.as_ptr().add(b);
        let km = vdupq_n_u64(PACK_KEY_MASK);
        let tgt = vdupq_n_u64(kh);
        loop {
            let p = (*bp).0.as_ptr();
            let v0 = vld1q_u64(p);
            let v1 = vld1q_u64(p.add(2));
            let v2 = vld1q_u64(p.add(4));
            let v3 = vld1q_u64(p.add(6));
            let c0 = vceqq_u64(vandq_u64(v0, km), tgt);
            let c1 = vceqq_u64(vandq_u64(v1, km), tgt);
            let c2 = vceqq_u64(vandq_u64(v2, km), tgt);
            let c3 = vceqq_u64(vandq_u64(v3, km), tgt);
            let h01 = vorrq_u64(vandq_u64(c0, v0), vandq_u64(c1, v1));
            let h23 = vorrq_u64(vandq_u64(c2, v2), vandq_u64(c3, v3));
            let hit = vorrq_u64(h01, h23);
            let val = vgetq_lane_u64::<0>(hit) | vgetq_lane_u64::<1>(hit);
            if val != 0 {
                return ((val as u32) & PACK_RANK_MASK) - 1;
            }
            if (*bp).0[7] == 0 {
                return u32::MAX;
            }
            bp = bp.add(1);
        }
    }

    /// x86_64 AVX-512F 8-wide branchless 64B-bucket probe: 1 aligned 64B load
    /// + `vpandq` + `vpcmpeqq` + `vpcompressq` extracts the matching rank in
    /// registers with 1 branch across all 8 slots and zero wrap masking.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,bmi1")]
    #[inline]
    unsafe fn probe_bucket_avx512(&self, kh: u64, b: usize) -> u32 {
        use std::arch::x86_64::*;
        let mut bp = self.buckets.as_ptr().add(b);
        let km = _mm512_set1_epi64(PACK_KEY_MASK as i64);
        let tgt = _mm512_set1_epi64(kh as i64);
        loop {
            let v = _mm512_load_si512(bp as *const _);
            let m = _mm512_cmpeq_epi64_mask(_mm512_and_si512(v, km), tgt);
            if m != 0 {
                let hit = _mm512_maskz_compress_epi64(m, v);
                let lo = _mm512_cvtsi512_si32(hit) as u32;
                return (lo & PACK_RANK_MASK) - 1;
            }
            if (*bp).0[7] == 0 {
                return u32::MAX;
            }
            bp = bp.add(1);
        }
    }

    /// x86_64 AVX2 8-wide branchless 64B-bucket probe: 2 aligned 32B loads +
    /// register OR-reduce with 1 branch across all 8 slots and zero wrap masking.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1")]
    #[inline]
    unsafe fn probe_bucket_avx2(&self, kh: u64, b: usize) -> u32 {
        use std::arch::x86_64::*;
        let mut bp = self.buckets.as_ptr().add(b);
        let km = _mm256_set1_epi64x(PACK_KEY_MASK as i64);
        let tgt = _mm256_set1_epi64x(kh as i64);
        loop {
            let p = bp as *const __m256i;
            let v0 = _mm256_load_si256(p);
            let v1 = _mm256_load_si256(p.add(1));
            let c0 = _mm256_cmpeq_epi64(_mm256_and_si256(v0, km), tgt);
            let c1 = _mm256_cmpeq_epi64(_mm256_and_si256(v1, km), tgt);
            let hit = _mm256_or_si256(_mm256_and_si256(c0, v0), _mm256_and_si256(c1, v1));
            let lo128 = _mm256_castsi256_si128(hit);
            let hi128 = _mm256_extracti128_si256::<1>(hit);
            let r128 = _mm_or_si128(lo128, hi128);
            let val = (_mm_cvtsi128_si64(r128) | _mm_extract_epi64::<1>(r128)) as u64;
            if val != 0 {
                return ((val as u32) & PACK_RANK_MASK) - 1;
            }
            if (*bp).0[7] == 0 {
                return u32::MAX;
            }
            bp = bp.add(1);
        }
    }

    #[inline(always)]
    fn probe_bucket_best(&self, kh: u64, b: usize) -> u32 {
        #[cfg(target_arch = "aarch64")]
        return unsafe { self.probe_bucket_neon(kh, b) };
        #[cfg(target_arch = "x86_64")]
        {
            match bpe_x86_tier() {
                3 => return unsafe { self.probe_bucket_avx512(kh, b) },
                2 => return unsafe { self.probe_bucket_avx2(kh, b) },
                _ => return self.probe_bucket_scalar(kh, b),
            }
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        self.probe_bucket_scalar(kh, b)
    }

    #[inline(always)]
    fn probe_table(&self, t1: u32, t2: u32, idx: usize) -> u32 {
        if !self.buckets.is_empty() {
            return self.probe_bucket_best(Self::pack_key(t1, t2), idx);
        }
        let mut idx = idx;
        let key = Self::key(t1, t2);
        loop {
            let e = unsafe { *self.entries.get_unchecked(idx) };
            if e.key == key {
                return e.rank;
            }
            if e.key == 0 {
                return u32::MAX;
            }
            idx = (idx + 1) & self.mask;
        }
    }

    #[inline(always)]
    fn prefetch_bucket(&self, b: usize) {
        if !self.buckets.is_empty() {
            // SAFETY: b < buckets.len() (computed via hash_packed >> bshift).
            let p = unsafe { self.buckets.as_ptr().add(b) as *const u8 };
            crate::chunk_table::prefetch(p);
        }
    }

    #[cold]
    #[inline(never)]
    fn probe_dense_overflow(&self, t1: u32, t2: u32) -> u32 {
        if !self.buckets.is_empty() {
            let kh = Self::pack_key(t1, t2);
            let h = Self::hash_packed(kh);
            return self.probe_bucket_best(kh, self.bucket_of(t1, t2, h));
        }
        let key = Self::key(t1, t2);
        let h = Self::hash_key(key);
        self.probe_table(t1, t2, (h >> self.shift) as usize)
    }

    #[inline(always)]
    pub fn lookup_rank(&self, t1: u32, t2: u32) -> u32 {
        if (t1 | t2) < DENSE_N {
            // SAFETY: t1, t2 < DENSE_N, so index < dense.len().
            return unsafe { self.lookup_dense(t1, t2) };
        } else if !self.can_merge(t1, t2) {
            return u32::MAX;
        }
        if !self.buckets.is_empty() {
            let kh = Self::pack_key(t1, t2);
            let h = Self::hash_packed(kh);
            let b = self.bucket_of(t1, t2, h);
            if !self.bloom_may_contain_b(h, b) {
                return u32::MAX;
            }
            return self.probe_bucket_best(kh, b);
        }
        let h = Self::hash_key(Self::key(t1, t2));
        if !self.bloom_may_contain(h) {
            return u32::MAX;
        }
        self.probe_table(t1, t2, (h >> self.shift) as usize)
    }

    #[inline(always)]
    fn lookup(&self, t1: u32, t2: u32) -> (u32, u32) {
        let r = self.lookup_rank(t1, t2);
        if r == u32::MAX {
            (u32::MAX, 0)
        } else {
            (r, self.target(r))
        }
    }
}

const NONE: u32 = u32::MAX;

/// Precomputed priority jump-tables for the O(n) streaming BPE merger.
/// (The streaming algorithm — tuetoken's "original" state machine — is used as a
/// fallback for long chunks, where the scan-and-merge in `merge` degrades to
/// O(n^2). For short post-pretokenization chunks, `merge` is faster.)
pub struct StreamTables {
    merges: Vec<(u32, u32, u32)>,
    initial_dest: Vec<u32>,     // token -> first (highest-priority) merge with this token1, or NONE
    next_token1_dest: Vec<u32>, // merge i -> next merge (index > i) with the same token1
    next_target_dest: Vec<u32>, // merge i -> first merge (index > i) with token1 == target_of_i
}

impl StreamTables {
    pub fn new(merges: &[(u32, u32, u32)], vocab_size: usize) -> Self {
        let nm = merges.len();
        // For each token value, the ascending list of merge indices where it is
        // token1 (ascending because we append in merge-index = priority order).
        let mut by_token1: Vec<Vec<u32>> = vec![Vec::new(); vocab_size];
        for (i, &(t1, _, _)) in merges.iter().enumerate() {
            if (t1 as usize) < vocab_size {
                by_token1[t1 as usize].push(i as u32);
            }
        }
        let mut initial_dest = vec![NONE; vocab_size];
        for (t, list) in by_token1.iter().enumerate() {
            if let Some(&f) = list.first() {
                initial_dest[t] = f;
            }
        }
        let mut next_token1_dest = vec![NONE; nm];
        for list in &by_token1 {
            for w in 0..list.len() {
                next_token1_dest[list[w] as usize] =
                    if w + 1 < list.len() { list[w + 1] } else { NONE };
            }
        }
        let mut next_target_dest = vec![NONE; nm];
        for (i, &(_, _, target)) in merges.iter().enumerate() {
            if (target as usize) < vocab_size {
                let list = &by_token1[target as usize];
                let pos = list.partition_point(|&x| x <= i as u32); // first index > i
                if pos < list.len() {
                    next_target_dest[i] = list[pos];
                }
            }
        }
        StreamTables {
            merges: merges.to_vec(),
            initial_dest,
            next_token1_dest,
            next_target_dest,
        }
    }

    #[inline]
    fn dest_for(&self, token: u32) -> u32 {
        let t = token as usize;
        if t < self.initial_dest.len() {
            self.initial_dest[t]
        } else {
            NONE
        }
    }

    /// Process one (token, destination) into `state`, emitting completed tokens
    /// to `out`. Iterative (an explicit `todo` stack) so deep merge chains can't
    /// blow the call stack on long inputs.
    fn push_token(&self, token: u32, dest: u32, state: &mut Vec<u32>,
                  todo: &mut Vec<(u32, u32)>, out: &mut Vec<u32>) {
        todo.clear();
        todo.push((token, dest));
        while let Some((tok, d)) = todo.pop() {
            match state.last().copied() {
                None => {
                    if d == NONE {
                        out.push(tok);
                    } else {
                        state.push(d);
                    }
                }
                Some(next) => {
                    if d != NONE && d < next {
                        state.push(d);
                    } else {
                        state.pop();
                        let (t1, t2, target) = self.merges[next as usize];
                        if tok == t2 {
                            todo.push((target, self.next_target_dest[next as usize]));
                        } else {
                            // process token1's fall-through first, then re-process tok
                            todo.push((tok, d));
                            todo.push((t1, self.next_token1_dest[next as usize]));
                        }
                    }
                }
            }
        }
    }

    fn flush(&self, state: &mut Vec<u32>, todo: &mut Vec<(u32, u32)>, out: &mut Vec<u32>) {
        while let Some(next) = state.pop() {
            let (t1, _, _) = self.merges[next as usize];
            self.push_token(t1, self.next_token1_dest[next as usize], state, todo, out);
        }
    }

    /// Streaming BPE of one chunk: O(chunk length). Appends merged tokens to `out`.
    pub fn merge_into(&self, tokens: &[u32], out: &mut Vec<u32>,
                      state: &mut Vec<u32>, todo: &mut Vec<(u32, u32)>) {
        state.clear();
        for &tok in tokens {
            let dest = self.dest_for(tok);
            self.push_token(tok, dest, state, todo, out);
        }
        self.flush(state, todo, out);
    }
}

/// Apply BPE merges to `tokens` in place. `pairs` is reusable scratch.
/// Scan-and-merge BPE with cached pair ranks. Each pass applies every pair of the
/// current lowest rank (left-to-right, non-overlapping). Instead of re-hashing all
/// n-1 pairs every pass, we keep each pair's rank and only recompute the few pairs
/// a merge actually changed (its neighbours); unchanged pairs reuse their cached
/// rank. Cuts hash lookups from O(passes * n) to ~O(n) — the win on merge text.
///
/// Returns `true` if the result equals canonical (HF/tiktoken) BPE. Applying every
/// lowest-rank pair in one pass equals canonical ONLY while the per-pass applied
/// rank is non-decreasing: if merging pair `(a,b)->t` at rank r creates a pair of
/// rank < r (a "rank inversion", e.g. gemma's whitespace-run tokens where
/// `(30-run)+(1-run)->31-run` has a far lower rank than `(1)+(1)->2`), canonical
/// would pursue that lower pair first, growing one token, while this batch merges
/// all `(a,b)` together. We detect that for FREE (the next pass's min drops below
/// the applied min), bail immediately, and let the caller redo the chunk with
/// `merge_canonical`. On monotonic vocabularies (every trained BPE) this never
/// fires, so the hot path is unchanged.
///
/// `ranks`/`nrank`: pair-rank buffers (ping-ponged). `ntok`: next tokens. `src`:
/// for each next token, its old index, or -1 if it was just merged.
const MAX_SMALL: usize = 64;

const INIT_PREV: [u8; MAX_SMALL + 1] = {
    let mut a = [0u8; MAX_SMALL + 1];
    a[0] = MAX_SMALL as u8;
    let mut i = 1;
    while i <= MAX_SMALL {
        a[i] = (i - 1) as u8;
        i += 1;
    }
    a
};

const INIT_NEXT: [u8; MAX_SMALL] = {
    let mut a = [0u8; MAX_SMALL];
    let mut i = 0;
    while i < MAX_SMALL {
        a[i] = (i + 1) as u8;
        i += 1;
    }
    a
};

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn bpe_x86_tier() -> u8 {
    use std::sync::atomic::{AtomicU8, Ordering};
    static TIER: AtomicU8 = AtomicU8::new(0);
    let t = TIER.load(Ordering::Relaxed);
    if t != 0 {
        return t;
    }
    let detected = if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("bmi1") {
        3
    } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("bmi1") {
        2
    } else if std::is_x86_feature_detected!("bmi1") {
        1
    } else {
        4
    };
    TIER.store(detected, Ordering::Relaxed);
    detected
}

/// Stack doubly-linked-list canonical BPE merge with zero-STLF `active: u64`
/// bitmask min-reduction and 64B cache-line SIMD bucket probing.
macro_rules! merge_small_ll_body {
    ($tokens:expr, $map:expr, $probe:ident) => {{
        let n = $tokens.len();
        debug_assert!(n >= 2 && n <= MAX_SMALL && !$map.buckets.is_empty());
        if n == 2 {
            let r = $map.lookup_rank(*$tokens.get_unchecked(0), *$tokens.get_unchecked(1));
            if r != u32::MAX {
                *$tokens.get_unchecked_mut(0) = $map.target(r);
                $tokens.truncate(1);
            }
            return;
        }
        let mut tok_buf = std::mem::MaybeUninit::<[u32; MAX_SMALL]>::uninit();
        let tok = tok_buf.as_mut_ptr() as *mut u32;
        let mut or_all = 0u32;
        for i in 0..n {
            let t = *$tokens.get_unchecked(i);
            *tok.add(i) = t;
            or_all |= t;
        }
        let nr = n - 1;
        let mut rnk_buf = std::mem::MaybeUninit::<[u32; MAX_SMALL]>::uninit();
        let rnk = rnk_buf.as_mut_ptr() as *mut u32;
        let mut best = u32::MAX;
        let mut active = 0u64;
        if or_all < DENSE_N {
            let dense = $map.dense.as_ptr();
            if !$map.has_dense_overflow {
                for i in 0..nr {
                    let a = *tok.add(i) as usize;
                    let b = *tok.add(i + 1) as usize;
                    let d = *dense.add((a << DENSE_BITS) | b);
                    if d != DENSE_NONE {
                        let p = ((d as u32) << 6) | (i as u32);
                        *rnk.add(i) = p;
                        active |= 1u64 << i;
                        best = best.min(p);
                    }
                }
            } else {
                for i in 0..nr {
                    let a = *tok.add(i);
                    let b = *tok.add(i + 1);
                    let d = *dense.add(((a as usize) << DENSE_BITS) | (b as usize));
                    let r = if d < DENSE_OVERFLOW {
                        d as u32
                    } else if d == DENSE_NONE {
                        u32::MAX
                    } else {
                        $map.probe_dense_overflow(a, b)
                    };
                    if r != u32::MAX {
                        let p = (r << 6) | (i as u32);
                        *rnk.add(i) = p;
                        active |= 1u64 << i;
                        best = best.min(p);
                    }
                }
            }
        } else {
            for i in 0..nr {
                let r = $map.lookup_rank(*tok.add(i), *tok.add(i + 1));
                if r != u32::MAX {
                    let p = (r << 6) | (i as u32);
                    *rnk.add(i) = p;
                    active |= 1u64 << i;
                    best = best.min(p);
                }
            }
        }
        if active == 0 {
            return;
        }
        let mut prev_buf = std::mem::MaybeUninit::<[u8; MAX_SMALL + 1]>::uninit();
        let mut next_buf = std::mem::MaybeUninit::<[u8; MAX_SMALL]>::uninit();
        let prev = prev_buf.as_mut_ptr() as *mut u8;
        let next = next_buf.as_mut_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(INIT_PREV.as_ptr(), prev, n);
        std::ptr::copy_nonoverlapping(INIT_NEXT.as_ptr(), next, n);
        loop {
            let rank = best >> 6;
            let bi = (best & 63) as usize;
            let merged = $map.target(rank);
            let r = *next.add(bi) as usize;
            let rr = *next.add(r) as usize;
            *tok.add(bi) = merged;
            *next.add(bi) = rr as u8;
            *prev.add(rr) = bi as u8;
            if bi == 0 && rr == n {
                break;
            }
            let l = *prev.add(bi) as usize;
            let lmask = if bi != 0 { 1u64 << l } else { 0 };
            active &= !((1u64 << r) | (1u64 << bi) | lmask);
            let mlr = $map.lr_bits(merged);
            if mlr != 0 {
                let do_l = bi != 0 && (mlr & 2) != 0 && ($map.lr_bits(*tok.add(l)) & 1) != 0;
                let do_r = rr < n && (mlr & 1) != 0 && ($map.lr_bits(*tok.add(rr)) & 2) != 0;
                let (rl, rr_val) = if merged >= DENSE_N {
                    let mut idx_l = usize::MAX;
                    let mut idx_r = usize::MAX;
                    let mut kh_l = 0u64;
                    let mut kh_r = 0u64;
                    if do_l {
                        let tl = *tok.add(l);
                        kh_l = MergeMap::pack_key(tl, merged);
                        let hl = MergeMap::hash_packed(kh_l);
                        let b = $map.bucket_of(tl, merged, hl);
                        if $map.bloom_may_contain_b(hl, b) {
                            idx_l = b;
                        }
                    }
                    if do_r {
                        let tr = *tok.add(rr);
                        kh_r = MergeMap::pack_key(merged, tr);
                        let hr = MergeMap::hash_packed(kh_r);
                        let b = $map.bucket_of(merged, tr, hr);
                        if $map.bloom_may_contain_b(hr, b) {
                            idx_r = b;
                        }
                    }
                    if idx_l != usize::MAX && idx_r != usize::MAX {
                        $map.prefetch_bucket(idx_l);
                        $map.prefetch_bucket(idx_r);
                        ($map.$probe(kh_l, idx_l), $map.$probe(kh_r, idx_r))
                    } else {
                        (
                            if idx_l != usize::MAX { $map.$probe(kh_l, idx_l) } else { u32::MAX },
                            if idx_r != usize::MAX { $map.$probe(kh_r, idx_r) } else { u32::MAX },
                        )
                    }
                } else {
                    (
                        if do_l { $map.lookup_rank(*tok.add(l), merged) } else { u32::MAX },
                        if do_r { $map.lookup_rank(merged, *tok.add(rr)) } else { u32::MAX },
                    )
                };
                if rl != u32::MAX {
                    *rnk.add(l) = (rl << 6) | (l as u32);
                    active |= 1u64 << l;
                }
                if rr_val != u32::MAX {
                    *rnk.add(bi) = (rr_val << 6) | (bi as u32);
                    active |= 1u64 << bi;
                }
            }
            if active == 0 {
                break;
            }
            let mut m = active;
            let i0 = m.trailing_zeros() as usize;
            m &= m - 1;
            best = *rnk.add(i0);
            while m != 0 {
                let i = m.trailing_zeros() as usize;
                m &= m - 1;
                best = best.min(*rnk.add(i));
            }
        }
        $tokens.clear();
        let mut cur = 0usize;
        while cur < n {
            $tokens.push(*tok.add(cur));
            cur = *next.add(cur) as usize;
        }
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,bmi1")]
unsafe fn merge_small_ll_avx512(tokens: &mut Vec<u32>, map: &MergeMap) {
    merge_small_ll_body!(tokens, map, probe_bucket_avx512)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1")]
unsafe fn merge_small_ll_avx2(tokens: &mut Vec<u32>, map: &MergeMap) {
    merge_small_ll_body!(tokens, map, probe_bucket_avx2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1")]
unsafe fn merge_small_ll_bmi1(tokens: &mut Vec<u32>, map: &MergeMap) {
    merge_small_ll_body!(tokens, map, probe_bucket_scalar)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn merge_small_ll_default(tokens: &mut Vec<u32>, map: &MergeMap) {
    merge_small_ll_body!(tokens, map, probe_bucket_neon)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn merge_small_ll_default(tokens: &mut Vec<u32>, map: &MergeMap) {
    merge_small_ll_body!(tokens, map, probe_bucket_scalar)
}

#[inline]
fn merge_small_canonical(tokens: &mut Vec<u32>, map: &MergeMap) {
    #[cfg(target_arch = "x86_64")]
    {
        match bpe_x86_tier() {
            3 => {
                unsafe { merge_small_ll_avx512(tokens, map) };
                return;
            }
            2 => {
                unsafe { merge_small_ll_avx2(tokens, map) };
                return;
            }
            1 => {
                unsafe { merge_small_ll_bmi1(tokens, map) };
                return;
            }
            _ => {}
        }
    }
    // SAFETY: invariants enforced by caller (`2 <= tokens.len() <= 64 && !map.buckets.is_empty()`).
    unsafe { merge_small_ll_default(tokens, map) };
}

/// Zero-copy direct byte-slice -> `out: &mut Vec<u32>` canonical BPE merge:
/// - Eliminates the 4 redundant `s.chunk` copies on every cold merge.
/// - Uses `dense_byte[(b0 << 8) | b1]` to break the `singleton -> dense` chain.
/// - Maintains a 64B stack `lr[65]` array with sentinel 0s at `n` and `MAX_SMALL`.
/// - Zero-STLF `active: u64` bitmask min-reduction with 2-way ILP unrolling.
/// - 64B cache-line 8-wide SIMD bucket probing (AVX-512F / AVX2 / NEON).
/// - Specializes `n == 1, 2, 3` with zero stack arrays or linked-list setup.
macro_rules! merge_bytes_ll_body {
    ($chunk:expr, $singleton:expr, $map:expr, $out:expr, $probe:ident) => {{
        let n = $chunk.len();
        debug_assert!(n >= 1 && n <= MAX_SMALL && !$map.dense_byte.is_empty());
        let cp = $chunk.as_ptr();
        if n == 1 {
            $out.push(*$singleton.get_unchecked(*cp as usize));
            return;
        }
        let db = $map.dense_byte.as_ptr();
        if n == 2 {
            let b0 = *cp as usize;
            let b1 = *cp.add(1) as usize;
            let d = *db.add((b0 << 8) | b1);
            if d < DENSE_OVERFLOW {
                $out.push($map.target(d as u32));
                return;
            }
            let t0 = *$singleton.get_unchecked(b0);
            let t1 = *$singleton.get_unchecked(b1);
            if d == DENSE_OVERFLOW {
                let r = $map.probe_dense_overflow(t0, t1);
                if r != u32::MAX {
                    $out.push($map.target(r));
                    return;
                }
            }
            $out.reserve(2);
            let base = $out.len();
            let wp = $out.as_mut_ptr().add(base);
            *wp = t0;
            *wp.add(1) = t1;
            $out.set_len(base + 2);
            return;
        }
        if n == 3 {
            let b0 = *cp as usize;
            let b1 = *cp.add(1) as usize;
            let b2 = *cp.add(2) as usize;
            let t0 = *$singleton.get_unchecked(b0);
            let t1 = *$singleton.get_unchecked(b1);
            let t2 = *$singleton.get_unchecked(b2);
            let d0 = *db.add((b0 << 8) | b1);
            let d1 = *db.add((b1 << 8) | b2);
            let r0 = if d0 < DENSE_OVERFLOW {
                d0 as u32
            } else if d0 == DENSE_NONE {
                u32::MAX
            } else {
                $map.probe_dense_overflow(t0, t1)
            };
            let r1 = if d1 < DENSE_OVERFLOW {
                d1 as u32
            } else if d1 == DENSE_NONE {
                u32::MAX
            } else {
                $map.probe_dense_overflow(t1, t2)
            };
            if r0 == u32::MAX && r1 == u32::MAX {
                $out.reserve(3);
                let base = $out.len();
                let wp = $out.as_mut_ptr().add(base);
                *wp = t0;
                *wp.add(1) = t1;
                *wp.add(2) = t2;
                $out.set_len(base + 3);
                return;
            }
            let (a, b) = if r0 <= r1 {
                ($map.target(r0), t2)
            } else {
                (t0, $map.target(r1))
            };
            let r2 = $map.lookup_rank(a, b);
            if r2 != u32::MAX {
                $out.push($map.target(r2));
            } else {
                $out.reserve(2);
                let base = $out.len();
                let wp = $out.as_mut_ptr().add(base);
                *wp = a;
                *wp.add(1) = b;
                $out.set_len(base + 2);
            }
            return;
        }
        let mut tok_buf = std::mem::MaybeUninit::<[u32; MAX_SMALL]>::uninit();
        let mut lr_buf = std::mem::MaybeUninit::<[u8; MAX_SMALL + 1]>::uninit();
        let tok = tok_buf.as_mut_ptr() as *mut u32;
        let lr = lr_buf.as_mut_ptr() as *mut u8;
        let blr = $map.byte_lr.as_ptr();
        let nr = n - 1;
        let mut rnk_buf = std::mem::MaybeUninit::<[u32; MAX_SMALL]>::uninit();
        let rnk = rnk_buf.as_mut_ptr() as *mut u32;
        let mut best = u32::MAX;
        let mut active = 0u64;
        let mut b0 = *cp as usize;
        if !$map.has_dense_byte_overflow {
            for i in 0..nr {
                let b1 = *cp.add(i + 1) as usize;
                *tok.add(i) = *$singleton.get_unchecked(b0);
                *lr.add(i) = *blr.add(b0);
                let d = *db.add((b0 << 8) | b1);
                if d != DENSE_NONE {
                    let p = ((d as u32) << 6) | (i as u32);
                    *rnk.add(i) = p;
                    active |= 1u64 << i;
                    best = best.min(p);
                }
                b0 = b1;
            }
        } else {
            for i in 0..nr {
                let b1 = *cp.add(i + 1) as usize;
                let t0 = *$singleton.get_unchecked(b0);
                *tok.add(i) = t0;
                *lr.add(i) = *blr.add(b0);
                let d = *db.add((b0 << 8) | b1);
                let r = if d < DENSE_OVERFLOW {
                    d as u32
                } else if d == DENSE_NONE {
                    u32::MAX
                } else {
                    $map.probe_dense_overflow(t0, *$singleton.get_unchecked(b1))
                };
                if r != u32::MAX {
                    let p = (r << 6) | (i as u32);
                    *rnk.add(i) = p;
                    active |= 1u64 << i;
                    best = best.min(p);
                }
                b0 = b1;
            }
        }
        *tok.add(nr) = *$singleton.get_unchecked(b0);
        *lr.add(nr) = *blr.add(b0);
        if active == 0 {
            $out.reserve(n);
            let base = $out.len();
            std::ptr::copy_nonoverlapping(tok, $out.as_mut_ptr().add(base), n);
            $out.set_len(base + n);
            return;
        }
        // Sentinels: prev[0] == MAX_SMALL and next[last] == n both map to lr == 0,
        // eliminating both boundary branches from `do_l` and `do_r` in the hot loop.
        *lr.add(n) = 0;
        *lr.add(MAX_SMALL) = 0;
        let mut prev_buf = std::mem::MaybeUninit::<[u8; MAX_SMALL + 1]>::uninit();
        let mut next_buf = std::mem::MaybeUninit::<[u8; MAX_SMALL]>::uninit();
        let prev = prev_buf.as_mut_ptr() as *mut u8;
        let next = next_buf.as_mut_ptr() as *mut u8;
        // Fixed-size SIMD copy (16B handles ~97% of cold chunks in 1 instruction!).
        if n <= 16 {
            std::ptr::copy_nonoverlapping(INIT_PREV.as_ptr(), prev, 16);
            std::ptr::copy_nonoverlapping(INIT_NEXT.as_ptr(), next, 16);
        } else if n <= 32 {
            std::ptr::copy_nonoverlapping(INIT_PREV.as_ptr(), prev, 32);
            std::ptr::copy_nonoverlapping(INIT_NEXT.as_ptr(), next, 32);
        } else {
            std::ptr::copy_nonoverlapping(INIT_PREV.as_ptr(), prev, 64);
            std::ptr::copy_nonoverlapping(INIT_NEXT.as_ptr(), next, 64);
        }
        let fast_target = $map.linear_off != u32::MAX && $map.all_targets_ge_256;
        let lin_off = $map.linear_off;
        loop {
            let rank = best >> 6;
            let bi = (best & 63) as usize;
            let merged = if fast_target { rank.wrapping_add(lin_off) } else { $map.target(rank) };
            let r = *next.add(bi) as usize;
            let rr = *next.add(r) as usize;
            *tok.add(bi) = merged;
            *next.add(bi) = rr as u8;
            *prev.add(rr) = bi as u8;
            // Single 99.99%-not-taken check (avoids 25%-mispredicted `bi == 0` branch).
            if (bi | (rr ^ n)) == 0 {
                break;
            }
            let l = *prev.add(bi) as usize;
            let lmask = if bi != 0 { 1u64 << l } else { 0 };
            active &= !((1u64 << r) | (1u64 << bi) | lmask);
            let mlr = $map.lr_bits_unchecked(merged);
            *lr.add(bi) = mlr;
            let (rl, rr_val) = if mlr != 0 {
                // Sentinel lr[MAX_SMALL] == 0 (when bi == 0) and lr[n] == 0 (when rr == n).
                let do_l = ((mlr >> 1) & *lr.add(l) & 1) != 0;
                let do_r = (mlr & (*lr.add(rr) >> 1) & 1) != 0;
                if do_l && do_r {
                    let tl = *tok.add(l);
                    let tr = *tok.add(rr);
                    if (merged | tl | tr) < DENSE_N {
                        // Direct L2 dense matrix: 2 x movzwl loads, ZERO hash/bloom/probe!
                        ($map.lookup_dense(tl, merged), $map.lookup_dense(merged, tr))
                    } else if (tl | merged) < DENSE_N {
                        let kh_r = MergeMap::pack_key(merged, tr);
                        let hr = MergeMap::hash_packed(kh_r);
                        let b_r = $map.bucket_of(merged, tr, hr);
                        $map.prefetch_bucket(b_r);
                        let rl = $map.lookup_dense(tl, merged);
                        let rr_v = if $map.bloom_may_contain_b(hr, b_r) {
                            $map.$probe(kh_r, b_r)
                        } else {
                            u32::MAX
                        };
                        (rl, rr_v)
                    } else if (merged | tr) < DENSE_N {
                        let kh_l = MergeMap::pack_key(tl, merged);
                        let hl = MergeMap::hash_packed(kh_l);
                        let b_l = $map.bucket_of(tl, merged, hl);
                        $map.prefetch_bucket(b_l);
                        let rr_v = $map.lookup_dense(merged, tr);
                        let rl = if $map.bloom_may_contain_b(hl, b_l) {
                            $map.$probe(kh_l, b_l)
                        } else {
                            u32::MAX
                        };
                        (rl, rr_v)
                    } else {
                        let kh_l = MergeMap::pack_key(tl, merged);
                        let kh_r = MergeMap::pack_key(merged, tr);
                        let hl = MergeMap::hash_packed(kh_l);
                        let hr = MergeMap::hash_packed(kh_r);
                        let b_l = $map.bucket_of(tl, merged, hl);
                        let b_r = $map.bucket_of(merged, tr, hr);
                        $map.prefetch_bucket(b_l);
                        $map.prefetch_bucket(b_r);
                        let bp = $map.bloom.as_ptr();
                        let wl = *bp.add(b_l);
                        let wr = *bp.add(b_r);
                        let ok_l = MergeMap::bloom_hit(wl, hl);
                        let ok_r = MergeMap::bloom_hit(wr, hr);
                        (
                            if ok_l { $map.$probe(kh_l, b_l) } else { u32::MAX },
                            if ok_r { $map.$probe(kh_r, b_r) } else { u32::MAX },
                        )
                    }
                } else if do_l {
                    let tl = *tok.add(l);
                    let rl = if (tl | merged) < DENSE_N {
                        $map.lookup_dense(tl, merged)
                    } else {
                        let kh_l = MergeMap::pack_key(tl, merged);
                        let hl = MergeMap::hash_packed(kh_l);
                        let b_l = $map.bucket_of(tl, merged, hl);
                        $map.prefetch_bucket(b_l);
                        if $map.bloom_may_contain_b(hl, b_l) {
                            $map.$probe(kh_l, b_l)
                        } else {
                            u32::MAX
                        }
                    };
                    (rl, u32::MAX)
                } else if do_r {
                    let tr = *tok.add(rr);
                    let rr_v = if (merged | tr) < DENSE_N {
                        $map.lookup_dense(merged, tr)
                    } else {
                        let kh_r = MergeMap::pack_key(merged, tr);
                        let hr = MergeMap::hash_packed(kh_r);
                        let b_r = $map.bucket_of(merged, tr, hr);
                        $map.prefetch_bucket(b_r);
                        if $map.bloom_may_contain_b(hr, b_r) {
                            $map.$probe(kh_r, b_r)
                        } else {
                            u32::MAX
                        }
                    };
                    (u32::MAX, rr_v)
                } else {
                    (u32::MAX, u32::MAX)
                }
            } else {
                (u32::MAX, u32::MAX)
            };
            // OOO latency-hiding: reduce untouched `active` BEFORE storing the
            // newly probed `rnk[l]`/`rnk[bi]`. Zero data dependency on `(rl, rr_val)`
            // lets the OOO core hide the min-scan behind L2/L3 dense/bloom/bucket
            // loads, eliminates STLF stalls, and scans 1-2 fewer active bits!
            let mut m = active;
            best = if m != 0 {
                let i0 = m.trailing_zeros() as usize;
                m &= m - 1;
                let mut b0 = *rnk.add(i0);
                if m != 0 {
                    let i1 = m.trailing_zeros() as usize;
                    m &= m - 1;
                    let mut b1 = *rnk.add(i1);
                    while m != 0 {
                        let ia = m.trailing_zeros() as usize;
                        m &= m - 1;
                        b0 = b0.min(*rnk.add(ia));
                        if m == 0 {
                            break;
                        }
                        let ib = m.trailing_zeros() as usize;
                        m &= m - 1;
                        b1 = b1.min(*rnk.add(ib));
                    }
                    b0.min(b1)
                } else {
                    b0
                }
            } else {
                u32::MAX
            };
            if rl != u32::MAX {
                let p = (rl << 6) | (l as u32);
                *rnk.add(l) = p;
                active |= 1u64 << l;
                best = best.min(p);
            }
            if rr_val != u32::MAX {
                let p = (rr_val << 6) | (bi as u32);
                *rnk.add(bi) = p;
                active |= 1u64 << bi;
                best = best.min(p);
            }
            if active == 0 {
                break;
            }
        }
        $out.reserve(n);
        let base = $out.len();
        let mut wp = $out.as_mut_ptr().add(base);
        let mut cur = 0usize;
        while cur < n {
            *wp = *tok.add(cur);
            wp = wp.add(1);
            cur = *next.add(cur) as usize;
        }
        $out.set_len(wp.offset_from($out.as_ptr()) as usize);
    }};
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,bmi1")]
unsafe fn merge_bytes_ll_avx512(chunk: &[u8], singleton: &[u32; 256], map: &MergeMap, out: &mut Vec<u32>) {
    merge_bytes_ll_body!(chunk, singleton, map, out, probe_bucket_avx512)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1")]
unsafe fn merge_bytes_ll_avx2(chunk: &[u8], singleton: &[u32; 256], map: &MergeMap, out: &mut Vec<u32>) {
    merge_bytes_ll_body!(chunk, singleton, map, out, probe_bucket_avx2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi1")]
unsafe fn merge_bytes_ll_bmi1(chunk: &[u8], singleton: &[u32; 256], map: &MergeMap, out: &mut Vec<u32>) {
    merge_bytes_ll_body!(chunk, singleton, map, out, probe_bucket_scalar)
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn merge_bytes_ll_default(chunk: &[u8], singleton: &[u32; 256], map: &MergeMap, out: &mut Vec<u32>) {
    merge_bytes_ll_body!(chunk, singleton, map, out, probe_bucket_neon)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn merge_bytes_ll_default(chunk: &[u8], singleton: &[u32; 256], map: &MergeMap, out: &mut Vec<u32>) {
    merge_bytes_ll_body!(chunk, singleton, map, out, probe_bucket_scalar)
}

/// Fast zero-copy canonical BPE for `1 <= chunk.len() <= 64` when
/// `map.init_byte_tables` succeeded. Appends merged tokens directly to `out`.
#[inline]
pub fn merge_bytes_into(chunk: &[u8], singleton: &[u32; 256], map: &MergeMap, out: &mut Vec<u32>) {
    #[cfg(target_arch = "x86_64")]
    {
        match bpe_x86_tier() {
            3 => {
                unsafe { merge_bytes_ll_avx512(chunk, singleton, map, out) };
                return;
            }
            2 => {
                unsafe { merge_bytes_ll_avx2(chunk, singleton, map, out) };
                return;
            }
            1 => {
                unsafe { merge_bytes_ll_bmi1(chunk, singleton, map, out) };
                return;
            }
            _ => {}
        }
    }
    unsafe { merge_bytes_ll_default(chunk, singleton, map, out) };
}


#[must_use]
pub fn merge(
    tokens: &mut Vec<u32>,
    map: &MergeMap,
    ranks: &mut Vec<u32>,
    ntok: &mut Vec<u32>,
    src: &mut Vec<i32>,
    nrank: &mut Vec<u32>,
) -> bool {
    let mut n = tokens.len();
    if n <= 1 {
        return true;
    }
    if n <= MAX_SMALL && !map.buckets.is_empty() {
        // Exact canonical stack linked-list + SIMD min merge — faster AND
        // canonical even on rank-inverted vocabularies (never needs to bail!).
        merge_small_canonical(tokens, map);
        return true;
    }
    ranks.clear();
    let mut min = u32::MAX;
    for i in 0..n - 1 {
        let (a, b) = unsafe { (*tokens.get_unchecked(i), *tokens.get_unchecked(i + 1)) };
        let r = map.lookup_rank(a, b);
        ranks.push(r);
        if r < min {
            min = r;
        }
    }

    while min != u32::MAX {
        let applied = min;
        let target = map.target(min);
        ntok.clear();
        src.clear();
        let mut i = 0usize;
        // Apply every pair whose rank == min, left to right (non-overlapping).
        unsafe {
            while i < n {
                if i + 1 < n && *ranks.get_unchecked(i) == min {
                    ntok.push(target);
                    src.push(-1); // freshly merged token
                    i += 2;
                } else {
                    ntok.push(*tokens.get_unchecked(i));
                    src.push(i as i32);
                    i += 1;
                }
            }
        }
        std::mem::swap(tokens, ntok);
        n = tokens.len();

        // Rebuild pair ranks: reuse the cached rank when both tokens are
        // unchanged copies that were adjacent before; otherwise look up.
        nrank.clear();
        min = u32::MAX;
        if n > 1 {
            for k in 0..n - 1 {
                let (sa, sb) = unsafe { (*src.get_unchecked(k), *src.get_unchecked(k + 1)) };
                let r = if sa >= 0 && sb == sa + 1 {
                    unsafe { *ranks.get_unchecked(sa as usize) }
                } else {
                    let (a, b) =
                        unsafe { (*tokens.get_unchecked(k), *tokens.get_unchecked(k + 1)) };
                    map.lookup_rank(a, b)
                };
                nrank.push(r);
                if r < min {
                    min = r;
                }
            }
        }
        std::mem::swap(ranks, nrank);
        // Rank inversion: a freshly created pair undercut the rank we just applied,
        // so the batch result is not canonical. Bail; the caller redoes canonically.
        if min < applied {
            return false;
        }
    }
    true
}

/// Reusable scratch for `merge_canonical` (doubly-linked list + packed-u64
/// min-heap with cached pair ranks for zero-lookup pop validation), so the
/// canonical path allocates nothing per chunk.
#[derive(Default)]
pub struct CanonScratch {
    prev: Vec<u32>,
    next: Vec<u32>,
    rnk: Vec<u32>,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<u64>>,
    out: Vec<u32>,
}

/// Canonical BPE (exact HF/tiktoken semantics): repeatedly merge the single
/// adjacent pair of globally-lowest rank, leftmost on ties. Doubly-linked list +
/// binary heap with cached pair ranks => O(n log n) with ZERO hash lookups on
/// heap pops (at most 2 lookups per actual merge), exact for ANY merge table
/// (monotonic or rank-inverted) and 20-100x faster than `StreamTables` on long
/// chunks (> 64 bytes) because it never walks `next_token1_dest` chains.
pub fn merge_canonical(tokens: &mut Vec<u32>, map: &MergeMap, s: &mut CanonScratch) {
    use std::cmp::Reverse;
    let n = tokens.len();
    if n <= 1 {
        return;
    }
    if n <= MAX_SMALL && !map.buckets.is_empty() {
        merge_small_canonical(tokens, map);
        return;
    }
    let none = n as u32; // sentinel for "no neighbour"
    s.prev.clear();
    s.next.clear();
    s.rnk.clear();
    s.heap.clear();
    s.prev.reserve(n);
    s.next.reserve(n);
    s.rnk.reserve(n);
    let mut or_all = 0u32;
    for i in 0..n {
        or_all |= unsafe { *tokens.get_unchecked(i) };
        s.prev.push(if i == 0 { none } else { (i - 1) as u32 });
        s.next.push((i + 1) as u32); // next[n - 1] == n == none
    }
    // Initial adjacent-pair ranks: direct L2 dense table when all tokens < DENSE_N.
    if or_all < DENSE_N {
        let dense = map.dense.as_ptr();
        for i in 0..n - 1 {
            let a = unsafe { *tokens.get_unchecked(i) };
            let b = unsafe { *tokens.get_unchecked(i + 1) };
            let d = unsafe { *dense.add(((a as usize) << DENSE_BITS) | (b as usize)) };
            let r = if d < DENSE_OVERFLOW {
                d as u32
            } else if d == DENSE_NONE {
                u32::MAX
            } else {
                map.probe_dense_overflow(a, b)
            };
            s.rnk.push(r);
            if r != u32::MAX {
                s.heap.push(Reverse(((r as u64) << 32) | (i as u64)));
            }
        }
    } else {
        for i in 0..n - 1 {
            let r = unsafe { map.lookup_rank(*tokens.get_unchecked(i), *tokens.get_unchecked(i + 1)) };
            s.rnk.push(r);
            if r != u32::MAX {
                s.heap.push(Reverse(((r as u64) << 32) | (i as u64)));
            }
        }
    }
    s.rnk.push(u32::MAX);

    // Pop lowest (rank, position). `rnk[i]` always equals the exact current
    // pair rank of `(tokens[i], tokens[next[i]])` (or `u32::MAX` if dead/tail),
    // validating pops with zero hash probes.
    while let Some(Reverse(packed)) = s.heap.pop() {
        let r = (packed >> 32) as u32;
        let i = packed as u32 as usize;
        if unsafe { *s.rnk.get_unchecked(i) } != r {
            continue;
        }
        let j = unsafe { *s.next.get_unchecked(i) } as usize;
        let t = map.target(r);
        unsafe {
            *tokens.get_unchecked_mut(i) = t;
            *s.rnk.get_unchecked_mut(j) = u32::MAX; // j folded into i (dead)
            let k = *s.next.get_unchecked(j);
            *s.next.get_unchecked_mut(i) = k;
            if k != none {
                *s.prev.get_unchecked_mut(k as usize) = i as u32;
            }
            let p = *s.prev.get_unchecked(i);
            if p != none {
                let pu = p as usize;
                let pr = map.lookup_rank(*tokens.get_unchecked(pu), t);
                *s.rnk.get_unchecked_mut(pu) = pr;
                if pr != u32::MAX {
                    s.heap.push(Reverse(((pr as u64) << 32) | (p as u64)));
                }
            }
            if k != none {
                let nr = map.lookup_rank(t, *tokens.get_unchecked(k as usize));
                *s.rnk.get_unchecked_mut(i) = nr;
                if nr != u32::MAX {
                    s.heap.push(Reverse(((nr as u64) << 32) | (i as u64)));
                }
            } else {
                *s.rnk.get_unchecked_mut(i) = u32::MAX;
            }
        }
    }
    // Gather survivors by walking the list from head 0 (never folded away).
    s.out.clear();
    let mut i = 0u32;
    loop {
        s.out.push(unsafe { *tokens.get_unchecked(i as usize) });
        let nx = unsafe { *s.next.get_unchecked(i as usize) };
        if nx == none {
            break;
        }
        i = nx;
    }
    std::mem::swap(tokens, &mut s.out);
}

/// Like `merge`, but carries a parallel `spans` array so each output token keeps
/// the byte range it came from. Same apply-all-lowest-rank result as `merge`
/// (verified by encode_with_offsets's ids matching encode_ordinary). Not the hot
/// path (only `encode_with_offsets` uses it), so kept simple.
pub fn merge_with_spans(
    tokens: &mut Vec<u32>,
    spans: &mut Vec<(usize, usize)>,
    map: &MergeMap,
    pairs: &mut Vec<(u32, u32)>,
) {
    let mut n = tokens.len();
    if n <= 1 {
        return;
    }
    loop {
        pairs.clear();
        let mut best = u32::MAX;
        for i in 0..n - 1 {
            let (r, t) = map.lookup(tokens[i], tokens[i + 1]);
            pairs.push((r, t));
            if r < best {
                best = r;
            }
        }
        if best == u32::MAX {
            break;
        }
        let mut w = 0usize;
        let mut i = 0usize;
        while i < n {
            if i + 1 < n && pairs[i].0 == best {
                tokens[w] = pairs[i].1;
                spans[w] = (spans[i].0, spans[i + 1].1);
                w += 1;
                i += 2;
            } else {
                tokens[w] = tokens[i];
                spans[w] = spans[i];
                w += 1;
                i += 1;
            }
        }
        tokens.truncate(w);
        spans.truncate(w);
        n = w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// Is a merge table monotonic (every token produced before consumed)? Mirrors
    /// the runtime flag; on monotonic tables the batch/streaming mergers are canonical.
    fn is_monotonic(merges: &[(u32, u32, u32)], vocab_size: usize) -> bool {
        let mut produce_max = vec![-1i64; vocab_size];
        let mut consume_min = vec![i64::MAX; vocab_size];
        for (i, &(a, b, t)) in merges.iter().enumerate() {
            let i = i as i64;
            produce_max[t as usize] = produce_max[t as usize].max(i);
            consume_min[a as usize] = consume_min[a as usize].min(i);
            consume_min[b as usize] = consume_min[b as usize].min(i);
        }
        (0..vocab_size).all(|t| produce_max[t] < 0 || consume_min[t] > produce_max[t])
    }

    /// Every engine must agree with brute-force canonical BPE: `merge_canonical`
    /// always; the batch `merge` whenever it reports success (no bail); and the
    /// streaming merger on monotonic tables. (Random tables exercise both monotonic
    /// and rank-inverted shapes, so this also covers the bail/redo path.)
    #[test]
    fn engines_match_canonical() {
        let mut seed = 0x1234_5678u64;
        for _round in 0..500 {
            let base = 8u32 + (lcg(&mut seed) % 56) as u32;
            let nmerges = 20 + (lcg(&mut seed) % 300) as u32;
            let mut merges: Vec<(u32, u32, u32)> = Vec::new();
            let mut seen: HashSet<(u32, u32)> = HashSet::new();
            let mut next_id = base;
            for _ in 0..nmerges {
                let t1 = (lcg(&mut seed) as u32) % next_id;
                let t2 = (lcg(&mut seed) as u32) % next_id;
                if seen.insert((t1, t2)) {
                    merges.push((t1, t2, next_id));
                    next_id += 1;
                }
            }
            let vocab_size = next_id as usize;
            let mut map = MergeMap::new(&merges);
            let mut sing_i32 = [0i32; 256];
            let mut sing_u32 = [0u32; 256];
            for b in 0..256 {
                let id = (b as u32) % base;
                sing_i32[b] = id as i32;
                sing_u32[b] = id;
            }
            let byte_ok = map.init_byte_tables(&sing_i32);
            let st = StreamTables::new(&merges, vocab_size);
            let mono = is_monotonic(&merges, vocab_size);

            for _ in 0..20 {
                let len = (lcg(&mut seed) % 400) as usize;
                let seq: Vec<u32> = (0..len).map(|_| (lcg(&mut seed) as u32) % base).collect();
                let want = canonical(&seq, &merges);
                if byte_ok && len >= 1 && len <= 64 {
                    let bytes: Vec<u8> = seq.iter().map(|&t| t as u8).collect();
                    let mut bout = Vec::new();
                    merge_bytes_into(&bytes, &sing_u32, &map, &mut bout);
                    assert_eq!(bout, want, "merge_bytes_into differs\nseq={:?}", seq);
                }

                // batch merge: result is canonical iff it reports success
                let mut a = seq.clone();
                let (mut r1, mut nt, mut sr, mut r2) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let ok = merge(&mut a, &map, &mut r1, &mut nt, &mut sr, &mut r2);
                if ok {
                    assert_eq!(a, want, "batch claimed canonical but differs\nseq={:?}", seq);
                } else {
                    assert!(!mono, "monotonic table must never bail\nmerges={:?}", merges);
                }

                // canonical merger: always exact
                let mut c = seq.clone();
                let mut cs = CanonScratch::default();
                merge_canonical(&mut c, &map, &mut cs);
                assert_eq!(c, want, "merge_canonical differs\nseq={:?}", seq);

                // streaming merger: canonical on monotonic tables
                if mono {
                    let mut b = Vec::new();
                    let (mut state, mut todo) = (Vec::new(), Vec::new());
                    st.merge_into(&seq, &mut b, &mut state, &mut todo);
                    assert_eq!(b, want, "streaming != canonical on monotonic table\nseq={:?}", seq);
                }
            }
        }
    }

    /// Brute-force canonical BPE (HF/tiktoken semantics): repeatedly merge the
    /// single adjacent pair of globally-lowest rank (leftmost on ties), re-evaluate.
    fn canonical(seq: &[u32], merges: &[(u32, u32, u32)]) -> Vec<u32> {
        use std::collections::HashMap;
        let mut rank: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for (i, &(a, b, t)) in merges.iter().enumerate() {
            rank.insert((a, b), (i as u32, t));
        }
        let mut v = seq.to_vec();
        loop {
            let mut best = u32::MAX;
            let mut bi = usize::MAX;
            for i in 0..v.len().saturating_sub(1) {
                if let Some(&(r, _)) = rank.get(&(v[i], v[i + 1])) {
                    if r < best { best = r; bi = i; }
                }
            }
            if bi == usize::MAX { break; }
            let t = rank[&(v[bi], v[bi + 1])].1;
            v[bi] = t;
            v.remove(bi + 1);
        }
        v
    }

    /// Rank-INVERTED run-length table (the gemma-whitespace shape): token `k`
    /// represents a run of length k+1; for each target length L, the merge
    /// `(L-1)+(1) -> L` is given a LOWER rank for LARGER L (extending a long run is
    /// top priority). Canonical BPE grows one token to MAX then repeats; the
    /// `merge_canonical` must match canonical on the gemma-whitespace shape, and the
    /// batch `merge` must BAIL (report non-canonical) there rather than return a wrong
    /// answer — the contract finish_chunk relies on to redo the chunk canonically.
    #[test]
    fn rank_inverted_canonical_and_bail() {
        const MAX: u32 = 31; // longest run token (ids 0..=MAX-1 => runs 1..=MAX)
        let vocab_size = MAX as usize;
        // merges: (run a)+(run b) -> run (a+b), for all a+b<=MAX. Rank so that the
        // pure "extend by one" (a=L-1,b=1) for large L is lowest. Use rank key =
        // (MAX - (a+b)) primary, then a, to make extension dominate.
        let mut tmp: Vec<((u32, u32), u32)> = Vec::new(); // ((idA,idB), targetId)
        for la in 1..MAX {
            for lb in 1..MAX {
                let l = la + lb;
                if l <= MAX {
                    tmp.push(((la - 1, lb - 1), l - 1));
                }
            }
        }
        // sort: larger total length first (lower rank), then larger la first
        tmp.sort_by_key(|&((a, _), t)| ((MAX - (t + 1)) as u64) << 20 | (MAX - 1 - a) as u64);
        let merges: Vec<(u32, u32, u32)> =
            tmp.iter().map(|&((a, b), t)| (a, b, t)).collect();
        let map = MergeMap::new(&merges);
        assert!(!is_monotonic(&merges, vocab_size), "table should be rank-inverted");

        let mut any_bail = false;
        for n in [2usize, 5, 33, 50, 62, 100, 200] {
            let seq = vec![0u32; n]; // n single-unit tokens
            let want = canonical(&seq, &merges);

            // batch merge must NOT silently return a non-canonical answer
            let mut a = seq.clone();
            let (mut r1, mut nt, mut sr, mut r2) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let ok = merge(&mut a, &map, &mut r1, &mut nt, &mut sr, &mut r2);
            if ok {
                assert_eq!(a, want, "batch claimed canonical but differs (n={n})");
            } else {
                any_bail = true;
            }

            // the canonical fallback must be exact (grows one token to MAX, repeats)
            let mut c = seq.clone();
            let mut cs = CanonScratch::default();
            merge_canonical(&mut c, &map, &mut cs);
            assert_eq!(c, want, "merge_canonical wrong on rank-inverted run (n={n})");
        }
        assert!(any_bail, "batch merge should bail on the rank-inverted runs");
    }

    /// (0,0) must not alias (0,1) in the merge map.
    #[test]
    fn merge_key_no_collision() {
        let merges = vec![(0u32, 0u32, 2u32), (0, 1, 3)];
        let map = MergeMap::new(&merges);
        assert_eq!(map.lookup(0, 0), (0, 2));
        assert_eq!(map.lookup(0, 1), (1, 3));
        assert_eq!(map.lookup(1, 0).0, u32::MAX); // absent
    }

    /// Streaming on a long repetitive run is O(n) and correct.
    #[test]
    fn streaming_long_run() {
        // a+a -> 256 ; collapses "aaaa..." pairwise
        let merges = vec![(97u32, 97u32, 256u32)];
        let st = StreamTables::new(&merges, 257);
        let seq: Vec<u32> = vec![97; 1000];
        let mut out = Vec::new();
        let (mut s, mut t) = (Vec::new(), Vec::new());
        st.merge_into(&seq, &mut out, &mut s, &mut t);
        // 1000 a's -> 500 X's (one pass merges all non-overlapping pairs)
        assert!(out.iter().all(|&x| x == 256));
        assert_eq!(out.len(), 500);
    }
}
