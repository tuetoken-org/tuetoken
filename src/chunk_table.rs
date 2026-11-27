//! Load-time "whole-chunk" table: pre-token bytes (≤ 15) -> its single token id.
//!
//! ~90% of pre-tokens in natural text encode to exactly ONE token, yet each one
//! would otherwise pay a full BPE merge (singleton lookup + pair-rank scans).
//! This table short-circuits them to one hash probe.
//!
//! Correct by construction: an entry `k -> t` is inserted only after running the
//! tokenizer's OWN per-chunk encoder on `k` at load time and observing `[t]`.
//! Because a chunk's tokens depend only on its bytes, `table[k]` can never differ
//! from what the merge path would produce — for any model family (byte-level,
//! byte_fallback/SentencePiece, ignore_merges, non-monotonic vocabularies). Keys
//! come from the vocabulary itself, so the table is complete for every chunk that
//! IS a token, and needs no warm-up: the first call on a cold process is as fast
//! as the millionth. It is read-only after load, so all threads share it with no
//! synchronization (no per-thread copies, no cache-line ping-pong).
//!
//! Layout: open addressing, linear probing, 24-byte entries (two u64 key words +
//! value) — compact, because lookups are bound by cache capacity, not by probe
//! branches (a 32-byte paired layout measured slower). Key packing: `lo` = bytes
//! 0..8, `hi` = bytes 8..15 | len << 56, so a real key always has `hi != 0`
//! (len ≥ 1); `hi == 0` marks an empty slot.

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Entry {
    lo: u64,
    hi: u64,
    val: u32,
}

pub const MAX_KEY: usize = 15;
pub const PACK20_MAX_TOK: u32 = (1 << 20) - 1;

#[cfg(test)]
pub static DENSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


pub struct ChunkTable {
    entries: HugeVec<Entry>,
    bloom: Vec<u64>,
    bloom_shift: u32,
    shift: u32,
    mask: usize,
}

/// Branch-minimal, zero-memcpy, page-boundary-safe packer for `1 <= len <= 15`
/// using at most 2 overlapping in-bounds GPR loads (zero stack buffer, zero
/// Store-To-Load Forwarding stall).
///
/// SAFETY: `p .. p+len` must be readable; `1 <= len <= 15`.
#[inline(always)]
pub unsafe fn pack_short_safe(p: *const u8, len: usize) -> (u64, u64) {
    debug_assert!(len >= 1 && len <= MAX_KEY);
    let (lo, hi_bytes) = if len >= 8 {
        let lo = u64::from_le((p as *const u64).read_unaligned());
        let hi_raw = u64::from_le((p.add(len - 8) as *const u64).read_unaligned());
        let hi_bytes = if len > 8 { hi_raw >> ((16 - len) * 8) } else { 0 };
        (lo, hi_bytes)
    } else if len >= 4 {
        let a = u32::from_le((p as *const u32).read_unaligned()) as u64;
        let b = u32::from_le((p.add(len - 4) as *const u32).read_unaligned()) as u64;
        (a | (b << ((len - 4) * 8)), 0)
    } else if len >= 2 {
        let a = u16::from_le((p as *const u16).read_unaligned()) as u64;
        let b = *p.add(len - 1) as u64;
        (a | (b << ((len - 1) * 8)), 0)
    } else {
        (*p as u64, 0)
    };
    (lo, hi_bytes | ((len as u64) << 56))
}

/// Pack ≤ 15 bytes into the (lo, hi) key. Caller guarantees 1 ≤ len ≤ 15.
#[inline(always)]
pub fn pack(chunk: &[u8]) -> (u64, u64) {
    let n = chunk.len();
    debug_assert!(n >= 1 && n <= MAX_KEY);
    // SAFETY: chunk has `n` readable bytes with 1 <= n <= MAX_KEY.
    unsafe { pack_short_safe(chunk.as_ptr(), n) }
}

/// Pack `len` bytes starting at `p` when at least 16 bytes are readable from `p`
/// using pure 64-bit GPR loads + masks (zero XMM->GPR domain crossing on x86_64
/// and single-`ldp` loads on ARM64).
///
/// SAFETY: `p .. p+16` must be readable; 1 ≤ len ≤ 15.
#[inline(always)]
pub unsafe fn pack_wide(p: *const u8, len: usize) -> (u64, u64) {
    debug_assert!(len >= 1 && len <= MAX_KEY);
    let (m0, m1) = *KEY_MASK64.get_unchecked(len);
    let lo = u64::from_le((p as *const u64).read_unaligned()) & m0;
    let hi = (u64::from_le((p.add(8) as *const u64).read_unaligned()) & m1) | ((len as u64) << 56);
    (lo, hi)
}

/// `pack_wide` for chunk `b[pos..pos + len]` (1 <= len <= MAX_KEY) wherever it
/// lies: within 16 bytes of the end of `b`, uses inline zero-memcpy overlapping
/// GPR loads (`pack_short_safe`) with zero stack spill / STLF stall.
#[inline(always)]
pub fn pack_at(b: &[u8], pos: usize, len: usize) -> (u64, u64) {
    if pos + 16 <= b.len() {
        // SAFETY: 16 bytes readable at pos.
        unsafe { pack_wide(b.as_ptr().add(pos), len) }
    } else {
        // SAFETY: pos + len <= b.len(), 1 <= len <= MAX_KEY.
        unsafe { pack_short_safe(b.as_ptr().add(pos), len) }
    }
}

static KEY_MASK64: [(u64, u64); 16] = {
    let mut m = [(0u64, 0u64); 16];
    let mut i = 1;
    while i < 16 {
        let full = (1u128 << (8 * i)) - 1;
        m[i] = (full as u64, (full >> 64) as u64);
        i += 1;
    }
    m
};

/// Portable read-prefetch hint (no-op where unsupported).
#[inline(always)]
pub fn prefetch(p: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}

#[inline(always)]
pub fn hash(lo: u64, hi: u64) -> u64 {
    // ONE multiply: callers index with the TOP bits, and every top bit of a
    // product depends on every bit of the multiplicand. `hi` is rotated so its
    // bytes (8..15 and the length) don't line up with `lo`'s. Collisions only
    // lengthen probes (keys are compared in full).
    (lo ^ hi.rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

impl ChunkTable {
    pub fn empty() -> Self {
        ChunkTable {
            entries: HugeVec::zeroed(2),
            bloom: vec![0u64; 2],
            bloom_shift: 63,
            shift: 63,
            mask: 1,
        }
    }

    /// Build from verified (key bytes, token) pairs. Duplicate keys keep the first.
    /// Also builds an L2-resident k=4 pure-GPR Blocked Bloom filter (~10.5 bits/entry,
    /// ~1% FPR: 256 KiB for o200k, 128 KiB for cl100k, 64 KiB for gpt2) so ~99% of
    /// non-vocab cold misses never touch the 12.6 MiB `entries` table in L3/DRAM.
    pub fn build(pairs: &[(&[u8], u32)]) -> Self {
        let mut cap = 16usize;
        #[cfg(test)]
        let dense = DENSE.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let dense = false;
        let spread = 2usize;
        while if dense { cap * 3 < pairs.len() * 4 } else { cap < pairs.len() * spread } {
            cap <<= 1;
        }
        let bits = cap.trailing_zeros();
        let bloom_words = (cap >> 4).max(2);
        let bloom_bits = bloom_words.trailing_zeros();
        let mut t = ChunkTable {
            entries: HugeVec::zeroed(cap),
            bloom: vec![0u64; bloom_words],
            bloom_shift: 64 - bloom_bits,
            shift: 64 - bits,
            mask: cap - 1,
        };
        for (k, v) in pairs {
            let (lo, hi) = pack(k);
            let h = hash(lo, hi);
            let bw = (h >> t.bloom_shift) as usize;
            t.bloom[bw] |= (1u64 << ((h >> 24) & 63))
                | (1u64 << ((h >> 30) & 63))
                | (1u64 << ((h >> 36) & 63))
                | (1u64 << ((h >> 42) & 63));
            let mut i = (h >> t.shift) as usize;
            loop {
                let e = &mut t.entries[i];
                if e.hi == 0 {
                    *e = Entry { lo, hi, val: *v };
                    break;
                }
                if e.lo == lo && e.hi == hi {
                    break;
                }
                i = (i + 1) & t.mask;
            }
        }
        t
    }

    #[inline(always)]
    pub fn get(&self, lo: u64, hi: u64) -> Option<u32> {
        self.get_h(lo, hi, hash(lo, hi))
    }

    #[cfg(test)]
    pub fn slots(&self) -> usize { self.entries.len() }
    #[cfg(test)]
    pub fn entry_bytes(&self) -> usize { std::mem::size_of_val(&self.entries[0]) }

    #[inline(always)]
    pub fn get_h(&self, lo: u64, hi: u64, h: u64) -> Option<u32> {
        // SAFETY: bloom_shift in 1..=63, so (h >> bloom_shift) < bloom.len().
        let w = unsafe { *self.bloom.get_unchecked((h >> self.bloom_shift) as usize) };
        if ((w >> ((h >> 24) & 63))
            & (w >> ((h >> 30) & 63))
            & (w >> ((h >> 36) & 63))
            & (w >> ((h >> 42) & 63))
            & 1)
            == 0
        {
            return None;
        }
        let mut i = (h >> self.shift) as usize;
        loop {
            // SAFETY: i is always masked to < entries.len().
            let e = unsafe { self.entries.get_unchecked(i) };
            if e.lo == lo && e.hi == hi {
                return Some(e.val);
            }
            if e.hi == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }
}

/// Hint Linux to back large (>= 2 MiB) random-access tables and output buffers
/// with 2 MiB Transparent Huge Pages (when THP is `[madvise]`, the Ubuntu/Linux
/// server default). Eliminates 4-level dTLB page-table walks on 16-128 MiB
/// `SharedTable` random probes and cuts 50,000 x 4 KiB output page faults down
/// to ~100 x 2 MiB faults. Safe no-op on non-Linux or unsupported kernels.
#[inline]
pub fn advise_hugepage(ptr: *mut u8, bytes: usize) {
    #[cfg(target_os = "linux")]
    if bytes >= (2 << 20) && !ptr.is_null() {
        const MADV_HUGEPAGE: i32 = 14;
        extern "C" {
            fn madvise(addr: *mut std::ffi::c_void, length: usize, advice: i32) -> i32;
        }
        let p = (ptr as usize) & !4095usize;
        let len = (ptr as usize + bytes) - p;
        unsafe {
            let _ = madvise(p as *mut std::ffi::c_void, len, MADV_HUGEPAGE);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (ptr, bytes);
}

/// Allocate zeroed memory for `layout`, advising Linux `MADV_HUGEPAGE` *before*
/// first-touch zeroing on `>= 2 MiB` 2 MiB-aligned tables so pages fault in as
/// genuine 2 MiB Transparent Huge Pages (eliminating 4-level dTLB walks).
#[inline]
unsafe fn alloc_zeroed_huge(layout: std::alloc::Layout) -> *mut u8 {
    #[cfg(target_os = "linux")]
    if layout.size() >= (2 << 20) && layout.align() >= (2 << 20) {
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        advise_hugepage(ptr, layout.size());
        std::ptr::write_bytes(ptr, 0, layout.size());
        return ptr;
    }
    let ptr = std::alloc::alloc_zeroed(layout);
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    advise_hugepage(ptr, layout.size());
    ptr
}

/// Owned contiguous buffer for `Copy` elements that aligns `>= 2 MiB`
/// allocations to 2 MiB on Linux and issues `MADV_HUGEPAGE` *before* first
/// touch, backing long-lived random-access hash tables (`ChunkTable`,
/// `MergeMap::packed`) with genuine 2 MiB Transparent Huge Pages.
/// On non-Linux targets uses normal 64B-aligned heap allocation.
pub struct HugeVec<T: Copy> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

unsafe impl<T: Copy + Send> Send for HugeVec<T> {}
unsafe impl<T: Copy + Sync> Sync for HugeVec<T> {}

impl<T: Copy> HugeVec<T> {
    #[inline]
    fn layout(cap: usize) -> std::alloc::Layout {
        let sz = std::mem::size_of::<T>();
        assert!(sz > 0, "HugeVec does not support ZSTs");
        let bytes = cap.checked_mul(sz).expect("HugeVec size overflow");
        #[cfg(target_os = "linux")]
        let align = if bytes >= (2 << 20) {
            2 << 20
        } else {
            std::mem::align_of::<T>().max(64)
        };
        #[cfg(not(target_os = "linux"))]
        let align = std::mem::align_of::<T>().max(64);
        std::alloc::Layout::from_size_align(bytes, align).expect("HugeVec layout")
    }

    #[inline]
    pub fn empty() -> Self {
        HugeVec { ptr: std::ptr::NonNull::dangling().as_ptr(), len: 0, cap: 0 }
    }

    /// Allocate `len` all-zero elements (with Linux 2 MiB THP before zeroing).
    #[inline]
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }
        let layout = Self::layout(len);
        // SAFETY: caller guarantees all-zero bit pattern is valid for `T`.
        let ptr = unsafe { alloc_zeroed_huge(layout) } as *mut T;
        HugeVec { ptr, len, cap: len }
    }
}

impl<T: Copy> std::ops::Deref for HugeVec<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: ptr is non-null & aligned (dangling when cap == 0); [0..len) initialized.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Copy> std::ops::DerefMut for HugeVec<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Copy> Drop for HugeVec<T> {
    fn drop(&mut self) {
        if self.cap != 0 {
            // SAFETY: allocated with Self::layout(self.cap).
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, Self::layout(self.cap)) };
        }
    }
}

/// Per-thread 2-way set-associative cache for the chunks the static table can't
/// answer (multi-token pre-tokens: rare words, numbers, identifiers).
///
/// * 32-byte entries, a set is one 64-byte line: one probe = one cache line.
/// * Up to 3 result tokens inline; longer results spill to a side arena.
/// * Insert = MRU into way 0, old way 0 demoted to way 1: bounded memory, no
///   rehash, nothing to clear on the hot path.
/// * Every entry carries a 24-bit owner tag (the tokenizer's uid), so threads
///   that alternate between tokenizers keep both warm and never mix results.
///   Tags are unique within an "epoch" (uid >> 24); the cache is wiped only when
///   the epoch changes, i.e. after 16.7M tokenizer constructions.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct DEntry {
    lo: u64,
    hi: u64,
    toks: [u32; 3],
    meta: u32, // bits 0..7: n tokens (0xFF = spilled), bits 8..31: owner tag
}

#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
struct DSet {
    w: [DEntry; 2],
}

const _: () = assert!(std::mem::size_of::<DSet>() == 64);

/// The cache starts at 2^DYN_MIN_BITS sets (256 KiB) and doubles — rehashing its
/// live entries — whenever it has evicted a quarter as many entries as it has
/// sets, up to
/// 2^DYN_MAX_BITS sets (8 MiB). Small workloads stay small; a large corpus with
/// many distinct pre-tokens gets a cache big enough to stop re-merging them.
const DYN_MIN_BITS: u32 = 12;
const DYN_MAX_BITS: u32 = 17;
pub const DYN_MAX_TOKS: usize = 3;
const SPILL: u32 = 0xFF;
const SPILL_MAX_TOKS: usize = 64;
const ARENA_CAP: usize = 1 << 20; // 4 MiB of spilled tokens, then wipe

/// `n` all-zero sets via the zeroing allocator (calloc / fresh mmap pages), so
/// untouched sets cost no RSS — `vec![DSet::default(); n]` would write all of it.
fn zeroed_sets(n: usize) -> Vec<DSet> {
    let layout = std::alloc::Layout::array::<DSet>(n).expect("cache layout");
    // SAFETY: layout has non-zero size; all-zero bytes are a valid DSet (plain
    // integers); the Vec takes ownership with exactly this layout (len == cap == n,
    // DSet's alignment), so its Drop frees it with the matching layout.
    unsafe {
        let p = std::alloc::alloc_zeroed(layout) as *mut DSet;
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Vec::from_raw_parts(p, n, n)
    }
}

#[derive(Default)]
pub struct DynCache {
    e: Vec<DSet>,
    bits: u32, // log2(sets); 0 = not allocated yet
    evictions: usize,
    arena: Vec<u32>,
    // Spilled entries record the arena generation they were written in; when
    // the arena fills it is cleared and the generation bumped, which retires
    // those entries WITHOUT wiping the rest of the cache.
    arena_gen: u32,
    epoch: u64,
    tag: u32, // current owner tag (24 bits), 0 = unclaimed
}

impl DynCache {
    /// Select the tokenizer this thread is about to encode with.
    #[inline]
    pub fn claim(&mut self, owner: u64) {
        let epoch = owner >> 24;
        if epoch != self.epoch {
            self.wipe();
            self.epoch = epoch;
        }
        self.tag = ((owner & 0xFF_FFFF) as u32) << 8;
    }

    #[inline(always)]
    pub fn owned_by(&self, owner: u64) -> bool {
        self.tag != 0 && self.tag == ((owner & 0xFF_FFFF) as u32) << 8 && self.epoch == owner >> 24
    }

    fn wipe(&mut self) {
        for x in self.e.iter_mut() {
            *x = DSet::default();
        }
        self.arena.clear();
    }


    #[inline(always)]
    pub fn get(&self, lo: u64, hi: u64) -> Option<&[u32]> {
        self.get_h(lo, hi, hash(lo, hi))
    }

    #[inline(always)]
    pub fn get_h(&self, lo: u64, hi: u64, h: u64) -> Option<&[u32]> {
        if self.e.is_empty() {
            return None;
        }
        // SAFETY: set index < 2^bits == e.len() once allocated.
        let set = unsafe { self.e.get_unchecked((h >> (64 - self.bits)) as usize) };
        for x in &set.w {
            if x.lo == lo && x.hi == hi && (x.meta & !0xFF) == self.tag && x.meta != 0 {
                let n = x.meta & 0xFF;
                if n == SPILL {
                    if x.toks[2] != self.arena_gen {
                        return None; // retired with an older arena
                    }
                    let (off, len) = (x.toks[0] as usize, x.toks[1] as usize);
                    return self.arena.get(off..off + len);
                }
                return Some(&x.toks[..n as usize]);
            }
        }
        None
    }

    #[inline]
    pub fn put(&mut self, lo: u64, hi: u64, toks: &[u32]) {
        self.put_h(lo, hi, hash(lo, hi), toks)
    }

    #[inline]
    pub fn put_h(&mut self, lo: u64, hi: u64, h: u64, toks: &[u32]) {
        if toks.is_empty() || toks.len() > SPILL_MAX_TOKS || self.tag == 0 {
            return;
        }
        if self.e.is_empty() {
            self.bits = DYN_MIN_BITS;
            self.e = zeroed_sets(1 << DYN_MIN_BITS);
        }
        let mut t = [0u32; 3];
        let n = if toks.len() <= DYN_MAX_TOKS {
            t[..toks.len()].copy_from_slice(toks);
            toks.len() as u32
        } else {
            if self.arena.len() + toks.len() > ARENA_CAP {
                self.arena.clear();
                self.arena_gen = self.arena_gen.wrapping_add(1);
                if self.arena_gen == 0 {
                    self.wipe(); // generation wrapped: retire everything to stay exact
                }
            }
            t[0] = self.arena.len() as u32;
            t[1] = toks.len() as u32;
            t[2] = self.arena_gen;
            self.arena.extend_from_slice(toks);
            SPILL
        };
        let i = (h >> (64 - self.bits)) as usize;
        let set = &mut self.e[i];
        if set.w[1].meta != 0 {
            self.evictions += 1;
        }
        set.w[1] = set.w[0];
        set.w[0] = DEntry { lo, hi, toks: t, meta: self.tag | n };
        if self.evictions >= self.e.len() / 4 && self.bits < DYN_MAX_BITS {
            self.grow();
        }
    }

    /// Double the number of sets and re-insert every live entry (older way first,
    /// so the most recent stays in way 0 when two land in the same set).
    #[cold]
    fn grow(&mut self) {
        let bits = self.bits + 1;
        let mut e = zeroed_sets(1 << bits);
        for set in &self.e {
            for x in set.w.iter().rev() {
                if x.meta != 0 {
                    let j = (hash(x.lo, x.hi) >> (64 - bits)) as usize;
                    let ns = &mut e[j];
                    ns.w[1] = ns.w[0];
                    ns.w[0] = *x;
                }
            }
        }
        self.e = e;
        self.bits = bits;
        self.evictions = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_wide_matches_pack() {
        let src: Vec<u8> = (0u8..64).map(|x| x.wrapping_mul(37).wrapping_add(1)).collect();
        for start in 0..32 {
            for len in 1..=MAX_KEY {
                let a = pack(&src[start..start + len]);
                let b = unsafe { pack_wide(src.as_ptr().add(start), len) };
                assert_eq!(a, b, "start={start} len={len}");
            }
        }
    }

    #[test]
    fn build_and_get() {
        let owned: Vec<(Vec<u8>, u32)> = (0..5000u32)
            .map(|i| (format!(" w{i}").into_bytes(), i))
            .collect();
        let pairs: Vec<(&[u8], u32)> = owned.iter().map(|(k, v)| (k.as_slice(), *v)).collect();
        let t = ChunkTable::build(&pairs);
        for (k, v) in &pairs {
            let (lo, hi) = pack(k);
            assert_eq!(t.get(lo, hi), Some(*v));
        }
        let (lo, hi) = pack(b" nope");
        assert_eq!(t.get(lo, hi), None);
        // a key that is a prefix with trailing zero bytes must not alias
        let (lo, hi) = pack(b" w1\0");
        assert_eq!(t.get(lo, hi), None);
    }
}

/// Zero-malloc open-addressing byte-slice -> u32-slice arena table powering
/// both `LongCache` and every `SpillMap` shard. Replaces per-entry
/// `Box<[u8]>` + `Box<[u32]>` heap allocations with two contiguous arenas.
#[derive(Clone, Copy, Default)]
struct ArenaSlot {
    hash: u64, // 0 = empty; real hashes OR'd with 1
    k_off: u32,
    t_off: u32,
    k_len: u16,
    t_len: u16,
}

#[derive(Default)]
struct ArenaMap {
    slots: Vec<ArenaSlot>,
    mask: usize,
    len: usize,
    keys: Vec<u8>,
    toks: Vec<u32>,
}

#[inline(always)]
pub fn hash_bytes(key: &[u8]) -> u64 {
    use std::hash::{BuildHasher, Hash, Hasher};
    let mut h = rustc_hash::FxBuildHasher.build_hasher();
    key.hash(&mut h);
    h.finish() | 1
}

impl ArenaMap {
    fn clear(&mut self) {
        for s in &mut self.slots {
            s.hash = 0;
        }
        self.len = 0;
        self.keys.clear();
        self.toks.clear();
    }

    #[inline]
    fn get_h(&self, key: &[u8], h: u64) -> Option<&[u32]> {
        if self.slots.is_empty() {
            return None;
        }
        let mut i = (h as usize) & self.mask;
        loop {
            let s = unsafe { *self.slots.get_unchecked(i) };
            if s.hash == 0 {
                return None;
            }
            if s.hash == h && s.k_len as usize == key.len() {
                let ko = s.k_off as usize;
                if unsafe { self.keys.get_unchecked(ko..ko + s.k_len as usize) } == key {
                    let to = s.t_off as usize;
                    return Some(unsafe { self.toks.get_unchecked(to..to + s.t_len as usize) });
                }
            }
            i = (i + 1) & self.mask;
        }
    }

    fn grow(&mut self) {
        let new_cap = (self.slots.len() * 2).max(64);
        let new_mask = new_cap - 1;
        let mut ns = vec![ArenaSlot::default(); new_cap];
        for &s in &self.slots {
            if s.hash != 0 {
                let mut i = (s.hash as usize) & new_mask;
                while ns[i].hash != 0 {
                    i = (i + 1) & new_mask;
                }
                ns[i] = s;
            }
        }
        self.slots = ns;
        self.mask = new_mask;
    }

    /// Insert if absent; returns true if newly inserted.
    #[inline]
    fn insert_h(&mut self, key: &[u8], h: u64, toks: &[u32]) -> bool {
        if key.len() > u16::MAX as usize || toks.len() > u16::MAX as usize {
            return false;
        }
        if self.keys.len() + key.len() > u32::MAX as usize || self.toks.len() + toks.len() > u32::MAX as usize {
            return false;
        }
        if self.slots.is_empty() || (self.len + 1) * 2 > self.slots.len() {
            self.grow();
        }
        let mut i = (h as usize) & self.mask;
        loop {
            let s = self.slots[i];
            if s.hash == 0 {
                let k_off = self.keys.len() as u32;
                let t_off = self.toks.len() as u32;
                self.keys.extend_from_slice(key);
                self.toks.extend_from_slice(toks);
                self.slots[i] = ArenaSlot {
                    hash: h,
                    k_off,
                    t_off,
                    k_len: key.len() as u16,
                    t_len: toks.len() as u16,
                };
                self.len += 1;
                return true;
            }
            if s.hash == h && s.k_len as usize == key.len() {
                let ko = s.k_off as usize;
                if &self.keys[ko..ko + s.k_len as usize] == key {
                    return false;
                }
            }
            i = (i + 1) & self.mask;
        }
    }
}

/// Per-thread exact cache for chunks LONGER than `MAX_KEY` bytes (long words,
/// identifiers, punctuation runs). Zero per-entry mallocs (arena-backed).
#[derive(Default)]
struct LongMap {
    owner: u64,
    map: ArenaMap,
}

#[derive(Default)]
pub struct LongCache {
    maps: Vec<LongMap>, // most recently claimed first
}

const LONG_MAPS: usize = 4;
const LONG_MAX_ENTRIES: usize = 1 << 17;
const LONG_MAX_ARENA: usize = 1 << 22; // u32s (16 MiB max; lazy-allocated only when long chunks exist)
const LONG_MAX_BYTES: usize = u16::MAX as usize; // matches ArenaSlot::k_len: u16; bounded by SpillMap/LongCache caps

impl LongCache {
    #[inline]
    pub fn claim(&mut self, owner: u64) {
        if self.maps.first().map_or(false, |m| m.owner == owner) {
            return;
        }
        if let Some(i) = self.maps.iter().position(|m| m.owner == owner) {
            let m = self.maps.remove(i);
            self.maps.insert(0, m);
        } else {
            if self.maps.len() >= LONG_MAPS {
                self.maps.pop();
            }
            self.maps.insert(0, LongMap { owner, ..Default::default() });
        }
    }

    #[inline]
    pub fn get_h(&self, owner: u64, key: &[u8], h: u64) -> Option<&[u32]> {
        let m = self.maps.first().filter(|m| m.owner == owner)?;
        m.map.get_h(key, h)
    }

    #[inline]
    pub fn put_h(&mut self, owner: u64, key: &[u8], h: u64, toks: &[u32]) {
        let Some(m) = self.maps.first_mut().filter(|m| m.owner == owner) else { return };
        if key.len() > LONG_MAX_BYTES {
            return;
        }
        if m.map.len >= LONG_MAX_ENTRIES || m.map.toks.len() + toks.len() > LONG_MAX_ARENA {
            m.map.clear();
        }
        let _ = m.map.insert_h(key, h, toks);
    }
}

// ---------------------------------------------------------------------------
// Shared learned table
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// One slot: 16-byte key, `meta` (0 = empty, 1 = being written, else
/// `2 | n << 2` = published with `n` tokens) and up to `W` inline tokens.
/// W = 3 -> 32 bytes, W = 11 -> 64 bytes. A slot is written exactly once
/// (claimed by CAS 0 -> 1, filled, then published with a Release store) and
/// never changes afterwards, so a reader that sees it published (Acquire)
/// reads a consistent entry.
#[repr(C)]
struct SSlot<const W: usize> {
    lo: AtomicU64,
    hi: AtomicU64,
    toks: [AtomicU32; W],
    meta: AtomicU32,
}

/// Its own 128-byte line (Apple M-series lines are 128 B): the insert counter
/// must not share a cache line with the fields every lookup reads.
#[repr(align(128))]
struct PaddedCount(AtomicUsize);

/// Read-only after construction except `count`, which is padded away from
/// `ptr`/`mask`/`shift` — otherwise every insert (any thread) would invalidate
/// the line every lookup (every thread) reads: classic false sharing.
#[repr(C, align(128))]
struct STable<const W: usize> {
    ptr: *mut SSlot<W>,
    len: usize,
    shift: u32,
    count: PaddedCount,
}

impl<const W: usize> STable<W> {
    fn layout(n: usize) -> std::alloc::Layout {
        let sz = std::mem::size_of::<SSlot<W>>();
        let bytes = sz * (n + PROBE_MAX);
        // 64B minimum alignment for cache-line pairing; on Linux, align >=2 MiB
        // tables to 2 MiB so `MADV_HUGEPAGE` actually backs them with 2 MiB THP
        // (unaligned glibc mmap + 16B header yields 0 KiB AnonHugePages!).
        #[cfg(target_os = "linux")]
        let align = if bytes >= (2 << 20) { 2 << 20 } else { sz.next_power_of_two().max(64) };
        #[cfg(not(target_os = "linux"))]
        let align = sz.next_power_of_two().max(64);
        std::alloc::Layout::from_size_align(bytes, align).expect("table layout")
    }

    fn new(bits: u32) -> Box<STable<W>> {
        let n = 1usize << bits;
        let layout = Self::layout(n);
        // SAFETY: non-zero size; all-zero bytes are a valid SSlot (atomics of
        // 0 = empty). Extra PROBE_MAX tail slots eliminate wrap-around checks
        // in every probe loop. Freed in Drop with the same layout.
        let ptr = unsafe { alloc_zeroed_huge(layout) } as *mut SSlot<W>;
        Box::new(STable { ptr, len: n, shift: 64 - bits, count: PaddedCount(AtomicUsize::new(0)) })
    }

    #[inline(always)]
    fn slot(&self, i: usize) -> &SSlot<W> {
        // SAFETY: callers pass i < len + PROBE_MAX; the allocation lives as long as self.
        unsafe { &*self.ptr.add(i) }
    }

    fn bits(&self) -> u32 {
        64 - self.shift
    }
}

impl<const W: usize> Drop for STable<W> {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with exactly this layout.
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, Self::layout(self.len)) };
    }
}

const PROBE_MAX: usize = 32;
/// Token count `SharedTable::get` reports for an entry stored in the SpillMap.
pub const SPILLED: usize = 63;
/// Outcome of `View::emit_inline_at` so `flush_recs` branches directly from
/// Probe-0 / Probe-1+ hits to `w += n; continue;` without a redundant `cmp < 6`.
#[derive(Clone, Copy)]
pub enum InlineHit {
    Hit(usize),
    Empty,
    Spilled,
}
const SHARED_MIN_BITS: u32 = 14;

/// Inserts a thread batches before publishing them to a table's count.
const COUNT_BATCH: usize = 64;
thread_local! {
    static PENDING: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// See [`SharedTable::view`]. Zero-wrap: `STable` allocates `+ PROBE_MAX` tail
/// slots so `View` fits in 2 words (`base`, `shift`) with no `end` pointer.
#[derive(Clone, Copy)]
pub struct View<'a, const W: usize> {
    base: *const SSlot<W>,
    shift: u32,
    _t: std::marker::PhantomData<&'a ()>,
}

/// Unpack a 20-bit packed 4..=6 token `SSlot<3>` (state `m & 3 == 3`) directly
/// into `dst` and return token count (`4..=6`).
/// SAFETY: `p` points at a published `SSlot<3>` (`m & 3 == 3`, write-once immutable),
/// and `dst` has capacity for at least 6 `u32`s.
#[inline(always)]
unsafe fn unpack_packed6(p: *const SSlot<3>, m: u32, dst: *mut u32) -> usize {
    let raw = ((p as *const u8).add(16) as *const [u32; 4]).read();
    let a = (raw[0] as u64) | ((raw[1] as u64) << 32);
    let b = raw[2];
    let t0 = (a & 0xF_FFFF) as u32;
    let t1 = ((a >> 20) & 0xF_FFFF) as u32;
    let t2 = ((a >> 40) & 0xF_FFFF) as u32;
    let t3 = b & 0xF_FFFF;
    let t4 = (b >> 20) | (((m >> 4) & 0xFF) << 12);
    let t5 = ((m >> 12) & 0xFFFF) | (((a >> 60) as u32) << 16);
    *dst = t0;
    *dst.add(1) = t1;
    *dst.add(2) = t2;
    *dst.add(3) = t3;
    *dst.add(4) = t4;
    *dst.add(5) = t5;
    4 + (((m >> 2) & 3) as usize)
}

#[inline(always)]
fn pack_6_words(toks: &[u32]) -> Option<([u32; 3], u32)> {
    let n = toks.len();
    if n < 4 || n > 6 {
        return None;
    }
    let t0 = toks[0];
    let t1 = toks[1];
    let t2 = toks[2];
    let t3 = toks[3];
    let t4 = if n > 4 { toks[4] } else { 0 };
    let t5 = if n > 5 { toks[5] } else { 0 };
    if (t0 | t1 | t2 | t3 | t4 | t5) > PACK20_MAX_TOK {
        return None;
    }
    let a = (t0 as u64)
        | ((t1 as u64) << 20)
        | ((t2 as u64) << 40)
        | ((((t5 >> 16) & 0xF) as u64) << 60);
    let b = t3 | ((t4 & 0xFFF) << 20);
    let meta = 3u32
        | (((n - 4) as u32) << 2)
        | (((t4 >> 12) & 0xFF) << 4)
        | ((t5 & 0xFFFF) << 12);
    Some(([a as u32, (a >> 32) as u32, b], meta))
}

impl View<'_, 3> {
    /// Address of the home slot for hash `h` (never 0).
    #[inline(always)]
    pub fn slot_addr(&self, h: u64) -> u64 {
        // SAFETY: (h >> shift) < 2^bits == table length.
        unsafe { self.base.add((h >> self.shift) as usize) as u64 }
    }

    #[inline(always)]
    fn load_meta(s: &SSlot<3>) -> u32 {
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            let val: u32;
            unsafe {
                std::arch::asm!(
                    "ldapr {val:w}, [{ptr}]",
                    val = out(reg) val,
                    ptr = in(reg) &s.meta,
                    options(nostack, readonly, preserves_flags),
                );
            }
            val
        }
        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        {
            s.meta.load(Ordering::Acquire)
        }
    }

    /// Probe starting from a home-slot address computed by [`Self::slot_addr`]
    /// and emit directly to `dst` on inline hit:
    /// - Probe 0 unrolled straight-line (hits ~80-90% of Zipf-hot lookups with
    ///   minimal L1d load port pressure and zero loop overhead).
    /// - 1..=3 tokens (`m & 3 == 2`): ONE 128-bit load + ONE 128-bit store.
    /// - 4..=6 tokens (`m & 3 == 3`): 20-bit bit-packed unpack from same 32B slot.
    /// Returns `InlineHit::Hit(1..=6)` on hit, `InlineHit::Empty` on empty slot, or `InlineHit::Spilled`.
    ///
    /// SAFETY: `dst` must have capacity for at least 6 `u32`s.
    #[inline(always)]
    pub unsafe fn emit_inline_at(&self, lo: u64, hi: u64, addr: u64, dst: *mut u32) -> InlineHit {
        const {
            assert!(std::mem::offset_of!(SSlot<3>, toks) == 16);
            assert!(std::mem::size_of::<SSlot<3>>() == 32);
        }
        let mut p = addr as *const SSlot<3>;
        // Probe 0: unrolled fast path for the prefetched home slot.
        {
            let s = &*p;
            let m = Self::load_meta(s);
            let (l, g) = (s.lo.load(Ordering::Relaxed), s.hi.load(Ordering::Relaxed));
            if (l ^ lo) | (g ^ hi) | (m.wrapping_sub(6) & !0b1100) as u64 == 0 {
                let t = ((p as *const u8).add(16) as *const [u32; 4]).read();
                (dst as *mut [u32; 4]).write_unaligned(t);
                return InlineHit::Hit((m >> 2) as usize);
            }
            if m == 0 {
                return InlineHit::Empty;
            }
            if l == lo && g == hi {
                if (m & 3) == 3 {
                    return InlineHit::Hit(unpack_packed6(p, m, dst));
                }
                return InlineHit::Spilled;
            }
        }
        // Probe 1..PROBE_MAX: zero-wrap linear probing into `+ PROBE_MAX` tail.
        for _ in 1..PROBE_MAX {
            p = p.add(1);
            let s = &*p;
            let m = Self::load_meta(s);
            let (l, g) = (s.lo.load(Ordering::Relaxed), s.hi.load(Ordering::Relaxed));
            if (l ^ lo) | (g ^ hi) | (m.wrapping_sub(6) & !0b1100) as u64 == 0 {
                let t = ((p as *const u8).add(16) as *const [u32; 4]).read();
                (dst as *mut [u32; 4]).write_unaligned(t);
                return InlineHit::Hit((m >> 2) as usize);
            }
            if m == 0 {
                return InlineHit::Empty;
            }
            if l == lo && g == hi {
                if (m & 3) == 3 {
                    return InlineHit::Hit(unpack_packed6(p, m, dst));
                }
                return InlineHit::Spilled;
            }
        }
        InlineHit::Spilled
    }
}

#[inline(always)]
fn spin_or_yield(spins: &mut u32) {
    *spins = spins.wrapping_add(1);
    if *spins < 64 {
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

/// Cooperative work-stealing growth job shared by all threads that reach
/// `put_n` while a `SharedTable` is doubling: every participating thread copies
/// disjoint 4096-slot stripes (`[s_lo..s_hi) -> [d_lo..d_hi)` with zero CAS)
/// and then inserts its pending entry into the new table (zero lost entries,
/// zero duplicate BPE storms during table growth).
struct GrowJob<const W: usize> {
    old: *const STable<W>,
    new: *mut STable<W>,
    next_stripe: AtomicUsize,
    done_stripes: AtomicUsize,
    copied: AtomicUsize,
    n_stripes: usize,
    stripe: usize,
    diff: u32,
    overflow: std::sync::Mutex<Vec<(u64, u64, [u32; W], u32, u64)>>,
    finished: AtomicBool,
}

unsafe impl<const W: usize> Send for GrowJob<W> {}
unsafe impl<const W: usize> Sync for GrowJob<W> {}

impl<const W: usize> GrowJob<W> {
    #[inline(never)]
    fn help(&self) {
        let old_t = unsafe { &*self.old };
        let new_t = unsafe { &*self.new };
        let mut local_copied = 0usize;
        let mut local_ov: Vec<(u64, u64, [u32; W], u32, u64)> = Vec::new();
        loop {
            let si = self.next_stripe.fetch_add(1, Ordering::Relaxed);
            if si >= self.n_stripes {
                break;
            }
            let s_lo = si * self.stripe;
            let s_hi = (s_lo + self.stripe).min(old_t.len + PROBE_MAX);
            let d_lo = s_lo.min(old_t.len) << self.diff;
            let d_hi = s_hi.min(old_t.len) << self.diff;
            for i in s_lo..s_hi {
                let s = old_t.slot(i);
                let mut m = s.meta.load(Ordering::Acquire);
                let mut sp = 0u32;
                while m == 1 {
                    spin_or_yield(&mut sp);
                    m = s.meta.load(Ordering::Acquire);
                }
                if m < 2 {
                    continue;
                }
                let (lo, hi) = (s.lo.load(Ordering::Relaxed), s.hi.load(Ordering::Relaxed));
                let h = hash(lo, hi);
                let mut j = (h >> new_t.shift) as usize;
                if j >= d_lo && j < d_hi {
                    while j < d_hi {
                        let d = new_t.slot(j);
                        if d.meta.load(Ordering::Relaxed) == 0 {
                            d.lo.store(lo, Ordering::Relaxed);
                            d.hi.store(hi, Ordering::Relaxed);
                            for (x, y) in d.toks.iter().zip(s.toks.iter()) {
                                x.store(y.load(Ordering::Relaxed), Ordering::Relaxed);
                            }
                            d.meta.store(m, Ordering::Relaxed);
                            local_copied += 1;
                            break;
                        }
                        j += 1;
                    }
                    if j == d_hi {
                        let mut tv = [0u32; W];
                        for (dst, src) in tv.iter_mut().zip(s.toks.iter()) {
                            *dst = src.load(Ordering::Relaxed);
                        }
                        local_ov.push((lo, hi, tv, m, h));
                    }
                } else {
                    let mut tv = [0u32; W];
                    for (dst, src) in tv.iter_mut().zip(s.toks.iter()) {
                        *dst = src.load(Ordering::Relaxed);
                    }
                    local_ov.push((lo, hi, tv, m, h));
                }
            }
            if !local_ov.is_empty() {
                let mut g = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
                g.append(&mut local_ov);
            }
            self.done_stripes.fetch_add(1, Ordering::Release);
        }
        if local_copied != 0 {
            self.copied.fetch_add(local_copied, Ordering::Relaxed);
        }
    }
}

/// Grow-only, lock-free table of LEARNED chunk encodings (multi-token results
/// the load-time table can't hold), shared by every thread using a tokenizer.
/// `W` = uncompressed tokens stored inline per entry (`W = 3` also adaptively
/// bit-packs 4..=6 tokens of <=20 bits into the same 32-byte slot using state 3!).
pub struct SharedTable<const W: usize = 3> {
    cur: AtomicPtr<STable<W>>,
    grow_job: AtomicPtr<GrowJob<W>>,
    retired: std::sync::Mutex<Vec<*mut STable<W>>>,
    retired_jobs: std::sync::Mutex<Vec<*mut GrowJob<W>>>,
    growing: AtomicBool,
    max_bits: AtomicU32,
}

unsafe impl<const W: usize> Send for SharedTable<W> {}
unsafe impl<const W: usize> Sync for SharedTable<W> {}

impl<const W: usize> SharedTable<W> {
    /// `max_mib`: memory cap of the live table (0 disables learning).
    pub fn new(max_mib: usize) -> Self {
        let t = SharedTable {
            cur: AtomicPtr::new(Box::into_raw(STable::<W>::new(SHARED_MIN_BITS))),
            grow_job: AtomicPtr::new(std::ptr::null_mut()),
            retired: std::sync::Mutex::new(Vec::new()),
            retired_jobs: std::sync::Mutex::new(Vec::new()),
            growing: AtomicBool::new(false),
            max_bits: AtomicU32::new(SHARED_MIN_BITS),
        };
        t.set_limit_mib(max_mib);
        t
    }

    /// Change the memory cap (MiB) of the live table. 0 = stop learning.
    pub fn set_limit_mib(&self, max_mib: usize) {
        let slots = (max_mib << 20) / std::mem::size_of::<SSlot<W>>();
        let bits = if slots < 2 { 0 } else { usize::BITS - 1 - slots.leading_zeros() };
        self.max_bits.store(bits, Ordering::Relaxed);
    }

    pub fn limit_mib(&self) -> usize {
        let b = self.max_bits.load(Ordering::Relaxed);
        if b == 0 { 0 } else { ((1usize << b) * std::mem::size_of::<SSlot<W>>()) >> 20 }
    }

    #[inline(always)]
    fn table(&self) -> &STable<W> {
        unsafe { &*self.cur.load(Ordering::Acquire) }
    }

    pub fn len(&self) -> usize {
        self.table().count.0.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn view(&self) -> View<'_, W> {
        let t = self.table();
        View {
            base: t.ptr as *const SSlot<W>,
            shift: t.shift,
            _t: std::marker::PhantomData,
        }
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn get(&self, lo: u64, hi: u64, h: u64) -> Option<([u32; W], usize)> {
        let t = self.table();
        let home = (h >> t.shift) as usize;
        for step in 0..PROBE_MAX {
            let s = t.slot(home + step);
            let m = s.meta.load(Ordering::Acquire);
            if m == 0 {
                return None;
            }
            if m & 3 == 2 && s.lo.load(Ordering::Relaxed) == lo && s.hi.load(Ordering::Relaxed) == hi {
                let mut v = [0u32; W];
                for (d, x) in v.iter_mut().zip(s.toks.iter()) {
                    *d = x.load(Ordering::Relaxed);
                }
                return Some((v, (m >> 2) as usize));
            }
        }
        None
    }

    /// Insert a learned encoding of 1..=W uncompressed tokens.
    pub fn put(&self, lo: u64, hi: u64, h: u64, toks: &[u32]) {
        if toks.is_empty() || toks.len() > W {
            return;
        }
        self.put_n(lo, hi, h, toks, 2 | ((toks.len() as u32) << 2));
    }

    /// Record that `key`'s encoding lives in the [`SpillMap`].
    pub fn put_spilled(&self, lo: u64, hi: u64, h: u64) {
        self.put_n(lo, hi, h, &[], 2 | ((SPILLED as u32) << 2));
    }

    #[cold]
    #[inline(never)]
    fn wait_or_help_grow(&self) {
        while self.growing.load(Ordering::Acquire) {
            let jp = self.grow_job.load(Ordering::Acquire);
            if !jp.is_null() {
                let job = unsafe { &*jp };
                job.help();
                let mut sp = 0u32;
                while !job.finished.load(Ordering::Acquire) {
                    spin_or_yield(&mut sp);
                }
                break;
            }
            // Initiator is allocating/madvise-ing the new table: yield instead of burning SMT/CPU.
            std::thread::yield_now();
        }
    }

    fn put_n(&self, lo: u64, hi: u64, h: u64, toks: &[u32], pub_meta: u32) {
        if self.max_bits.load(Ordering::Relaxed) == 0 {
            return;
        }
        let t = self.table();
        let count = t.count.0.load(Ordering::Relaxed);
        if count * 8 >= t.len * 5 {
            if t.bits() < self.max_bits.load(Ordering::Relaxed) {
                self.grow(t);
            } else {
                return;
            }
        }
        if self.growing.load(Ordering::Acquire) {
            self.wait_or_help_grow();
        }
        let t = self.table();
        let home = (h >> t.shift) as usize;
        for step in 0..PROBE_MAX {
            let s = t.slot(home + step);
            let mut m = s.meta.load(Ordering::Acquire);
            if m == 0 {
                match s.meta.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed) {
                    Ok(_) => {
                        s.lo.store(lo, Ordering::Relaxed);
                        s.hi.store(hi, Ordering::Relaxed);
                        for (k, d) in s.toks.iter().enumerate() {
                            d.store(toks.get(k).copied().unwrap_or(0), Ordering::Relaxed);
                        }
                        s.meta.store(pub_meta, Ordering::Release);
                        PENDING.with(|c| {
                            let v = c.get() + 1;
                            if v >= COUNT_BATCH {
                                t.count.0.fetch_add(v, Ordering::Relaxed);
                                c.set(0);
                            } else {
                                c.set(v);
                            }
                        });
                        return;
                    }
                    Err(cur) => m = cur,
                }
            }
            let mut sp = 0u32;
            while m == 1 {
                spin_or_yield(&mut sp);
                m = s.meta.load(Ordering::Acquire);
            }
            if m >= 2 && s.lo.load(Ordering::Relaxed) == lo && s.hi.load(Ordering::Relaxed) == hi {
                return;
            }
        }
    }

    #[cfg(test)]
    pub fn presize(&self, bits: u32) {
        let old = self.cur.swap(Box::into_raw(STable::<W>::new(bits)), Ordering::AcqRel);
        self.retired.lock().unwrap().push(old);
    }

    #[cold]
    fn grow(&self, old: &STable<W>) {
        self.grow_to(old, old.bits() + 1);
    }

    /// Cooperative lock-free disjoint-stripe doubling: every home slot `h_old`
    /// maps to `[h_old * scale .. (h_old + 1) * scale)`, so all threads reaching
    /// `put_n` during growth cooperatively copy disjoint 4096-slot stripes with
    /// zero CAS and zero lost inserts.
    #[cold]
    #[inline(never)]
    fn grow_to(&self, old: &STable<W>, bits: u32) {
        if self.growing.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            return;
        }
        let cur = self.cur.load(Ordering::Acquire);
        if !std::ptr::eq(cur, old) || unsafe { &*cur }.bits() >= bits {
            self.growing.store(false, Ordering::Release);
            return;
        }
        #[cfg(test)]
        let t0 = std::time::Instant::now();
        let newp = Box::into_raw(STable::<W>::new(bits));
        let diff = bits - old.bits();
        let stripe = 4096usize.min(old.len).max(1);
        let n_stripes = (old.len + PROBE_MAX).div_ceil(stripe);
        let jobp = Box::into_raw(Box::new(GrowJob {
            old: old as *const STable<W>,
            new: newp,
            next_stripe: AtomicUsize::new(0),
            done_stripes: AtomicUsize::new(0),
            copied: AtomicUsize::new(0),
            n_stripes,
            stripe,
            diff,
            overflow: std::sync::Mutex::new(Vec::new()),
            finished: AtomicBool::new(false),
        }));
        self.grow_job.store(jobp, Ordering::Release);
        let job = unsafe { &*jobp };
        job.help();
        let mut sp = 0u32;
        while job.done_stripes.load(Ordering::Acquire) < n_stripes {
            spin_or_yield(&mut sp);
        }
        let new = unsafe { &*newp };
        let mut total = job.copied.load(Ordering::Relaxed);
        let ov = std::mem::take(&mut *job.overflow.lock().unwrap_or_else(|e| e.into_inner()));
        for (lo, hi, tv, m, h) in ov {
            let home = (h >> new.shift) as usize;
            for step in 0..PROBE_MAX {
                let d = new.slot(home + step);
                if d.meta.load(Ordering::Relaxed) == 0 {
                    d.lo.store(lo, Ordering::Relaxed);
                    d.hi.store(hi, Ordering::Relaxed);
                    for (x, &y) in d.toks.iter().zip(tv.iter()) {
                        x.store(y, Ordering::Relaxed);
                    }
                    d.meta.store(m, Ordering::Relaxed);
                    total += 1;
                    break;
                }
            }
        }
        new.count.0.store(total, Ordering::Relaxed);
        #[cfg(test)]
        if std::env::var("TT_DIAG").is_ok() {
            eprintln!("grow -> 2^{} ({total} entries) {:?}", new.bits(), t0.elapsed());
        }
        self.cur.store(newp, Ordering::Release);
        self.grow_job.store(std::ptr::null_mut(), Ordering::Release);
        job.finished.store(true, Ordering::Release);
        self.retired.lock().unwrap_or_else(|e| e.into_inner()).push(cur);
        self.retired_jobs.lock().unwrap_or_else(|e| e.into_inner()).push(jobp);
        self.growing.store(false, Ordering::Release);
    }
}

impl SharedTable<3> {
    /// Lookup returning up to 6 inline tokens (uncompressed 1..=3 OR 20-bit packed
    /// 4..=6), or count == `SPILLED` when stored in `SpillMap`.
    #[inline(always)]
    pub fn get_6(&self, lo: u64, hi: u64, h: u64) -> Option<([u32; 6], usize)> {
        let t = self.table();
        let home = (h >> t.shift) as usize;
        for step in 0..PROBE_MAX {
            let s = t.slot(home + step);
            let m = View::load_meta(s);
            if m == 0 {
                return None;
            }
            if m >= 2 && s.lo.load(Ordering::Relaxed) == lo && s.hi.load(Ordering::Relaxed) == hi {
                let mut v = [0u32; 6];
                if (m & 3) == 2 {
                    v[0] = s.toks[0].load(Ordering::Relaxed);
                    v[1] = s.toks[1].load(Ordering::Relaxed);
                    v[2] = s.toks[2].load(Ordering::Relaxed);
                    return Some((v, (m >> 2) as usize));
                }
                if (m & 3) == 3 {
                    let c = unsafe { unpack_packed6(s as *const SSlot<3>, m, v.as_mut_ptr()) };
                    return Some((v, c));
                }
            }
        }
        None
    }

    /// Insert 1..=3 tokens uncompressed or 4..=6 tokens (<= 20 bits) bit-packed
    /// in the same 32-byte slot. Returns `true` if stored inline, `false` if
    /// marked spilled (caller should store in `dyn_cache`/`spill`).
    #[inline]
    pub fn put_6(&self, lo: u64, hi: u64, h: u64, toks: &[u32]) -> bool {
        let n = toks.len();
        if n >= 1 && n <= 3 {
            self.put_n(lo, hi, h, toks, 2 | ((n as u32) << 2));
            return true;
        }
        if let Some((words, pub_meta)) = pack_6_words(toks) {
            self.put_n(lo, hi, h, &words, pub_meta);
            return true;
        }
        self.put_spilled(lo, hi, h);
        false
    }
}

impl<const W: usize> Drop for SharedTable<W> {
    fn drop(&mut self) {
        // SAFETY: exclusive access; every pointer came from Box::into_raw and is
        // freed exactly once.
        unsafe {
            drop(Box::from_raw(*self.cur.get_mut()));
            for p in self.retired.get_mut().unwrap_or_else(|e| e.into_inner()).drain(..) {
                drop(Box::from_raw(p));
            }
            for jp in self.retired_jobs.get_mut().unwrap_or_else(|e| e.into_inner()).drain(..) {
                drop(Box::from_raw(jp));
            }
        }
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;

    #[test]
    fn shared_slot_sizes() {
        assert_eq!(std::mem::size_of::<SSlot<3>>(), 32);
        assert_eq!(std::mem::size_of::<SSlot<11>>(), 64);
        let t: SharedTable<11> = SharedTable::new(8);
        let v: Vec<u32> = (1..=11).collect();
        t.put(5, 1 << 56, hash(5, 1 << 56), &v);
        assert_eq!(t.get(5, 1 << 56, hash(5, 1 << 56)), Some(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 11)));
    }

    /// At the memory cap the table stops inserting at 5/8 load instead of
    /// filling up (which would make every probe walk long clusters).
    #[test]
    fn shared_table_stops_at_cap() {
        let t: SharedTable = SharedTable::new(1); // 1 MiB = 32768 slots
        for k in 1..200_000u64 {
            t.put(k, 1 << 56, hash(k, 1 << 56), &[1]);
        }
        let slots = (1usize << 20) / 32;
        assert!(t.len() <= slots * 5 / 8 + 1, "len {} > 5/8 of {slots}", t.len());
        assert!(t.len() >= slots / 2);
        // what is stored is still found
        let mut found = 0;
        for k in 1..200_000u64 {
            if t.get(k, 1 << 56, hash(k, 1 << 56)).is_some() { found += 1; }
        }
        assert_eq!(found, t.len());
    }

    #[test]
    fn shared_table_concurrent_insert_get() {
        let t: SharedTable = SharedTable::new(64);
        let keys: Vec<(u64, u64)> = (0..200_000u64).map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15), (i % 15 + 1) << 56)).collect();
        std::thread::scope(|sc| {
            for w in 0..8 {
                let (t, keys) = (&t, &keys);
                sc.spawn(move || {
                    for (j, &(lo, hi)) in keys.iter().enumerate() {
                        if j % 8 == w {
                            let v = [j as u32, (j as u32) ^ 0xABCD, 7];
                            t.put(lo, hi, hash(lo, hi), &v[..(j % 3) + 1]);
                        }
                        // concurrent readers must only ever see exact entries
                        if let Some((v, n)) = t.get(lo, hi, hash(lo, hi)) {
                            assert_eq!(n, (j % 3) + 1);
                            assert_eq!(v[0], j as u32);
                            if n > 1 { assert_eq!(v[1], (j as u32) ^ 0xABCD); }
                        }
                    }
                });
            }
        });
        // every entry present after the concurrent phase is exact (coverage under
        // concurrent growth is best-effort: a lost insert only costs a re-merge)
        for (j, &(lo, hi)) in keys.iter().enumerate() {
            if let Some((v, n)) = t.get(lo, hi, hash(lo, hi)) {
                assert_eq!((v[0], n), (j as u32, (j % 3) + 1));
            }
        }
        // a quiet re-insert pass fills the gaps: then (nearly) everything is found
        for (j, &(lo, hi)) in keys.iter().enumerate() {
            let v = [j as u32, (j as u32) ^ 0xABCD, 7];
            t.put(lo, hi, hash(lo, hi), &v[..(j % 3) + 1]);
        }
        let found = keys.iter().filter(|&&(lo, hi)| t.get(lo, hi, hash(lo, hi)).is_some()).count();
        assert!(found * 100 >= keys.len() * 99, "found {found} of {}", keys.len());
    }

    #[test]
    fn shared_table_adaptive_packed6() {
        let t: SharedTable<3> = SharedTable::new(32);
        for cnt in 1..=6usize {
            let key = format!("chunk_{cnt:02}_abc");
            let (lo, hi) = pack(key.as_bytes());
            let h = hash(lo, hi);
            let want: Vec<u32> = (0..cnt as u32).map(|i| (i * 199_991 + 17) & PACK20_MAX_TOK).collect();
            assert!(t.put_6(lo, hi, h, &want));
            let (got, gn) = t.get_6(lo, hi, h).unwrap();
            assert_eq!((gn, &got[..gn]), (cnt, want.as_slice()));
            let view = t.view();
            let mut out = [0u32; 8];
            let InlineHit::Hit(vc) =
                (unsafe { view.emit_inline_at(lo, hi, view.slot_addr(h), out.as_mut_ptr()) })
            else {
                panic!("expected InlineHit::Hit");
            };
            assert_eq!((vc, &out[..vc]), (cnt, want.as_slice()));
        }
        // 7 tokens -> spilled
        let (lo, hi) = pack(b"seven_tokens_xx");
        let h = hash(lo, hi);
        assert!(!t.put_6(lo, hi, h, &[1, 2, 3, 4, 5, 6, 7]));
        assert_eq!(t.get_6(lo, hi, h).unwrap().1, SPILLED);
        // 4 tokens with >20-bit token -> spilled safely
        let (lo2, hi2) = pack(b"big_tok_id_test");
        let h2 = hash(lo2, hi2);
        assert!(!t.put_6(lo2, hi2, h2, &[1, 2, 3, PACK20_MAX_TOK + 1]));
        assert_eq!(t.get_6(lo2, hi2, h2).unwrap().1, SPILLED);
        // 3 tokens with >20-bit token -> still stored inline uncompressed!
        let (lo3, hi3) = pack(b"big_3_tok_test!");
        let h3 = hash(lo3, hi3);
        assert!(t.put_6(lo3, hi3, h3, &[1, 2, PACK20_MAX_TOK + 99]));
        let (got3, gn3) = t.get_6(lo3, hi3, h3).unwrap();
        assert_eq!((gn3, &got3[..gn3]), (3, &[1, 2, PACK20_MAX_TOK + 99][..]));
    }
}

// ---------------------------------------------------------------------------
// Shared spill map: learned encodings that don't fit the inline table
// ---------------------------------------------------------------------------

/// Learned encodings of chunks the [`SharedTable`] can't hold — longer than 15
/// bytes or more than its inline token count (URLs, long words, numbers,
/// identifiers). Shared by all threads and kept for the tokenizer's lifetime
/// (up to a memory cap), behind the bounded per-thread caches.
///
/// These chunks are rare (~0.1% of chunks on web text) but each costs a full
/// merge, and their distinct count is large: a bounded per-thread cache that
/// is cleared when full re-merged ~58k of them on EVERY pass over 300 MB.
/// Sharded by key hash into 64 RwLock'd maps so concurrent readers rarely
/// touch the same lock word. Lookups and inserts never wait (try-lock; a busy
/// shard is a miss / a skipped insert), so the map can't stall a worker or
/// deadlock a forked child.
pub struct SpillMap {
    shards: Box<[std::sync::RwLock<ArenaMap>]>,
    bytes: PaddedCount,
    limit: AtomicUsize,
}

const SPILL_SHARDS: usize = 64;
#[cfg(test)]
pub static SPILL_GETS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
pub static SPILL_PUTS: AtomicU64 = AtomicU64::new(0);

impl SpillMap {
    pub fn new(limit_bytes: usize) -> Self {
        SpillMap {
            shards: (0..SPILL_SHARDS).map(|_| std::sync::RwLock::new(ArenaMap::default())).collect(),
            bytes: PaddedCount(AtomicUsize::new(0)),
            limit: AtomicUsize::new(limit_bytes),
        }
    }

    pub fn set_limit(&self, limit_bytes: usize) {
        self.limit.store(limit_bytes, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_into_h(&self, key: &[u8], h: u64, out: &mut Vec<u32>) -> bool {
        #[cfg(test)]
        SPILL_GETS.fetch_add(1, Ordering::Relaxed);
        let sh = &self.shards[(h >> 58) as usize];
        let g = match sh.try_read() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return false,
        };
        match g.get_h(key, h) {
            Some(t) => {
                out.extend_from_slice(t);
                true
            }
            None => false,
        }
    }

    /// Append the stored encoding of `key` to `out`; false if not stored.
    #[inline]
    pub fn get_into(&self, key: &[u8], out: &mut Vec<u32>) -> bool {
        self.get_into_h(key, hash_bytes(key), out)
    }

    pub fn put_h(&self, key: &[u8], h: u64, toks: &[u32]) -> bool {
        if key.len() > LONG_MAX_BYTES {
            return false;
        }
        let cost = key.len() + 4 * toks.len() + 24;
        if self.bytes.0.load(Ordering::Relaxed) + cost > self.limit.load(Ordering::Relaxed) {
            return false;
        }
        #[cfg(test)]
        SPILL_PUTS.fetch_add(1, Ordering::Relaxed);
        let sh = &self.shards[(h >> 58) as usize];
        let mut g = match sh.try_write() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return false,
        };
        if g.insert_h(key, h, toks) {
            self.bytes.0.fetch_add(cost, Ordering::Relaxed);
        }
        true
    }

    /// Store `key`'s encoding; false if it isn't stored (over the cap / too long).
    pub fn put(&self, key: &[u8], toks: &[u32]) -> bool {
        self.put_h(key, hash_bytes(key), toks)
    }
}

// ---------------------------------------------------------------------------
// Lock-free lazy SharedLongTable for 16..=31 byte chunks (99.8% of >15B words)
// ---------------------------------------------------------------------------

pub const MAX_LONG_KEY: usize = 31;
const LONG_MIN_BITS: u32 = 11; // 2048 slots * 64B = 128 KiB initial (lazy)
thread_local! {
    static PENDING_LONG: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Pack a 16..=31 byte slice into a lossless 32-byte key (`buf[31] = len`).
#[inline(always)]
pub fn pack32_at(b: &[u8], pos: usize, len: usize) -> [u64; 4] {
    debug_assert!(len >= 16 && len <= MAX_LONG_KEY);
    if pos + 32 <= b.len() {
        // SAFETY: 32 bytes readable at pos; `len - 16` is in 0..=15.
        unsafe {
            let p = b.as_ptr().add(pos);
            let (m0, m1) = *KEY_MASK64.get_unchecked(len - 16);
            [
                u64::from_le((p as *const u64).read_unaligned()),
                u64::from_le((p.add(8) as *const u64).read_unaligned()),
                u64::from_le((p.add(16) as *const u64).read_unaligned()) & m0,
                (u64::from_le((p.add(24) as *const u64).read_unaligned()) & m1)
                    | ((len as u64) << 56),
            ]
        }
    } else {
        pack32_tail(&b[pos..pos + len])
    }
}

#[cold]
#[inline(never)]
fn pack32_tail(c: &[u8]) -> [u64; 4] {
    let mut buf = [0u8; 32];
    buf[..c.len()].copy_from_slice(c);
    buf[31] = c.len() as u8;
    [
        u64::from_le_bytes(buf[0..8].try_into().unwrap()),
        u64::from_le_bytes(buf[8..16].try_into().unwrap()),
        u64::from_le_bytes(buf[16..24].try_into().unwrap()),
        u64::from_le_bytes(buf[24..32].try_into().unwrap()),
    ]
}

#[inline(always)]
pub fn hash32(k: [u64; 4]) -> u64 {
    let a = (k[0] ^ k[2].rotate_left(17)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let b = (k[1] ^ k[3].rotate_left(31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (a ^ b.rotate_left(29)).wrapping_mul(0x94D0_49BB_1331_11EB)
}

/// One 64-byte cache-line slot for a 16..=31 byte chunk:
/// - `k`: 32-byte lossless key (`len` in byte 31, always 16..=31)
/// - `toks`: 1..=7 uncompressed tokens (`meta & 3 == 2`), OR 8..=11 20-bit
///   packed tokens (`meta & 3 == 3`)
/// - `meta`: 0 = empty, 1 = being written, else published.
#[repr(C, align(64))]
struct LSlot {
    k: [AtomicU64; 4],
    toks: [AtomicU32; 7],
    meta: AtomicU32,
}
const _: () = assert!(std::mem::size_of::<LSlot>() == 64);

#[repr(C, align(128))]
struct LTable {
    ptr: *mut LSlot,
    len: usize,
    mask: usize,
    shift: u32,
    count: PaddedCount,
}

impl LTable {
    fn layout(n: usize) -> std::alloc::Layout {
        let bytes = 64 * n;
        #[cfg(target_os = "linux")]
        let align = if bytes >= (2 << 20) { 2 << 20 } else { 64 };
        #[cfg(not(target_os = "linux"))]
        let align = 64;
        std::alloc::Layout::from_size_align(bytes, align).expect("ltable layout")
    }

    fn new(bits: u32) -> Box<LTable> {
        let n = 1usize << bits;
        let layout = Self::layout(n);
        let ptr = unsafe { alloc_zeroed_huge(layout) } as *mut LSlot;
        Box::new(LTable { ptr, len: n, mask: n - 1, shift: 64 - bits, count: PaddedCount(AtomicUsize::new(0)) })
    }

    #[inline(always)]
    fn slot(&self, i: usize) -> &LSlot {
        unsafe { &*self.ptr.add(i) }
    }

    fn bits(&self) -> u32 {
        64 - self.shift
    }
}

impl Drop for LTable {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, Self::layout(self.len)) };
    }
}

#[inline(always)]
fn pack_11_words(toks: &[u32]) -> Option<([u32; 7], u32)> {
    let n = toks.len();
    if n < 8 || n > 11 {
        return None;
    }
    let mut or_all = 0u32;
    let mut t = [0u32; 11];
    for (i, &v) in toks.iter().enumerate() {
        t[i] = v;
        or_all |= v;
    }
    if or_all > PACK20_MAX_TOK {
        return None;
    }
    let w0 = (t[0] as u64) | ((t[1] as u64) << 20) | ((t[2] as u64) << 40) | (((t[9] & 0xF) as u64) << 60);
    let w1 = (t[3] as u64) | ((t[4] as u64) << 20) | ((t[5] as u64) << 40) | ((((t[9] >> 4) & 0xF) as u64) << 60);
    let w2 = (t[6] as u64) | ((t[7] as u64) << 20) | ((t[8] as u64) << 40) | ((((t[9] >> 8) & 0xF) as u64) << 60);
    let w3 = t[10] | (((t[9] >> 12) & 0xFF) << 20);
    let words = [
        w0 as u32,
        (w0 >> 32) as u32,
        w1 as u32,
        (w1 >> 32) as u32,
        w2 as u32,
        (w2 >> 32) as u32,
        w3,
    ];
    let meta = 3u32 | (((n - 8) as u32) << 2);
    Some((words, meta))
}

#[inline(always)]
unsafe fn unpack_11_into(s: &LSlot, m: u32, out: &mut Vec<u32>) {
    let n = 8 + (((m >> 2) & 3) as usize);
    let raw = [
        s.toks[0].load(Ordering::Relaxed),
        s.toks[1].load(Ordering::Relaxed),
        s.toks[2].load(Ordering::Relaxed),
        s.toks[3].load(Ordering::Relaxed),
        s.toks[4].load(Ordering::Relaxed),
        s.toks[5].load(Ordering::Relaxed),
        s.toks[6].load(Ordering::Relaxed),
    ];
    let w0 = (raw[0] as u64) | ((raw[1] as u64) << 32);
    let w1 = (raw[2] as u64) | ((raw[3] as u64) << 32);
    let w2 = (raw[4] as u64) | ((raw[5] as u64) << 32);
    let w3 = raw[6];
    let t9 = ((w0 >> 60) as u32)
        | (((w1 >> 60) as u32) << 4)
        | (((w2 >> 60) as u32) << 8)
        | (((w3 >> 20) & 0xFF) << 12);
    let t = [
        (w0 & 0xF_FFFF) as u32,
        ((w0 >> 20) & 0xF_FFFF) as u32,
        ((w0 >> 40) & 0xF_FFFF) as u32,
        (w1 & 0xF_FFFF) as u32,
        ((w1 >> 20) & 0xF_FFFF) as u32,
        ((w1 >> 40) & 0xF_FFFF) as u32,
        (w2 & 0xF_FFFF) as u32,
        ((w2 >> 20) & 0xF_FFFF) as u32,
        ((w2 >> 40) & 0xF_FFFF) as u32,
        t9,
        w3 & 0xF_FFFF,
    ];
    out.extend_from_slice(&t[..n]);
}

/// Lookup result for `SharedLongTable::get_into`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LongHit {
    Hit,
    Empty,
    Spilled,
}

/// Read-only snapshot view of [`SharedLongTable`] for 256-deep pipelined prefetching.
#[derive(Clone, Copy)]
pub struct LongView<'a> {
    base: *const LSlot,
    end: *const LSlot,
    shift: u32,
    _t: std::marker::PhantomData<&'a ()>,
}

impl LongView<'_> {
    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        !self.base.is_null()
    }

    /// True iff `table` has not grown since this `LongView` was captured (1 L1 load +
    /// pointer compare), letting `emit_long_at` skip a redundant second probe on cold misses.
    #[inline(always)]
    pub fn is_current(&self, table: &SharedLongTable) -> bool {
        let cur = table.cur.load(Ordering::Acquire);
        !cur.is_null() && std::ptr::eq(self.base, unsafe { (*cur).ptr as *const LSlot })
    }

    /// Address of the home 64-byte `LSlot` for hash `h`.
    #[inline(always)]
    pub fn slot_addr(&self, h: u64) -> u64 {
        debug_assert!(!self.base.is_null());
        unsafe { self.base.add((h >> self.shift) as usize) as u64 }
    }

    /// Probe starting from a prefetched `LSlot` address `addr` with unrolled
    /// Probe-0 and two-128-bit-store token copy for 1..=7 uncompressed tokens.
    #[inline(always)]
    pub unsafe fn get_into_at(&self, k: [u64; 4], addr: u64, out: &mut Vec<u32>) -> LongHit {
        const {
            assert!(std::mem::offset_of!(LSlot, toks) == 32);
            assert!(std::mem::size_of::<LSlot>() == 64);
        }
        // Probe 0: prefetched home 64B cache line.
        let mut p = addr as *const LSlot;
        let s0 = &*p;
        let m0 = SharedLongTable::load_meta(s0);
        if m0 == 0 {
            return LongHit::Empty;
        }
        if ((s0.k[0].load(Ordering::Relaxed) ^ k[0])
            | (s0.k[1].load(Ordering::Relaxed) ^ k[1])
            | (s0.k[2].load(Ordering::Relaxed) ^ k[2])
            | (s0.k[3].load(Ordering::Relaxed) ^ k[3]))
            == 0
        {
            if (m0 & 3) == 2 {
                let n = (m0 >> 2) as usize;
                if n <= 7 {
                    let base = out.len();
                    out.reserve(8);
                    let raw = ((p as *const u8).add(32) as *const [u32; 8]).read();
                    (out.as_mut_ptr().add(base) as *mut [u32; 8]).write_unaligned(raw);
                    out.set_len(base + n);
                    return LongHit::Hit;
                }
                return LongHit::Spilled;
            }
            if (m0 & 3) == 3 {
                unpack_11_into(s0, m0, out);
                return LongHit::Hit;
            }
        }
        // Probe 1..PROBE_MAX: linear scan.
        for _ in 1..PROBE_MAX {
            p = p.add(1);
            if p == self.end {
                p = self.base;
            }
            let s = &*p;
            let m = SharedLongTable::load_meta(s);
            if m == 0 {
                return LongHit::Empty;
            }
            if m >= 2
                && ((s.k[0].load(Ordering::Relaxed) ^ k[0])
                    | (s.k[1].load(Ordering::Relaxed) ^ k[1])
                    | (s.k[2].load(Ordering::Relaxed) ^ k[2])
                    | (s.k[3].load(Ordering::Relaxed) ^ k[3]))
                    == 0
            {
                if (m & 3) == 2 {
                    let n = (m >> 2) as usize;
                    if n <= 7 {
                        let base = out.len();
                        out.reserve(8);
                        let raw = ((p as *const u8).add(32) as *const [u32; 8]).read();
                        (out.as_mut_ptr().add(base) as *mut [u32; 8]).write_unaligned(raw);
                        out.set_len(base + n);
                        return LongHit::Hit;
                    }
                    return LongHit::Spilled;
                }
                if (m & 3) == 3 {
                    unpack_11_into(s, m, out);
                    return LongHit::Hit;
                }
            }
        }
        LongHit::Spilled
    }
}

struct LGrowJob {
    old: *const LTable,
    new: *mut LTable,
    next_stripe: AtomicUsize,
    done_stripes: AtomicUsize,
    copied: AtomicUsize,
    n_stripes: usize,
    stripe: usize,
    diff: u32,
    overflow: std::sync::Mutex<Vec<([u64; 4], [u32; 7], u32, u64)>>,
    finished: AtomicBool,
}

unsafe impl Send for LGrowJob {}
unsafe impl Sync for LGrowJob {}

impl LGrowJob {
    #[inline(never)]
    fn help(&self) {
        let old_t = unsafe { &*self.old };
        let new_t = unsafe { &*self.new };
        let mut local_copied = 0usize;
        let mut local_ov: Vec<([u64; 4], [u32; 7], u32, u64)> = Vec::new();
        loop {
            let si = self.next_stripe.fetch_add(1, Ordering::Relaxed);
            if si >= self.n_stripes {
                break;
            }
            let s_lo = si * self.stripe;
            let s_hi = (s_lo + self.stripe).min(old_t.len);
            let d_lo = s_lo << self.diff;
            let d_hi = s_hi << self.diff;
            for i in s_lo..s_hi {
                let s = old_t.slot(i);
                let mut m = s.meta.load(Ordering::Acquire);
                let mut sp = 0u32;
                while m == 1 {
                    spin_or_yield(&mut sp);
                    m = s.meta.load(Ordering::Acquire);
                }
                if m < 2 {
                    continue;
                }
                let k = [
                    s.k[0].load(Ordering::Relaxed),
                    s.k[1].load(Ordering::Relaxed),
                    s.k[2].load(Ordering::Relaxed),
                    s.k[3].load(Ordering::Relaxed),
                ];
                let h = hash32(k);
                let mut j = (h >> new_t.shift) as usize;
                if j >= d_lo && j < d_hi {
                    while j < d_hi {
                        let d = new_t.slot(j);
                        if d.meta.load(Ordering::Relaxed) == 0 {
                            for x in 0..4 {
                                d.k[x].store(k[x], Ordering::Relaxed);
                            }
                            for x in 0..7 {
                                d.toks[x].store(s.toks[x].load(Ordering::Relaxed), Ordering::Relaxed);
                            }
                            d.meta.store(m, Ordering::Relaxed);
                            local_copied += 1;
                            break;
                        }
                        j += 1;
                    }
                    if j == d_hi {
                        let mut tv = [0u32; 7];
                        for x in 0..7 {
                            tv[x] = s.toks[x].load(Ordering::Relaxed);
                        }
                        local_ov.push((k, tv, m, h));
                    }
                } else {
                    let mut tv = [0u32; 7];
                    for x in 0..7 {
                        tv[x] = s.toks[x].load(Ordering::Relaxed);
                    }
                    local_ov.push((k, tv, m, h));
                }
            }
            if !local_ov.is_empty() {
                let mut g = self.overflow.lock().unwrap_or_else(|e| e.into_inner());
                g.append(&mut local_ov);
            }
            self.done_stripes.fetch_add(1, Ordering::Release);
        }
        if local_copied != 0 {
            self.copied.fetch_add(local_copied, Ordering::Relaxed);
        }
    }
}

/// Lock-free, lazy-allocated shared table for 16..=31 byte chunks.
/// Allocates 0 bytes until the first 16..=31 byte chunk is stored, then starts
/// at 128 KiB and doubles with cooperative lock-free disjoint stripes up to `max_mib`.
/// Eliminates per-thread `LongCache` wipes and `SpillMap` RwLock contention
/// on German/compound-word corpora while staying tiny on English.
pub struct SharedLongTable {
    cur: AtomicPtr<LTable>,
    grow_job: AtomicPtr<LGrowJob>,
    retired: std::sync::Mutex<Vec<*mut LTable>>,
    retired_jobs: std::sync::Mutex<Vec<*mut LGrowJob>>,
    growing: AtomicBool,
    max_bits: AtomicU32,
}

unsafe impl Send for SharedLongTable {}
unsafe impl Sync for SharedLongTable {}

impl SharedLongTable {
    pub fn new(max_mib: usize) -> Self {
        let t = SharedLongTable {
            cur: AtomicPtr::new(std::ptr::null_mut()),
            grow_job: AtomicPtr::new(std::ptr::null_mut()),
            retired: std::sync::Mutex::new(Vec::new()),
            retired_jobs: std::sync::Mutex::new(Vec::new()),
            growing: AtomicBool::new(false),
            max_bits: AtomicU32::new(LONG_MIN_BITS),
        };
        t.set_limit_mib(max_mib);
        t
    }

    pub fn set_limit_mib(&self, max_mib: usize) {
        let slots = (max_mib << 20) / 64;
        let bits = if slots < (1 << LONG_MIN_BITS) {
            0
        } else {
            usize::BITS - 1 - slots.leading_zeros()
        };
        self.max_bits.store(bits, Ordering::Relaxed);
    }

    #[inline(always)]
    fn load_meta(s: &LSlot) -> u32 {
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            let val: u32;
            unsafe {
                std::arch::asm!(
                    "ldapr {val:w}, [{ptr}]",
                    val = out(reg) val,
                    ptr = in(reg) &s.meta,
                    options(nostack, readonly, preserves_flags),
                );
            }
            val
        }
        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        {
            s.meta.load(Ordering::Acquire)
        }
    }

    #[inline(always)]
    pub fn view(&self) -> LongView<'_> {
        let tp = self.cur.load(Ordering::Acquire);
        if tp.is_null() {
            return LongView {
                base: std::ptr::null(),
                end: std::ptr::null(),
                shift: 0,
                _t: std::marker::PhantomData,
            };
        }
        let t = unsafe { &*tp };
        LongView {
            base: t.ptr as *const LSlot,
            end: unsafe { (t.ptr as *const LSlot).add(t.len) },
            shift: t.shift,
            _t: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn get_into(&self, k: [u64; 4], h: u64, out: &mut Vec<u32>) -> LongHit {
        let v = self.view();
        if !v.is_valid() {
            return LongHit::Empty;
        }
        unsafe { v.get_into_at(k, v.slot_addr(h), out) }
    }

    #[cold]
    #[inline(never)]
    fn wait_or_help_grow(&self) {
        while self.growing.load(Ordering::Acquire) {
            let jp = self.grow_job.load(Ordering::Acquire);
            if !jp.is_null() {
                let job = unsafe { &*jp };
                job.help();
                let mut sp = 0u32;
                while !job.finished.load(Ordering::Acquire) {
                    spin_or_yield(&mut sp);
                }
                break;
            }
            // Initiator is allocating/madvise-ing the new table: yield instead of burning SMT/CPU.
            std::thread::yield_now();
        }
    }

    /// Store 1..=11 tokens inline (1..=7 uncompressed, 8..=11 20-bit packed),
    /// or record a `SPILLED` marker and return `false` so caller uses SpillMap.
    pub fn put(&self, k: [u64; 4], h: u64, toks: &[u32]) -> bool {
        let max_b = self.max_bits.load(Ordering::Relaxed);
        if max_b == 0 {
            return false;
        }
        let mut tp = self.cur.load(Ordering::Acquire);
        if tp.is_null() {
            tp = self.init_table();
            if tp.is_null() {
                return false;
            }
        }
        let t = unsafe { &*tp };
        if t.count.0.load(Ordering::Relaxed) * 8 >= t.len * 5 {
            if t.bits() < max_b {
                self.grow_to(t, t.bits() + 1);
            } else {
                return false;
            }
        }
        if self.growing.load(Ordering::Acquire) {
            self.wait_or_help_grow();
        }
        let t = unsafe { &*self.cur.load(Ordering::Acquire) };
        let (words, pub_meta, inline) = if !toks.is_empty() && toks.len() <= 7 {
            let mut w = [0u32; 7];
            w[..toks.len()].copy_from_slice(toks);
            (w, 2 | ((toks.len() as u32) << 2), true)
        } else if let Some((w, m)) = pack_11_words(toks) {
            (w, m, true)
        } else {
            ([0u32; 7], 2 | ((SPILLED as u32) << 2), false)
        };
        let mut i = (h >> t.shift) as usize;
        for _ in 0..PROBE_MAX {
            let s = t.slot(i);
            let mut m = s.meta.load(Ordering::Acquire);
            if m == 0 {
                match s.meta.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed) {
                    Ok(_) => {
                        for j in 0..4 {
                            s.k[j].store(k[j], Ordering::Relaxed);
                        }
                        for j in 0..7 {
                            s.toks[j].store(words[j], Ordering::Relaxed);
                        }
                        s.meta.store(pub_meta, Ordering::Release);
                        PENDING_LONG.with(|c| {
                            let v = c.get() + 1;
                            if v >= 16 {
                                t.count.0.fetch_add(v, Ordering::Relaxed);
                                c.set(0);
                            } else {
                                c.set(v);
                            }
                        });
                        return inline;
                    }
                    Err(cur) => m = cur,
                }
            }
            let mut sp = 0u32;
            while m == 1 {
                spin_or_yield(&mut sp);
                m = s.meta.load(Ordering::Acquire);
            }
            if m >= 2
                && s.k[0].load(Ordering::Relaxed) == k[0]
                && s.k[1].load(Ordering::Relaxed) == k[1]
                && s.k[2].load(Ordering::Relaxed) == k[2]
                && s.k[3].load(Ordering::Relaxed) == k[3]
            {
                return inline;
            }
            i = (i + 1) & t.mask;
        }
        false
    }

    #[cold]
    #[inline(never)]
    fn init_table(&self) -> *mut LTable {
        if self.growing.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            while self.growing.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            return self.cur.load(Ordering::Acquire);
        }
        let cur = self.cur.load(Ordering::Acquire);
        let out = if cur.is_null() {
            let p = Box::into_raw(LTable::new(LONG_MIN_BITS));
            self.cur.store(p, Ordering::Release);
            p
        } else {
            cur
        };
        self.growing.store(false, Ordering::Release);
        out
    }

    #[cold]
    #[inline(never)]
    fn grow_to(&self, old: &LTable, bits: u32) {
        if self.growing.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            return;
        }
        let cur = self.cur.load(Ordering::Acquire);
        if !std::ptr::eq(cur, old) || unsafe { &*cur }.bits() >= bits {
            self.growing.store(false, Ordering::Release);
            return;
        }
        let newp = Box::into_raw(LTable::new(bits));
        let diff = bits - old.bits();
        let stripe = 2048usize.min(old.len).max(1);
        let n_stripes = old.len.div_ceil(stripe);
        let jobp = Box::into_raw(Box::new(LGrowJob {
            old: old as *const LTable,
            new: newp,
            next_stripe: AtomicUsize::new(0),
            done_stripes: AtomicUsize::new(0),
            copied: AtomicUsize::new(0),
            n_stripes,
            stripe,
            diff,
            overflow: std::sync::Mutex::new(Vec::new()),
            finished: AtomicBool::new(false),
        }));
        self.grow_job.store(jobp, Ordering::Release);
        let job = unsafe { &*jobp };
        job.help();
        let mut sp = 0u32;
        while job.done_stripes.load(Ordering::Acquire) < n_stripes {
            spin_or_yield(&mut sp);
        }
        let new = unsafe { &*newp };
        let mut total = job.copied.load(Ordering::Relaxed);
        let ov = std::mem::take(&mut *job.overflow.lock().unwrap_or_else(|e| e.into_inner()));
        for (k, tv, m, h) in ov {
            let mut j = (h >> new.shift) as usize;
            loop {
                let d = new.slot(j);
                if d.meta.load(Ordering::Relaxed) == 0 {
                    for x in 0..4 {
                        d.k[x].store(k[x], Ordering::Relaxed);
                    }
                    for x in 0..7 {
                        d.toks[x].store(tv[x], Ordering::Relaxed);
                    }
                    d.meta.store(m, Ordering::Relaxed);
                    total += 1;
                    break;
                }
                j = (j + 1) & new.mask;
            }
        }
        new.count.0.store(total, Ordering::Relaxed);
        self.cur.store(newp, Ordering::Release);
        self.grow_job.store(std::ptr::null_mut(), Ordering::Release);
        job.finished.store(true, Ordering::Release);
        self.retired.lock().unwrap_or_else(|e| e.into_inner()).push(cur);
        self.retired_jobs.lock().unwrap_or_else(|e| e.into_inner()).push(jobp);
        self.growing.store(false, Ordering::Release);
    }
}

impl Drop for SharedLongTable {
    fn drop(&mut self) {
        unsafe {
            let cur = *self.cur.get_mut();
            if !cur.is_null() {
                drop(Box::from_raw(cur));
            }
            for p in self.retired.get_mut().unwrap_or_else(|e| e.into_inner()).drain(..) {
                drop(Box::from_raw(p));
            }
            for jp in self.retired_jobs.get_mut().unwrap_or_else(|e| e.into_inner()).drain(..) {
                drop(Box::from_raw(jp));
            }
        }
    }
}

#[cfg(test)]
mod long_table_tests {
    use super::*;

    #[test]
    fn shared_long_table_roundtrip_and_pack11() {
        let t = SharedLongTable::new(16);
        let mut out = Vec::new();
        for len in 16..=31usize {
            let key: Vec<u8> = (0..len).map(|i| b'a' + ((i + len) % 26) as u8).collect();
            let k = pack32_at(&key, 0, len);
            // Also verify wide load path when padded with >=32 bytes
            let mut padded = key.clone();
            padded.extend_from_slice(&[0xFF; 32]);
            assert_eq!(k, pack32_at(&padded, 0, len));
            let h = hash32(k);
            let ntok = 1 + (len % 11); // tests 1..=11 tokens!
            let want: Vec<u32> = (0..ntok as u32).map(|i| (i * 173_111 + len as u32) & PACK20_MAX_TOK).collect();
            assert_eq!(t.get_into(k, h, &mut out), LongHit::Empty);
            assert!(t.put(k, h, &want));
            out.clear();
            assert_eq!(t.get_into(k, h, &mut out), LongHit::Hit);
            assert_eq!(out, want);
        }
        // 12 tokens -> spilled
        let key12 = b"twelve_tokens_long_word!";
        let k12 = pack32_at(key12, 0, key12.len());
        let h12 = hash32(k12);
        assert!(!t.put(k12, h12, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
        assert_eq!(t.get_into(k12, h12, &mut out), LongHit::Spilled);
    }

    #[test]
    fn shared_long_table_concurrent_grow() {
        let t = SharedLongTable::new(32);
        let keys: Vec<([u64; 4], u64)> = (0..40_000u64)
            .map(|i| {
                let k = [
                    i.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    i.wrapping_mul(0xBF58_476D_1CE4_E5B9),
                    i ^ 0x1234_5678,
                    (20u64 << 56) | (i & 0xFFFF_FFFF),
                ];
                (k, hash32(k))
            })
            .collect();
        std::thread::scope(|sc| {
            for w in 0..8 {
                let (t, keys) = (&t, &keys);
                sc.spawn(move || {
                    let mut out = Vec::new();
                    for (j, &(k, h)) in keys.iter().enumerate() {
                        if j % 8 == w || j < 200 {
                            let n = 1 + (j % 11);
                            let v: Vec<u32> = (0..n as u32).map(|x| ((j as u32) * 7 + x) & PACK20_MAX_TOK).collect();
                            let _ = t.put(k, h, &v);
                        }
                        out.clear();
                        if t.get_into(k, h, &mut out) == LongHit::Hit {
                            let n = 1 + (j % 11);
                            assert_eq!(out.len(), n);
                            assert_eq!(out[0], ((j as u32) * 7) & PACK20_MAX_TOK);
                        }
                    }
                });
            }
        });
        let mut out = Vec::new();
        let mut found = 0usize;
        for (j, &(k, h)) in keys.iter().enumerate() {
            out.clear();
            if t.get_into(k, h, &mut out) == LongHit::Hit {
                assert_eq!(out.len(), 1 + (j % 11));
                found += 1;
            }
        }
        assert!(found * 100 >= keys.len() * 99, "found only {found} of {}", keys.len());
    }
}
