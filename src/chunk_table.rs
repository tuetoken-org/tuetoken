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

#[cfg(test)]
pub static DENSE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


pub struct ChunkTable {
    entries: Vec<Entry>,
    shift: u32,
    mask: usize,
}

/// Pack ≤ 15 bytes into the (lo, hi) key. Caller guarantees 1 ≤ len ≤ 15.
#[inline(always)]
pub fn pack(chunk: &[u8]) -> (u64, u64) {
    let n = chunk.len();
    debug_assert!(n >= 1 && n <= MAX_KEY);
    let mut buf = [0u8; 16];
    buf[..n].copy_from_slice(chunk);
    let lo = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let hi = u64::from_le_bytes(buf[8..16].try_into().unwrap()) | ((n as u64) << 56);
    (lo, hi)
}

/// Pack `len` bytes starting at `p` when at least 16 bytes are readable from `p`
/// (one unaligned 16-byte load + mask instead of a variable-length memcpy).
///
/// SAFETY: `p .. p+16` must be readable; 1 ≤ len ≤ 15.
#[inline(always)]
pub unsafe fn pack_wide(p: *const u8, len: usize) -> (u64, u64) {
    debug_assert!(len >= 1 && len <= MAX_KEY);
    // one 16-byte load, keep the first `len` bytes (branch-free; len <= 15 so the
    // shift is < 128)
    let raw = u128::from_le((p as *const u128).read_unaligned());
    let k = raw & ((1u128 << (8 * len)) - 1);
    (k as u64, ((k >> 64) as u64) | ((len as u64) << 56))
}

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
    // Two multiplies; callers use the TOP bits, which mix every input bit.
    (lo.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ hi).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
}

impl ChunkTable {
    pub fn empty() -> Self {
        ChunkTable { entries: vec![Entry::default(); 2], shift: 63, mask: 1 }
    }

    /// Build from verified (key bytes, token) pairs. Duplicate keys keep the first.
    pub fn build(pairs: &[(&[u8], u32)]) -> Self {
        let mut cap = 16usize;
        #[cfg(test)]
        let dense = DENSE.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let dense = false;
        while if dense { cap * 3 < pairs.len() * 4 } else { cap < pairs.len() * 2 } {
            cap <<= 1;
        }
        let bits = cap.trailing_zeros();
        let mut t = ChunkTable {
            entries: vec![Entry::default(); cap],
            shift: 64 - bits,
            mask: cap - 1,
        };
        for (k, v) in pairs {
            let (lo, hi) = pack(k);
            let mut i = (hash(lo, hi) >> t.shift) as usize;
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

    #[inline(always)]
    pub fn get_h(&self, lo: u64, hi: u64, h: u64) -> Option<u32> {
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

/// Per-thread exact cache for chunks LONGER than `MAX_KEY` bytes (long words,
/// identifiers, punctuation runs): ~0.1% of chunks, but each costs a full merge
/// of hundreds of ns. Keyed by the chunk bytes (exact compare); results live in
/// an arena. One small map per tokenizer (the 4 most recently used), so a thread
/// alternating between models keeps them all warm. Bounded per map; a full map
/// is cleared.
#[derive(Default)]
struct LongMap {
    owner: u64,
    map: rustc_hash::FxHashMap<Box<[u8]>, (u32, u32)>, // bytes -> (offset, len)
    arena: Vec<u32>,
}

#[derive(Default)]
pub struct LongCache {
    maps: Vec<LongMap>, // most recently claimed first
}

const LONG_MAPS: usize = 4;
const LONG_MAX_ENTRIES: usize = 1 << 14;
const LONG_MAX_ARENA: usize = 1 << 20; // u32s (4 MiB)
const LONG_MAX_BYTES: usize = 1024; // longer chunks are not worth caching

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
    pub fn get(&self, owner: u64, key: &[u8]) -> Option<&[u32]> {
        let m = self.maps.first().filter(|m| m.owner == owner)?;
        let &(off, len) = m.map.get(key)?;
        Some(&m.arena[off as usize..(off + len) as usize])
    }

    #[inline]
    pub fn put(&mut self, owner: u64, key: &[u8], toks: &[u32]) {
        let Some(m) = self.maps.first_mut().filter(|m| m.owner == owner) else { return };
        if key.len() > LONG_MAX_BYTES {
            return;
        }
        if m.map.len() >= LONG_MAX_ENTRIES || m.arena.len() + toks.len() > LONG_MAX_ARENA {
            m.map.clear();
            m.arena.clear();
        }
        let off = m.arena.len() as u32;
        m.arena.extend_from_slice(toks);
        m.map.insert(key.into(), (off, toks.len() as u32));
    }
}

// ---------------------------------------------------------------------------
// Shared learned table
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};

/// One slot, 32 bytes. `meta`: 0 = empty, 1 = being written, else
/// `2 | n << 2` = published with `n` (1..=3) tokens. A slot is written exactly
/// once (claimed by CAS 0 -> 1, filled, then published with a Release store),
/// and never changes afterwards, so a reader that sees it published (Acquire)
/// reads a consistent entry.
#[repr(C, align(32))]
struct SSlot {
    meta: AtomicU32,
    t2: AtomicU32,
    lo: AtomicU64,
    hi: AtomicU64,
    t01: AtomicU64,
}

const _: () = assert!(std::mem::size_of::<SSlot>() == 32);

struct STable {
    slots: Vec<SSlot>,
    mask: usize,
    shift: u32,
    count: AtomicUsize,
}

impl STable {
    fn new(bits: u32) -> Box<STable> {
        let n = 1usize << bits;
        let layout = std::alloc::Layout::array::<SSlot>(n).expect("table layout");
        // SAFETY: all-zero bytes are a valid SSlot (atomics of 0 = empty); the
        // Vec owns exactly this layout (len == cap == n, SSlot's alignment).
        let slots = unsafe {
            let p = std::alloc::alloc_zeroed(layout) as *mut SSlot;
            if p.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Vec::from_raw_parts(p, n, n)
        };
        Box::new(STable { slots, mask: n - 1, shift: 64 - bits, count: AtomicUsize::new(0) })
    }

    fn bits(&self) -> u32 {
        64 - self.shift
    }
}

const PROBE_MAX: usize = 32;
const SHARED_MIN_BITS: u32 = 14; // 512 KiB

/// Grow-only, lock-free table of LEARNED chunk encodings (multi-token results
/// the load-time table can't hold), shared by every thread using a tokenizer.
///
/// Natural text has a very long tail of distinct multi-token pre-tokens (names,
/// numbers, URL pieces, typos). Bounded per-thread caches keep evicting them —
/// and each thread re-learns what the others already merged. This table keeps
/// everything any thread has learned, growing on demand up to a memory cap.
///
/// * Reads take no lock: one Acquire load of the table pointer, then linear
///   probing over published slots.
/// * Inserts claim an empty slot with one CAS; entries are never overwritten.
///   Two threads racing on the same key may both insert it (harmless).
/// * Growth: when half full, one thread builds a 2x table, copies the entries
///   and swaps the pointer. Inserts that land in the old table meanwhile may be
///   lost — fine, because every entry is just `f(bytes)` for the exact merge
///   function: a missing entry only means that chunk gets merged again.
/// * Old tables stay allocated until the tokenizer is dropped (so a concurrent
///   reader can never see freed memory); geometric growth bounds that overhead
///   to the size of the final table.
pub struct SharedTable {
    cur: AtomicPtr<STable>,
    retired: std::sync::Mutex<Vec<*mut STable>>,
    growing: AtomicBool,
    max_bits: AtomicU32,
}

// SAFETY: all shared state is atomic or behind the Mutex; the raw table
// pointers are owned by this struct and freed only in Drop (exclusive access).
unsafe impl Send for SharedTable {}
unsafe impl Sync for SharedTable {}

impl SharedTable {
    /// `max_mib`: memory cap of the live table (0 disables learning).
    pub fn new(max_mib: usize) -> Self {
        let t = SharedTable {
            cur: AtomicPtr::new(Box::into_raw(STable::new(SHARED_MIN_BITS))),
            retired: std::sync::Mutex::new(Vec::new()),
            growing: AtomicBool::new(false),
            max_bits: AtomicU32::new(SHARED_MIN_BITS),
        };
        t.set_limit_mib(max_mib);
        t
    }

    /// Change the memory cap (MiB) of the live table. 0 = stop learning.
    pub fn set_limit_mib(&self, max_mib: usize) {
        let slots = (max_mib << 20) / std::mem::size_of::<SSlot>();
        let bits = if slots < 2 { 0 } else { usize::BITS - 1 - slots.leading_zeros() };
        self.max_bits.store(bits, Ordering::Relaxed);
    }

    pub fn limit_mib(&self) -> usize {
        let b = self.max_bits.load(Ordering::Relaxed);
        if b == 0 { 0 } else { ((1usize << b) * std::mem::size_of::<SSlot>()) >> 20 }
    }

    #[inline(always)]
    fn table(&self) -> &STable {
        // SAFETY: `cur` always points to a live table; tables are freed only in Drop.
        unsafe { &*self.cur.load(Ordering::Acquire) }
    }

    /// Learned entries currently stored (approximate under concurrency).
    pub fn len(&self) -> usize {
        self.table().count.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn get(&self, lo: u64, hi: u64, h: u64) -> Option<([u32; 3], usize)> {
        let t = self.table();
        let mut i = (h >> t.shift) as usize;
        for _ in 0..PROBE_MAX {
            // SAFETY: i is masked to < slots.len().
            let s = unsafe { t.slots.get_unchecked(i) };
            let m = s.meta.load(Ordering::Acquire);
            if m == 0 {
                return None;
            }
            if m & 3 == 2 && s.lo.load(Ordering::Relaxed) == lo && s.hi.load(Ordering::Relaxed) == hi {
                let t01 = s.t01.load(Ordering::Relaxed);
                return Some(([t01 as u32, (t01 >> 32) as u32, s.t2.load(Ordering::Relaxed)], (m >> 2) as usize));
            }
            i = (i + 1) & t.mask;
        }
        None
    }

    /// Hint the slot a lookup with hash `h` reads first into cache.
    #[inline(always)]
    pub fn prefetch(&self, h: u64) {
        let t = self.table();
        let i = (h >> t.shift) as usize;
        // SAFETY: i < slots.len().
        prefetch(unsafe { t.slots.as_ptr().add(i) } as *const u8);
    }

    /// Insert a learned encoding of 1..=3 tokens (longer ones are ignored).
    pub fn put(&self, lo: u64, hi: u64, h: u64, toks: &[u32]) {
        let n = toks.len();
        if n == 0 || n > 3 || self.max_bits.load(Ordering::Relaxed) == 0 {
            return;
        }
        let t = self.table();
        if t.count.load(Ordering::Relaxed) * 2 >= t.slots.len() {
            if t.bits() < self.max_bits.load(Ordering::Relaxed) {
                self.grow(t);
            }
            // at the cap: keep inserting until probes get long (PROBE_MAX)
        }
        // A growth in progress is copying the old table; an insert landing there
        // now could be missed by the copy. Don't wait for it (threads would pile
        // up): skip — the chunk is simply merged again if it recurs.
        if self.growing.load(Ordering::Acquire) {
            return;
        }
        let t = self.table();
        let mut i = (h >> t.shift) as usize;
        for _ in 0..PROBE_MAX {
            // SAFETY: i is masked to < slots.len().
            let s = unsafe { t.slots.get_unchecked(i) };
            let mut m = s.meta.load(Ordering::Relaxed);
            if m == 0 {
                match s.meta.compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed) {
                    Ok(_) => {
                        s.lo.store(lo, Ordering::Relaxed);
                        s.hi.store(hi, Ordering::Relaxed);
                        let t0 = toks[0] as u64;
                        let t1 = toks.get(1).copied().unwrap_or(0) as u64;
                        s.t01.store(t0 | t1 << 32, Ordering::Relaxed);
                        s.t2.store(toks.get(2).copied().unwrap_or(0), Ordering::Relaxed);
                        s.meta.store(2 | (n as u32) << 2, Ordering::Release);
                        t.count.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    Err(cur) => m = cur,
                }
            }
            if m & 3 == 2 && s.lo.load(Ordering::Relaxed) == lo && s.hi.load(Ordering::Relaxed) == hi {
                return; // already learned
            }
            i = (i + 1) & t.mask;
        }
    }

    /// Double the table (one thread at a time; others keep using the old one).
    #[cold]
    fn grow(&self, old: &STable) {
        self.grow_to(old, old.bits() + 1)
    }

    #[cold]
    fn grow_to(&self, old: &STable, bits: u32) {
        if self.growing.swap(true, Ordering::Acquire) {
            return;
        }
        let cur = self.cur.load(Ordering::Acquire);
        if !std::ptr::eq(cur, old) {
            self.growing.store(false, Ordering::Release);
            return;
        }
        let new = STable::new(bits.max(old.bits() + 1));
        let mut copied = 0usize;
        for s in old.slots.iter() {
            let m = s.meta.load(Ordering::Acquire);
            if m & 3 != 2 {
                continue;
            }
            let (lo, hi) = (s.lo.load(Ordering::Relaxed), s.hi.load(Ordering::Relaxed));
            let mut i = (hash(lo, hi) >> new.shift) as usize;
            loop {
                let d = &new.slots[i];
                if d.meta.load(Ordering::Relaxed) == 0 {
                    d.lo.store(lo, Ordering::Relaxed);
                    d.hi.store(hi, Ordering::Relaxed);
                    d.t01.store(s.t01.load(Ordering::Relaxed), Ordering::Relaxed);
                    d.t2.store(s.t2.load(Ordering::Relaxed), Ordering::Relaxed);
                    d.meta.store(m, Ordering::Relaxed);
                    copied += 1;
                    break;
                }
                i = (i + 1) & new.mask;
            }
        }
        new.count.store(copied, Ordering::Relaxed);
        let newp = Box::into_raw(new);
        // Release: the copied slots are visible to any thread that loads `cur`.
        self.cur.store(newp, Ordering::Release);
        self.retired.lock().unwrap_or_else(|e| e.into_inner()).push(cur);
        self.growing.store(false, Ordering::Release);
    }
}

impl Drop for SharedTable {
    fn drop(&mut self) {
        // SAFETY: exclusive access; every pointer came from Box::into_raw and is
        // freed exactly once.
        unsafe {
            drop(Box::from_raw(*self.cur.get_mut()));
            for p in self.retired.get_mut().unwrap_or_else(|e| e.into_inner()).drain(..) {
                drop(Box::from_raw(p));
            }
        }
    }
}

#[cfg(test)]
mod shared_tests {
    use super::*;

    #[test]
    fn shared_table_concurrent_insert_get() {
        let t = SharedTable::new(64);
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
}
