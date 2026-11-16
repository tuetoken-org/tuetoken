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
                self.wipe();
            }
            t[0] = self.arena.len() as u32;
            t[1] = toks.len() as u32;
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
