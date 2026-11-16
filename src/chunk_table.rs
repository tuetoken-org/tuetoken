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
    let lo_raw = (p as *const u64).read_unaligned().to_le();
    let hi_raw = (p.add(8) as *const u64).read_unaligned().to_le();
    // keep the first `len` bytes: bits [0, 8*len) of the 128-bit little-endian word
    let (lo, hi) = if len >= 8 {
        let keep_hi = 8 * (len - 8) as u32; // 0..=56 bits of hi
        let m = if keep_hi == 0 { 0 } else { u64::MAX >> (64 - keep_hi) };
        (lo_raw, hi_raw & m)
    } else {
        (lo_raw & (u64::MAX >> (64 - 8 * len as u32)), 0)
    };
    (lo, hi | ((len as u64) << 56))
}

#[inline(always)]
pub fn hash(lo: u64, hi: u64) -> u64 {
    // Two multiplies, mixed: cheap and good enough for byte-string keys.
    let a = lo.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let b = hi.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    (a ^ b.rotate_left(31)).wrapping_mul(0x1656_67B1_9E37_79F9)
}

impl ChunkTable {
    pub fn empty() -> Self {
        ChunkTable { entries: vec![Entry::default(); 2], shift: 63, mask: 1 }
    }

    /// Build from verified (key bytes, token) pairs. Duplicate keys keep the first.
    pub fn build(pairs: &[(&[u8], u32)]) -> Self {
        let mut cap = 16usize;
        while cap < pairs.len() * 2 {
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
        let mut i = (hash(lo, hi) >> self.shift) as usize;
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

pub const DYN_SET_BITS: u32 = 15; // 32K sets x 64 B = 2 MiB per thread, calloc'd (pages touched on use)
pub const DYN_MAX_TOKS: usize = 3;
const SPILL: u32 = 0xFF;
const SPILL_MAX_TOKS: usize = 64;
const ARENA_CAP: usize = 1 << 22; // 16 MiB of spilled tokens, then wipe

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
    fn set_index(lo: u64, hi: u64) -> usize {
        (hash(lo, hi) >> (64 - DYN_SET_BITS)) as usize
    }

    #[inline(always)]
    pub fn get(&self, lo: u64, hi: u64) -> Option<&[u32]> {
        if self.e.is_empty() {
            return None;
        }
        // SAFETY: set index < 2^DYN_SET_BITS == e.len() once allocated.
        let set = unsafe { self.e.get_unchecked(Self::set_index(lo, hi)) };
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
        if toks.is_empty() || toks.len() > SPILL_MAX_TOKS || self.tag == 0 {
            return;
        }
        if self.e.is_empty() {
            self.e = zeroed_sets(1 << DYN_SET_BITS);
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
        let i = Self::set_index(lo, hi);
        let set = &mut self.e[i];
        set.w[1] = set.w[0];
        set.w[0] = DEntry { lo, hi, toks: t, meta: self.tag | n };
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
