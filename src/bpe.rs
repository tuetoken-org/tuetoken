//! BPE merge: open-addressing hash map keyed (t1<<32|t2), with the same
//! scan-all-pairs / apply-all-lowest-rank-in-one-pass loop as the C engine.

/// One hash-map slot, AoS (16 bytes) so a probe hit is a single cache line —
/// matching the C engine. (An SoA layout splits a hit across 3 cache lines,
/// which is slower for the merge-heavy workload.)
#[derive(Clone, Copy)]
struct Entry {
    key: u64, // 0 = empty slot
    rank: u32,
    target: u32,
}

pub struct MergeMap {
    entries: Vec<Entry>,
    mask: usize,
}

impl MergeMap {
    pub fn new(merges: &[(u32, u32, u32)]) -> Self {
        let mut cap = 16usize;
        while cap < merges.len().saturating_mul(3) {
            cap <<= 1;
        }
        let mut m = MergeMap {
            entries: vec![
                Entry {
                    key: 0,
                    rank: 0,
                    target: 0
                };
                cap
            ],
            mask: cap - 1,
        };
        for (i, &(t1, t2, tg)) in merges.iter().enumerate() {
            let key = Self::key(t1, t2);
            let mut idx = (Self::hash(key) as usize) & m.mask;
            while m.entries[idx].key != 0 {
                idx = (idx + 1) & m.mask;
            }
            m.entries[idx] = Entry {
                key,
                rank: i as u32,
                target: tg,
            };
        }
        m
    }

    /// (0,0) packs to 0 (the empty sentinel); remap it to u64::MAX so it does
    /// not alias the common pair (0,1) — the same fix as the C version.
    #[inline]
    fn key(t1: u32, t2: u32) -> u64 {
        let k = ((t1 as u64) << 32) | (t2 as u64);
        if k == 0 {
            u64::MAX
        } else {
            k
        }
    }

    #[inline]
    fn hash(mut k: u64) -> u64 {
        k ^= k >> 33;
        k = k.wrapping_mul(0xff51afd7ed558ccd);
        k ^= k >> 33;
        k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
        k ^= k >> 33;
        k
    }

    #[inline]
    fn lookup(&self, t1: u32, t2: u32) -> (u32, u32) {
        let key = Self::key(t1, t2);
        let mut idx = (Self::hash(key) as usize) & self.mask;
        loop {
            // SAFETY: idx is always `& self.mask`, so idx < capacity == entries.len().
            let e = unsafe { *self.entries.get_unchecked(idx) };
            if e.key == key {
                return (e.rank, e.target);
            }
            if e.key == 0 {
                return (u32::MAX, 0);
            }
            idx = (idx + 1) & self.mask;
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
    ranks.clear();
    let mut min = u32::MAX;
    for i in 0..n - 1 {
        let (a, b) = unsafe { (*tokens.get_unchecked(i), *tokens.get_unchecked(i + 1)) };
        let r = map.lookup(a, b).0;
        ranks.push(r);
        if r < min {
            min = r;
        }
    }

    while min != u32::MAX {
        let applied = min;
        ntok.clear();
        src.clear();
        let mut i = 0usize;
        // Apply every pair whose rank == min, left to right (non-overlapping).
        unsafe {
            while i < n {
                if i + 1 < n && *ranks.get_unchecked(i) == min {
                    let t = map
                        .lookup(*tokens.get_unchecked(i), *tokens.get_unchecked(i + 1))
                        .1;
                    ntok.push(t);
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
                    map.lookup(a, b).0
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

/// Reusable scratch for `merge_canonical` (doubly-linked list + a min-heap),
/// so the canonical path allocates nothing per chunk.
#[derive(Default)]
pub struct CanonScratch {
    prev: Vec<u32>,
    next: Vec<u32>,
    alive: Vec<bool>,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, u32)>>,
    out: Vec<u32>,
    ranks: Vec<u32>, // cached pair ranks for the small-chunk path
}

/// Chunks at/under this length take the simple cached-rank canonical merge (no
/// heap / linked-list setup — cheaper for the typical short word-chunk); longer
/// ones use the O(n log n) heap path. Almost all post-pretokenization chunks are
/// short, so the heap machinery is reserved for rare long runs.
const SMALL_CANON: usize = 256;

/// Canonical BPE (exact HF/tiktoken semantics): repeatedly merge the single
/// adjacent pair of globally-lowest rank, leftmost on ties. Doubly-linked list +
/// binary heap => O(n log n), correct for ANY merge table — including the
/// rank-inverted vocabularies (gemma whitespace) where the batch `merge` is not
/// canonical. Used only for non-monotonic vocabularies, so the common hot path
/// (every normal trained BPE) stays on the faster batch/streaming mergers.
pub fn merge_canonical(tokens: &mut Vec<u32>, map: &MergeMap, s: &mut CanonScratch) {
    use std::cmp::Reverse;
    let n = tokens.len();
    if n <= 1 {
        return;
    }
    if n <= SMALL_CANON {
        // Cached-rank canonical: merge the leftmost lowest-rank pair, repeat. Each
        // merge updates only its two neighbour ranks (≤3 lookups), and the min is a
        // cheap linear scan over the small rank array — no heap/list allocation.
        s.ranks.clear();
        for i in 0..n - 1 {
            s.ranks.push(map.lookup(tokens[i], tokens[i + 1]).0);
        }
        loop {
            let mut best = u32::MAX;
            let mut bi = usize::MAX;
            for (i, &r) in s.ranks.iter().enumerate() {
                if r < best {
                    best = r;
                    bi = i;
                }
            }
            if bi == usize::MAX {
                break;
            }
            let t = map.lookup(tokens[bi], tokens[bi + 1]).1;
            tokens[bi] = t;
            tokens.remove(bi + 1);
            s.ranks.remove(bi);
            if bi > 0 {
                s.ranks[bi - 1] = map.lookup(tokens[bi - 1], tokens[bi]).0;
            }
            if bi < s.ranks.len() {
                s.ranks[bi] = map.lookup(tokens[bi], tokens[bi + 1]).0;
            }
        }
        return;
    }
    let none = n as u32; // sentinel for "no neighbour"
    s.prev.clear();
    s.next.clear();
    s.alive.clear();
    s.heap.clear();
    for i in 0..n as u32 {
        s.prev.push(if i == 0 { none } else { i - 1 });
        s.next.push(if i + 1 == none { none } else { i + 1 });
        s.alive.push(true);
    }
    // (rank, position) — min-heap via Reverse; ties break on lower position (leftmost).
    for i in 0..n - 1 {
        let r = map.lookup(tokens[i], tokens[i + 1]).0;
        if r != u32::MAX {
            s.heap.push(Reverse((r, i as u32)));
        }
    }
    while let Some(Reverse((r, pi))) = s.heap.pop() {
        let i = pi as usize;
        if !s.alive[i] || s.next[i] == none {
            continue;
        }
        let j = s.next[i] as usize;
        if !s.alive[j] {
            continue;
        }
        let (cr, t) = map.lookup(tokens[i], tokens[j]);
        if cr != r {
            continue; // stale heap entry (this pair changed since it was queued)
        }
        // fold the right token j into the left token i
        tokens[i] = t;
        s.alive[j] = false;
        let k = s.next[j];
        s.next[i] = k;
        if k != none {
            s.prev[k as usize] = pi;
        }
        // queue the two new adjacencies around the merged token
        let p = s.prev[i];
        if p != none {
            let pr = map.lookup(tokens[p as usize], t).0;
            if pr != u32::MAX {
                s.heap.push(Reverse((pr, p)));
            }
        }
        if s.next[i] != none {
            let nr = map.lookup(t, tokens[s.next[i] as usize]).0;
            if nr != u32::MAX {
                s.heap.push(Reverse((nr, pi)));
            }
        }
    }
    // gather survivors by walking the list from the head (index 0 is never folded
    // away — only right-hand tokens are, and 0 is always leftmost).
    s.out.clear();
    let mut i = 0u32;
    loop {
        s.out.push(tokens[i as usize]);
        let nx = s.next[i as usize];
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
            let map = MergeMap::new(&merges);
            let st = StreamTables::new(&merges, vocab_size);
            let mono = is_monotonic(&merges, vocab_size);

            for _ in 0..20 {
                let len = (lcg(&mut seed) % 400) as usize;
                let seq: Vec<u32> = (0..len).map(|_| (lcg(&mut seed) as u32) % base).collect();
                let want = canonical(&seq, &merges);

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
