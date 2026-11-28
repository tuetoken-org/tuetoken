use super::*;
pub static LONG_MERGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Interleaved cold A/B (fresh tokenizer per run, one thread): spill map on/off.
#[test]
#[ignore]
fn ab_cold() {
    use std::sync::atomic::Ordering::Relaxed;
    let path = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let mut end = (300usize << 20).min(data.len());
    while end < data.len() && data[end] & 0xC0 == 0x80 { end += 1; }
    let b = &data[..end];
    let mut best = [f64::MAX; 2];
    for round in 0..8 {
        let spill = round % 2;
        let t = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
        if spill == 0 { t.spill.set_limit(0); }
        let mut s = Scratch::default();
        s.dyn_cache.claim(t.uid);
        let t0 = std::time::Instant::now();
        t.encode_range(b, 0, b.len(), &mut s);
        best[spill] = best[spill].min(t0.elapsed().as_secs_f64());
        if round == 1 {
            println!("spill gets {} puts {} | merges {} long merges {}", chunk_table::SPILL_GETS.load(Relaxed),
                chunk_table::SPILL_PUTS.load(Relaxed), MERGES.load(Relaxed), LONG_MERGES.load(Relaxed));
        }
        chunk_table::SPILL_GETS.store(0, Relaxed); chunk_table::SPILL_PUTS.store(0, Relaxed);
        MERGES.store(0, Relaxed); LONG_MERGES.store(0, Relaxed);
    }
    for spill in 0..2 {
        println!("cold 1 thread, spill map {}: {:.0} MB/s", ["off", "on "][spill], b.len() as f64 / best[spill] / 1e6);
    }
}

pub static NO_PIPE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static NO_SCAN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Component costs of the batched pipeline (single core, warm).
#[test]
#[ignore]
fn split_costs() {
    let path = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let b = &data[..300 << 20];
    let n = b.len();
    let t = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
    let mut s = Scratch::default();
    s.dyn_cache.claim(t.uid);
    t.encode_range(b, 0, n, &mut s);
    let mb = |d: f64| n as f64 / d / 1e6;
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let mut ends = vec![0u32; 512 + blockscan::SCAN_SLACK];
        let mut sc = blockscan::Scan::new(0);
        let mut acc = 0usize;
        loop { let k = sc.fill(t.kind, b, n, &mut ends, 0, 512); if k == 0 { break; } acc += k; }
        let d_scan = t0.elapsed().as_secs_f64();
        // rules only: block_m over every block, carry in place
        let t0 = std::time::Instant::now();
        let mut c = blockscan::Carry::default();
        let mut i = 0usize;
        let mut z = 0u64;
        while i + 64 <= n { let (st, bd) = blockscan::block_m(t.kind, b, i, &mut c); z = z.wrapping_add(st ^ bd); i += 64; }
        let d_rules = t0.elapsed().as_secs_f64();
        println!("  rules only {:.0} MB/s ({z}) -> extraction+driver {:.3} ns/B", mb(d_rules), (d_scan - d_rules) / n as f64 * 1e9);
        let asc: Vec<u8> = b.iter().map(|&x| if x >= 0x80 { b'q' } else { x }).collect();
        let t0 = std::time::Instant::now();
        let mut c = blockscan::Carry::default();
        let mut i = 0usize;
        while i + 64 <= n { let (st, bd) = blockscan::block_m(t.kind, &asc, i, &mut c); z = z.wrapping_add(st ^ bd); i += 64; }
        println!("  rules only, all-ASCII text {:.0} MB/s ({z})", mb(t0.elapsed().as_secs_f64()));
        let t0 = std::time::Instant::now();
        let mut sc = blockscan::Scan::new(0);
        let view = t.learned.view();
        let mut recs = vec![(0usize, 0usize, 0u64, 0u64, 0u64); 512 + blockscan::SCAN_SLACK];
        let mut a = 0usize;
        let bp = b.as_ptr();
        let mut x = 0u64;
        loop {
            let k = sc.fill(t.kind, b, n, &mut ends, 0, 512);
            if k == 0 { break; }
            for (c, &e) in ends[..k].iter().enumerate() {
                let e = e as usize;
                let len = e - a;
                if a + 16 > n || len.wrapping_sub(1) >= 15 { a = e; continue; }
                let (lo, hi) = unsafe { chunk_table::pack_wide(bp.add(a), len) };
                let p = view.slot_addr(chunk_table::hash(lo, hi));
                chunk_table::prefetch(p as *const u8);
                recs[c] = (a, e, lo, hi, p);
                a = e;
            }
            x = x.wrapping_add(recs[k / 2].4);
        }
        let d_fill = t0.elapsed().as_secs_f64();
        s.out.clear();
        let t0 = std::time::Instant::now();
        t.encode_range(b, 0, n, &mut s);
        let d_full = t0.elapsed().as_secs_f64();
        println!("scan {:.0} MB/s | +fill {:.0} | full {:.0} || ns/B: scan {:.3} fill {:.3} probe+emit {:.3} ({acc} {x})",
            mb(d_scan), mb(d_fill), mb(d_full), d_scan / n as f64 * 1e9, (d_fill - d_scan) / n as f64 * 1e9, (d_full - d_fill) / n as f64 * 1e9);
    }
}

#[test]
#[ignore]
fn ab_scan() {
    use std::sync::atomic::Ordering::Relaxed;
    let path = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let b = &data[..300 << 20];
    for p in path.split(',') {
        let t = Tokenizer::new(p).unwrap();
        let mut s = Scratch::default();
        s.dyn_cache.claim(t.uid);
        t.encode_range(b, 0, b.len(), &mut s);
        let mut best = [f64::MAX; 2];
        let mut outs: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for round in 0..12 {
            let v = round % 2;
            NO_SCAN.store(v == 1, Relaxed);
            s.out.clear();
            let t0 = std::time::Instant::now();
            t.encode_range(b, 0, b.len(), &mut s);
            best[v] = best[v].min(t0.elapsed().as_secs_f64());
            if round < 2 { outs[v] = s.out.clone(); }
        }
        NO_SCAN.store(false, Relaxed);
        assert!(outs[0] == outs[1], "output differs");
        println!("{:28} batched+straight-line fill {:.0} MB/s | per-block {:.0} MB/s", p.rsplit('/').nth(2).unwrap_or(p).chars().take(28).collect::<String>(),
            b.len() as f64 / best[0] / 1e6, b.len() as f64 / best[1] / 1e6);
    }
}


/// Profiling loop: pipelined chunk engine only, warm, repeated.
#[test]
#[ignore]
fn pipe_loop() {
    let toks = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let mut end = (300usize << 20).min(data.len());
    while end < data.len() && data[end] & 0xC0 == 0x80 { end += 1; }
    let b = &data[..end];
    let t = Tokenizer::new(toks.split(',').next().unwrap()).unwrap();
    let mut s = Scratch::default();
    s.dyn_cache.claim(t.uid);
    for _ in 0..6 {
        s.out.clear();
        t.encode_range(b, 0, b.len(), &mut s);
    }
}

/// Interleaved A/B on SHORT inputs (one encode call per text): pipelined vs immediate.
#[test]
#[ignore]
fn ab_short() {
    let toks = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read_to_string(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let t = Tokenizer::new(toks.split(',').next().unwrap()).unwrap();
    for len in [16usize, 64, 256, 512, 1024, 2048, 8192] {
        let mut texts = Vec::new();
        let mut i = 0usize;
        while texts.len() < 200_000 && i + len < data.len() {
            let mut e = i + len;
            while !data.is_char_boundary(e) { e += 1; }
            texts.push(&data[i..e]);
            i = e + 7;
            while !data.is_char_boundary(i) { i += 1; }
        }
        let mut s = Scratch::default();
        s.dyn_cache.claim(t.uid);
        let mut best = [f64::MAX; 2];
        for round in 0..8 {
            let v = round % 2;
            NO_PIPE.store(v == 1, std::sync::atomic::Ordering::Relaxed);
            let t0 = std::time::Instant::now();
            for x in &texts { s.out.clear(); t.encode_range(x.as_bytes(), 0, x.len(), &mut s); }
            best[v] = best[v].min(t0.elapsed().as_secs_f64());
        }
        NO_PIPE.store(false, std::sync::atomic::Ordering::Relaxed);
        println!("{len:5} B texts: pipelined {:.0} ns/text | immediate {:.0} ns/text", best[0] / texts.len() as f64 * 1e9, best[1] / texts.len() as f64 * 1e9);
    }
}

/// Interleaved A/B: pipelined chunk resolution vs immediate.
#[test]
#[ignore]
fn ab_pipe() {
    let toks = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let mut end = (300usize << 20).min(data.len());
    while end < data.len() && data[end] & 0xC0 == 0x80 { end += 1; }
    let b = &data[..end];
    for path in toks.split(',') {
        let t = Tokenizer::new(path).unwrap();
        let mut s = Scratch::default();
        s.dyn_cache.claim(t.uid);
        let mut best = [f64::MAX; 2];
        let mut outs: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for round in 0..8 {
            let v = round % 2;
            NO_PIPE.store(v == 1, std::sync::atomic::Ordering::Relaxed);
            s.out.clear();
            let t0 = std::time::Instant::now();
            t.encode_range(b, 0, b.len(), &mut s);
            best[v] = best[v].min(t0.elapsed().as_secs_f64());
            if round < 2 { outs[v] = s.out.clone(); }
        }
        NO_PIPE.store(false, std::sync::atomic::Ordering::Relaxed);
        assert!(outs[0] == outs[1], "pipelined output differs");
        println!("{:45} pipelined {:6.1} MB/s | immediate {:6.1} MB/s",
            path.rsplit('/').nth(1).unwrap_or(path), b.len() as f64 / best[0] / 1e6, b.len() as f64 / best[1] / 1e6);
    }
}

pub static MERGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DYN_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[test]
#[ignore]
fn dyn_cache_smoke() {
    use std::sync::atomic::Ordering::Relaxed;
    let path = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let t = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
    let text = " xyzzyplughq".repeat(1000) + "                ";
    let mut s = Scratch::default();
    s.dyn_cache.claim(t.uid);
    MERGES.store(0, Relaxed);
    DYN_HITS.store(0, Relaxed);
    t.encode_range(text.as_bytes(), 0, text.len(), &mut s);
    println!("owned_by={} merges={} dyn_hits={} tokens={}", s.dyn_cache.owned_by(t.uid),
        MERGES.load(Relaxed), DYN_HITS.load(Relaxed), s.out.len());
}

/// PROTOTYPE (falsification test): word-level ("unit") memoization.
/// Units = text between spaces preceded by an ASCII non-whitespace byte
/// (proven chunk boundaries). Boundaries from a SWAR mask; each unit is one
/// probe into a 64-byte-slot table (key + up to 11 tokens inline); misses run
/// the exact serial chunk path on the unit and insert. Verifies output.
#[test]
#[ignore]
fn unit_proto() {
    // 32-byte slot: 16-byte key + count + 3 inline tokens (96.6% of unit hits);
    // longer results spill to an arena (count = SPILL, toks = [offset, len, _])
    const SPILL: u32 = u32::MAX;
    #[derive(Clone, Copy, Default)]
    #[repr(C, align(32))]
    struct Slot { lo: u64, hi: u64, n: u32, toks: [u32; 3] }
    struct UTable { e: Vec<Slot>, mask: usize, shift: u32, len: usize, arena: Vec<u32> }
    impl UTable {
        fn new(bits: u32) -> Self { UTable { e: vec![Slot::default(); 1 << bits], mask: (1 << bits) - 1, shift: 64 - bits, len: 0, arena: Vec::new() } }
        #[inline(always)]
        fn get(&self, lo: u64, hi: u64, h: u64) -> Option<&Slot> {
            let mut i = (h >> self.shift) as usize;
            loop {
                let x = unsafe { self.e.get_unchecked(i) };
                if x.lo == lo && x.hi == hi { return Some(x); }
                if x.hi == 0 { return None; }
                i = (i + 1) & self.mask;
            }
        }
        fn put(&mut self, lo: u64, hi: u64, h: u64, t: &[u32]) {
            if t.is_empty() || t.len() > 64 || self.len * 2 > self.e.len() { return; }
            let mut i = (h >> self.shift) as usize;
            while self.e[i].hi != 0 { i = (i + 1) & self.mask; }
            let (n, toks) = if t.len() <= 3 {
                let mut toks = [0u32; 3];
                toks[..t.len()].copy_from_slice(t);
                (t.len() as u32, toks)
            } else {
                let off = self.arena.len() as u32;
                self.arena.extend_from_slice(t);
                (SPILL, [off, t.len() as u32, 0])
            };
            self.e[i] = Slot { lo, hi, n, toks };
            self.len += 1;
        }
        #[inline(always)]
        fn prefetch(&self, h: u64) {
            chunk_table::prefetch(unsafe { self.e.as_ptr().add((h >> self.shift) as usize) } as *const u8);
        }
    }
    const HI: u64 = 0x8080_8080_8080_8080;
    const ONES: u64 = 0x0101_0101_0101_0101;
    #[inline(always)]
    fn rng(x: u64, lo: u8, hi: u8) -> u64 {
        let y = x | HI;
        (y.wrapping_sub(ONES * lo as u64)) & !(y.wrapping_sub(ONES * (hi as u64 + 1))) & !x & HI
    }
    let is_nw = |c: u8| c < 0x80 && !(c == b' ' || (0x09..=0x0d).contains(&c));
    // unit-boundary bitmask for bytes [i, i+64) (needs i >= 1, i + 64 <= n)
    let unit_mask = |b: &[u8], i: usize| -> u64 {
        let mut m = 0u64;
        let mut carry = is_nw(b[i - 1]) as u64;
        for w in 0..8 {
            let x = u64::from_le(unsafe { (b.as_ptr().add(i + 8 * w) as *const u64).read_unaligned() });
            let sp = rng(x, b' ', b' ');
            let nw = !x & HI & !sp & !rng(x, 0x09, 0x0d);
            let prev = (nw << 8) | (carry << 7);
            let bits = (((sp & prev) >> 7).wrapping_mul(0x0102_0408_1020_4080)) >> 56;
            m |= bits << (8 * w);
            carry = nw >> 63;
        }
        m
    };
    let path = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let mb: usize = std::env::var("PROTO_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(500);
    let mut end = (mb << 20).min(data.len());
    while end < data.len() && data[end] & 0xC0 == 0x80 { end += 1; }
    let b = &data[..end];
    let n = b.len();
    let t = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
    let mut sref = Scratch::default();
    sref.dyn_cache.claim(t.uid);
    t.encode_range(b, 0, n, &mut sref);
    let want = std::mem::take(&mut sref.out);
    let mut table = UTable::new(23);
    let mut hist = [0usize; 12];
    let admit = std::env::var("ADMIT").is_ok();
    let mut seen = vec![0u64; 1 << 20]; // 2^26-bit "seen once" filter (8 MiB) // 16M slots x 64 B = 1 GiB virtual, lazily touched
    for pass in 0..3 {
        let mut s = Scratch::default();
        s.dyn_cache.claim(t.uid);
        s.out = Vec::with_capacity(n / 3 + 64);
        let (mut misses, mut units) = (0usize, 0usize);
        let t0 = std::time::Instant::now();
        let mut pos = 0usize;
        let mut i = 1usize;
        const GROUP: usize = 8; // blocks per phase: ~80 units of prefetch lead
        let mut ends = [0usize; 64 * GROUP];
        let mut keys = [(0u64, 0u64, 0u64); 64 * GROUP];
        while i + 64 <= n {
            let mut k = 0;
            let mut a = pos;
            let mut g = 0;
            while g < GROUP && i + 64 <= n {
                let mut m = unit_mask(b, i);
                while m != 0 {
                    let q = i + m.trailing_zeros() as usize;
                    m &= m - 1;
                    let len = q - a;
                    keys[k] = if len <= chunk_table::MAX_KEY && a + 16 <= n {
                        let (lo, hi) = unsafe { chunk_table::pack_wide(b.as_ptr().add(a), len) };
                        let h = chunk_table::hash(lo, hi) | 1;
                        table.prefetch(h);
                        (lo, hi, h)
                    } else { (0, 0, 0) };
                    ends[k] = q;
                    k += 1;
                    a = q;
                }
                i += 64;
                g += 1;
            }
            units += k;
            // branch-light emit: always store 4 tokens, advance by the count
            s.out.reserve(4 * k + 16);
            let mut w = s.out.len();
            let mut dst = s.out.as_mut_ptr();
            for j in 0..k {
                let q = ends[j];
                let (lo, hi, h) = keys[j];
                if h != 0 {
                    if let Some(x) = table.get(lo, hi, h) {
                        if x.n <= 3 {
                            hist[x.n as usize] += 1;
                            unsafe { (dst.add(w) as *mut [u32; 4]).write_unaligned([x.toks[0], x.toks[1], x.toks[2], 0]) };
                            w += x.n as usize;
                            pos = q;
                            continue;
                        }
                        let (off, len) = (x.toks[0] as usize, x.toks[1] as usize);
                        hist[len.min(11)] += 1;
                        unsafe { s.out.set_len(w) };
                        s.out.extend_from_slice(&table.arena[off..off + len]);
                        s.out.reserve(4 * (k - j) + 16);
                        w = s.out.len();
                        dst = s.out.as_mut_ptr();
                        pos = q;
                        continue;
                    }
                }
                unsafe { s.out.set_len(w) };
                misses += 1;
                let st = s.out.len();
                let mut p = pos;
                while p < q {
                    let e = pretok::next_chunk_end(t.kind, b, p);
                    t.emit_chunk(b, p, e, n, &mut s);
                    p = e;
                }
                if h != 0 {
                    // admission: store a unit only on its second sighting
                    let bit = (h >> 38) as usize;
                    let (wi, bi) = (bit >> 6, bit & 63);
                    if !admit || seen[wi] >> bi & 1 == 1 {
                        table.put(lo, hi, h, &s.out[st..]);
                    } else {
                        seen[wi] |= 1 << bi;
                    }
                }
                pos = q;
                s.out.reserve(4 * (k - j) + 16);
                w = s.out.len();
                dst = s.out.as_mut_ptr();
            }
            unsafe { s.out.set_len(w) };
        }
        // tail: exact chunk path
        let mut p = pos;
        while p < n {
            let e = pretok::next_chunk_end(t.kind, b, p);
            t.emit_chunk(b, p, e, n, &mut s);
            p = e;
        }
        let dt = t0.elapsed().as_secs_f64();
        assert!(s.out == want, "unit prototype output differs from the chunk engine");
        println!("pass {pass} ({}): {:.0} MB/s single thread | units {} | miss {:.2}% | table entries {}",
            if pass == 0 { "cold" } else { "warm" }, n as f64 / dt / 1e6, units,
            misses as f64 / units as f64 * 100.0, table.len);
    }
    {
        // ablation on the warm table: 0 = boundaries only, 1 = + keys/hash/prefetch, 2 = + probe
        for stage in 0..3 {
            let t0 = std::time::Instant::now();
            let (mut acc, mut pos, mut i) = (0u64, 0usize, 1usize);
            while i + 64 <= n {
                let mut m = unit_mask(b, i);
                let mut a = pos;
                while m != 0 {
                    let q = i + m.trailing_zeros() as usize;
                    m &= m - 1;
                    let len = q - a;
                    if stage >= 1 && len <= chunk_table::MAX_KEY && a + 16 <= n {
                        let (lo, hi) = unsafe { chunk_table::pack_wide(b.as_ptr().add(a), len) };
                        let h = chunk_table::hash(lo, hi) | 1;
                        table.prefetch(h);
                        acc = acc.wrapping_add(h);
                        if stage >= 2 {
                            if let Some(x) = table.get(lo, hi, h) { acc = acc.wrapping_add(x.toks[0] as u64); }
                        }
                    }
                    acc = acc.wrapping_add(q as u64);
                    a = q;
                }
                pos = a;
                i += 64;
            }
            println!("  ablation stage {stage} ({}): {:.0} MB/s ({acc})",
                ["boundaries", "+key/hash/prefetch", "+probe (no emit)"][stage], n as f64 / t0.elapsed().as_secs_f64() / 1e6);
        }
    }
    let tot: usize = hist.iter().sum();
    println!("unit hits by token count: {}", (1..12).map(|k| format!("{k}:{:.1}%", hist[k] as f64 / tot as f64 * 100.0)).collect::<Vec<_>>().join(" "));
    let mut s = Scratch::default();
    s.dyn_cache.claim(t.uid);
    t.encode_range(b, 0, n, &mut s);
    s.out.clear();
    let t0 = std::time::Instant::now();
    t.encode_range(b, 0, n, &mut s);
    println!("chunk engine (warm): {:.0} MB/s", n as f64 / t0.elapsed().as_secs_f64() / 1e6);
}

/// Phase timing for one big blob through the direct-output engine.
#[test]
#[ignore]
fn blob_phases() {
    let path = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read_to_string(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let t = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let (v, _) = t.encode_direct(&[data.as_str()], 0, true);
        let t1 = t0.elapsed();
        let t2 = t1;
        drop(v);
        let t3 = t0.elapsed();
        println!("encode {:?} ({} task bufs) | gather {:?} | free {:?} | total {:.2} GB/s",
            t1, 0, t2 - t1, t3 - t2, data.len() as f64 / t3.as_secs_f64() / 1e9);
    }
}

/// Single-core stage costs of the chunk engine on a warm tokenizer.
#[test]
#[ignore]
fn stage_ablate() {
    let toks = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let mut end = (300usize << 20).min(data.len());
    while end < data.len() && data[end] & 0xC0 == 0x80 { end += 1; }
    let b = &data[..end];
    let n = b.len();
    let t = Tokenizer::new(toks.split(',').next().unwrap()).unwrap();
    let mut s = Scratch::default();
    s.dyn_cache.claim(t.uid);
    t.encode_range(b, 0, n, &mut s);
    let ntok = s.out.len();
    let mb = |dt: std::time::Duration| n as f64 / dt.as_secs_f64() / 1e6;
    for _ in 0..2 {
        let t0 = std::time::Instant::now();
        let mut acc = 0usize;
        blockscan::chunks_batched(t.kind, b, 0, n, |_, ends| { acc += ends.len(); });
        println!("scan only            {:6.0} MB/s ({acc} chunks, {:.2} B/chunk, {:.3} tok/chunk)", mb(t0.elapsed()), n as f64 / acc as f64, ntok as f64 / acc as f64);
        let t0 = std::time::Instant::now();
        let mut acc = 0u64;
        blockscan::chunks_batched(t.kind, b, 0, n, |first, ends| {
            let mut a = first;
            for &e in ends {
                let len = e - a;
                if len.wrapping_sub(1) < chunk_table::MAX_KEY && a + 16 <= n {
                    let (lo, hi) = unsafe { chunk_table::pack_wide(b.as_ptr().add(a), len) };
                    acc = acc.wrapping_add(chunk_table::hash(lo, hi));
                }
                a = e;
            }
        });
        println!("scan+key+hash        {:6.0} MB/s ({acc})", mb(t0.elapsed()));
        let t0 = std::time::Instant::now();
        let mut acc = 0u64;
        blockscan::chunks_batched(t.kind, b, 0, n, |first, ends| {
            let mut a = first;
            for &e in ends {
                let len = e - a;
                if len.wrapping_sub(1) < chunk_table::MAX_KEY && a + 16 <= n {
                    let (lo, hi) = unsafe { chunk_table::pack_wide(b.as_ptr().add(a), len) };
                    let h = chunk_table::hash(lo, hi) | 1;
                    acc = acc.wrapping_add(t.chunk_table.get_h(lo, hi, h).unwrap_or(7) as u64);
                }
                a = e;
            }
        });
        println!("scan+vocab probe     {:6.0} MB/s ({acc})", mb(t0.elapsed()));
        s.out.clear();
        MERGES.store(0, std::sync::atomic::Ordering::Relaxed);
        LONG_MERGES.store(0, std::sync::atomic::Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        t.encode_range(b, 0, n, &mut s);
        println!("full engine          {:6.0} MB/s | warm merges: short>3tok {} long>15B {} (of {} chunks)", mb(t0.elapsed()),
            MERGES.load(std::sync::atomic::Ordering::Relaxed), LONG_MERGES.load(std::sync::atomic::Ordering::Relaxed), 65345842);
    }
    println!("chunk table: {} slots x {} B", t.chunk_table.slots(), t.chunk_table.entry_bytes());
    // block statistics + ASCII-only variant
    let mut nb = 0usize; let mut nbad = 0usize;
    let mut i = 0usize;
    let mut carry = blockscan::Carry::default();
    while i + 64 <= n { let (_, bad, c) = blockscan::block(t.kind, b, i, carry); carry = c; nb += 1; if bad != 0 { nbad += 1; } i += 64; }
    let nonascii = b.iter().filter(|&&x| x >= 0x80).count();
    println!("blocks {nb} | with BAD bits {:.1}% | non-ASCII bytes {:.2}%", nbad as f64 / nb as f64 * 100.0, nonascii as f64 / n as f64 * 100.0);
    let a: Vec<u8> = b.iter().map(|&x| if x >= 0x80 { b'q' } else { x }).collect();
    for _ in 0..2 {
        let t0 = std::time::Instant::now();
        let mut acc = 0usize;
        blockscan::chunks_batched(t.kind, &a, 0, n, |_, ends| { acc += ends.len(); });
        println!("scan only, ASCII-fied {:6.0} MB/s", mb(t0.elapsed()));
        let t0 = std::time::Instant::now();
        let mut i = 0usize; let mut carry = blockscan::Carry::default(); let mut acc = 0u64;
        while i + 64 <= n { let (st, bad, c) = blockscan::block(t.kind, &a, i, carry); carry = c; acc = acc.wrapping_add(st ^ bad); i += 64; }
        println!("block() only, ASCII   {:6.0} MB/s ({acc})", mb(t0.elapsed()));
        let t0 = std::time::Instant::now();
        let mut i = 0usize; let mut acc = 0u64;
        while i + 64 <= n { let c = blockscan::classify(&a, i); acc = acc.wrapping_add(c.l ^ c.sp ^ c.d ^ c.na); i += 64; }
        println!("classify() only       {:6.0} MB/s ({acc})", mb(t0.elapsed()));
        let t0 = std::time::Instant::now();
        let mut i = 0usize; let mut carry = blockscan::Carry::default(); let mut acc = 0u64;
        while i + 64 <= n { let (st, bad, c) = blockscan::block(t.kind, b, i, carry); carry = c; acc = acc.wrapping_add(st ^ bad); i += 64; }
        println!("block() only, REAL    {:6.0} MB/s ({acc})", mb(t0.elapsed()));
    }
}

#[test]
#[ignore]
fn seg_size() {
    let path = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read_to_string(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let t = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
    let _ = t.encode_direct(&[data.as_str()], 0, false);
    let mut end = 300usize << 20;
    while !data.is_char_boundary(end) { end += 1; }
    let small = &data[..end];
    let mut s = Scratch::default();
    s.dyn_cache.claim(t.uid);
    let t0 = std::time::Instant::now();
    t.encode_range(small.as_bytes(), 0, small.len(), &mut s);
    println!("one encode_range call, 300 MB, 1 thread: {:.0} MB/s", small.len() as f64 / t0.elapsed().as_secs_f64() / 1e6);
    for (label, text, nt) in [("1 thread, 300 MB", small, 1usize), ("all threads, 2 GB", data.as_str(), 0)] {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let _ = t.encode_direct(&[text], nt, true);
            best = best.min(t0.elapsed().as_secs_f64());
        }
        println!("SEG {:>8}: {label}: {:.0} MB/s", std::env::var("TT_SEG").unwrap_or("65536".into()), text.len() as f64 / best / 1e6);
    }
}

/// `split_on` == `bytes.split` semantics (leftmost, non-overlapping) for
/// inputs large enough to take the parallel path, with matches placed
/// across every 8 MiB piece boundary, incl. self-overlapping separators.
#[test]
fn split_on_matches_naive() {
    fn naive(b: &[u8], sep: &[u8]) -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        let (mut st, mut i) = (0usize, 0usize);
        while i + sep.len() <= b.len() {
            if &b[i..i + sep.len()] == sep {
                v.push((st, i));
                i += sep.len();
                st = i;
            } else {
                i += 1;
            }
        }
        v.push((st, b.len()));
        v
    }
    const P: usize = 8 << 20;
    let mut seed = 7u64;
    for (sep, fill) in [(&b"<|endoftext|>"[..], b'x'), (&b"aa"[..], b'a'), (&b"aba"[..], b'b'), (&b"\n"[..], b'q')] {
        let n = 3 * P + 12345;
        let mut b: Vec<u8> = (0..n).map(|i| if i % 7 == 0 { b' ' } else { fill }).collect();
        // separators straddling / touching every piece cut, plus random ones
        for cut in [P, 2 * P, 3 * P] {
            for off in 0..=sep.len() {
                let at = cut - off + 40 * off;
                if at + sep.len() <= n {
                    b[at..at + sep.len()].copy_from_slice(sep);
                }
            }
            let at = cut - sep.len() / 2 - 1;
            b[at..at + sep.len()].copy_from_slice(sep);
        }
        for _ in 0..2000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let at = (seed >> 20) as usize % (n - sep.len());
            b[at..at + sep.len()].copy_from_slice(sep);
        }
        assert_eq!(split_on(&b, sep, 0), naive(&b, sep), "sep {:?}", sep);
        assert_eq!(split_on(&b, sep, 1), naive(&b, sep), "sep {:?} serial", sep);
    }
}

/// Cold vs warm learning cost on one big blob: merges per call by thread count.
#[test]
#[ignore]
fn cold_learn() {
    use std::sync::atomic::Ordering::Relaxed;
    let path = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read_to_string(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let want = std::env::var("TT_OUT").is_ok();
    let run = |t: &Tokenizer, nt: usize, label: &str| {
        MERGES.store(0, Relaxed);
        let t0 = std::time::Instant::now();
        let (v, c) = t.encode_direct(&[data.as_str()], nt, want);
        let dt = t0.elapsed().as_secs_f64();
        drop(v);
        println!("{label:40} {:.2} GB/s | merges {:.3}%", data.len() as f64 / dt / 1e9, MERGES.load(Relaxed) as f64 / c[0] as f64 * 100.0);
    };
    if std::env::var("TT_ISO").is_ok() {
        let a = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
        run(&a, 14, "A: fresh tokenizer, fresh threads, 14t");
        run(&a, 14, "A: second call");
        let b = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
        run(&b, 14, "B: fresh tokenizer, warm threads, 14t");
        let d = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
        d.learned.presize(22);
        run(&d, 14, "D: fresh, learned table pre-sized 2^22, 14t");
        let e = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
        run(&e, 1, "E: fresh, 1 thread (x14 = ideal)");
        return;
    }
    for nt in [1usize, 14] {
        let t = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
        for pass in 0..2 {
            MERGES.store(0, Relaxed);
            let t0 = std::time::Instant::now();
            let (v, c) = t.encode_direct(&[data.as_str()], nt, want);
            let dt = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            drop(v);
            if want { eprintln!("  drop {:?}", t1.elapsed()); }
            println!("threads {nt:2} pass {pass}: {:.2} GB/s | merges {:.3}% of {} tokens | learned {}",
                data.len() as f64 / dt / 1e9, MERGES.load(Relaxed) as f64 / c[0] as f64 * 100.0, c[0], t.learned.len());
        }
    }
}

/// Multi-thread run over a document file (`<|endoftext|>`-separated), like a
/// data-prep job: throughput and per-thread cache miss rate, cold pass then
/// warm pass. `TUETOKEN_DOCS=owt.txt TUETOKEN_TOKENIZERS=gpt2.json`
#[test]
#[ignore]
fn mt_docs() {
    use std::sync::atomic::Ordering::Relaxed;
    let path = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let data = std::fs::read_to_string(std::env::var("TUETOKEN_DOCS").expect("set TUETOKEN_DOCS")).unwrap();
    let docs: Vec<&str> = data.split("<|endoftext|>").filter(|d| !d.is_empty()).collect();
    let t = Tokenizer::new(path.split(',').next().unwrap()).unwrap();
    let mut chunks_est = 0usize;
    let nthreads: usize = std::env::var("TUETOKEN_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    for pass in 0..2 {
        MERGES.store(0, Relaxed);
        DYN_HITS.store(0, Relaxed);
        let t0 = std::time::Instant::now();
        let (_, counts) = t.encode_direct(&docs, nthreads, true);
        let dt = t0.elapsed().as_secs_f64();
        let toks: usize = counts.iter().sum();
        if pass == 0 {
            chunks_est = toks; // ~1.09 tokens per chunk; tokens as the denominator
        }
        println!(
            "pass {pass}: {:.2} GB/s ({} docs, {:.2} GB, {toks} tokens) | per-thread cache hits {:.2}% | merges {:.3}% of tokens",
            data.len() as f64 / dt / 1e9, docs.len(), data.len() as f64 / 1e9,
            DYN_HITS.load(Relaxed) as f64 / chunks_est as f64 * 100.0,
            MERGES.load(Relaxed) as f64 / chunks_est as f64 * 100.0
        );
    }
}

/// Interleaved A/B of the load-time table's load factor (<=50% vs <=75%):
/// two tokenizers built in one process, encodes alternated so background
/// load hits both equally.
#[test]
#[ignore]
fn ab_table_load() {
    let toks = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let corpus = std::fs::read(std::env::var("TUETOKEN_CORPUS").expect("set TUETOKEN_CORPUS")).unwrap();
    let text = String::from_utf8_lossy(&corpus[..corpus.len().min(30_000_000)]).into_owned();
    let b = text.as_bytes();
    for path in toks.split(',') {
        chunk_table::DENSE.store(false, std::sync::atomic::Ordering::Relaxed);
        let ta = Tokenizer::new(path).unwrap();
        chunk_table::DENSE.store(true, std::sync::atomic::Ordering::Relaxed);
        let tb = Tokenizer::new(path).unwrap();
        chunk_table::DENSE.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut best = [f64::MAX; 2];
        for round in 0..10 {
            let t = if round % 2 == 0 { &ta } else { &tb };
            let mut s = Scratch::default();
            s.dyn_cache.claim(t.uid);
        s.long_cache.claim(t.uid);
            t.encode_range(b, 0, b.len(), &mut s); // warm this tokenizer's cache
            s.out.clear();
            let t0 = std::time::Instant::now();
            t.encode_range(b, 0, b.len(), &mut s);
            best[round % 2] = best[round % 2].min(t0.elapsed().as_secs_f64());
        }
        println!("{:45} load<=50% {:6.1} MB/s | load<=75% {:6.1} MB/s",
            path.rsplit('/').nth(1).unwrap_or(path), b.len() as f64 / best[0] / 1e6, b.len() as f64 / best[1] / 1e6);
    }
}

/// Core single-thread encode throughput, no Python in the loop:
/// `TUETOKEN_TOKENIZERS=a.json,b.json TUETOKEN_CORPUS=corpus.txt \
///  cargo test --release core_perf -- --ignored --nocapture`
#[test]
#[ignore]
fn core_perf() {
    let toks = std::env::var("TUETOKEN_TOKENIZERS").expect("set TUETOKEN_TOKENIZERS");
    let corpus = std::fs::read(std::env::var("TUETOKEN_CORPUS").expect("set TUETOKEN_CORPUS")).unwrap();
    let text = String::from_utf8_lossy(&corpus[..corpus.len().min(30_000_000)]).into_owned();
    let b = text.as_bytes();
    for path in toks.split(',') {
        let t = Tokenizer::new(path).unwrap();
        let mut s = Scratch::default();
        s.dyn_cache.claim(t.uid);
        s.long_cache.claim(t.uid);
        let mut best = f64::MAX;
        let mut ntok = 0;
        for _ in 0..4 {
            s.out.clear();
            let t0 = std::time::Instant::now();
            t.encode_range(b, 0, b.len(), &mut s);
            best = best.min(t0.elapsed().as_secs_f64());
            ntok = s.out.len();
        }
        println!("{:60} {:7.1} MB/s  ({ntok} tokens)", path.rsplit('/').nth(1).unwrap_or(path), b.len() as f64 / best / 1e6);
        {
            let n = b.len();
            for v in 0..3 {
                let mut best = f64::MAX;
                let mut acc = 0u64;
                for _ in 0..3 {
                    let t0 = std::time::Instant::now();
                    blockscan::chunks(t.kind, b, 0, n, |a, e| {
                        let len = e - a;
                        if v >= 1 && len.wrapping_sub(1) < chunk_table::MAX_KEY && a + 16 <= n {
                            let (lo, hi) = unsafe { chunk_table::pack_wide(b.as_ptr().add(a), len) };
                            if v == 1 {
                                acc = acc.wrapping_add(chunk_table::hash(lo, hi));
                            } else if let Some(x) = t.chunk_table.get(lo, hi) {
                                acc = acc.wrapping_add(x as u64);
                            }
                        }
                        acc = acc.wrapping_add(e as u64);
                    });
                    best = best.min(t0.elapsed().as_secs_f64());
                }
                println!("    ablation {} {:7.1} MB/s ({acc})", ["boundaries", "+pack+hash", "+probe"][v], n as f64 / best / 1e6);
            }
            // warm-pass outcome counts (s's dyn cache is warm from the runs above)
            let (mut st, mut dy, mut mg, mut lg, mut tot) = (0usize, 0usize, 0usize, 0usize, 0usize);
            let mut merge_ns = 0f64;
            let mut long_ns = 0f64;
            blockscan::chunks(t.kind, b, 0, n, |a, e| {
                tot += 1;
                let len = e - a;
                if len.wrapping_sub(1) < chunk_table::MAX_KEY && a + 16 <= n {
                    let (lo, hi) = unsafe { chunk_table::pack_wide(b.as_ptr().add(a), len) };
                    if t.chunk_table.get(lo, hi).is_some() { st += 1; return; }
                    if s.dyn_cache.get(lo, hi).is_some() { dy += 1; return; }
                    mg += 1;
                    let t0 = std::time::Instant::now();
                    let k = s.out.len();
                    t.process_chunk(&b[a..e], &mut s, Merge::Auto);
                    merge_ns += t0.elapsed().as_nanos() as f64;
                    s.out.truncate(k);
                } else {
                    lg += 1;
                    let t0 = std::time::Instant::now();
                    let k = s.out.len();
                    t.emit(&b[a..e], &mut s, Merge::Auto);
                    long_ns += t0.elapsed().as_nanos() as f64;
                    s.out.truncate(k);
                }
            });
            println!("    long chunks: avg {:.0} ns each => {:.2} ns per chunk overall; merges => {:.2} ns per chunk",
                long_ns / lg.max(1) as f64, long_ns / tot as f64, merge_ns / tot as f64);
            println!("    warm: static {:.2}% dyn {:.2}% merge {:.3}% (avg {:.0} ns) long {:.3}%",
                st as f64 / tot as f64 * 100.0, dy as f64 / tot as f64 * 100.0,
                mg as f64 / tot as f64 * 100.0, merge_ns / mg.max(1) as f64, lg as f64 / tot as f64 * 100.0);
        }
    }
}

#[test]
fn simd_wide_to_utf8_matches_scalar() {
    for len in 0..80usize {
        let mut src = vec![b'a'; len];
        for pos in (0..len).step_by(7) {
            src[pos] = 0x80 + ((pos * 19 + len) % 128) as u8;
        }
        let mut got = b"pfx:".to_vec();
        latin1_to_utf8_append(&src, &mut got);
        let mut want = vec![0u8; src.len() * 2];
        let w = unsafe { units_to_utf8(&src, want.as_mut_ptr()) }.unwrap();
        assert_eq!(&got[4..], &want[..w], "latin1 mismatch len={len}");
    }
    for off in 0..32usize {
        let mut src = vec![b'x'; 48];
        src[off] = 0xFC;
        let mut got = Vec::new();
        latin1_to_utf8_append(&src, &mut got);
        let want_s: String = src.iter().map(|&b| b as char).collect();
        assert_eq!(got, want_s.as_bytes(), "latin1 off={off}");
    }
    let all_hi: Vec<u8> = (0x80u8..=0xFF).collect();
    let mut got_hi = Vec::new();
    latin1_to_utf8_append(&all_hi, &mut got_hi);
    let want_hi: String = all_hi.iter().map(|&b| b as char).collect();
    assert_eq!(got_hi, want_hi.as_bytes());

    for len in 0..80usize {
        let mut src: Vec<u16> = (0..len).map(|i| b'A' as u16 + (i % 26) as u16).collect();
        for pos in (0..len).step_by(5) {
            src[pos] = match pos % 15 {
                0 => 0x00E4,
                5 => 0x201E,
                _ => 0x4E2D,
            };
        }
        let mut got = b"pfx:".to_vec();
        ucs2_to_utf8_append(&src, &mut got).unwrap();
        let mut want = vec![0u8; src.len() * 3];
        let w = unsafe { units_to_utf8(&src, want.as_mut_ptr()) }.unwrap();
        assert_eq!(&got[4..], &want[..w], "ucs2 mismatch len={len}");
    }
    for off in 0..32usize {
        let mut src = vec![b'z' as u16; 48];
        src[off] = 0x201C;
        let mut got = Vec::new();
        ucs2_to_utf8_append(&src, &mut got).unwrap();
        let want_s: String = src.iter().map(|&c| char::from_u32(c as u32).unwrap()).collect();
        assert_eq!(got, want_s.as_bytes(), "ucs2 off={off}");

        src[off] = 0xD800 + (off as u16 % 0x800);
        let mut err_buf = b"keep".to_vec();
        assert_eq!(ucs2_to_utf8_append(&src, &mut err_buf), Err(off));
        assert_eq!(err_buf, b"keep");
    }
}

#[test]
fn merge_tree_chunk_table_matches_bruteforce() {
    // Adversarial synthetic ByteLevel BPE covering:
    // - "aa" + "a" = "aaa" (leftmost wins, self-encodes)
    // - "a" + "aa" = "aaa2" (preempted by leftmost "aa", encodes to "aaa" not "aaa2")
    // - "aa" + "aa" = "aaaa" (equal-rank tie on left/right spines, self-encodes)
    // - "ab" + "cd" = "abcd" where "bc" has lower rank (cross-boundary preemption!)
    // - "xy" + "zw" = "xyzw" where "yz" has higher rank (no preemption, self-encodes)
    // - duplicate 1-byte token + unmerged multi-byte token
    let c2b = bytes_to_unicode_table();
    let mut b2c = ['\0'; 256];
    for (cp, &b) in c2b.iter().enumerate() {
        if b != u16::MAX {
            b2c[b as usize] = char::from_u32(cp as u32).unwrap();
        }
    }
    let enc_bl = |s: &[u8]| -> String { s.iter().map(|&b| b2c[b as usize]).collect() };

    let mut vocab = serde_json::Map::new();
    for b in 0u8..=255 {
        vocab.insert(enc_bl(&[b]), serde_json::json!(b as u32));
    }
    let extra: &[&[u8]] = &[
        b"aa",   // 256
        b"bc",   // 257 (lower rank than ab/cd -> preempts ab+cd!)
        b"ab",   // 258
        b"cd",   // 259
        b"abcd", // 260 (from ab+cd, MUST NOT self-encode to 260 because bc fires first!)
        b"aaa",  // 261 (from aa+a, self-encodes!)
        b"aaaa", // 262 (from aa+aa, self-encodes!)
        b"xy",   // 263
        b"zw",   // 264
        b"yz",   // 265 (higher rank than xy/zw -> does NOT preempt xy+zw!)
        b"xyzw", // 266 (from xy+zw, self-encodes!)
        b"a2",   // 267 (from a+aa -> actually test "b"+"aa" vs "a"+"aa" via another pair)
        b"bb",   // 268
        b"bbb",  // 269 (from b+bb -> PREEMPTED by leftmost b+b -> bb+b, does NOT self-encode!)
        b"unmerged_tok", // 270 (no merge produces it -> encodes to singletons)
    ];
    for (i, &tok) in extra.iter().enumerate() {
        vocab.insert(enc_bl(tok), serde_json::json!(256 + i));
    }
    let merges: Vec<String> = vec![
        (b"a".as_slice(), b"a".as_slice()),    // -> 256 "aa"
        (b"b", b"c"),                          // -> 257 "bc"
        (b"a", b"b"),                          // -> 258 "ab"
        (b"c", b"d"),                          // -> 259 "cd"
        (b"ab", b"cd"),                        // -> 260 "abcd" (preempted by bc!)
        (b"aa", b"a"),                         // -> 261 "aaa" (valid: leftmost wins)
        (b"aa", b"aa"),                        // -> 262 "aaaa" (valid: equal-rank spine tie)
        (b"x", b"y"),                          // -> 263 "xy"
        (b"z", b"w"),                          // -> 264 "zw"
        (b"y", b"z"),                          // -> 265 "yz" (higher rank than xy, zw)
        (b"xy", b"zw"),                        // -> 266 "xyzw" (valid!)
        (b"b", b"b"),                          // -> 268 "bb"
        (b"b", b"bb"),                         // -> 269 "bbb" (preempted by leftmost b+b!)
    ]
    .into_iter()
    .map(|(a, b)| format!("{} {}", enc_bl(a), enc_bl(b)))
    .collect();

    let doc = serde_json::json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": merges },
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "use_regex": true },
        "decoder": { "type": "ByteLevel" },
        "added_tokens": []
    });
    let tmp = std::env::temp_dir().join(format!("tuetoken_mergetree_{}.json", std::process::id()));
    std::fs::write(&tmp, doc.to_string()).unwrap();
    let tk = Tokenizer::new(tmp.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_file(&tmp);

    assert!(tk.byte_fast && tk.monotonic);
    // Verify EVERY vocab token in chunk_table matches exact merge_bytes_into!
    let mut s = Scratch::default();
    for id in 0..tk.byte_span.len() as u32 {
        let (off, len) = tk.byte_span[id as usize];
        if len >= 1 && (len as usize) <= chunk_table::MAX_KEY {
            let b = &tk.byte_arena[off as usize..(off + len) as usize];
            s.out.clear();
            tk.process_chunk(b, &mut s, Merge::Scan);
            let want = if s.out.len() == 1 { Some(s.out[0]) } else { None };
            let (lo, hi) = chunk_table::pack(b);
            let got = tk.chunk_table.get(lo, hi);
            assert_eq!(got, want, "mismatch on id={id} bytes={:?}", std::str::from_utf8(b));
        }
    }
    // Spot-check the tricky cases explicitly:
    let lookup = |b: &[u8]| {
        let (lo, hi) = chunk_table::pack(b);
        tk.chunk_table.get(lo, hi)
    };
    assert_eq!(lookup(b"abcd"), None);       // preempted by "bc" -> [a, bc, d]
    assert_eq!(lookup(b"aaa"), Some(261));   // leftmost aa+a succeeds
    assert_eq!(lookup(b"aaaa"), Some(262));  // equal-rank aa+aa succeeds
    assert_eq!(lookup(b"xyzw"), Some(266));  // yz has higher rank -> xy+zw succeeds
    assert_eq!(lookup(b"bbb"), None);        // b+bb preempted by leftmost b+b -> [bb, b]
    assert_eq!(lookup(b"unmerged_tok"), None);
}

#[test]
fn special_prefilter_classification_and_scan() {
    let p_whisper = SpecialPrefilter::from_patterns(&[
        "<|endoftext|>".to_string(),
        "<|startoftranscript|>".to_string(),
        "<|0.00|>".to_string(),
        "<|30.00|>".to_string(),
    ]);
    assert_eq!(p_whisper, SpecialPrefilter::Prefix2(b'<', b'|'));
    assert_eq!(p_whisper.find_first(b"plain hello world < not pipe"), None);
    assert_eq!(p_whisper.find_first(b"a < b <|0.00|> c"), Some(6));
    assert_eq!(p_whisper.find_first(b"<"), None);

    let p_mask = SpecialPrefilter::from_patterns(&[
        "<s>".to_string(),
        "<|im_start|>".to_string(),
    ]);
    assert!(matches!(p_mask, SpecialPrefilter::Prefix1Mask(b'<', _)));
    assert_eq!(p_mask.find_first(b"no angle brackets here"), None);
    assert_eq!(p_mask.find_first(b"x < 5 and a < b"), None);
    assert_eq!(p_mask.find_first(b"hi <s>"), Some(3));
    assert_eq!(p_mask.find_first(b"x < 5 <|im_start|>"), Some(6));

    let p_one = SpecialPrefilter::from_patterns(&[
        "<".to_string(),
        "<|im_start|>".to_string(),
    ]);
    assert_eq!(p_one, SpecialPrefilter::Bytes1(b'<'));
    assert_eq!(p_one.find_first(b"no angle brackets here"), None);
    assert_eq!(p_one.find_first(b"hi <s>"), Some(3));

    let p_two = SpecialPrefilter::from_patterns(&[
        "<s>".to_string(),
        "[INST]".to_string(),
    ]);
    assert_eq!(p_two, SpecialPrefilter::Bytes2(b'<', b'['));
    assert_eq!(p_two.find_first(b"plain text"), None);
    assert_eq!(p_two.find_first(b"abc [INST]"), Some(4));

    let p_three = SpecialPrefilter::from_patterns(&[
        "<s>".to_string(),
        "[CLS]".to_string(),
        "{X}".to_string(),
    ]);
    assert_eq!(p_three, SpecialPrefilter::Bytes3(b'<', b'[', b'{'));
    assert_eq!(p_three.find_first(b"plain text"), None);
    assert_eq!(p_three.find_first(b"a {X}"), Some(2));

    // 4..=8 distinct first bytes use the exact SIMD nibble-table set prefilter
    // with per-slot required second-byte filtering (like Gemma-2/3).
    let p_gemma = SpecialPrefilter::from_patterns(&[
        "\t\t".to_string(),
        "\n".to_string(),
        "<bos>".to_string(),
        "<eos>".to_string(),
        "[multimodal]".to_string(),
        "▁▁".to_string(),
    ]);
    assert!(matches!(p_gemma, SpecialPrefilter::SimdSet { .. }));
    // Skips plain ASCII, em-dash (0xe2 0x80 != 0xe2 0x96), single tab, and `[1]` across SIMD lanes
    let clean = format!(
        "{} — hello [1] \t world {} — end",
        "x".repeat(40),
        "y".repeat(40)
    );
    assert_eq!(p_gemma.find_first(clean.as_bytes()), None);
    let with_sp = format!("{} — [1] [multimodal] tail", "z".repeat(35));
    assert_eq!(p_gemma.find_first(with_sp.as_bytes()), Some(with_sp.find("[multimodal]").unwrap()));
    let with_meta = format!("{} — ▁▁ tail", "a".repeat(33));
    assert_eq!(p_gemma.find_first(with_meta.as_bytes()), Some(with_meta.find("▁▁").unwrap()));
    let with_nl = format!("{}a\nb", "q".repeat(31));
    assert_eq!(p_gemma.find_first(with_nl.as_bytes()), Some(32));

    let p_many = SpecialPrefilter::from_patterns(&[
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "d".to_string(),
        "e".to_string(),
        "f".to_string(),
        "g".to_string(),
        "h".to_string(),
        "i".to_string(),
    ]);
    assert_eq!(p_many, SpecialPrefilter::None);
    assert_eq!(p_many.find_first(b"xyz"), Some(0));
}

#[test]
fn whisper_gather_bytes_opts_timestamps_and_prompt_strip() {
    let mut vocab = serde_json::Map::new();
    let c2b = bytes_to_unicode_table();
    let mut b2u = ['\0'; 256];
    for (cp, &bv) in c2b.iter().enumerate() {
        if bv != u16::MAX {
            b2u[bv as usize] = char::from_u32(cp as u32).unwrap();
        }
    }
    for b in 0u8..=255 {
        vocab.insert(b2u[b as usize].to_string(), serde_json::json!(b as u32));
    }
    // Special tokens: 256=<|endoftext|>, 257=<|startoftranscript|>, 258=<|startofprev|>
    // Timestamps (special=false): 260=<|0.00|>, 260+k=<|0.02*k|>
    let mut added = vec![
        serde_json::json!({"id": 256, "content": "<|endoftext|>", "special": true, "lstrip": false, "rstrip": false}),
        serde_json::json!({"id": 257, "content": "<|startoftranscript|>", "special": true, "lstrip": false, "rstrip": false}),
        serde_json::json!({"id": 258, "content": "<|startofprev|>", "special": true, "lstrip": false, "rstrip": false}),
    ];
    for k in 0..=1500u32 {
        let label = format!("<|{:.2}|>", k as f64 * 0.02);
        added.push(serde_json::json!({"id": 260 + k, "content": label, "special": false, "lstrip": false, "rstrip": false}));
    }
    let doc = serde_json::json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": [] },
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "use_regex": true },
        "decoder": { "type": "ByteLevel" },
        "added_tokens": added
    });
    let tmp = std::env::temp_dir().join(format!("tuetoken_whisper_dec_{}.json", std::process::id()));
    std::fs::write(&tmp, doc.to_string()).unwrap();
    let mut tk = Tokenizer::new(tmp.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_file(&tmp);
    tk.set_whisper_decode(260, 258, 257);

    let ts = |sec_fiftieths: u32| 260 + sec_fiftieths;
    let ids_prompt_and_segs: Vec<u32> = vec![
        258, b'p' as u32, b'r' as u32, // <|startofprev|>pr
        257,                           // <|startoftranscript|>
        ts(0), b'A' as u32, ts(1000),  // <|0.00|>A<|20.00|>
        ts(1000), ts(250), b'B' as u32, ts(500), // consecutive boundary -> rewind second 20.00, shift +20.00 -> <|25.00|>B<|30.00|>
        256,                           // <|endoftext|>
    ];
    // 1. skip_special=true, decode_with_timestamps=false: strips prompt, specials, AND timestamps
    let (b1, a1) = tk.gather_bytes_fast_opts(&ids_prompt_and_segs, true, false, 0.02, 1500);
    assert!(a1);
    assert_eq!(String::from_utf8(b1).unwrap(), "AB");
    // 2. skip_special=false, decode_with_timestamps=false: keeps prompt & specials, strips timestamps
    let (b2, a2) = tk.gather_bytes_fast_opts(&ids_prompt_and_segs, false, false, 0.02, 1500);
    assert!(a2);
    assert_eq!(String::from_utf8(b2).unwrap(), "<|startofprev|>pr<|startoftranscript|>AB<|endoftext|>");
    // 3. skip_special=true, decode_with_timestamps=true: strips prompt & specials, accumulates rollover!
    let (b3, a3) = tk.gather_bytes_fast_opts(&ids_prompt_and_segs, true, true, 0.02, 1500);
    assert!(a3);
    assert_eq!(String::from_utf8(b3).unwrap(), "<|0.00|>A<|20.00|><|25.00|>B<|30.00|>");
    // 4. Single-ending segment rollover (+30.00s): <|0.00|>X<|29.00|><|1.00|>Y<|5.00|>
    let ids_single_end = vec![257, ts(0), b'X' as u32, ts(1450), ts(50), b'Y' as u32, ts(250), 256];
    let (b4, a4) = tk.gather_bytes_fast_opts(&ids_single_end, true, true, 0.02, 1500);
    assert!(a4);
    assert_eq!(String::from_utf8(b4).unwrap(), "<|0.00|>X<|29.00|><|31.00|>Y<|35.00|>");
    // 5. Prompt with NO <|startoftranscript|> -> empty when skip_special=true
    let ids_no_sot = vec![258, b'x' as u32, b'y' as u32];
    let (b5, a5) = tk.gather_bytes_fast_opts(&ids_no_sot, true, false, 0.02, 1500);
    assert!(a5);
    assert_eq!(String::from_utf8(b5).unwrap(), "");
}

#[test]
fn non_monotonic_chunk_table_matches_bruteforce() {
    // Adversarial non-monotonic ByteLevel vocab with BOTH:
    // - a rank inversion (`(aa, a) -> aaa` at rank 0 BEFORE `(a, a) -> aa` at rank 1!),
    // - multiple merges targeting the same token (`(x, yz) -> xyz` AND `(xy, z) -> xyz`),
    // - ordinary monotonic tokens (`p+q -> pq`, `pq+r -> pqr`).
    // Verifies our generalized local-monotonic DAG builder matches exact canonical BPE
    // on EVERY token in the vocabulary!
    let c2b = bytes_to_unicode_table();
    let mut b2c = ['\0'; 256];
    for (cp, &b) in c2b.iter().enumerate() {
        if b != u16::MAX {
            b2c[b as usize] = char::from_u32(cp as u32).unwrap();
        }
    }
    let enc_bl = |s: &[u8]| -> String { s.iter().map(|&b| b2c[b as usize]).collect() };

    let mut vocab = serde_json::Map::new();
    for b in 0u8..=255 {
        vocab.insert(enc_bl(&[b]), serde_json::json!(b as u32));
    }
    let extra: &[&[u8]] = &[
        b"aa",   // 256
        b"aaa",  // 257 (inverted rank: (aa, a)->aaa at rank 0 < (a, a)->aa at rank 1)
        b"aaaa", // 258 ((aaa, a)->aaaa at rank 2: canonical BPE on "aaaa" does a+a->aa, then aa+a->aaa, then aaa+a->aaaa!)
        b"xy",   // 259
        b"yz",   // 260
        b"xyz",  // 261 (multi-prod: (xy, z) at rank 5 AND (x, yz) at rank 6)
        b"pq",   // 262
        b"pqr",  // 263 (clean monotonic)
    ];
    for (i, &tok) in extra.iter().enumerate() {
        vocab.insert(enc_bl(tok), serde_json::json!(256 + i));
    }
    let merges: Vec<String> = vec![
        (b"aa".as_slice(), b"a".as_slice()), // rank 0 -> 257 "aaa" (RANK INVERSION!)
        (b"a", b"a"),                        // rank 1 -> 256 "aa"
        (b"aaa", b"a"),                      // rank 2 -> 258 "aaaa"
        (b"x", b"y"),                        // rank 3 -> 259 "xy"
        (b"y", b"z"),                        // rank 4 -> 260 "yz"
        (b"xy", b"z"),                       // rank 5 -> 261 "xyz" (first prod)
        (b"x", b"yz"),                       // rank 6 -> 261 "xyz" (second prod!)
        (b"p", b"q"),                        // rank 7 -> 262 "pq"
        (b"pq", b"r"),                       // rank 8 -> 263 "pqr"
    ]
    .into_iter()
    .map(|(a, b)| format!("{} {}", enc_bl(a), enc_bl(b)))
    .collect();

    let doc = serde_json::json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": merges },
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "use_regex": true },
        "decoder": { "type": "ByteLevel" },
        "added_tokens": []
    });
    let tmp = std::env::temp_dir().join(format!("tuetoken_nonmono_{}.json", std::process::id()));
    std::fs::write(&tmp, doc.to_string()).unwrap();
    let tk = Tokenizer::new(tmp.to_str().unwrap()).unwrap();
    let _ = std::fs::remove_file(&tmp);

    assert!(!tk.monotonic, "synthetic vocab must be non-monotonic");
    assert!(tk.byte_len_exact, "all 256 byte singletons + additive merges -> byte_len_exact");
    let mut s = Scratch::default();
    for id in 0..tk.byte_span.len() as u32 {
        let (off, len) = tk.byte_span[id as usize];
        if len >= 1 && (len as usize) <= chunk_table::MAX_KEY {
            let b = &tk.byte_arena[off as usize..(off + len) as usize];
            s.out.clear();
            tk.process_chunk(b, &mut s, Merge::Scan);
            let want = if s.out.len() == 1 { Some(s.out[0]) } else { None };
            let (lo, hi) = chunk_table::pack(b);
            let got = tk.chunk_table.get(lo, hi);
            assert_eq!(got, want, "non-mono mismatch on id={id} bytes={:?}", std::str::from_utf8(b));
        }
    }
    let lookup = |b: &[u8]| {
        let (lo, hi) = chunk_table::pack(b);
        tk.chunk_table.get(lo, hi)
    };
    assert_eq!(lookup(b"aaa"), Some(257));
    assert_eq!(lookup(b"aaaa"), Some(258));
    assert_eq!(lookup(b"xyz"), Some(261));
    assert_eq!(lookup(b"pqr"), Some(263));
}
