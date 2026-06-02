//! tuetoken — a Rust/PyO3 port of tuetoken's BPE engine, implementing the
//! same algorithm (state-machine pre-tokenization + open-addressing hash-map
//! BPE) for a head-to-head comparison with the C version. Batch encoding uses
//! rayon work-stealing (Rust's genuine edge over the C even-shard pthreads).
//!
//! Coverage: ByteLevel BPE (fast hand-written machines + a fancy-regex Split
//! fallback for unrecognized patterns/multi-Split) AND byte_fallback/metaspace
//! SentencePiece models (char-lookup encode + NFC normalize + metaspace). The
//! only pre-tokenizer still declined is the Punctuation pre-tokenizer.

mod bpe;
mod cjk;
mod pretok;
mod unicode;
mod unicode_tables;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use pretok::Pretok;

/// Cached rayon thread pools keyed by thread count. Building a pool spawns OS
/// threads, so creating one per `encode_ordinary_batch` call made 2-thread runs
/// SLOWER than serial on short docs. We build each pool once and reuse it.
/// Returns None if the OS can't spawn the threads (caller falls back to serial).
fn fixed_pool(n: usize) -> Option<Arc<rayon::ThreadPool>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Option<Arc<rayon::ThreadPool>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // Recover from a poisoned lock instead of aborting the interpreter.
    let mut map = cache.lock().unwrap_or_else(|e| e.into_inner());
    map.entry(n)
        .or_insert_with(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .ok()
                .map(Arc::new)
        })
        .clone()
}

fn bytes_to_unicode_c2b() -> HashMap<char, u8> {
    let mut bs: Vec<u32> = Vec::new();
    for (lo, hi) in [(0x21u32, 0x7e), (0xa1, 0xac), (0xae, 0xff)] {
        for b in lo..=hi {
            bs.push(b);
        }
    }
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut c2b = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        c2b.insert(char::from_u32(*c).unwrap(), *b as u8);
    }
    c2b
}

/// Build a Python str from decoded bytes with one UTF-8 validation and one copy
/// (the `String` round-trip pyo3 does for a returned `String` validates+copies
/// twice). Invalid UTF-8 (truncated token sequences) falls back to lossy, as
/// tiktoken/HF do.
fn bytes_to_pystr<'py>(py: Python<'py>, buf: &[u8]) -> Bound<'py, PyString> {
    match std::str::from_utf8(buf) {
        Ok(s) => PyString::new(py, s),
        Err(_) => PyString::new(py, &String::from_utf8_lossy(buf)),
    }
}

fn decode_bytelevel(s: &str, c2b: &HashMap<char, u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        if let Some(&b) = c2b.get(&ch) {
            out.push(b);
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}

/// Decode a byte_fallback (SentencePiece) vocab token to raw bytes:
/// `<0xXX>` -> that byte, U+2581 -> space, otherwise the token's UTF-8 bytes.
fn decode_byte_fallback(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    if b.len() == 6 && &b[..3] == b"<0x" && b[5] == b'>' {
        if let Ok(v) = u8::from_str_radix(&s[3..5], 16) {
            return vec![v];
        }
    }
    s.replace('\u{2581}', " ").into_bytes()
}

/// Metaspace prepend mode (matches the C `_apply_metaspace` variants):
/// 0 = none, 1 = replace-only, 2 = always-prepend (Replace+Prepend normalizers),
/// 3 = prepend-first (a Metaspace pre-tokenizer).
fn detect_meta_mode(cfg: &Value) -> u8 {
    let has_metaspace = |p: &Value| -> bool {
        match p.get("type").and_then(|v| v.as_str()) {
            Some("Metaspace") => true,
            Some("Sequence") => p
                .get("pretokenizers")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().any(|x| x.get("type").and_then(|v| v.as_str()) == Some("Metaspace")))
                .unwrap_or(false),
            _ => false,
        }
    };
    if cfg.get("pre_tokenizer").map(has_metaspace).unwrap_or(false) {
        return 3;
    }
    if let Some(norm) = cfg.get("normalizer") {
        let list: Vec<&Value> = if norm.get("type").and_then(|v| v.as_str()) == Some("Sequence") {
            norm.get("normalizers").and_then(|v| v.as_array()).map(|a| a.iter().collect()).unwrap_or_default()
        } else {
            vec![norm]
        };
        let mut has_prepend = false;
        let mut has_replace = false;
        for nm in list {
            if nm.get("type").and_then(|v| v.as_str()) == Some("Prepend")
                && nm.get("prepend").and_then(|v| v.as_str()) == Some("\u{2581}")
            {
                has_prepend = true;
            }
            if nm.get("type").and_then(|v| v.as_str()) == Some("Replace")
                && nm.get("content").and_then(|v| v.as_str()) == Some("\u{2581}")
                && nm.get("pattern").and_then(|p| p.get("String")).and_then(|v| v.as_str()) == Some(" ")
            {
                has_replace = true;
            }
        }
        if has_replace {
            return if has_prepend { 2 } else { 1 };
        }
    }
    0
}

/// SentencePiece decoders end with `Strip{content:" ", start:1}` — strip the one
/// leading space the metaspace prepend added back (matches C/HF decode).
fn detect_decode_strip(cfg: &Value) -> bool {
    let dec = match cfg.get("decoder") {
        Some(d) => d,
        None => return false,
    };
    let is_strip = |p: &Value| {
        p.get("type").and_then(|v| v.as_str()) == Some("Strip")
            && p.get("content").and_then(|v| v.as_str()) == Some(" ")
    };
    if is_strip(dec) {
        return true;
    }
    if dec.get("type").and_then(|v| v.as_str()) == Some("Sequence") {
        if let Some(arr) = dec.get("decoders").and_then(|v| v.as_array()) {
            return arr.iter().any(is_strip);
        }
    }
    false
}

/// 0 = none, 1 = NFC, 2 = NFD, 3 = NFKC, 4 = NFKD.
fn detect_norm_form(cfg: &Value) -> u8 {
    let code = |s: &str| match s {
        "NFC" => 1,
        "NFD" => 2,
        "NFKC" => 3,
        "NFKD" => 4,
        _ => 0,
    };
    let norm = match cfg.get("normalizer") {
        Some(n) => n,
        None => return 0,
    };
    if let Some(s) = norm.get("type").and_then(|v| v.as_str()) {
        let c = code(s);
        if c != 0 {
            return c;
        }
        if s == "Sequence" {
            if let Some(arr) = norm.get("normalizers").and_then(|v| v.as_array()) {
                for nm in arr {
                    if let Some(t) = nm.get("type").and_then(|v| v.as_str()) {
                        let c = code(t);
                        if c != 0 {
                            return c;
                        }
                    }
                }
            }
        }
    }
    0
}

/// Fail-closed normalizer check: return the type of the first normalizer stage we
/// don't actually reproduce, or None if every stage is one we apply. We model
/// exactly three kinds of stage — Unicode normalization (NFC/NFD/NFKC/NFKD, via
/// `normalize`) and the two SentencePiece metaspace stages (`Prepend{"▁"}` and
/// `Replace{" " -> "▁"}`, via `apply_metaspace`). Anything else (Lowercase,
/// StripAccents, Strip, a non-metaspace Replace, Precompiled, BertNormalizer, …)
/// rewrites the text in a way we'd silently ignore, mistokenizing every input it
/// touches — so we reject the model at load instead of lying about its output.
fn unsupported_normalizer_stage(cfg: &Value) -> Option<String> {
    let norm = cfg.get("normalizer")?;
    if norm.is_null() {
        return None;
    }
    let stages: Vec<&Value> = if norm.get("type").and_then(|v| v.as_str()) == Some("Sequence") {
        norm.get("normalizers").and_then(|v| v.as_array()).map(|a| a.iter().collect()).unwrap_or_default()
    } else {
        vec![norm]
    };
    for st in stages {
        let t = st.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let handled = match t {
            "NFC" | "NFD" | "NFKC" | "NFKD" => true,
            "Prepend" => st.get("prepend").and_then(|v| v.as_str()) == Some("\u{2581}"),
            "Replace" => {
                st.get("content").and_then(|v| v.as_str()) == Some("\u{2581}")
                    && st.get("pattern").and_then(|p| p.get("String")).and_then(|v| v.as_str())
                        == Some(" ")
            }
            _ => false,
        };
        if !handled {
            return Some(if t.is_empty() { "<unknown>".to_string() } else { t.to_string() });
        }
    }
    None
}

fn detect_pretok(pre: &Value) -> Option<Pretok> {
    let t = pre.get("type")?.as_str()?;
    if t == "ByteLevel" {
        let use_regex = pre.get("use_regex").and_then(|v| v.as_bool()).unwrap_or(true);
        return if use_regex { Some(Pretok::Word(false)) } else { None };
    }
    if t == "Sequence" {
        let pts = pre.get("pretokenizers")?.as_array()?;
        let mut has_bl = false;
        let mut bl_use_regex = false;
        let mut indiv_digits = false;
        let mut split_pat: Option<&str> = None;
        let mut n_split = 0u32;
        for pt in pts {
            match pt.get("type").and_then(|v| v.as_str()) {
                Some("ByteLevel") => {
                    has_bl = true;
                    bl_use_regex =
                        pt.get("use_regex").and_then(|v| v.as_bool()).unwrap_or(true);
                }
                Some("Digits") => {
                    indiv_digits = pt
                        .get("individual_digits")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                }
                Some("Split") => {
                    n_split += 1;
                    split_pat = pt
                        .get("pattern")
                        .and_then(|p| p.get("Regex"))
                        .and_then(|v| v.as_str());
                }
                // Any other pre-tokenizer (Punctuation, Metaspace, ...) changes
                // chunking in a way we don't model — fail closed (e.g. a
                // Punctuation+ByteLevel+Digits sequence must NOT be treated as a
                // plain word grammar).
                _ => return None,
            }
        }
        // Multiple Split stages compose in ways a single machine can't reproduce —
        // only one Split is modeled.
        if !has_bl || n_split > 1 {
            return None;
        }
        // Digits{individual_digits} + ByteLevel{use_regex}: the plain word grammar
        // with each digit isolated (common in code tokenizers).
        if split_pat.is_none() && bl_use_regex && indiv_digits {
            return Some(Pretok::Word(true));
        }
        // A Split stage AND a ByteLevel that itself splits (`use_regex`) would
        // double-split: each Split chunk is further cut by ByteLevel's GPT-2
        // regex. Our machines apply only the Split pattern, so we'd produce
        // different boundaries — fail closed. (Real Split-based models set
        // `use_regex:false`; this only rejects the unhandled composition.)
        if split_pat.is_some() && bl_use_regex {
            return None;
        }
        let pat = split_pat?;
        // Case-aware CamelCase letter runs (`[\p{Lu}\p{Lt}…]*[\p{Ll}…]+`).
        // Contractions are optional; the digit cap comes from the quantifier
        // (`\p{N}{1,3}` -> 3, `\p{N}+` -> unlimited, otherwise single-digit).
        if pat.contains("[\\p{Lu}\\p{Lt}") {
            let contractions = pat.contains("(?i:");
            let max_digits = if pat.contains("\\p{N}{1,3}") {
                3
            } else if pat.contains("\\p{N}+") {
                0
            } else {
                1
            };
            return Some(Pretok::CamelCase { contractions, max_digits });
        }
        // The canonical plain-word pattern (case-sensitive contractions, exact match).
        let canonical = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
        if pat == canonical {
            return Some(Pretok::Word(indiv_digits));
        }
        // Generic family: case-insensitive contractions + plain `\p{L}+` letter
        // run, digit cap from the quantifier.
        let digit_cap = || {
            if pat.contains("\\p{N}{1,3}") {
                3
            } else if pat.contains("\\p{N}+") {
                0
            } else {
                1
            }
        };
        // Mark-inclusive variant (e.g. Qwen3.6): letter runs are `[\p{L}\p{M}]+`
        // AND the symbol class excludes Marks (`[^\s\p{L}\p{M}\p{N}]`). Require
        // BOTH so we only claim the shape the mark-aware machine reproduces;
        // any other mark handling falls through to the regex fallback.
        if pat.contains("(?i:")
            && pat.contains("[\\p{L}\\p{M}]+")
            && pat.contains("[^\\s\\p{L}\\p{M}\\p{N}]")
        {
            return Some(Pretok::Generic { max_digits: digit_cap(), marks: true });
        }
        if pat.contains("(?i:") && pat.contains("\\p{L}+") {
            return Some(Pretok::Generic { max_digits: digit_cap(), marks: false });
        }
    }
    None
}

/// Recognize the exact DeepSeek-V3/R1/V4 pre-tokenizer: a 3-stage Isolated Split
/// sequence (`\p{N}{1,3}`, a CJK/kana run, then the mark-inclusive main pattern)
/// followed by a non-regex ByteLevel. Matched byte-exact and fail-closed: any
/// deviation returns false and the model takes the (correct, slower) fancy-regex
/// fallback instead. When it matches, `cjk::chunk_ranges` reproduces it ~10x
/// faster. See src/cjk.rs.
fn is_cjk_pretok(pre: &Value) -> bool {
    const STAGE1: &str = "\\p{N}{1,3}";
    const STAGE2: &str = "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+";
    const STAGE3: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";
    if pre.get("type").and_then(|v| v.as_str()) != Some("Sequence") {
        return false;
    }
    let pts = match pre.get("pretokenizers").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return false,
    };
    if pts.len() != 4 {
        return false;
    }
    let is_split = |pt: &Value, expect: &str| {
        pt.get("type").and_then(|v| v.as_str()) == Some("Split")
            && pt.get("behavior").and_then(|v| v.as_str()) == Some("Isolated")
            && !pt.get("invert").and_then(|v| v.as_bool()).unwrap_or(false)
            && pt.get("pattern").and_then(|p| p.get("Regex")).and_then(|v| v.as_str())
                == Some(expect)
    };
    is_split(&pts[0], STAGE1)
        && is_split(&pts[1], STAGE2)
        && is_split(&pts[2], STAGE3)
        && pts[3].get("type").and_then(|v| v.as_str()) == Some("ByteLevel")
        && !pts[3].get("use_regex").and_then(|v| v.as_bool()).unwrap_or(true)
}

/// For a ByteLevel pre-tokenizer the fast machines don't recognize, return the
/// ordered list of split regexes to run via fancy-regex (one per Split / Digits
/// stage). Returns None if it isn't a generically-handleable ByteLevel pipeline
/// (e.g. byte_fallback handled elsewhere, Metaspace, Punctuation, non-Isolated
/// Split behavior, or a ByteLevel that also applies use_regex alongside splits).
fn extract_split_fallback(pre: &Value) -> Option<Vec<String>> {
    let t = pre.get("type")?.as_str()?;
    if t != "Sequence" {
        return None;
    }
    let pts = pre.get("pretokenizers")?.as_array()?;
    let mut stages: Vec<String> = Vec::new();
    let mut has_bl = false;
    for pt in pts {
        match pt.get("type").and_then(|v| v.as_str()) {
            Some("ByteLevel") => {
                has_bl = true;
                // If ByteLevel itself splits (use_regex) on top of explicit
                // Splits, the composition is ambiguous — bail.
                if pt.get("use_regex").and_then(|v| v.as_bool()).unwrap_or(true)
                    && !stages.is_empty()
                {
                    return None;
                }
            }
            Some("Split") => {
                // Only the standard Isolated, non-inverted Split is a simple
                // "matches are the pre-tokens, gaps kept" segmentation.
                if pt.get("behavior").and_then(|v| v.as_str()) != Some("Isolated")
                    || pt.get("invert").and_then(|v| v.as_bool()).unwrap_or(false)
                {
                    return None;
                }
                let re = pt.get("pattern").and_then(|p| p.get("Regex")).and_then(|v| v.as_str())?;
                // \A / \G / \z / \Z anchor against the WHOLE input or the previous
                // match end. We apply each Split stage chunk-wise (HF "Isolated"),
                // so those anchors would bind to chunk boundaries instead — silently
                // mis-grouping (e.g. right-aligned 3-digit number grouping). Reject
                // rather than mistokenize.
                if re.contains("\\A") || re.contains("\\G")
                    || re.contains("\\z") || re.contains("\\Z")
                {
                    return None;
                }
                stages.push(re.to_string());
            }
            Some("Digits") => {
                let indiv = pt.get("individual_digits").and_then(|v| v.as_bool()).unwrap_or(false);
                stages.push(if indiv { "\\p{N}".to_string() } else { "\\p{N}+".to_string() });
            }
            // Anything else (Metaspace, Punctuation, ...) we can't reproduce here.
            _ => return None,
        }
    }
    if has_bl && !stages.is_empty() {
        Some(stages)
    } else {
        None
    }
}

/// Chunks longer than this use the O(n) streaming merger; shorter ones use the
/// cache-hot scan-and-merge (faster for typical short post-pretok chunks).
const STREAM_THRESHOLD: usize = 256;

/// Which BPE merge implementation to run. `Auto` picks per chunk (streaming for
/// long byte-level chunks, scan otherwise); `Stream`/`Scan` force one — used by
/// the `_encode_ordinary_stream`/`_encode_ordinary_scan` parity-test entry points
/// so both implementations can be checked head-to-head on identical input.
#[derive(Clone, Copy, PartialEq)]
enum Merge {
    Auto,
    Stream,
    Scan,
}

#[pyclass(module = "tuetoken")]
struct Tokenizer {
    kind: Pretok,
    singleton: [i32; 256],
    map: bpe::MergeMap,
    stream: bpe::StreamTables,
    // Decode: all token bytes in ONE contiguous arena (cache-friendly), indexed
    // by (offset, len) per token id — instead of a Vec<Vec<u8>> of scattered
    // heap allocations. Matches the C engine's vocab arena.
    byte_arena: Vec<u8>,
    byte_span: Vec<(u32, u32)>,
    ignore_lookup: Option<FxHashMap<Box<[u8]>, u32>>,
    vocab_size: usize,
    // Non-empty => use the generic regex pre-tokenizer (fancy-regex) instead of
    // `kind`'s state machine, for ByteLevel Split patterns we don't hand-code.
    fallback_re: Vec<fancy_regex::Regex>,
    // --- SentencePiece / byte_fallback path ---
    byte_fallback: bool,
    meta_mode: u8, // 0 none, 1 replace-only, 2 always-prepend, 3 prepend-first
    norm_form: u8, // 0 none, 1 NFC, 2 NFD, 3 NFKC, 4 NFKD
    // char (codepoint) -> token id for single-character vocab tokens. ASCII goes
    // in the array (hot path), the rest in the map.
    char_ascii: [i32; 128],
    char_map: FxHashMap<u32, u32>,
    decode_strip_space: bool, // SentencePiece: strip one leading space on decode
    source_path: String,      // tokenizer.json path (for repr + pickle)
    is_byte_level: bool,      // true ByteLevel (not byte_fallback) — offsets supported
    // True when every token is produced before it is consumed (`produce_max <
    // consume_min`) — i.e. the merge ranks are monotonic, which guarantees the fast
    // batch `merge` and streaming merger equal canonical BPE. False for vocabularies
    // with rank inversions (e.g. gemma's whitespace-run tokens), which route to the
    // canonical merger. Every normal trained BPE is monotonic; see bpe::merge.
    monotonic: bool,
    // Use the hand-written DeepSeek 3-stage chunker (src/cjk.rs) instead of the
    // fancy-regex fallback — set only when the pre-tokenizer matches it byte-exact.
    cjk: bool,
    // Special-token splitter for the AutoTokenizer path: a leftmost-longest matcher
    // over the added-token surfaces, so `encode_special` tokenizes a whole chat
    // string (specials + gaps) in one Rust pass instead of Python regex + a call
    // per fragment. Set once via `set_special_tokens`; None until then.
    special_ac: Option<aho_corasick::AhoCorasick>,
    special_ids: Vec<u32>,     // pattern i -> token id
    special_lstrip: Vec<bool>, // absorb whitespace to the left of the match
    special_rstrip: Vec<bool>, // absorb whitespace to the right
    prepend_scheme: u8,        // 0 none, 1 always, 2 first (metaspace prepend)
}

#[derive(Default)]
struct Scratch {
    out: Vec<u32>,
    chunk: Vec<u32>,
    // reusable scratch for the canonical merger (non-monotonic vocabularies).
    canon: bpe::CanonScratch,
    // scan-and-merge buffers: cached pair ranks (+ ping-pong), next tokens, src
    ranks: Vec<u32>,
    ntok: Vec<u32>,
    src: Vec<i32>,
    nrank: Vec<u32>,
    state: Vec<u32>,
    todo: Vec<(u32, u32)>,
}

impl Tokenizer {
    /// Map `f` over `items` in parallel, sizing rayon tasks by WORK (each task
    /// gets ~MIN_TASK_BYTES of `weight`) so cheap per-item work doesn't over-split
    /// — the same workload-independent rule used by encode_ordinary_batch. Falls
    /// back to serial for 1 thread / a failed pool. Used by all *_batch methods.
    fn par_over<T: Sync, R: Send>(
        &self,
        num_threads: usize,
        items: &[T],
        weight: impl Fn(&T) -> usize + Sync,
        f: impl Fn(&T) -> R + Sync,
    ) -> Vec<R> {
        const MIN_TASK_BYTES: usize = 4096;
        let nt = if num_threads == 0 { 0 } else { num_threads.min(1024) };
        if nt == 1 || items.len() <= 1 {
            return items.iter().map(f).collect();
        }
        let total: usize = items.iter().map(&weight).sum();
        let avg = (total / items.len().max(1)).max(1);
        let min_len = (MIN_TASK_BYTES / avg).max(1);
        let run = || {
            items
                .par_iter()
                .with_min_len(min_len)
                .map(&f)
                .collect::<Vec<R>>()
        };
        if nt == 0 {
            run()
        } else {
            match fixed_pool(nt) {
                Some(pool) => pool.install(run),
                None => run(),
            }
        }
    }

    /// Gather the raw bytes for `ids` from the contiguous token-byte arena.
    /// Two passes: sum lengths for one exact allocation, then memcpy each span.
    fn gather_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let nspans = self.byte_span.len();
        let mut total = 0usize;
        for &t in ids {
            if (t as usize) < nspans {
                total += unsafe { self.byte_span.get_unchecked(t as usize).1 } as usize;
            }
        }
        let mut buf: Vec<u8> = Vec::with_capacity(total);
        for &t in ids {
            let ti = t as usize;
            if ti < nspans {
                // SAFETY: ti < nspans; span (off,len) points inside byte_arena.
                let (off, len) = unsafe { *self.byte_span.get_unchecked(ti) };
                let (off, len) = (off as usize, len as usize);
                buf.extend_from_slice(unsafe { self.byte_arena.get_unchecked(off..off + len) });
            }
        }
        buf
    }

    /// SentencePiece decode post-step: strip one leading space (the metaspace
    /// prepend) when the model's decoder declares a `Strip{" "}`.
    #[inline]
    fn strip_decoded<'a>(&self, buf: &'a [u8]) -> &'a [u8] {
        if self.decode_strip_space && buf.first() == Some(&b' ') {
            &buf[1..]
        } else {
            buf
        }
    }

    /// Run BPE merges over the initial tokens in `s.chunk` into `s.out`.
    #[inline]
    fn finish_chunk(&self, s: &mut Scratch, mode: Merge) {
        if s.chunk.is_empty() {
            return;
        }
        let long = s.chunk.len() > STREAM_THRESHOLD;
        // The streaming merger reproduces scan-and-merge exactly for byte-level
        // merge tables, but NOT for the char-based byte_fallback/metaspace tables
        // (on a long ▁-run it under-merges), and NOT for non-monotonic vocabularies
        // (it isn't canonical there). So it's byte-level + monotonic only; everything
        // else scans, exactly like the C engine.
        let stream = !self.byte_fallback
            && self.monotonic
            && match mode {
                Merge::Stream => true,
                Merge::Scan => false,
                Merge::Auto => long,
            };
        if stream {
            // O(n) streaming — avoids the scan-and-merge O(n^2) on long chunks
            self.stream
                .merge_into(&s.chunk, &mut s.out, &mut s.state, &mut s.todo);
            return;
        }
        if self.monotonic {
            // Monotonic vocab (every normal trained BPE): the batch merge equals
            // canonical. debug-only: it must never report a rank inversion here.
            let ok = bpe::merge(&mut s.chunk, &self.map, &mut s.ranks, &mut s.ntok,
                                &mut s.src, &mut s.nrank);
            debug_assert!(ok, "monotonic vocab hit a rank inversion");
            let _ = ok;
            s.out.extend_from_slice(&s.chunk);
        } else {
            // Non-monotonic vocab (rank-inverted, e.g. gemma whitespace): the batch
            // and streaming mergers aren't canonical, so use the O(n log n) canonical
            // merger. Only these rare vocabularies pay for it.
            bpe::merge_canonical(&mut s.chunk, &self.map, &mut s.canon);
            s.out.extend_from_slice(&s.chunk);
        }
    }

    /// Emit one pre-token chunk: byte_fallback (char lookup) or byte-level.
    #[inline]
    fn emit(&self, chunk: &[u8], s: &mut Scratch, mode: Merge) {
        if self.byte_fallback {
            self.process_chunk_bf(chunk, s, mode);
        } else {
            self.process_chunk(chunk, s, mode);
        }
    }

    /// ByteLevel chunk: each byte -> its singleton token, then merge.
    #[inline]
    fn process_chunk(&self, chunk: &[u8], s: &mut Scratch, mode: Merge) {
        if chunk.is_empty() {
            return;
        }
        if let Some(map) = &self.ignore_lookup {
            if let Some(&id) = map.get(chunk) {
                s.out.push(id);
                return;
            }
        }
        s.chunk.clear();
        for &b in chunk {
            let t = self.singleton[b as usize];
            if t >= 0 {
                s.chunk.push(t as u32);
            }
        }
        self.finish_chunk(s, mode);
    }

    /// byte_fallback chunk: each codepoint -> its char token, else its UTF-8
    /// bytes -> `<0xXX>` byte tokens; then merge. Mirrors the C engine exactly.
    #[inline]
    fn process_chunk_bf(&self, chunk: &[u8], s: &mut Scratch, mode: Merge) {
        s.chunk.clear();
        let n = chunk.len();
        let mut i = 0usize;
        while i < n {
            let (cp, cl) = unicode::decode(chunk, i);
            let tid = if cp < 128 {
                self.char_ascii[cp as usize]
            } else {
                self.char_map.get(&cp).map(|&x| x as i32).unwrap_or(-1)
            };
            if tid >= 0 {
                s.chunk.push(tid as u32);
            } else {
                let end = (i + cl).min(n);
                for &b in &chunk[i..end] {
                    let t = self.singleton[b as usize];
                    if t >= 0 {
                        s.chunk.push(t as u32);
                    }
                }
            }
            i += cl;
        }
        self.finish_chunk(s, mode);
    }

    /// Unicode-normalize input (NFC/NFD/NFKC/NFKD) if the model declares one.
    /// The quick-check short-circuits already-normalized text to a borrow (no
    /// allocation), so the common case — ASCII / already-NFC input on a model that
    /// declares an NFC normalizer — stays zero-copy on the hot path.
    #[inline]
    fn normalize<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        use std::borrow::Cow;
        use unicode_normalization::{
            is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick, IsNormalized,
            UnicodeNormalization,
        };
        let yes = IsNormalized::Yes;
        match self.norm_form {
            1 if is_nfc_quick(text.chars()) == yes => Cow::Borrowed(text),
            1 => Cow::Owned(text.nfc().collect()),
            2 if is_nfd_quick(text.chars()) == yes => Cow::Borrowed(text),
            2 => Cow::Owned(text.nfd().collect()),
            3 if is_nfkc_quick(text.chars()) == yes => Cow::Borrowed(text),
            3 => Cow::Owned(text.nfkc().collect()),
            4 if is_nfkd_quick(text.chars()) == yes => Cow::Borrowed(text),
            4 => Cow::Owned(text.nfkd().collect()),
            _ => Cow::Borrowed(text),
        }
    }

    /// SentencePiece metaspace: ' ' -> U+2581, with the model's prepend variant.
    /// `prepend` gates the leading metaspace: the HF `AutoTokenizer` wrapper passes
    /// false for fragments after a special token under `prepend_scheme="first"`
    /// (only the first fragment of the whole sequence gets the ▁).
    fn apply_metaspace(&self, text: &str, prepend: bool) -> String {
        if text.is_empty() {
            return String::new(); // empty input -> no tokens (no stray prepend)
        }
        let mut out = String::with_capacity(text.len() + 4);
        if prepend && self.meta_mode == 2 {
            out.push('\u{2581}'); // always-prepend
        }
        for ch in text.chars() {
            if ch == ' ' {
                out.push('\u{2581}');
            } else {
                out.push(ch);
            }
        }
        if prepend && self.meta_mode == 3 && !out.starts_with('\u{2581}') {
            out.insert(0, '\u{2581}'); // prepend-first
        }
        out
    }

    /// Split metaspace text on U+2581 boundaries (`▁* non-▁*`) and emit each.
    fn encode_meta(&self, meta: &str, s: &mut Scratch, mode: Merge) {
        let b = meta.as_bytes();
        let n = b.len();
        let is_ms = |p: usize| p + 2 < n && b[p] == 0xE2 && b[p + 1] == 0x96 && b[p + 2] == 0x81;
        let mut pos = 0usize;
        while pos < n {
            let start = pos;
            while is_ms(pos) {
                pos += 3; // leading ▁ run
            }
            if pos > start && pos >= n {
                self.emit(&b[start..pos], s, mode);
                break;
            }
            while pos < n {
                if is_ms(pos) {
                    if pos == start {
                        pos += 3;
                        continue;
                    }
                    break;
                }
                let c = b[pos];
                pos += if c < 0x80 { 1 } else if c < 0xE0 { 2 } else if c < 0xF0 { 3 } else { 4 };
                if pos > n {
                    pos = n;
                }
            }
            if pos == start {
                pos = (start + 1).min(n);
            }
            self.emit(&b[start..pos], s, mode);
        }
    }

    /// Fast path: hand-written state machine produces chunk boundaries.
    #[inline]
    fn encode_bytes(&self, bytes: &[u8], s: &mut Scratch, mode: Merge) {
        let n = bytes.len();
        let mut pos = 0usize;
        while pos < n {
            let end = pretok::next_chunk_end(self.kind, bytes, pos);
            self.emit(&bytes[pos..end], s, mode);
            pos = end;
        }
    }

    /// Fallback path: pre-tokenize via the model's actual Split regex(es)
    /// (`fancy-regex`), composing multiple Split stages with HF "Isolated"
    /// semantics (each match isolated, gaps kept). Used only for ByteLevel
    /// models whose pattern the fast machines don't recognize.
    /// Byte ranges of the pre-token pieces for the fancy-regex Split fallback,
    /// composing each Split stage with HF "Isolated" semantics (matches isolated,
    /// gaps kept). Shared by `encode_text_regex` and `encode_with_offsets`.
    fn regex_chunk_ranges(&self, text: &str) -> Vec<(usize, usize)> {
        let mut pieces: Vec<(usize, usize)> = vec![(0, text.len())];
        let mut next: Vec<(usize, usize)> = Vec::new();
        for re in &self.fallback_re {
            next.clear();
            for &(ps, pe) in &pieces {
                let sub = &text[ps..pe];
                let mut last = ps;
                for m in re.find_iter(sub) {
                    let m = match m {
                        Ok(m) => m,
                        Err(_) => break, // backtracking blow-up: stop refining
                    };
                    let (ms, me) = (ps + m.start(), ps + m.end());
                    if me == ms {
                        continue; // skip zero-width matches
                    }
                    if ms > last {
                        next.push((last, ms)); // gap before the match
                    }
                    next.push((ms, me));
                    last = me;
                }
                if last < pe {
                    next.push((last, pe));
                }
            }
            std::mem::swap(&mut pieces, &mut next);
        }
        pieces
    }

    fn encode_text_regex(&self, text: &str, s: &mut Scratch, mode: Merge) {
        let bytes = text.as_bytes();
        for (ps, pe) in self.regex_chunk_ranges(text) {
            self.emit(&bytes[ps..pe], s, mode);
        }
    }

    /// DeepSeek path: hand-written 3-stage chunker, then emit each piece.
    fn encode_cjk(&self, text: &str, s: &mut Scratch, mode: Merge) {
        let bytes = text.as_bytes();
        let (mut cur, mut nxt) = (Vec::new(), Vec::new());
        cjk::chunk_ranges(text, &mut cur, &mut nxt);
        for &(ps, pe) in &cur {
            self.emit(&bytes[ps..pe], s, mode);
        }
    }

    /// BPE one ByteLevel chunk while tracking each token's byte span in the input.
    fn emit_chunk_offsets(
        &self,
        bytes: &[u8],
        cs: usize,
        ce: usize,
        ids: &mut Vec<u32>,
        spans: &mut Vec<(usize, usize)>,
        toks: &mut Vec<u32>,
        sp: &mut Vec<(usize, usize)>,
        pairs: &mut Vec<(u32, u32)>,
    ) {
        let chunk = &bytes[cs..ce];
        if chunk.is_empty() {
            return;
        }
        if let Some(map) = &self.ignore_lookup {
            if let Some(&id) = map.get(chunk) {
                ids.push(id);
                spans.push((cs, ce));
                return;
            }
        }
        toks.clear();
        sp.clear();
        for (j, &b) in chunk.iter().enumerate() {
            let t = self.singleton[b as usize];
            if t >= 0 {
                toks.push(t as u32);
                sp.push((cs + j, cs + j + 1));
            }
        }
        bpe::merge_with_spans(toks, sp, &self.map, pairs);
        ids.extend_from_slice(toks);
        spans.extend_from_slice(sp);
    }

    fn encode_one(&self, text: &str, mode: Merge) -> Vec<u32> {
        self.encode_one_p(text, mode, true)
    }

    /// `prepend`: apply the SentencePiece metaspace prepend (true) or skip it
    /// (false, for non-first fragments under `prepend_scheme="first"`). Only the
    /// metaspace path uses it; all other paths ignore it.
    fn encode_one_p(&self, text: &str, mode: Merge, prepend: bool) -> Vec<u32> {
        let mut s = Scratch::default();
        // The normalizer runs first in the HF pipeline (normalizer -> pre_tokenizer
        // -> model), for ByteLevel models too — not just SentencePiece. Skipping it
        // silently mistokenizes decomposed/combining Unicode (NFD text, jamo, Å, …)
        // on every model that declares one (a large share of popular models).
        let norm = self.normalize(text);
        let text = norm.as_ref();
        if self.meta_mode != 0 {
            // SentencePiece: metaspace -> split on U+2581 -> emit.
            let meta = self.apply_metaspace(text, prepend);
            self.encode_meta(&meta, &mut s, mode);
        } else if self.byte_fallback {
            // byte_fallback without metaspace (Split pretok, or whole-text).
            if self.fallback_re.is_empty() {
                self.emit(text.as_bytes(), &mut s, mode);
            } else {
                self.encode_text_regex(text, &mut s, mode);
            }
        } else if self.cjk {
            self.encode_cjk(text, &mut s, mode);
        } else if !self.fallback_re.is_empty() {
            self.encode_text_regex(text, &mut s, mode);
        } else {
            self.encode_bytes(text.as_bytes(), &mut s, mode);
        }
        s.out
    }

    /// BPE one inter-special fragment and append to `out`. `at_start` => the
    /// fragment begins at byte 0 (gets the metaspace prefix under scheme "first").
    #[inline]
    fn encode_fragment(&self, frag: &str, at_start: bool, out: &mut Vec<u32>) {
        if frag.is_empty() {
            return;
        }
        let prepend = self.prepend_scheme != 2 || at_start;
        out.extend(self.encode_one_p(frag, Merge::Auto, prepend));
    }

    /// Special-token-aware encode of one string (no GIL handling) — shared by the
    /// `encode_special` and `encode_special_batch` entry points.
    fn encode_special_one(&self, text: &str) -> Vec<u32> {
        let ac = match &self.special_ac {
            Some(a) => a,
            None => return self.encode_one(text, Merge::Auto),
        };
        let b = text.as_bytes();
        let n = b.len();
        let is_ws = |x: u8| matches!(x, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c);
        let mut out: Vec<u32> = Vec::new();
        let mut last = 0usize;
        for m in ac.find_iter(text) {
            let pat = m.pattern().as_usize();
            let (mut start, mut end) = (m.start(), m.end());
            if self.special_lstrip[pat] {
                while start > last && is_ws(b[start - 1]) {
                    start -= 1;
                }
            }
            if self.special_rstrip[pat] {
                while end < n && is_ws(b[end]) {
                    end += 1;
                }
            }
            if start > last {
                self.encode_fragment(&text[last..start], last == 0, &mut out);
            }
            out.push(self.special_ids[pat]);
            last = end;
        }
        if last < n {
            self.encode_fragment(&text[last..], last == 0, &mut out);
        }
        out
    }
}

#[pymethods]
impl Tokenizer {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("cannot read {path}: {e}")))?;
        let cfg: Value = serde_json::from_str(&data)
            .map_err(|e| PyValueError::new_err(format!("bad JSON: {e}")))?;
        let model = cfg.get("model").ok_or_else(|| PyValueError::new_err("no model"))?;

        // Fail closed on any normalizer stage we don't actually apply, rather
        // than silently ignoring it and mistokenizing (e.g. a Lowercase model).
        if let Some(t) = unsupported_normalizer_stage(&cfg) {
            return Err(PyValueError::new_err(format!(
                "unsupported normalizer stage {t:?}: this model rewrites text in a \
                 way tuetoken does not reproduce (only NFC/NFD/NFKC/NFKD and \
                 the SentencePiece metaspace Prepend/Replace are modeled)"
            )));
        }

        let byte_fallback = model.get("byte_fallback").and_then(|v| v.as_bool()).unwrap_or(false);
        let meta_mode = detect_meta_mode(&cfg);
        let norm_form = detect_norm_form(&cfg);
        let decode_strip_space = detect_decode_strip(&cfg);
        let pre = cfg.get("pre_tokenizer").unwrap_or(&Value::Null);

        let compile_split = |pats: &[String]| -> PyResult<Vec<fancy_regex::Regex>> {
            let mut res = Vec::with_capacity(pats.len());
            for p in pats {
                res.push(fancy_regex::Regex::new(p).map_err(|e| {
                    PyValueError::new_err(format!("cannot compile split regex {p:?}: {e}"))
                })?);
            }
            Ok(res)
        };

        // Chunking source: a byte-level machine, a fancy-regex Split fallback,
        // metaspace (▁-split), or whole-text. byte_fallback only affects how each
        // chunk is turned into initial tokens (char lookup vs raw bytes).
        let (kind, fallback_re) = if byte_fallback {
            if meta_mode != 0 {
                (Pretok::Word(false), Vec::new()) // chunker = encode_meta
            } else {
                match extract_split_fallback(pre) {
                    Some(pats) => (Pretok::Word(false), compile_split(&pats)?),
                    // A missing/null pre_tokenizer is a legitimate whole-text
                    // SentencePiece model (one chunk). But a pre_tokenizer we DON'T
                    // recognize (e.g. Punctuation) must fail closed — collapsing it
                    // to one chunk would silently mistokenize. Same fail-closed rule
                    // the ByteLevel path already applies below.
                    None if pre.is_null() => (Pretok::Word(false), Vec::new()),
                    None => {
                        return Err(PyValueError::new_err(
                            "unsupported pre_tokenizer for byte_fallback model \
                             (cannot be reproduced; refusing to mistokenize)",
                        ))
                    }
                }
            }
        } else if is_cjk_pretok(pre) {
            // Hand-written 3-stage DeepSeek chunker (cjk::chunk_ranges).
            (Pretok::Word(false), Vec::new())
        } else {
            // Fast path: a recognized machine. Otherwise fall back to running the
            // model's actual Split regex(es) via fancy-regex (universal ByteLevel).
            match detect_pretok(pre) {
                Some(k) => (k, Vec::new()),
                None => {
                    let pats = extract_split_fallback(pre).ok_or_else(|| {
                        PyValueError::new_err(
                            "unsupported pre_tokenizer (ByteLevel or byte_fallback)",
                        )
                    })?;
                    (Pretok::Word(false), compile_split(&pats)?)
                }
            }
        };
        let ignore_merges = model.get("ignore_merges").and_then(|v| v.as_bool()).unwrap_or(false);

        let c2b = bytes_to_unicode_c2b();
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| PyValueError::new_err("no vocab"))?;
        let mut vocab: FxHashMap<String, u32> = FxHashMap::default();
        let mut max_id = 0u32;
        for (k, v) in vocab_obj {
            let id = v.as_u64().unwrap_or(0) as u32;
            vocab.insert(k.clone(), id);
            if id > max_id {
                max_id = id;
            }
        }
        let mut vocab_size = (max_id as usize) + 1;

        // added_tokens may extend the id space
        if let Some(arr) = cfg.get("added_tokens").and_then(|v| v.as_array()) {
            for at in arr {
                if let Some(id) = at.get("id").and_then(|v| v.as_u64()) {
                    vocab_size = vocab_size.max(id as usize + 1);
                }
            }
        }

        let mut token_to_bytes: Vec<Vec<u8>> = vec![Vec::new(); vocab_size];
        // A single-byte token is "canonical" iff its string is exactly the
        // byte-level alphabet char for that byte (e.g. 'Ġ' for 0x20). A stray raw
        // char (e.g. ' ', which UTF-8s to 0x20) is unreachable once input is
        // byte-level encoded, so it must not shadow the canonical token below.
        let mut canonical = vec![false; vocab_size];
        // byte_fallback: codepoint -> token id for single-character vocab tokens.
        let mut char_ascii = [-1i32; 128];
        let mut char_map: FxHashMap<u32, u32> = FxHashMap::default();
        for (k, &id) in &vocab {
            token_to_bytes[id as usize] = if byte_fallback {
                decode_byte_fallback(k)
            } else {
                decode_bytelevel(k, &c2b)
            };
            let mut chs = k.chars();
            match (chs.next(), chs.next()) {
                (Some(c), None) => {
                    // single-character token
                    if byte_fallback {
                        let cp = c as u32;
                        if cp < 128 {
                            char_ascii[cp as usize] = id as i32;
                        } else {
                            char_map.insert(cp, id);
                        }
                    } else {
                        canonical[id as usize] = c2b.contains_key(&c);
                    }
                }
                _ => {}
            }
        }
        if let Some(arr) = cfg.get("added_tokens").and_then(|v| v.as_array()) {
            for at in arr {
                if let (Some(content), Some(id)) = (
                    at.get("content").and_then(|v| v.as_str()),
                    at.get("id").and_then(|v| v.as_u64()),
                ) {
                    let id = id as usize;
                    if id < vocab_size && token_to_bytes[id].is_empty() {
                        token_to_bytes[id] = content.as_bytes().to_vec();
                    }
                }
            }
        }

        let mut singleton = [-1i32; 256];
        let mut singleton_canon = [false; 256];
        for id in 0..vocab_size {
            let b = &token_to_bytes[id];
            if b.len() == 1 {
                let bv = b[0] as usize;
                let canon = canonical[id];
                // prefer canonical; among equal canonicity prefer the smaller id
                if singleton[bv] < 0
                    || (canon && !singleton_canon[bv])
                    || (canon == singleton_canon[bv] && (id as i32) < singleton[bv])
                {
                    singleton[bv] = id as i32;
                    singleton_canon[bv] = canon;
                }
            }
        }

        let mut merges: Vec<(u32, u32, u32)> = Vec::new();
        if let Some(arr) = model.get("merges").and_then(|v| v.as_array()) {
            for m in arr {
                let pair = if let Some(a) = m.as_array() {
                    match (a.get(0).and_then(|x| x.as_str()), a.get(1).and_then(|x| x.as_str())) {
                        (Some(x), Some(y)) => Some((x.to_string(), y.to_string())),
                        _ => None,
                    }
                } else if let Some(s) = m.as_str() {
                    s.find(' ').map(|sp| (s[..sp].to_string(), s[sp + 1..].to_string()))
                } else {
                    None
                };
                if let Some((t1s, t2s)) = pair {
                    let merged = format!("{t1s}{t2s}");
                    if let (Some(&t1), Some(&t2), Some(&tg)) =
                        (vocab.get(&t1s), vocab.get(&t2s), vocab.get(&merged))
                    {
                        merges.push((t1, t2, tg));
                    }
                }
            }
        }
        let map = bpe::MergeMap::new(&merges);
        let stream = bpe::StreamTables::new(&merges, vocab_size);

        // Monotonicity: a vocab is "monotonic" when every token is fully produced
        // (its highest-rank producing merge) before it is first consumed (its
        // lowest-rank consuming merge). That guarantees the fast batch/streaming
        // mergers equal canonical BPE. Every normal trained BPE satisfies it; a few
        // models hand-assign inverted ranks (gemma's whitespace-run tokens give
        // `(30-run)+(1-run)->31-run` a near-top rank) and route to the canonical
        // merger instead. See bpe::merge / finish_chunk.
        let monotonic = {
            let mut produce_max: Vec<i64> = vec![-1; vocab_size];
            let mut consume_min: Vec<i64> = vec![i64::MAX; vocab_size];
            for (i, &(a, b, t)) in merges.iter().enumerate() {
                let i = i as i64;
                if (t as usize) < vocab_size {
                    produce_max[t as usize] = produce_max[t as usize].max(i);
                }
                if (a as usize) < vocab_size {
                    consume_min[a as usize] = consume_min[a as usize].min(i);
                }
                if (b as usize) < vocab_size {
                    consume_min[b as usize] = consume_min[b as usize].min(i);
                }
            }
            (0..vocab_size).all(|t| produce_max[t] < 0 || consume_min[t] > produce_max[t])
        };

        let ignore_lookup = if ignore_merges {
            let mut m: FxHashMap<Box<[u8]>, u32> = FxHashMap::default();
            for id in 0..vocab_size {
                let b = &token_to_bytes[id];
                if !b.is_empty() {
                    m.entry(b.clone().into_boxed_slice()).or_insert(id as u32);
                }
            }
            Some(m)
        } else {
            None
        };

        // Flatten token_to_bytes into a contiguous arena for fast decode.
        let total: usize = token_to_bytes.iter().map(|b| b.len()).sum();
        let mut byte_arena: Vec<u8> = Vec::with_capacity(total);
        let mut byte_span: Vec<(u32, u32)> = Vec::with_capacity(token_to_bytes.len());
        for b in &token_to_bytes {
            byte_span.push((byte_arena.len() as u32, b.len() as u32));
            byte_arena.extend_from_slice(b);
        }

        Ok(Tokenizer {
            kind,
            singleton,
            map,
            stream,
            byte_arena,
            byte_span,
            ignore_lookup,
            vocab_size,
            fallback_re,
            byte_fallback,
            meta_mode,
            norm_form,
            char_ascii,
            char_map,
            decode_strip_space,
            source_path: path.to_string(),
            is_byte_level: !byte_fallback,
            monotonic,
            cjk: !byte_fallback && is_cjk_pretok(pre),
            special_ac: None,
            special_ids: Vec::new(),
            special_lstrip: Vec::new(),
            special_rstrip: Vec::new(),
            prepend_scheme: 0,
        })
    }

    /// `repr(tok)` — show what it is.
    fn __repr__(&self) -> String {
        format!(
            "Tokenizer(n_vocab={}, path={:?})",
            self.vocab_size, self.source_path
        )
    }

    /// `len(tok)` -> vocab size.
    fn __len__(&self) -> usize {
        self.vocab_size
    }

    /// Pickle support: reconstruct from the tokenizer.json path. Enables sending
    /// a Tokenizer to multiprocessing workers / datasets.map(num_proc=...).
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let cls: Py<PyAny> = py.get_type::<Tokenizer>().into_any().unbind();
        Ok((cls, (self.source_path.clone(),)))
    }

    /// Load by HuggingFace repo id (downloads `tokenizer.json`). One-liner like
    /// HF's `from_pretrained`. Requires `huggingface_hub`.
    #[staticmethod]
    #[pyo3(signature = (repo_id, revision = None))]
    fn from_pretrained(py: Python<'_>, repo_id: &str, revision: Option<&str>) -> PyResult<Self> {
        let hub = py.import("huggingface_hub").map_err(|_| {
            PyValueError::new_err(
                "from_pretrained requires huggingface_hub (pip install huggingface_hub)",
            )
        })?;
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(rev) = revision {
            kwargs.set_item("revision", rev)?;
        }
        let path: String = hub
            .call_method("hf_hub_download", (repo_id, "tokenizer.json"), Some(&kwargs))?
            .extract()?;
        Self::new(&path)
    }

    /// Load an OpenAI tiktoken encoding by name (e.g. "cl100k_base", "o200k_base").
    /// Downloads the rank file and builds a tokenizer.json (cached).
    #[staticmethod]
    fn from_tiktoken(py: Python<'_>, name: &str) -> PyResult<Self> {
        let helper = py.import("tuetoken._tiktoken")?;
        let path: String = helper
            .call_method1("get_cached_tokenizer_path", (name,))?
            .extract()?;
        Self::new(&path)
    }

    fn encode_ordinary(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.detach(|| self.encode_one(text, Merge::Auto))
    }

    /// Like `encode_ordinary` but WITHOUT the SentencePiece metaspace prepend —
    /// used by the AutoTokenizer wrapper for fragments after a special token under
    /// `prepend_scheme="first"`. Identical to `encode_ordinary` for non-metaspace
    /// models.
    fn encode_ordinary_no_prefix(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.detach(|| self.encode_one_p(text, Merge::Auto, false))
    }

    /// Register the AutoTokenizer's special tokens so `encode_special` can split on
    /// them in Rust. Parallel arrays `contents`/`ids`/`lstrip`/`rstrip`; `scheme` is
    /// the metaspace prepend rule for fragments (0 none, 1 always, 2 first).
    fn set_special_tokens(&mut self, contents: Vec<String>, ids: Vec<u32>,
                          lstrip: Vec<bool>, rstrip: Vec<bool>, scheme: u8) -> PyResult<()> {
        use aho_corasick::{AhoCorasick, MatchKind};
        self.special_ac = if contents.is_empty() {
            None
        } else {
            Some(
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(&contents)
                    .map_err(|e| PyValueError::new_err(format!("special-token matcher: {e}")))?,
            )
        };
        self.special_ids = ids;
        self.special_lstrip = lstrip;
        self.special_rstrip = rstrip;
        self.prepend_scheme = scheme;
        Ok(())
    }

    /// Special-token-aware encode (WITHOUT the post-processor's BOS/EOS): split on
    /// the registered added tokens (leftmost-longest, honouring lstrip/rstrip), BPE
    /// each gap, splice the special ids — all in one Rust pass. Same ids as the
    /// Python fragment loop, but no per-fragment Python↔Rust crossing.
    fn encode_special(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.detach(|| self.encode_special_one(text))
    }

    /// Batch of `encode_special`, parallel over rayon (GIL released) — so the
    /// AutoTokenizer's `__call__` on a list is one Rust pass instead of a Python
    /// loop. `num_threads` 0 = all cores.
    #[pyo3(signature = (texts, num_threads = 0))]
    fn encode_special_batch(&self, py: Python<'_>, texts: Vec<String>, num_threads: usize) -> Vec<Vec<u32>> {
        py.detach(|| self.par_over(num_threads, &texts, |t| t.len(), |t| self.encode_special_one(t)))
    }

    /// Buffer fast-path for the output side: ordinary-encode to a raw
    /// native-endian uint32 byte buffer (one memcpy) instead of a Python list,
    /// skipping the per-token `int` boxing that costs ~20% of `encode_ordinary`
    /// on long inputs. Wrap with `numpy.frombuffer(buf, dtype=numpy.uint32)`.
    /// The encode counterpart to `decode_array` (and matches C `encode_to_bytes`).
    fn encode_to_bytes<'py>(&self, py: Python<'py>, text: &str) -> Bound<'py, PyBytes> {
        let toks = py.detach(|| self.encode_one(text, Merge::Auto));
        // SAFETY: reinterpret &[u32] as &[u8]; u32 has no padding/invalid bytes
        // and the slice is 4*len bytes, all initialized. Native-endian, matching
        // numpy.frombuffer on the same machine.
        let bytes =
            unsafe { std::slice::from_raw_parts(toks.as_ptr() as *const u8, toks.len() * 4) };
        PyBytes::new(py, bytes)
    }

    /// Force the streaming merger on every chunk — for parity testing only.
    fn _encode_ordinary_stream(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.detach(|| self.encode_one(text, Merge::Stream))
    }

    /// Force scan-and-merge on every chunk (never stream) — for parity testing
    /// only. Lets tests pit scan vs stream on identical long inputs, which the
    /// default path can't (it auto-streams chunks over the threshold).
    fn _encode_ordinary_scan(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.detach(|| self.encode_one(text, Merge::Scan))
    }

    fn count_tokens(&self, py: Python<'_>, text: &str) -> usize {
        py.detach(|| self.encode_one(text, Merge::Auto).len())
    }

    /// Encode and return, for each token, its byte span `(start, end)` in `text`
    /// — for highlighting, NER alignment, char-level truncation. Returns
    /// `(ids, offsets)`. ByteLevel models only (byte_fallback/SentencePiece
    /// transform the text so byte offsets wouldn't map back; raises for those).
    fn encode_with_offsets(
        &self,
        py: Python<'_>,
        text: &str,
    ) -> PyResult<(Vec<u32>, Vec<(usize, usize)>)> {
        if !self.is_byte_level {
            return Err(PyValueError::new_err(
                "encode_with_offsets is only available for ByteLevel models \
                 (byte_fallback/SentencePiece transform the text, so offsets \
                 wouldn't map to the original)",
            ));
        }
        // A normalizer rewrites the text (e.g. NFC composes decomposed chars),
        // which shifts byte positions — offsets into the original would be wrong.
        // Already-normalized input is unchanged, so offsets stay valid; only fail
        // when normalization would actually alter the bytes.
        if self.norm_form != 0 && self.normalize(text) != text {
            return Err(PyValueError::new_err(
                "encode_with_offsets: input is not already normalized and this \
                 model applies a Unicode normalizer; byte offsets into the \
                 original text are undefined (normalize the text first)",
            ));
        }
        // The offset path uses the batch span-merger (`merge_with_spans`), which
        // reproduces HF only on monotonic merge tables — the same condition the
        // ordinary path checks before routing non-monotonic vocabs to the
        // canonical merger. There is no canonical span-merger, so the ids/offsets
        // would diverge from `encode`. Fail closed rather than return wrong spans.
        if !self.monotonic {
            return Err(PyValueError::new_err(
                "encode_with_offsets: this model's merge table is non-monotonic \
                 and requires the canonical merger, which the offset path does \
                 not implement; offsets would not match encode() (encode/decode \
                 are unaffected)",
            ));
        }
        let bytes = text.as_bytes();
        Ok(py.detach(|| {
            let mut ids = Vec::new();
            let mut spans = Vec::new();
            let (mut toks, mut sp, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
            if self.cjk {
                let (mut cur, mut nxt) = (Vec::new(), Vec::new());
                cjk::chunk_ranges(text, &mut cur, &mut nxt);
                for &(ps, pe) in &cur {
                    self.emit_chunk_offsets(
                        bytes, ps, pe, &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                    );
                }
            } else if self.fallback_re.is_empty() {
                let n = bytes.len();
                let mut pos = 0usize;
                while pos < n {
                    let end = pretok::next_chunk_end(self.kind, bytes, pos);
                    self.emit_chunk_offsets(
                        bytes, pos, end, &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                    );
                    pos = end;
                }
            } else {
                for (ps, pe) in self.regex_chunk_ranges(text) {
                    self.emit_chunk_offsets(
                        bytes, ps, pe, &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                    );
                }
            }
            (ids, spans)
        }))
    }

    #[pyo3(signature = (texts, num_threads = 0))]
    fn encode_ordinary_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        num_threads: usize,
    ) -> Vec<Vec<u32>> {
        // Clamp absurd thread counts so we never try to spawn millions of OS
        // threads (0 = use the global pool / all cores).
        let num_threads = if num_threads == 0 { 0 } else { num_threads.min(1024) };
        py.detach(|| {
            if num_threads == 1 {
                return texts.iter().map(|t| self.encode_one(t, Merge::Auto)).collect();
            }
            // Size each rayon task by WORK, not item count: group docs so every
            // task encodes at least ~MIN_TASK_BYTES. Encoding cost is ~linear in
            // bytes, and rayon's per-task scheduling overhead is sub-microsecond,
            // so a few KB of work per task makes that overhead negligible (<1%).
            // This is workload-independent: tiny docs batch up (no over-split, so
            // 2 threads beat serial), large docs run ~one-per-task, and the task
            // count stays high enough for good work-stealing at ANY thread count
            // — no thread-count-tuned constant.
            const MIN_TASK_BYTES: usize = 4096;
            let total: usize = texts.iter().map(|t| t.len()).sum();
            let avg = (total / texts.len().max(1)).max(1);
            let min_len = (MIN_TASK_BYTES / avg).max(1);
            let run = || -> Vec<Vec<u32>> {
                texts
                    .par_iter()
                    .with_min_len(min_len)
                    .map(|t| self.encode_one(t, Merge::Auto))
                    .collect()
            };
            if num_threads == 0 {
                run()
            } else {
                // Reuse a cached pool; if the OS can't spawn threads, run serial.
                match fixed_pool(num_threads) {
                    Some(pool) => pool.install(run),
                    None => run(),
                }
            }
        })
    }

    /// Token counts for many texts, in parallel (no Python list of ids built).
    #[pyo3(signature = (texts, num_threads = 0))]
    fn count_tokens_batch(&self, py: Python<'_>, texts: Vec<String>, num_threads: usize) -> Vec<usize> {
        py.detach(|| {
            self.par_over(num_threads, &texts, |t| t.len(), |t| self.encode_one(t, Merge::Auto).len())
        })
    }

    /// Training-loader helper: encode `texts` and return a fixed-width, padded
    /// batch as numpy arrays. Encoding runs in parallel without the GIL; the
    /// padded buffers are built in Rust and handed to numpy in one copy.
    ///
    /// `max_length` sets the column width: longer sequences are truncated, shorter
    /// ones padded with `pad_id`. If `max_length` is None the width is the longest
    /// sequence in the batch (dynamic padding). Returns a dict with
    /// `input_ids` (uint32, shape [rows, width]) and `attention_mask`
    /// (uint8, 1 = real token, 0 = pad).
    #[pyo3(signature = (texts, max_length = None, pad_id = 0, num_threads = 0))]
    fn encode_batch<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        max_length: Option<usize>,
        pad_id: u32,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seqs: Vec<Vec<u32>> =
            py.detach(|| self.par_over(num_threads, &texts, |t| t.len(), |t| self.encode_one(t, Merge::Auto)));
        let rows = seqs.len();
        let width = max_length.unwrap_or_else(|| seqs.iter().map(|s| s.len()).max().unwrap_or(0));

        // Pad/truncate into flat row-major buffers (built without numpy).
        let mut ids_flat = vec![pad_id; rows * width];
        let mut mask_flat = vec![0u8; rows * width];
        for (r, seq) in seqs.iter().enumerate() {
            let len = seq.len().min(width);
            let base = r * width;
            ids_flat[base..base + len].copy_from_slice(&seq[..len]);
            for m in &mut mask_flat[base..base + len] {
                *m = 1;
            }
        }

        // Hand the buffers to numpy in a single copy each (the `.copy()` detaches
        // the array from the temporary PyBytes, leaving a writable, owned array).
        let np = py.import("numpy").map_err(|_| {
            PyValueError::new_err("encode_batch requires numpy (pip install numpy)")
        })?;
        let ids_u8 = unsafe {
            std::slice::from_raw_parts(ids_flat.as_ptr() as *const u8, ids_flat.len() * 4)
        };
        let ids_arr = np
            .call_method1("frombuffer", (PyBytes::new(py, ids_u8), "uint32"))?
            .call_method1("reshape", ((rows, width),))?
            .call_method0("copy")?;
        let mask_arr = np
            .call_method1("frombuffer", (PyBytes::new(py, &mask_flat), "uint8"))?
            .call_method1("reshape", ((rows, width),))?
            .call_method0("copy")?;

        let out = PyDict::new(py);
        out.set_item("input_ids", ids_arr)?;
        out.set_item("attention_mask", mask_arr)?;
        Ok(out)
    }

    /// All-Rust padded batch for AutoTokenizer.__call__(return_tensors=...): special
    /// -aware encode (parallel), splice the post-processor's `prefix`/`suffix`
    /// special ids around each row, truncate the CONTENT to fit `max_length`
    /// (reserving the specials), pad to a rectangular `[rows, width]`, and build the
    /// attention mask — handed to numpy as int64 with no Python per-row work. Single
    /// -sequence post-processors only (pairs stay on the Python path).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (texts, prefix, suffix, max_length = None, pad_id = 0,
                        truncation = false, pad_to_max = false, pad_left = false,
                        trunc_left = false, num_threads = 0))]
    fn encode_special_padded<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
        prefix: Vec<i64>,
        suffix: Vec<i64>,
        max_length: Option<usize>,
        pad_id: i64,
        truncation: bool,
        pad_to_max: bool,
        pad_left: bool,
        trunc_left: bool,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n_special = prefix.len() + suffix.len();
        let contents: Vec<Vec<u32>> = py.detach(|| {
            self.par_over(num_threads, &texts, |t| t.len(), |t| self.encode_special_one(t))
        });
        let rows = contents.len();
        // content length per row after truncation (reserving the specials).
        let trunc_cap = if truncation {
            max_length.map(|ml| ml.saturating_sub(n_special))
        } else {
            None
        };
        let content_len = |c: &Vec<u32>| trunc_cap.map_or(c.len(), |cap| c.len().min(cap));
        let width = if pad_to_max {
            max_length.unwrap_or(0)
        } else {
            contents.iter().map(|c| n_special + content_len(c)).max().unwrap_or(0)
        };

        let (ids, mask) = py.detach(|| {
            let mut ids = vec![pad_id; rows * width];
            let mut mask = vec![0i64; rows * width];
            for (r, c) in contents.iter().enumerate() {
                // clamp content so prefix+content+suffix never overruns the row.
                let cl = content_len(c).min(width.saturating_sub(n_special));
                let cslice: &[u32] = if trunc_left && cl < c.len() {
                    &c[c.len() - cl..]
                } else {
                    &c[..cl]
                };
                let total = n_special + cl;
                let off = if pad_left { width - total } else { 0 };
                let mut w = r * width + off;
                for &p in &prefix {
                    ids[w] = p;
                    w += 1;
                }
                for &x in cslice {
                    ids[w] = x as i64;
                    w += 1;
                }
                for &s in &suffix {
                    ids[w] = s;
                    w += 1;
                }
                for k in 0..total {
                    mask[r * width + off + k] = 1;
                }
            }
            (ids, mask)
        });

        let np = py.import("numpy").map_err(|_| {
            PyValueError::new_err("return_tensors requires numpy (pip install numpy)")
        })?;
        let to_np = |buf: &[i64]| -> PyResult<Bound<'py, PyAny>> {
            let u8s = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 8) };
            np.call_method1("frombuffer", (PyBytes::new(py, u8s), "int64"))?
                .call_method1("reshape", ((rows, width),))?
                .call_method0("copy")
        };
        let out = PyDict::new(py);
        out.set_item("input_ids", to_np(&ids)?)?;
        out.set_item("attention_mask", to_np(&mask)?)?;
        Ok(out)
    }

    /// Decode many token sequences, in parallel. The byte gather runs without the
    /// GIL; the Python strings are built once at the end.
    #[pyo3(signature = (sequences, num_threads = 0))]
    fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<Vec<u32>>,
        num_threads: usize,
    ) -> Bound<'py, PyList> {
        let bufs: Vec<Vec<u8>> =
            py.detach(|| self.par_over(num_threads, &sequences, |ids| ids.len() * 4, |ids| self.gather_bytes(ids)));
        let out = PyList::empty(py);
        for buf in &bufs {
            out.append(bytes_to_pystr(py, self.strip_decoded(buf))).unwrap();
        }
        out
    }

    fn decode<'py>(
        &self,
        py: Python<'py>,
        tokens: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyString>> {
        // Fast list extraction: pyo3's `Vec<u32>` FromPyObject is ~2x slower than
        // the raw CPython list API (it dominates ~44% of list-decode time). For a
        // plain list of ints, read each element directly with PyList_GET_ITEM +
        // PyLong_AsLong (matching the C engine); fall back to generic extraction
        // for other sequence types.
        let ptr = tokens.as_ptr();
        let ids: Vec<u32> = unsafe {
            if pyo3::ffi::PyList_Check(ptr) != 0 {
                let n = pyo3::ffi::PyList_GET_SIZE(ptr) as usize;
                let mut v: Vec<u32> = Vec::with_capacity(n);
                let mut bad = false;
                for i in 0..n {
                    // borrowed item ref; read its value as a signed 64-bit int.
                    let item = pyo3::ffi::PyList_GET_ITEM(ptr, i as pyo3::ffi::Py_ssize_t);
                    let val = pyo3::ffi::PyLong_AsLongLong(item);
                    // Reject out-of-range ids (negative or > u32::MAX) instead of
                    // letting `as u32` silently wrap them into valid-looking token
                    // data — bail to the checked extractor, which raises like
                    // tiktoken. A non-int element also lands here (val == -1 with a
                    // Python error set), and is likewise re-raised cleanly below.
                    if val < 0 || val > u32::MAX as i64 {
                        bad = true;
                        break;
                    }
                    v.push(val as u32);
                }
                if bad || !pyo3::ffi::PyErr_Occurred().is_null() {
                    pyo3::ffi::PyErr_Clear();
                    tokens.extract::<Vec<u32>>()?
                } else {
                    v
                }
            } else {
                tokens.extract::<Vec<u32>>()?
            }
        };
        let buf = py.detach(|| self.gather_bytes(&ids));
        Ok(bytes_to_pystr(py, self.strip_decoded(&buf)))
    }

    /// Buffer fast-path: decode from a contiguous uint32 buffer (numpy uint32
    /// array, array.array('I'), torch CPU int tensor exposing the buffer
    /// protocol) without unboxing a Python list element-by-element. The decode
    /// counterpart to `encode_to_numpy` for ML pipelines that hold token arrays.
    fn decode_array<'py>(
        &self,
        py: Python<'py>,
        tokens: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyString>> {
        let view = PyBuffer::<u32>::get(tokens)?;
        if !view.is_c_contiguous() {
            return Err(PyValueError::new_err("decode_array requires a C-contiguous uint32 buffer"));
        }
        // Copy the ids into an owned Vec WHILE HOLDING THE GIL — `view.to_vec`
        // validated the u32 element type, and reading under the GIL means another
        // thread can't resize/free the source buffer mid-read. (Reading a raw
        // pointer into the buffer after `py.detach` released the GIL would be
        // unsound for buffer exporters that don't honour the export count.) The
        // copy is a single memcpy, cheap next to the gather + str build.
        let ids: Vec<u32> = view.to_vec(py)?;
        let buf = py.detach(|| self.gather_bytes(&ids));
        Ok(bytes_to_pystr(py, self.strip_decoded(&buf)))
    }

    #[getter]
    fn n_vocab(&self) -> usize {
        self.vocab_size
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
