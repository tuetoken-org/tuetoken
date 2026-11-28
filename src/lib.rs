//! tuetoken — a Rust/PyO3 port of tuetoken's BPE engine, implementing the
//! same algorithm (state-machine pre-tokenization + open-addressing hash-map
//! BPE) for a head-to-head comparison with the C version. Batch encoding uses
//! rayon work-stealing (Rust's genuine edge over the C even-shard pthreads).
//!
//! Coverage: ByteLevel BPE (fast hand-written machines + a fancy-regex Split
//! fallback for unrecognized patterns/multi-Split) AND byte_fallback/metaspace
//! SentencePiece models (char-lookup encode + NFC normalize + metaspace). The
//! only pre-tokenizer still declined is the Punctuation pre-tokenizer.

mod blockscan;
mod bpe;
mod chunk_table;
mod cjk;
mod fast_load;
mod norm;
mod pretok;
mod pretok_pipe;
mod unicode;
mod unicode_tables;
mod utf8_simd;

use utf8_simd::{latin1_to_utf8_append, ucs2_to_utf8_append, ucs4_to_utf8_append, ucs4_to_utf8_ptr, units_to_utf8, utf8_len};
#[cfg(target_arch = "x86_64")]
use utf8_simd::has_avx2_bmi1;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use pretok::Pretok;

#[cfg(target_arch = "x86_64")]
const X86_PF_DIST: usize = 24;
#[cfg(not(target_arch = "x86_64"))]
const X86_PF_DIST: usize = 0;

/// Fork safety: rayon pools (global or cached) are threads of the process that
/// created them; after `fork()` the child inherits the pool's state but not its
/// threads, so submitting work there deadlocks. We remember the pid of the
/// first process to go parallel and run everything serially in any other
/// process (e.g. a multiprocessing child that forked after the parent encoded).
fn parallel_ok() -> bool {
    static PID: OnceLock<u32> = OnceLock::new();
    *PID.get_or_init(std::process::id) == std::process::id()
}

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

/// Flat `[u16; 324]` ByteLevel Unicode-codepoint -> raw byte table (`u16::MAX`
/// = not in the GPT-2 byte-level alphabet). Replaces `HashMap<char, u8>` SipHash
/// during load-time vocab decoding (1 array load per char vs SipHash).
fn bytes_to_unicode_table() -> [u16; 324] {
    let mut bs = [0u32; 256];
    let mut len = 0usize;
    for (lo, hi) in [(0x21u32, 0x7e), (0xa1, 0xac), (0xae, 0xff)] {
        for b in lo..=hi {
            bs[len] = b;
            len += 1;
        }
    }
    let mut cs = bs;
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs[..len].contains(&b) {
            bs[len] = b;
            cs[len] = 256 + n;
            len += 1;
            n += 1;
        }
    }
    let mut c2b = [u16::MAX; 324];
    for i in 0..256 {
        c2b[cs[i] as usize] = bs[i] as u16;
    }
    c2b
}

/// Build a Python str from decoded bytes in one CPython pass (UTF-8 validation,
/// U+FFFD replacement, and PyUnicode construction in a single scan, avoiding
/// double validation and avoiding `String::from_utf8_lossy` heap allocation on
/// invalid UTF-8).
#[inline]
fn bytes_to_pystr<'py>(py: Python<'py>, buf: &[u8]) -> Bound<'py, PyString> {
    if buf.is_empty() {
        return PyString::new(py, "");
    }
    unsafe {
        let obj = pyo3::ffi::PyUnicode_DecodeUTF8(
            buf.as_ptr() as *const std::os::raw::c_char,
            buf.len() as pyo3::ffi::Py_ssize_t,
            c"replace".as_ptr(),
        );
        if !obj.is_null() {
            return Bound::from_owned_ptr(py, obj).cast_into_unchecked();
        }
        pyo3::ffi::PyErr_Clear();
    }
    match std::str::from_utf8(buf) {
        Ok(s) => PyString::new(py, s),
        Err(_) => PyString::new(py, &String::from_utf8_lossy(buf)),
    }
}

/// Build a Python ASCII `str` directly into `PyUnicode_New(len, 127)` when the
/// caller has already proven every byte is `< 0x80` (zero UTF-8 scan).
#[inline]
fn ascii_bytes_to_pystr<'py>(py: Python<'py>, buf: &[u8]) -> Bound<'py, PyString> {
    #[cfg(not(any(PyPy, GraalPy, Py_LIMITED_API)))]
    unsafe {
        let obj = pyo3::ffi::PyUnicode_New(buf.len() as pyo3::ffi::Py_ssize_t, 127);
        if !obj.is_null() {
            if !buf.is_empty() {
                let dst = pyo3::ffi::PyUnicode_1BYTE_DATA(obj);
                std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len());
            }
            return Bound::from_owned_ptr(py, obj).cast_into_unchecked();
        }
        pyo3::ffi::PyErr_Clear();
    }
    bytes_to_pystr(py, buf)
}

/// Compact 16-byte cache-aligned decode table entry: tokens `<= 12` bytes
/// (~98.5% of real LLM tokens) live completely inline with zero `byte_arena`
/// indirection and copy in a single 16-byte SIMD unaligned store.
const DEC_FLAG_NON_ASCII: u8 = 1;
const DEC_FLAG_SPECIAL: u8 = 2;
const DEC_OUTLINE_LEN: u8 = 255;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct DecodeEntry {
    /// Inline token bytes (`0..=12`) when `len != DEC_OUTLINE_LEN`;
    /// otherwise `[off_u32_ne (0..4), len_u32_ne (4..8)]` into `byte_arena`.
    data: [u8; 12],
    len: u8,
    flags: u8,
    _pad: u16,
}

impl DecodeEntry {
    #[inline(always)]
    fn outline_span(&self) -> (usize, usize) {
        let off = u32::from_ne_bytes([self.data[0], self.data[1], self.data[2], self.data[3]]) as usize;
        let len = u32::from_ne_bytes([self.data[4], self.data[5], self.data[6], self.data[7]]) as usize;
        (off, len)
    }
}

/// Runtime-verified direct `PyLongObject` 1-digit fast reader for CPython 3.12+
/// 64-bit (handles `0 <= id < 2^30` in 3 instructions with zero C-API calls).
/// Self-verifies layout once against live `PyLong` objects so any future ABI
/// change automatically and safely disables the inline path.
#[cfg(all(Py_3_12, target_pointer_width = "64", not(any(PyPy, GraalPy, Py_LIMITED_API, Py_GIL_DISABLED))))]
#[inline]
fn pylong_inline_ok(_py: Python<'_>) -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0); // 0=unchecked, 1=no, 2=yes
    let s = STATE.load(Ordering::Relaxed);
    if s != 0 {
        return s == 2;
    }
    let ok = unsafe {
        use pyo3::ffi;
        let lt = &raw mut ffi::PyLong_Type;
        let p0 = ffi::PyLong_FromLong(0);
        let p1 = ffi::PyLong_FromLong(1);
        let pbig = ffi::PyLong_FromLong(199_999);
        let pneg = ffi::PyLong_FromLong(-1);
        let p2d = ffi::PyLong_FromLongLong(1i64 << 30);
        let mut good = false;
        if !p0.is_null() && !p1.is_null() && !pbig.is_null() && !pneg.is_null() && !p2d.is_null() {
            let tag = |p: *mut ffi::PyObject| *(p as *const usize).add(2) & !4usize;
            let dig = |p: *mut ffi::PyObject| *(p as *const u32).add(6);
            good = ffi::Py_TYPE(p0) == lt
                && ffi::Py_TYPE(p1) == lt
                && ffi::Py_TYPE(pbig) == lt
                && tag(p0) == 1
                && dig(p0) == 0
                && tag(p1) == 8
                && dig(p1) == 1
                && tag(pbig) == 8
                && dig(pbig) == 199_999
                && tag(pneg) != 1
                && tag(pneg) != 8
                && tag(p2d) != 1
                && tag(p2d) != 8;
        }
        if !p0.is_null() { ffi::Py_DECREF(p0); }
        if !p1.is_null() { ffi::Py_DECREF(p1); }
        if !pbig.is_null() { ffi::Py_DECREF(pbig); }
        if !pneg.is_null() { ffi::Py_DECREF(pneg); }
        if !p2d.is_null() { ffi::Py_DECREF(p2d); }
        good
    };
    STATE.store(if ok { 2 } else { 1 }, Ordering::Relaxed);
    ok
}

#[cfg(not(all(Py_3_12, target_pointer_width = "64", not(any(PyPy, GraalPy, Py_LIMITED_API, Py_GIL_DISABLED)))))]
#[inline]
fn pylong_inline_ok(_py: Python<'_>) -> bool {
    false
}

/// Extract a plain Python `list[int]` into `dst` (writing `n` elements at `dst`,
/// which must have capacity `>= n`), using the self-verified inline `PyLong`
/// digit path with lookahead prefetching when available. Returns `true` on
/// success or `false` on any non-u32 / non-int element (clearing no PyErr).
#[inline]
unsafe fn extract_pylist_u32_into(
    list_ptr: *mut pyo3::ffi::PyObject,
    n: usize,
    dst: *mut u32,
    inline_long: bool,
) -> bool {
    use pyo3::ffi;
    #[cfg(all(Py_3_12, target_pointer_width = "64", not(any(PyPy, GraalPy, Py_LIMITED_API, Py_GIL_DISABLED))))]
    if inline_long {
        let lt = &raw mut ffi::PyLong_Type;
        for i in 0..n {
            if i + 16 < n {
                let pf = ffi::PyList_GET_ITEM(list_ptr, (i + 16) as ffi::Py_ssize_t);
                chunk_table::prefetch(pf as *const u8);
            }
            let item = ffi::PyList_GET_ITEM(list_ptr, i as ffi::Py_ssize_t);
            if ffi::Py_TYPE(item) == lt {
                let tag = *(item as *const usize).add(2) & !4usize;
                if tag == 8 {
                    *dst.add(i) = *(item as *const u32).add(6);
                    continue;
                }
                if tag == 1 {
                    *dst.add(i) = 0;
                    continue;
                }
            }
            let val = ffi::PyLong_AsLongLong(item);
            if val < 0 || val > u32::MAX as i64 {
                return false;
            }
            *dst.add(i) = val as u32;
        }
        return ffi::PyErr_Occurred().is_null();
    }
    let _ = inline_long;
    for i in 0..n {
        let item = ffi::PyList_GET_ITEM(list_ptr, i as ffi::Py_ssize_t);
        let val = ffi::PyLong_AsLongLong(item);
        if val < 0 || val > u32::MAX as i64 {
            return false;
        }
        *dst.add(i) = val as u32;
    }
    ffi::PyErr_Occurred().is_null()
}

/// RAII guard for a CPython `Py_buffer` acquired via `PyObject_GetBuffer`.
struct PyBufGuard {
    view: pyo3::ffi::Py_buffer,
    active: bool,
}

impl Drop for PyBufGuard {
    fn drop(&mut self) {
        if self.active {
            unsafe { pyo3::ffi::PyBuffer_Release(&mut self.view) };
        }
    }
}

#[derive(Clone, Copy)]
enum IntBufKind {
    U32,
    I32,
    I64,
    U64,
    U16,
    I16,
}

/// Inspect a `Py_buffer`'s format string + itemsize and return its native-endian
/// integer element kind (if supported for zero-copy / bulk decode).
unsafe fn classify_int_buffer(view: &pyo3::ffi::Py_buffer) -> Option<IntBufKind> {
    if view.buf.is_null() {
        return None;
    }
    let fmt = if view.format.is_null() {
        b"B".as_slice()
    } else {
        std::ffi::CStr::from_ptr(view.format).to_bytes()
    };
    let code = match fmt {
        [c] => *c,
        [b'@' | b'=', c] => *c,
        [b'<', c] if cfg!(target_endian = "little") => *c,
        [b'>', c] if cfg!(target_endian = "big") => *c,
        _ => return None,
    };
    match (code, view.itemsize) {
        (b'I' | b'L', 4) => Some(IntBufKind::U32),
        (b'i' | b'l', 4) => Some(IntBufKind::I32),
        (b'q' | b'l', 8) => Some(IntBufKind::I64),
        (b'Q' | b'L', 8) => Some(IntBufKind::U64),
        (b'H', 2) => Some(IntBufKind::U16),
        (b'h', 2) => Some(IntBufKind::I16),
        _ => None,
    }
}

/// Copy `count` typed integer elements from `src` into `dst` (`Vec<u32>`),
/// checking non-negative / `<= u32::MAX` bounds. Returns `false` if out-of-range.
unsafe fn copy_int_buf_to_u32(src: *const u8, count: usize, kind: IntBufKind, dst: &mut Vec<u32>) -> bool {
    dst.clear();
    dst.reserve(count);
    let out = dst.as_mut_ptr();
    match kind {
        IntBufKind::U32 => {
            std::ptr::copy_nonoverlapping(src as *const u32, out, count);
        }
        IntBufKind::I32 => {
            let p = src as *const i32;
            let mut neg = 0i32;
            for i in 0..count {
                let v = *p.add(i);
                neg |= v;
                *out.add(i) = v as u32;
            }
            if neg < 0 {
                return false;
            }
        }
        IntBufKind::I64 => {
            let p = src as *const i64;
            let mut bad = 0u64;
            for i in 0..count {
                let v = *p.add(i);
                bad |= (v as u64) >> 32;
                *out.add(i) = v as u32;
            }
            if bad != 0 {
                return false;
            }
        }
        IntBufKind::U64 => {
            let p = src as *const u64;
            let mut bad = 0u64;
            for i in 0..count {
                let v = *p.add(i);
                bad |= v >> 32;
                *out.add(i) = v as u32;
            }
            if bad != 0 {
                return false;
            }
        }
        IntBufKind::U16 => {
            let p = src as *const u16;
            for i in 0..count {
                *out.add(i) = *p.add(i) as u32;
            }
        }
        IntBufKind::I16 => {
            let p = src as *const i16;
            let mut neg = 0i16;
            for i in 0..count {
                let v = *p.add(i);
                neg |= v;
                *out.add(i) = v as u32;
            }
            if neg < 0 {
                return false;
            }
        }
    }
    dst.set_len(count);
    true
}

#[inline]
fn decode_bytelevel_into(s: &str, c2b: &[u16; 324], out: &mut Vec<u8>) {
    for ch in s.chars() {
        let cp = ch as usize;
        if cp < 324 {
            let v = c2b[cp];
            if v != u16::MAX {
                out.push(v as u8);
                continue;
            }
        }
        let mut buf = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
    }
}

/// Decode a byte_fallback (SentencePiece) vocab token directly into `out`:
/// `<0xXX>` -> that byte, U+2581 -> space, otherwise the token's UTF-8 bytes.
#[inline]
fn decode_byte_fallback_into(s: &str, out: &mut Vec<u8>) {
    let b = s.as_bytes();
    if b.len() == 6 && &b[..3] == b"<0x" && b[5] == b'>' {
        if let Ok(v) = u8::from_str_radix(&s[3..5], 16) {
            out.push(v);
            return;
        }
    }
    if s.contains('\u{2581}') {
        for ch in s.chars() {
            if ch == '\u{2581}' {
                out.push(b' ');
            } else {
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    } else {
        out.extend_from_slice(b);
    }
}

/// Zero-copy streaming deserialization of HuggingFace `tokenizer.json`:
/// avoids building a 200K-node `BTreeMap<String, serde_json::Value>` and
/// eliminates ~600K heap allocations while resolving merges on the fly.
#[derive(serde::Deserialize)]
struct RawAddedToken {
    id: Option<u64>,
    content: Option<String>,
    #[serde(default)]
    special: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
}

#[derive(Eq, PartialEq, Hash)]
struct BorrowedStr<'a>(std::borrow::Cow<'a, str>);

impl BorrowedStr<'_> {
    #[inline]
    fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

impl std::borrow::Borrow<str> for BorrowedStr<'_> {
    #[inline]
    fn borrow(&self) -> &str {
        self.0.as_ref()
    }
}

impl<'de: 'a, 'a> serde::Deserialize<'de> for BorrowedStr<'a> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Vis<'a>(std::marker::PhantomData<&'a ()>);
        impl<'de: 'a, 'a> serde::de::Visitor<'de> for Vis<'a> {
            type Value = BorrowedStr<'a>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a string")
            }
            fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E> {
                Ok(BorrowedStr(std::borrow::Cow::Borrowed(v)))
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
                Ok(BorrowedStr(std::borrow::Cow::Owned(v.to_owned())))
            }
            fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
                Ok(BorrowedStr(std::borrow::Cow::Owned(v)))
            }
        }
        deserializer.deserialize_str(Vis(std::marker::PhantomData))
    }
}

struct RawModel<'a> {
    model_type: Option<String>,
    vocab: Option<FxHashMap<BorrowedStr<'a>, u32>>,
    max_id: u32,
    merges: Vec<(u32, u32, u32)>,
    has_merges: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
}

struct VocabSeed<'m, 'a> {
    vocab: &'m mut FxHashMap<BorrowedStr<'a>, u32>,
    max_id: &'m mut u32,
}

impl<'de: 'a, 'a> serde::de::DeserializeSeed<'de> for VocabSeed<'_, 'a> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        struct Vis<'m, 'a>(&'m mut FxHashMap<BorrowedStr<'a>, u32>, &'m mut u32);
        impl<'de: 'a, 'a> serde::de::Visitor<'de> for Vis<'_, 'a> {
            type Value = ();
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a vocab object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
                if let Some(sz) = map.size_hint() {
                    self.0.reserve(sz);
                }
                while let Some((k, id_u64)) = map.next_entry::<BorrowedStr<'a>, u64>()? {
                    let id = id_u64 as u32;
                    if id > *self.1 {
                        *self.1 = id;
                    }
                    self.0.insert(k, id);
                }
                Ok(())
            }
        }
        deserializer.deserialize_map(Vis(self.vocab, self.max_id))
    }
}

enum MergeItem<'a> {
    Str(std::borrow::Cow<'a, str>),
    Pair(BorrowedStr<'a>, BorrowedStr<'a>),
    Other,
}

impl<'de: 'a, 'a> serde::Deserialize<'de> for MergeItem<'a> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Vis<'a>(std::marker::PhantomData<&'a ()>);
        impl<'de: 'a, 'a> serde::de::Visitor<'de> for Vis<'a> {
            type Value = MergeItem<'a>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a merge string or 2-element string array")
            }
            fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E> {
                Ok(MergeItem::Str(std::borrow::Cow::Borrowed(v)))
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
                Ok(MergeItem::Str(std::borrow::Cow::Owned(v.to_owned())))
            }
            fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
                Ok(MergeItem::Str(std::borrow::Cow::Owned(v)))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let a = seq.next_element::<BorrowedStr<'a>>()?;
                let b = seq.next_element::<BorrowedStr<'a>>()?;
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                match (a, b) {
                    (Some(x), Some(y)) => Ok(MergeItem::Pair(x, y)),
                    _ => Ok(MergeItem::Other),
                }
            }
        }
        deserializer.deserialize_any(Vis(std::marker::PhantomData))
    }
}

struct MergesSeed<'m, 'a> {
    vocab: Option<&'m FxHashMap<BorrowedStr<'a>, u32>>,
    merges: &'m mut Vec<(u32, u32, u32)>,
    deferred: &'m mut Vec<(String, String)>,
}

impl<'de: 'a, 'a> serde::de::DeserializeSeed<'de> for MergesSeed<'_, 'a> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        struct Vis<'m, 'a> {
            vocab: Option<&'m FxHashMap<BorrowedStr<'a>, u32>>,
            merges: &'m mut Vec<(u32, u32, u32)>,
            deferred: &'m mut Vec<(String, String)>,
        }
        impl<'de: 'a, 'a> serde::de::Visitor<'de> for Vis<'_, 'a> {
            type Value = ();
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a merges array")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
                let mut merged_buf = String::with_capacity(64);
                while let Some(item) = seq.next_element::<MergeItem<'a>>()? {
                    let pair: Option<(&str, &str)> = match &item {
                        MergeItem::Str(s) => s.split_once(' '),
                        MergeItem::Pair(a, b) => Some((a.as_str(), b.as_str())),
                        MergeItem::Other => None,
                    };
                    if let Some((t1s, t2s)) = pair {
                        if let Some(vocab) = self.vocab {
                            if let (Some(&t1), Some(&t2)) = (vocab.get(t1s), vocab.get(t2s)) {
                                merged_buf.clear();
                                merged_buf.push_str(t1s);
                                merged_buf.push_str(t2s);
                                if let Some(&tg) = vocab.get(merged_buf.as_str()) {
                                    self.merges.push((t1, t2, tg));
                                }
                            }
                        } else {
                            self.deferred.push((t1s.to_owned(), t2s.to_owned()));
                        }
                    }
                }
                Ok(())
            }
        }
        deserializer.deserialize_seq(Vis {
            vocab: self.vocab,
            merges: self.merges,
            deferred: self.deferred,
        })
    }
}

impl<'de: 'a, 'a> serde::Deserialize<'de> for RawModel<'a> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Vis<'a>(std::marker::PhantomData<&'a ()>);
        impl<'de: 'a, 'a> serde::de::Visitor<'de> for Vis<'a> {
            type Value = RawModel<'a>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a tokenizer model object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<RawModel<'a>, A::Error> {
                let mut model_type: Option<String> = None;
                let mut vocab: Option<FxHashMap<BorrowedStr<'a>, u32>> = None;
                let mut max_id = 0u32;
                let mut merges: Vec<(u32, u32, u32)> = Vec::new();
                let mut deferred_merges: Vec<(String, String)> = Vec::new();
                let mut has_merges = false;
                let mut byte_fallback = false;
                let mut ignore_merges = false;
                let mut continuing_subword_prefix: Option<String> = None;
                let mut end_of_word_suffix: Option<String> = None;
                while let Some(key) = map.next_key::<BorrowedStr<'a>>()? {
                    match key.as_str() {
                        "type" => {
                            model_type = map.next_value::<Option<String>>()?;
                        }
                        "vocab" => {
                            if let Some(ref mt) = model_type {
                                if mt != "BPE" {
                                    let _ = map.next_value::<serde::de::IgnoredAny>()?;
                                    continue;
                                }
                            }
                            let mut v = FxHashMap::with_capacity_and_hasher(65536, Default::default());
                            map.next_value_seed(VocabSeed { vocab: &mut v, max_id: &mut max_id })?;
                            vocab = Some(v);
                        }
                        "merges" => {
                            has_merges = true;
                            map.next_value_seed(MergesSeed {
                                vocab: vocab.as_ref(),
                                merges: &mut merges,
                                deferred: &mut deferred_merges,
                            })?;
                        }
                        "byte_fallback" => {
                            byte_fallback = map.next_value::<Option<bool>>()?.unwrap_or(false);
                        }
                        "ignore_merges" => {
                            ignore_merges = map.next_value::<Option<bool>>()?.unwrap_or(false);
                        }
                        "continuing_subword_prefix" => {
                            continuing_subword_prefix = map.next_value::<Option<String>>()?;
                        }
                        "end_of_word_suffix" => {
                            end_of_word_suffix = map.next_value::<Option<String>>()?;
                        }
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }
                if !deferred_merges.is_empty() {
                    if let Some(ref v) = vocab {
                        let mut buf = String::with_capacity(64);
                        merges.reserve(deferred_merges.len());
                        for (t1s, t2s) in &deferred_merges {
                            if let (Some(&t1), Some(&t2)) = (v.get(t1s.as_str()), v.get(t2s.as_str())) {
                                buf.clear();
                                buf.push_str(t1s);
                                buf.push_str(t2s);
                                if let Some(&tg) = v.get(buf.as_str()) {
                                    merges.push((t1, t2, tg));
                                }
                            }
                        }
                    }
                    deferred_merges.clear();
                }
                Ok(RawModel {
                    model_type,
                    vocab,
                    max_id,
                    merges,
                    has_merges,
                    byte_fallback,
                    ignore_merges,
                    continuing_subword_prefix,
                    end_of_word_suffix,
                })
            }
        }
        deserializer.deserialize_map(Vis(std::marker::PhantomData))
    }
}

#[derive(serde::Deserialize)]
struct RawTokenizerJson<'a> {
    #[serde(borrow)]
    model: Option<RawModel<'a>>,
    #[serde(default)]
    normalizer: Option<Value>,
    #[serde(default)]
    pre_tokenizer: Option<Value>,
    #[serde(default)]
    decoder: Option<Value>,
    #[serde(default)]
    post_processor: Option<Value>,
    #[serde(default)]
    added_tokens: Option<Vec<RawAddedToken>>,
}

/// Compact metadata extracted during `Tokenizer::new` so `AutoTokenizer` skips
/// re-parsing the full 12+ MB `tokenizer.json` in Python.
struct AutoMeta {
    added_tokens: Vec<(String, u32, bool, bool, bool)>,
    post_processor_json: Option<String>,
    prepend_scheme: String,
    known_token_ids: Vec<(String, u32)>,
}

fn detect_prepend_scheme_str(pre: &Value, norm: &Value) -> String {
    fn find_ms(p: &Value) -> Option<&Value> {
        match p.get("type").and_then(|v| v.as_str()) {
            Some("Metaspace") => Some(p),
            Some("Sequence") => p
                .get("pretokenizers")
                .and_then(|v| v.as_array())
                .and_then(|a| a.iter().find(|x| x.get("type").and_then(|v| v.as_str()) == Some("Metaspace"))),
            _ => None,
        }
    }
    if let Some(ms) = find_ms(pre) {
        return ms
            .get("prepend_scheme")
            .and_then(|v| v.as_str())
            .unwrap_or("always")
            .to_string();
    }
    let has_prepend = |st: &Value| {
        st.get("type").and_then(|v| v.as_str()) == Some("Prepend")
            && st.get("prepend").and_then(|v| v.as_str()) == Some("\u{2581}")
    };
    let has_prepend_norm = if norm.get("type").and_then(|v| v.as_str()) == Some("Sequence") {
        norm.get("normalizers")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().any(has_prepend))
            .unwrap_or(false)
    } else {
        has_prepend(norm)
    };
    if has_prepend_norm {
        "always".to_string()
    } else {
        "none".to_string()
    }
}

/// SentencePiece decoders end with `Strip{content:" ", start:1}` — strip the one
/// leading space the metaspace prepend added back (matches C/HF decode).
fn detect_decode_strip(dec: &Value) -> bool {
    if dec.is_null() {
        return false;
    }
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

/// True if `val` (a pre_tokenizer or decoder JSON node) contains a `ByteLevel` stage.
fn contains_bytelevel(val: &Value, list_key: &str) -> bool {
    match val.get("type").and_then(|v| v.as_str()) {
        Some("ByteLevel") => true,
        Some("Sequence") => val
            .get(list_key)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().any(|x| contains_bytelevel(x, list_key)))
            .unwrap_or(false),
        _ => false,
    }
}

/// Normalize literal CR/LF bytes inside a regex pattern string to `\\r`/`\\n`
/// so JSON strings using raw `\n` and escaped `\\n` compare identically.
fn norm_regex_escapes(pat: &str) -> String {
    let mut out = String::with_capacity(pat.len());
    for ch in pat.chars() {
        match ch {
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            _ => out.push(ch),
        }
    }
    out
}

/// Parse an exact digit-branch pattern for `Pretok::Generic` / `Pretok::CamelCase`.
/// Returns `Some(max_digits)` (`0` = unlimited, `1` = single digit, `3` = up to 3).
fn parse_exact_digit_branch(s: &str) -> Option<u32> {
    match s {
        r"\p{N}{1,3}" | r"\p{N}{1,3}+" => Some(3),
        r"\p{N}" | r"\p{N}{1}" => Some(1),
        r"\p{N}+" | r"\p{N}++" => Some(0),
        _ => None,
    }
}

/// Match an exact proven-equivalent whitespace tail for `Generic` / `CamelCase`.
/// Note: `\s*[\r\n]` equals `\s*[\r\n]+` because `[\r\n] ⊂ \s` and `\s*` is greedy;
/// `\s++$` before `\s*[\r\n]` only triggers at EOF with no trailing newline, where
/// `\s+` already consumes the entire trailing whitespace run.
fn is_exact_ws_tail(s: &str) -> bool {
    matches!(
        s,
        r"\s*[\r\n]+|\s+(?!\S)|\s+"
            | r"\s*[\r\n]|\s+(?!\S)|\s+"
            | r"\s++$|\s*[\r\n]|\s+(?!\S)|\s"
            | r"\s++$|\s*[\r\n]+|\s+(?!\S)|\s+"
    )
}

/// Strictly match a single `Split` regex string to one of the hand-written
/// `pretok` / `blockscan` grammars (`Pretok`). Used both by `detect_pretok`
/// (to dispatch top-level models to SIMD `blockscan`) and by `PretokPipe`
/// (to accelerate any `Split` stage inside a multi-stage `Sequence`, such as
/// `Falcon-H1`'s `[Punctuation, Split(CamelCase), ByteLevel, Split([0-9])]`).
pub(crate) fn match_fast_split_regex(raw_pat: &str) -> Option<Pretok> {
    let pat = norm_regex_escapes(raw_pat);

    // 1. Canonical GPT-2 / r50k / p50k plain-word grammar.
    if matches!(
        pat.as_str(),
        r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            | r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s+"
            | r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"
    ) {
        return Some(Pretok::Word(false));
    }

    // 2. Generic family (cl100k, Llama-3, Qwen2.5/3/3.6, SmolLM3, GLM, OLMo, Phi-4, ...).
    let rest_after_contr = pat
        .strip_prefix(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|")
        .or_else(|| pat.strip_prefix(r"'(?i:[sdmt]|ll|ve|re)|"));
    if let Some(rest) = rest_after_contr {
        // Plain letter class (`marks = false`) vs mark-inclusive (`marks = true`).
        let (marks, rest) = if let Some(r) = rest
            .strip_prefix(r"[^\r\n\p{L}\p{N}]?\p{L}+|")
            .or_else(|| rest.strip_prefix(r"[^\r\n\p{L}\p{N}]?+\p{L}+|"))
            .or_else(|| rest.strip_prefix(r"[^\r\n\p{L}\p{N}]?+\p{L}++|"))
        {
            (false, r)
        } else {
            (true, rest.strip_prefix(r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|")?)
        };
        let (dig_part, rest) = rest.split_once('|')?;
        let max_digits = parse_exact_digit_branch(dig_part)?;
        let (no_nl, rest) = if marks {
            if let Some(r) = rest.strip_prefix(r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|") {
                (false, r)
            } else {
                let r = rest
                    .strip_prefix(r" ?[^\s\p{L}\p{M}\p{N}\r\n]+|")
                    .or_else(|| rest.strip_prefix(r" ?[^\s\p{L}\p{M}\p{N}]+|"))?;
                (true, r)
            }
        } else if let Some(r) = rest
            .strip_prefix(r" ?[^\s\p{L}\p{N}]+[\r\n]*|")
            .or_else(|| rest.strip_prefix(r" ?[^\s\p{L}\p{N}]++[\r\n]*|"))
            .or_else(|| rest.strip_prefix(r" ?[^\s\p{L}\p{N}]++[\r\n]*+|"))
        {
            (false, r)
        } else {
            let r = rest
                .strip_prefix(r" ?[^\s\p{L}\p{N}\r\n]+|")
                .or_else(|| rest.strip_prefix(r" ?[^\s\p{L}\p{N}]+|"))?;
            (true, r)
        };
        if is_exact_ws_tail(rest) {
            return Some(Pretok::Generic { max_digits, marks, no_nl });
        }
        return None;
    }

    // 3. CamelCase family (o200k, gpt-oss, Llama-4, MiniMax, Phi-4-mini,
    //    Mistral Nemo/Small/Devstral/Magistral/Tekken, Nemotron-Nano).
    const CAMEL_CONTR: &str = concat!(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    );
    const CAMEL_NO_CONTR: &str = concat!(
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    );
    let (contractions, rest) = if let Some(r) = pat.strip_prefix(CAMEL_CONTR) {
        (true, r)
    } else {
        (false, pat.strip_prefix(CAMEL_NO_CONTR)?)
    };
    let (dig_part, rest) = rest.split_once('|')?;
    let max_digits = parse_exact_digit_branch(dig_part)?;
    let rest = rest.strip_prefix(r" ?[^\s\p{L}\p{N}]+[\r\n/]*|")?;
    if is_exact_ws_tail(rest) {
        return Some(Pretok::CamelCase { contractions, max_digits });
    }
    None
}

/// Strictly match a ByteLevel pre-tokenizer to one of the hand-written SIMD
/// `blockscan` grammars (`Pretok`). Returns `None` on ANY deviation — e.g.,
/// `add_prefix_space: true`, custom symbol/newline rules like Seed-Coder's
/// ` ?[^\s\p{L}\p{N}\r\n]+`, multi-stage pipelines, or non-equivalent behaviors —
/// so unrecognized models seamlessly route to the exact universal `PretokPipe`
/// instead of silently mistokenizing.
fn detect_pretok(pre: &Value) -> Option<Pretok> {
    let t = pre.get("type")?.as_str()?;
    if t == "ByteLevel" {
        let aps = pre.get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(false);
        let use_regex = pre.get("use_regex").and_then(|v| v.as_bool()).unwrap_or(true);
        return if !aps && use_regex { Some(Pretok::Word(false)) } else { None };
    }
    if t != "Sequence" {
        return None;
    }
    let pts = pre.get("pretokenizers")?.as_array()?;
    if pts.len() != 2 {
        return None;
    }
    // Second stage MUST be ByteLevel with add_prefix_space == false.
    if pts[1].get("type").and_then(|v| v.as_str()) != Some("ByteLevel") {
        return None;
    }
    let bl_aps = pts[1].get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(false);
    let bl_use_regex = pts[1].get("use_regex").and_then(|v| v.as_bool()).unwrap_or(true);
    if bl_aps {
        return None;
    }

    match pts[0].get("type").and_then(|v| v.as_str())? {
        "Digits" => {
            let indiv = pts[0]
                .get("individual_digits")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            // `[Digits(individual_digits=true), ByteLevel(add_prefix_space=false, use_regex=true)]`
            if indiv && bl_use_regex {
                Some(Pretok::Word(true))
            } else {
                None
            }
        }
        "Split" => {
            // Double-splitting (`Split` followed by `ByteLevel(use_regex=true)`) must
            // go to `PretokPipe` so both stages compose accurately.
            if bl_use_regex {
                return None;
            }
            // All grammars matched below are total (every non-empty byte is matched),
            // so `Isolated + invert=false` and `Removed + invert=true` are identical.
            let beh = pts[0].get("behavior").and_then(|v| v.as_str()).unwrap_or("Isolated");
            let inv = pts[0].get("invert").and_then(|v| v.as_bool()).unwrap_or(false);
            let total_ok = (beh == "Isolated" && !inv) || (beh == "Removed" && inv);
            if !total_ok {
                return None;
            }
            let raw_pat = pts[0].get("pattern")?.get("Regex")?.as_str()?;
            match_fast_split_regex(raw_pat)
        }
        _ => None,
    }
}

/// Recognize the exact DeepSeek-V3/R1/V3.1/V4 pre-tokenizer: a 3-stage Isolated Split
/// sequence (`\p{N}{1,3}`, a CJK/kana run, then the mark-inclusive main pattern)
/// followed by a non-regex ByteLevel. Matched byte-exact and fail-closed: any
/// deviation returns false and the model takes the universal `PretokPipe` instead.
/// When it matches, `cjk::chunk_ranges` reproduces it ~10x faster. See src/cjk.rs.
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
        && !pts[3].get("add_prefix_space").and_then(|v| v.as_bool()).unwrap_or(false)
        && !pts[3].get("use_regex").and_then(|v| v.as_bool()).unwrap_or(true)
}

/// Zero-allocation SIMD SentencePiece fast-path mode (enabled only when
/// load-time verification proves `safe_meta_split == Some('▁')` and the
/// normalizer/pre-tokenizer strictly match one of the standard SP metaspace
/// pipelines). Keeps 1-byte `' '` as a lossless surrogate for 3-byte `'▁'`
/// across `ChunkTable`, `SharedTable`, and `SharedLongTable`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SpFastMeta {
    None,
    /// `norm = [Prepend("▁"), Replace(" ", "▁")]`, `pre = None` (Phi-3.5, Llama-2, SOLAR, Zamba2).
    NormPrependAlways,
    /// `norm = [Replace(" ", "▁")]`, `pre = None | Split(" ", MergedWithPrevious)` (Yi-1.5, Gemma-2/3, ERNIE-4.5).
    NormNoPrepend,
    /// `norm = None`, `pre = Metaspace("▁", split=false, prepend_scheme)` (Mistral-7B-v0.3, Nemotron-Mini).
    Metaspace { prepend_scheme: u8 },
}

fn detect_sp_fast_meta(
    byte_fallback: bool,
    ignore_merges: bool,
    has_eow: bool,
    sp_split_ok: bool,
    normalizer: &norm::Normalizer,
    pretok_pipe: Option<&pretok_pipe::PretokPipe>,
    pre: &Value,
) -> SpFastMeta {
    if !byte_fallback || ignore_merges || has_eow || !sp_split_ok {
        return SpFastMeta::None;
    }
    match (normalizer.sp_metaspace_kind(), pretok_pipe) {
        (1, None) => SpFastMeta::NormPrependAlways,
        (2, None) => SpFastMeta::NormNoPrepend,
        (2, Some(_))
            if pre.get("type").and_then(|v| v.as_str()) == Some("Split")
                && pre.get("pattern").and_then(|p| p.get("String")).and_then(|v| v.as_str())
                    == Some(" ")
                && pre.get("behavior").and_then(|v| v.as_str()) == Some("MergedWithPrevious")
                && !pre.get("invert").and_then(|v| v.as_bool()).unwrap_or(false) =>
        {
            SpFastMeta::NormNoPrepend
        }
        (0, Some(pipe)) if normalizer.is_identity() => match pipe.single_unsplit_u2581_prepend() {
            Some(ps) => SpFastMeta::Metaspace { prepend_scheme: ps },
            None => SpFastMeta::None,
        },
        _ => SpFastMeta::None,
    }
}

/// Analyze all merges producing tokens with internal `'▁'` (after leading `'▁'`s).
/// Returns `(safe_unconditional, sp_split_ok, cross_prev_mask)`:
/// - `safe_unconditional`: true iff ZERO merges contain an internal `'▁'`.
/// - `sp_split_ok`: true iff `meta_split_cand == Some('▁')` and `<= 16` distinct
///   bytes ever precede an internal `'▁'` in any merge target (e.g. Gemma-3 has 1: `'>'`).
/// - `cross_prev_mask`: 256-bit set of bytes `c` that appear immediately before `' '`
///   in any decoded internal-`'▁'` merge target (sound superset of cross-space merges).
fn analyze_sp_cross_prev(
    meta_split_cand: Option<char>,
    merges: &[(u32, u32, u32)],
    has_internal_ms: &[bool],
    byte_arena: &[u8],
    byte_span: &[(u32, u32)],
) -> (Option<char>, bool, [u64; 4]) {
    let Some(rep) = meta_split_cand else {
        return (None, false, [0u64; 4]);
    };
    let n = has_internal_ms.len();
    let mut any_internal = false;
    let mut cross = [0u64; 4];
    let mut can_guard = rep == '\u{2581}';
    for &(_, _, tg) in merges {
        let tgi = tg as usize;
        if tgi < n && has_internal_ms[tgi] {
            any_internal = true;
            if !can_guard {
                break;
            }
            if tgi >= byte_span.len() {
                can_guard = false;
                break;
            }
            let (off, len) = byte_span[tgi];
            if len == 0 {
                can_guard = false;
                break;
            }
            let b = &byte_arena[off as usize..(off + len) as usize];
            let mut found_edge = false;
            for j in 1..b.len() {
                if b[j] == b' ' && b[j - 1] != b' ' {
                    let prev = b[j - 1] as usize;
                    cross[prev >> 6] |= 1u64 << (prev & 63);
                    found_edge = true;
                }
            }
            if !found_edge {
                can_guard = false;
                break;
            }
        }
    }
    if !any_internal {
        return (Some(rep), rep == '\u{2581}', [0u64; 4]);
    }
    let cnt: u32 = cross.iter().map(|w| w.count_ones()).sum();
    let sp_ok = can_guard && cnt <= 16;
    (None, sp_ok, if sp_ok { cross } else { [0u64; 4] })
}

/// Pre-tokenizer dispatch mode selected once at load time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FastPretok {
    /// Hand-written 64B SIMD `blockscan` + 256-deep pipelined `SharedTable` path.
    BlockScan,
    /// Hand-written 3-stage DeepSeek-V3/R1 chunker (`src/cjk.rs`).
    Cjk,
    /// Universal composable HF pre-tokenizer pipeline (`src/pretok_pipe.rs`).
    Pipe,
    /// Null / empty pre-tokenizer (whole normalized text as one chunk).
    WholeText,
}

/// Default memory cap (MiB) of each tokenizer's shared learned-chunk table.
/// It starts at 512 KiB and grows only as new distinct chunks are seen.
const DEFAULT_CACHE_MIB: usize = 128;

/// Documents at least this long are encoded as parallel segments.
const SPLIT_MIN: usize = 1 << 18;

/// Which BPE merge implementation to run. `Auto` picks the fastest exact merger
/// per chunk (`merge_bytes_into` for <=64B, O(n log n) `merge_canonical` for
/// longer chunks); `Stream`/`Scan` force one — used by the
/// `_encode_ordinary_stream`/`_encode_ordinary_scan` parity-test entry points.
#[derive(Clone, Copy, PartialEq)]
enum Merge {
    Auto,
    Stream,
    Scan,
}

#[pyclass(module = "tuetoken")]
struct Tokenizer {
    kind: Pretok,
    fast_pretok: FastPretok,
    singleton: [i32; 256],
    singleton_u32: [u32; 256],
    /// Optional end-of-word singleton table for BPE models with `end_of_word_suffix`
    /// (e.g. `</w>` in CLIP/SDXL/FLUX): applied to the final byte of each chunk.
    singleton_eow: Option<[i32; 256]>,
    byte_fast: bool,
    map: bpe::MergeMap,
    raw_merges: Vec<(u32, u32, u32)>,
    stream: OnceLock<bpe::StreamTables>,
    // Decode: all token bytes in ONE contiguous arena (cache-friendly), indexed
    // by (offset, len) per token id — instead of a Vec<Vec<u8>> of scattered
    // heap allocations. Matches the C engine's vocab arena.
    byte_arena: Vec<u8>,
    byte_span: Vec<(u32, u32)>,
    c2b: [u16; 324],
    ignore_merges: bool,
    ignore_lookup: Option<FxHashMap<Box<[u8]>, u32>>,
    vocab_size: usize,
    // Composable HF normalizer & pre-tokenizer pipelines.
    normalizer: norm::Normalizer,
    pretok_pipe: Option<pretok_pipe::PretokPipe>,
    // --- SentencePiece / byte_fallback path ---
    byte_fallback: bool,
    // Proven-exact metaspace chunk split char (e.g. Some('▁')): set only when
    // load-time verification proves no merge crosses `(non-rep | rep)` boundaries.
    safe_meta_split: Option<char>,
    sp_fast_meta: SpFastMeta,
    /// 256-bit bitmap of bytes `c` that appear immediately before an internal `'▁'`
    /// in any merge target (e.g. `{ b'>' }` for Gemma-3's `">▁</"` merge). Any
    /// `(non-' ', ' ')` boundary where `prev` is NOT in this bitmap is mathematically
    /// proven uncrossable by BPE; boundaries where `prev` IS set stay unsplit.
    sp_cross_prev: [u64; 4],
    sp_has_cross: bool,
    sp_rep_id: i32,
    // char (codepoint) -> token id for single-character vocab tokens. ASCII goes
    // in the array (hot path), the rest in the map.
    char_ascii: [i32; 128],
    char_map: FxHashMap<u32, u32>,
    decode_strip_space: bool, // SentencePiece: strip one leading space on decode
    source_path: String,      // tokenizer.json path (for repr + pickle)
    is_byte_level: bool,      // true ByteLevel (not byte_fallback) — offsets supported
    // True when `is_byte_level && singleton_eow.is_none()`, all 256 byte singletons
    // decode to length 1, and every merge `(a, b) -> tg` satisfies
    // `len(tg) == len(a) + len(b)`. Guarantees every token's byte span in
    // `encode_with_offsets` is an exact contiguous prefix sum of `byte_span[t].1`,
    // unlocking the 10 GB/s SIMD+SharedTable+parallel encoder (and non-monotonic
    // ByteLevel models like Qwen3.6, Llama-4, gpt-oss-20b!).
    byte_len_exact: bool,
    // True when every token is produced before it is consumed (`produce_max <
    // consume_min`) — i.e. the merge ranks are monotonic, which guarantees the fast
    // batch `merge` and streaming merger equal canonical BPE. False for vocabularies
    // with rank inversions (e.g. gemma's whitespace-run tokens), which route to the
    // canonical merger. Every normal trained BPE is monotonic; see bpe::merge.
    monotonic: bool,
    // Special-token splitter for the AutoTokenizer path: a leftmost-longest matcher
    // over the added-token surfaces, so `encode_special` tokenizes a whole chat
    // string (specials + gaps) in one Rust pass instead of Python regex + a call
    // per fragment. Set once via `set_special_tokens`; None until then.
    special_ac: Option<aho_corasick::AhoCorasick>,
    special_prefilter: SpecialPrefilter,
    special_ids: Vec<u32>,     // pattern i -> token id
    special_lstrip: Vec<bool>, // absorb whitespace to the left of the match
    special_rstrip: Vec<bool>, // absorb whitespace to the right
    special_max_len: usize,    // max byte length among registered special tokens
    special_skip_mask: Vec<bool>, // token id -> true iff `special == true` in added_tokens
    whisper_ts_begin: u32,     // `<|0.00|>` token id for Whisper models, else u32::MAX
    whisper_sop_id: u32,       // `<|startofprev|>` token id for Whisper prompt stripping
    whisper_sot_id: u32,       // `<|startoftranscript|>` token id for Whisper prompt stripping
    prepend_scheme: u8,        // 0 none, 1 always, 2 first, 3 never (metaspace prepend)
    // Load-time verified whole-chunk table (≤15-byte pre-token -> its single
    // token). See chunk_table.rs. Only consulted under Merge::Auto.
    chunk_table: chunk_table::ChunkTable,
    uid: u64,
    // Grow-only table of learned multi-token chunk encodings, shared by all
    // threads (see chunk_table::SharedTable). Memory-capped.
    learned: chunk_table::SharedTable,
    // Lock-free lazy shared table for 16..=31 byte chunks (99.8% of >15B words).
    long_table: chunk_table::SharedLongTable,
    // Learned encodings too long for `learned`/`long_table` (> 31B or > 11 toks),
    // shared by all threads, behind the per-thread caches.
    spill: chunk_table::SpillMap,
    // Pre-built Python ints for every token id (lazily, under the GIL), so a
    // result list is filled with INCREF'd pointers instead of allocating a fresh
    // PyLong per token — the dominant cost of returning list[int].
    int_table: OnceLock<Vec<Py<PyAny>>>,
    // Lazily built 16-byte inline decode table (zero startup cost if unused).
    decode_table: OnceLock<Vec<DecodeEntry>>,
    // Compact metadata for AutoTokenizer so Python skips re-parsing tokenizer.json.
    auto_meta: AutoMeta,
}

/// Fast SIMD prefilter for `encode_special_one`: before invoking the Aho-Corasick
/// automaton (which may hold 1,600+ patterns, e.g. Whisper's `<|0.00|>`..`<|30.00|>`),
/// scan for a shared 2-byte prefix or 1..=3 possible first bytes using `memchr`.
/// On plain text with no special tokens, this bypasses AC completely and runs the
/// SIMD BPE fast path directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpecialPrefilter {
    None,
    Prefix2(u8, u8),
    /// All patterns are `>= 2` bytes and share a common first byte `b0`, while
    /// the second byte must have its bit set in `b1_mask` (e.g. Qwen2.5 `<`
    /// followed only by `/`, `t`, `|`, bypassing `x < 5` without AC).
    Prefix1Mask(u8, [u64; 4]),
    Bytes1(u8),
    Bytes2(u8, u8),
    Bytes3(u8, u8, u8),
    /// Exact SIMD nibble-table set prefilter for 4..=8 distinct first bytes
    /// (e.g. Gemma-2/3's `{'\t', '\n', '<', '[', 0xe2}`).
    /// `lo[b & 0xf] & hi[b >> 4]` has bit `i` set iff `b == b_i` (zero false
    /// positives for <= 8 bytes). `req_b1[i]` holds `0x100` (any next byte) or
    /// the single required second byte `c` when all patterns starting with `b_i`
    /// share `b[1] == c` (e.g. Gemma's `0xe2` -> `0x96`, skipping `—`/`’`/`“`!).
    SimdSet {
        lo: [u8; 16],
        hi: [u8; 16],
        req_b1: [u16; 8],
    },
}

#[inline(always)]
fn simd_set_confirm(hay: &[u8], idx: usize, m: u8, req_b1: &[u16; 8]) -> bool {
    let r = req_b1[m.trailing_zeros() as usize];
    r == 0x100 || (idx + 1 < hay.len() && hay[idx + 1] as u16 == r)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_simd_set_avx2(
    hay: &[u8],
    lo: &[u8; 16],
    hi: &[u8; 16],
    req_b1: &[u16; 8],
) -> Option<usize> {
    use std::arch::x86_64::*;
    let lo_v = _mm256_broadcastsi128_si256(_mm_loadu_si128(lo.as_ptr() as *const __m128i));
    let hi_v = _mm256_broadcastsi128_si256(_mm_loadu_si128(hi.as_ptr() as *const __m128i));
    let mask_0f = _mm256_set1_epi8(0x0f);
    let zero = _mm256_setzero_si256();
    let n = hay.len();
    let ptr = hay.as_ptr();
    let mut i = 0usize;
    while i + 32 <= n {
        let v = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let lo_n = _mm256_and_si256(v, mask_0f);
        let hi_n = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask_0f);
        let hit = _mm256_and_si256(
            _mm256_shuffle_epi8(lo_v, lo_n),
            _mm256_shuffle_epi8(hi_v, hi_n),
        );
        let nz = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(hit, zero)) as u32);
        if nz != 0 {
            let mut bits = nz;
            while bits != 0 {
                let k = bits.trailing_zeros() as usize;
                let b = *ptr.add(i + k);
                let m = lo[(b & 0x0f) as usize] & hi[(b >> 4) as usize];
                if simd_set_confirm(hay, i + k, m, req_b1) {
                    return Some(i + k);
                }
                bits &= bits - 1;
            }
        }
        i += 32;
    }
    while i < n {
        let b = *ptr.add(i);
        let m = lo[(b & 0x0f) as usize] & hi[(b >> 4) as usize];
        if m != 0 && simd_set_confirm(hay, i, m, req_b1) {
            return Some(i);
        }
        i += 1;
    }
    None
}

impl SpecialPrefilter {
    fn from_patterns(patterns: &[String]) -> Self {
        let mut first_bytes = [0u8; 8];
        let mut req_b1 = [0xFFFFu16; 8];
        let mut n_first = 0usize;
        let mut overflow = false;
        let mut common2: Option<(u8, u8)> = None;
        let mut all_ge2 = true;
        let mut same2 = true;
        let mut b1_mask = [0u64; 4];
        for p in patterns {
            let b = p.as_bytes();
            if b.is_empty() {
                return Self::None;
            }
            if b.len() < 2 {
                all_ge2 = false;
            } else {
                let b1 = b[1] as usize;
                b1_mask[b1 >> 6] |= 1u64 << (b1 & 63);
                if same2 {
                    match common2 {
                        None => common2 = Some((b[0], b[1])),
                        Some(pair) if pair != (b[0], b[1]) => same2 = false,
                        _ => {}
                    }
                }
            }
            let b0 = b[0];
            let slot = if let Some(pos) = first_bytes[..n_first].iter().position(|&x| x == b0) {
                pos
            } else if n_first < 8 {
                let pos = n_first;
                first_bytes[pos] = b0;
                n_first += 1;
                pos
            } else {
                overflow = true;
                8
            };
            if slot < 8 {
                if b.len() < 2 {
                    req_b1[slot] = 0x100;
                } else if req_b1[slot] == 0xFFFF {
                    req_b1[slot] = b[1] as u16;
                } else if req_b1[slot] != b[1] as u16 {
                    req_b1[slot] = 0x100;
                }
            }
        }
        if let Some((b0, b1)) = common2 {
            if all_ge2 && same2 {
                return Self::Prefix2(b0, b1);
            }
        }
        if !overflow && n_first == 1 && all_ge2 {
            return Self::Prefix1Mask(first_bytes[0], b1_mask);
        }
        if overflow {
            return Self::None;
        }
        match n_first {
            1 => Self::Bytes1(first_bytes[0]),
            2 => Self::Bytes2(first_bytes[0], first_bytes[1]),
            3 => Self::Bytes3(first_bytes[0], first_bytes[1], first_bytes[2]),
            4..=8 => {
                let mut lo = [0u8; 16];
                let mut hi = [0u8; 16];
                for (slot, &b0) in first_bytes[..n_first].iter().enumerate() {
                    let bit = 1u8 << slot;
                    lo[(b0 & 0x0f) as usize] |= bit;
                    hi[(b0 >> 4) as usize] |= bit;
                }
                Self::SimdSet { lo, hi, req_b1 }
            }
            _ => Self::None,
        }
    }

    #[inline]
    fn find_first(self, hay: &[u8]) -> Option<usize> {
        match self {
            Self::None => Some(0),
            Self::Prefix2(b0, b1) => {
                if hay.len() < 2 {
                    return None;
                }
                let mut pos = 0usize;
                let limit = hay.len() - 1;
                while pos < limit {
                    let off = memchr::memchr(b0, &hay[pos..limit])?;
                    let idx = pos + off;
                    if hay[idx + 1] == b1 {
                        return Some(idx);
                    }
                    pos = idx + 1;
                }
                None
            }
            Self::Prefix1Mask(b0, mask) => {
                if hay.len() < 2 {
                    return None;
                }
                let mut pos = 0usize;
                let limit = hay.len() - 1;
                while pos < limit {
                    let off = memchr::memchr(b0, &hay[pos..limit])?;
                    let idx = pos + off;
                    let b1 = hay[idx + 1] as usize;
                    if (mask[b1 >> 6] >> (b1 & 63)) & 1 != 0 {
                        return Some(idx);
                    }
                    pos = idx + 1;
                }
                None
            }
            Self::Bytes1(b0) => memchr::memchr(b0, hay),
            Self::Bytes2(b0, b1) => memchr::memchr2(b0, b1, hay),
            Self::Bytes3(b0, b1, b2) => memchr::memchr3(b0, b1, b2, hay),
            Self::SimdSet { lo, hi, req_b1 } => {
                let n = hay.len();
                let mut i = 0usize;
                #[cfg(target_arch = "aarch64")]
                if n >= 16 {
                    use std::arch::aarch64::*;
                    unsafe {
                        let lo_v = vld1q_u8(lo.as_ptr());
                        let hi_v = vld1q_u8(hi.as_ptr());
                        let mask_0f = vdupq_n_u8(0x0f);
                        let ptr = hay.as_ptr();
                        while i + 16 <= n {
                            let v = vld1q_u8(ptr.add(i));
                            let hit = vandq_u8(
                                vqtbl1q_u8(lo_v, vandq_u8(v, mask_0f)),
                                vqtbl1q_u8(hi_v, vshrq_n_u8::<4>(v)),
                            );
                            if vmaxvq_u8(hit) != 0 {
                                let mut buf = [0u8; 16];
                                vst1q_u8(buf.as_mut_ptr(), hit);
                                for (k, &m) in buf.iter().enumerate() {
                                    if m != 0 && simd_set_confirm(hay, i + k, m, &req_b1) {
                                        return Some(i + k);
                                    }
                                }
                            }
                            i += 16;
                        }
                    }
                }
                #[cfg(target_arch = "x86_64")]
                if n >= 32 && std::arch::is_x86_feature_detected!("avx2") {
                    return unsafe { find_simd_set_avx2(hay, &lo, &hi, &req_b1) };
                }
                while i < n {
                    let b = hay[i];
                    let m = lo[(b & 0x0f) as usize] & hi[(b >> 4) as usize];
                    if m != 0 && simd_set_confirm(hay, i, m, &req_b1) {
                        return Some(i);
                    }
                    i += 1;
                }
                None
            }
        }
    }
}

/// Compact 24-byte pipeline record (down from 40B; chunk byte spans are
/// reconstructed from `ends[]` only on the <1% cold miss path).
#[derive(Clone, Copy)]
#[repr(C)]
struct PipeRec {
    lo: u64,
    hi: u64,
    p: u64,
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
    // per-thread memo of multi-token chunks (see chunk_table::DynCache)
    dyn_cache: chunk_table::DynCache,
    // per-thread exact memo of chunks longer than the table key (see LongCache)
    long_cache: chunk_table::LongCache,
    // reusable L1/L2 buffer for streaming WideStr -> UTF-8 transcoding
    utf8: Vec<u8>,
    // reusable ping-pong arenas for PretokPipe & DeepSeek CJK chunkers
    pipe: pretok_pipe::PipeScratch,
    cjk_cur: Vec<(usize, usize)>,
    cjk_nxt: Vec<(usize, usize)>,
}

thread_local! {
    /// One reusable Scratch per OS thread: no per-call allocation of the merge
    /// buffers, and the DynCache stays warm across calls on the same thread.
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::default());
}

/// A batch input item: a Python `str`, or `bytes` holding UTF-8 text. `bytes`
/// and ASCII `str` are borrowed without a copy, so a multi-GB `bytes` read
/// straight from a file costs nothing to pass in. A non-ASCII `str` is kept as
/// its raw code units (`Wide`) and converted to UTF-8 by `transcode_wide`, in
/// parallel with the GIL released.
enum TextArg {
    Str(PyText),
    Bytes(pyo3::pybacked::PyBackedBytes),
}

// Hand-written instead of `#[derive(FromPyObject)]`: the derive tries `str`
// first and, for a `bytes` item, builds (and drops) a Python exception before
// trying `bytes` — ~0.5 us per item, i.e. 200 ms on a 400k-document batch.
// A type check picks the right variant directly.
impl<'a, 'py> FromPyObject<'a, 'py> for TextArg {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<pyo3::types::PyBytes>() || obj.is_instance_of::<pyo3::types::PyByteArray>() {
            Ok(TextArg::Bytes(obj.extract().map_err(Into::<PyErr>::into)?))
        } else {
            Ok(TextArg::Str(obj.extract()?))
        }
    }
}

/// A `str` batch item. CPython stores a non-ASCII `str` as fixed-width code
/// units (1, 2 or 4 bytes each) and builds its UTF-8 form only on request —
/// on the calling thread, under the GIL, caching a copy on the object. For a
/// batch of non-ASCII documents that serial conversion cost more than the
/// whole parallel encode (1 GB of OpenWebText: 1.9 GB/s on the first call vs
/// 7 GB/s once Python had cached the UTF-8). So such strings are kept as their
/// code units and converted by `transcode_wide` on all cores instead.
/// ASCII strings, and strings whose UTF-8 Python already cached, are
/// borrowed as UTF-8 directly.
enum PyText {
    Utf8(PyBackedStr),
    Wide(WideStr),
}

/// The code units of a non-ASCII `str`; `_obj` keeps the (immutable) string
/// and therefore `data` alive.
struct WideStr {
    _obj: Py<PyAny>,
    kind: u8,
    data: *const u8,
    len: usize,
}
// SAFETY: `data` points into an immutable Python str owned by `_obj`; CPython
// never modifies the code units of a str that has other references, so it is
// read-only memory for as long as this value lives.
unsafe impl Send for WideStr {}
unsafe impl Sync for WideStr {}

impl<'a, 'py> FromPyObject<'a, 'py> for PyText {
    type Error = PyErr;
    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        #[cfg(not(any(PyPy, GraalPy, Py_LIMITED_API, Py_GIL_DISABLED)))]
        // SAFETY: CPython-layout accessors on an object checked to be a str.
        unsafe {
            use pyo3::ffi;
            let p = obj.as_ptr();
            if ffi::PyUnicode_Check(p) != 0 {
                #[cfg(not(Py_3_12))]
                let ready = ffi::PyUnicode_READY(p) == 0;
                #[cfg(Py_3_12)]
                let ready = true;
                if ready {
                    let data = ffi::PyUnicode_DATA(p) as *const u8;
                    #[cfg(not(Py_3_14))]
                    let compact_non_ascii =
                        ffi::PyUnicode_IS_COMPACT(p) != 0 && ffi::PyUnicode_IS_ASCII(p) == 0;
                    #[cfg(Py_3_14)]
                    let compact_non_ascii = (data as usize).wrapping_sub(p as usize)
                        == std::mem::size_of::<ffi::PyCompactUnicodeObject>();
                    if compact_non_ascii
                        && (*(p as *mut ffi::PyCompactUnicodeObject)).utf8.is_null()
                    {
                        return Ok(PyText::Wide(WideStr {
                            _obj: obj.to_owned().unbind(),
                            kind: ffi::PyUnicode_KIND(p) as u8,
                            data,
                            len: ffi::PyUnicode_GET_LENGTH(p) as usize,
                        }));
                    }
                }
            }
        }
        Ok(PyText::Utf8(obj.extract()?))
    }
}

impl PyText {
    fn wide(&self) -> Option<&WideStr> {
        match self {
            PyText::Wide(w) => Some(w),
            PyText::Utf8(_) => None,
        }
    }
}

impl WideStr {
    /// Exact UTF-8 length of code units [lo, hi).
    fn utf8_len(&self, lo: usize, hi: usize) -> usize {
        // SAFETY: `data` holds `len` units of `kind` bytes each (see WideStr).
        unsafe {
            match self.kind {
                1 => utf8_len(&std::slice::from_raw_parts(self.data, self.len)[lo..hi]),
                2 => utf8_len(&std::slice::from_raw_parts(self.data as *const u16, self.len)[lo..hi]),
                _ => utf8_len(&std::slice::from_raw_parts(self.data as *const u32, self.len)[lo..hi]),
            }
        }
    }

    /// Write the UTF-8 of code units [lo, hi) to `dst`; Err(position) on a surrogate.
    /// SAFETY: `dst` valid for `self.utf8_len(lo, hi)` writes.
    unsafe fn write_utf8(&self, lo: usize, hi: usize, dst: *mut u8) -> Result<usize, usize> {
        match self.kind {
            1 => units_to_utf8(&std::slice::from_raw_parts(self.data, self.len)[lo..hi], dst),
            2 => units_to_utf8(&std::slice::from_raw_parts(self.data as *const u16, self.len)[lo..hi], dst),
            _ => ucs4_to_utf8_ptr(&std::slice::from_raw_parts(self.data as *const u32, self.len)[lo..hi], dst),
        }
        .map_err(|i| lo + i)
    }

    /// Stream-append UTF-8 of code units `[lo, hi)` into a reusable `Vec<u8>`.
    /// Never calls Python's allocator or mutates the `str` object.
    fn append_utf8(&self, lo: usize, hi: usize, dst: &mut Vec<u8>) -> Result<(), usize> {
        let count = hi - lo;
        if count == 0 {
            return Ok(());
        }
        unsafe {
            match self.kind {
                1 => {
                    let sl = &std::slice::from_raw_parts(self.data, self.len)[lo..hi];
                    latin1_to_utf8_append(sl, dst);
                    Ok(())
                }
                2 => {
                    let sl = &std::slice::from_raw_parts(self.data as *const u16, self.len)[lo..hi];
                    ucs2_to_utf8_append(sl, dst).map_err(|i| lo + i)
                }
                _ => {
                    let sl = &std::slice::from_raw_parts(self.data as *const u32, self.len)[lo..hi];
                    ucs4_to_utf8_append(sl, dst).map_err(|i| lo + i)
                }
            }
        }
    }

    /// Proven chunk boundaries (`'\n'` followed by ASCII letter) in native code
    /// units, matching `Tokenizer::split_points` on the transcoded UTF-8.
    fn split_points(&self, seg: usize) -> Vec<usize> {
        fn find<T: Copy + Into<u32>>(u: &[T], seg: usize) -> Vec<usize> {
            let n = u.len();
            let mut pts = vec![0usize];
            let mut target = seg;
            while target < n {
                let lim = (target + seg).min(n);
                let mut found = None;
                let mut i = target.max(1);
                while i < lim {
                    let prev: u32 = u[i - 1].into();
                    let cur: u32 = u[i].into();
                    if prev == b'\n' as u32 && cur < 128 && (cur as u8).is_ascii_alphabetic() {
                        found = Some(i);
                        break;
                    }
                    i += 1;
                }
                match found {
                    Some(p) => {
                        pts.push(p);
                        target = p + seg;
                    }
                    None => target = lim + 1,
                }
            }
            pts.push(n);
            pts
        }
        unsafe {
            match self.kind {
                1 => find(std::slice::from_raw_parts(self.data, self.len), seg),
                2 => find(std::slice::from_raw_parts(self.data as *const u16, self.len), seg),
                _ => find(std::slice::from_raw_parts(self.data as *const u32, self.len), seg),
            }
        }
    }
}

/// UTF-8 views of the wide strs of a batch (in order). They point into each
/// str's own UTF-8 cache, so they live as long as the batch's items.
#[derive(Default)]
struct WideCopies {
    spans: Vec<(usize, usize)>,
}
// SAFETY: read-only views of immutable UTF-8 kept alive by the batch items.
unsafe impl Send for WideCopies {}
unsafe impl Sync for WideCopies {}

impl WideCopies {
    fn get(&self, k: usize) -> &str {
        let (p, n) = self.spans[k];
        // SAFETY: (p, n) is a str's cached UTF-8 (written by units_to_utf8 or
        // by CPython), alive while the batch items are.
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(p as *const u8, n)) }
    }
    /// `&str` of each wide item in order.
    fn iter(&self) -> impl Iterator<Item = &str> {
        (0..self.spans.len()).map(move |k| self.get(k))
    }
}

/// Give every wide str among `items` its UTF-8 form, exactly as CPython's
/// `PyUnicode_AsUTF8AndSize` would — the bytes go into the str's own UTF-8
/// cache, allocated with Python's allocator on this (GIL-holding) thread — but
/// with the conversion itself done on all cores. Afterwards the strings are
/// in the same state as after the serial CPython conversion (so a later call
/// borrows them for free). Small batches use CPython's conversion directly.
/// A string with a lone surrogate raises Python's own UnicodeEncodeError.
fn transcode_wide<'a>(py: Python<'_>, items: impl Iterator<Item = Option<&'a WideStr>>, num_threads: usize) -> PyResult<WideCopies> {
    use pyo3::ffi;
    const BIG: usize = 4 << 20;
    const PIECE: usize = 1 << 20;
    let all: Vec<&WideStr> = items.flatten().collect();
    if all.is_empty() {
        return Ok(WideCopies::default());
    }
    let total: usize = all.iter().map(|w| w.len).sum();
    let par = num_threads != 1 && total >= 1 << 16 && parallel_ok();
    let cpython = |w: &WideStr| -> PyResult<(usize, usize)> {
        let mut n: ffi::Py_ssize_t = 0;
        // SAFETY: a live str object; GIL held.
        let p = unsafe { ffi::PyUnicode_AsUTF8AndSize(w._obj.as_ptr(), &mut n) };
        if p.is_null() {
            Err(PyErr::fetch(py))
        } else {
            Ok((p as usize, n as usize))
        }
    };
    if !par {
        return Ok(WideCopies { spans: all.iter().map(|w| cpython(w)).collect::<PyResult<_>>()? });
    }
    // the same str object may appear many times: convert each object once
    let mut wides: Vec<&WideStr> = Vec::with_capacity(all.len());
    let mut uniq_of: Vec<u32> = Vec::with_capacity(all.len());
    {
        let mut idx: rustc_hash::FxHashMap<usize, u32> =
            rustc_hash::FxHashMap::with_capacity_and_hasher(all.len(), Default::default());
        for w in &all {
            let u = *idx.entry(w._obj.as_ptr() as usize).or_insert_with(|| {
                wides.push(w);
                (wides.len() - 1) as u32
            });
            uniq_of.push(u);
        }
    }
    // 1. exact UTF-8 length of each string (per 1M-unit piece for big ones)
    let lens: Vec<Vec<usize>> = in_pool(num_threads, || {
        wides
            .par_iter()
            .with_min_len(8)
            .map(|w| {
                if w.len >= BIG {
                    (0..w.len.div_ceil(PIECE)).into_par_iter().map(|p| w.utf8_len(p * PIECE, ((p + 1) * PIECE).min(w.len))).collect()
                } else {
                    vec![w.utf8_len(0, w.len)]
                }
            })
            .collect()
    });
    // 2. allocate each cache buffer with Python's allocator (CPython frees it
    //    with PyObject_Free when the str dies), on this thread
    let mut bufs: Vec<usize> = Vec::with_capacity(wides.len());
    for l in &lens {
        let n: usize = l.iter().sum();
        // SAFETY: GIL held.
        let p = unsafe { ffi::PyObject_Malloc(n + 1) } as usize;
        if p == 0 {
            for &q in &bufs {
                // SAFETY: allocated above, not published.
                unsafe { ffi::PyObject_Free(q as *mut std::ffi::c_void) };
            }
            return Err(pyo3::exceptions::PyMemoryError::new_err("out of memory converting str to UTF-8"));
        }
        bufs.push(p);
    }
    // 3. convert in parallel (workers only write the fresh buffers)
    let res = in_pool(num_threads, || -> Result<(), usize> {
        (0..wides.len()).into_par_iter().with_min_len(8).try_for_each(|k| {
            let w = wides[k];
            let dst = bufs[k] as *mut u8;
            let l = &lens[k];
            // SAFETY: piece p writes exactly l[p] bytes at its offset; the
            // buffer holds sum(l) + 1 bytes.
            unsafe {
                if l.len() == 1 {
                    w.write_utf8(0, w.len, dst).map_err(|_| k)?;
                } else {
                    let offs: Vec<usize> = l.iter().scan(0usize, |a, &x| { let o = *a; *a += x; Some(o) }).collect();
                    let d = dst as usize;
                    (0..l.len()).into_par_iter().try_for_each(|p| {
                        w.write_utf8(p * PIECE, ((p + 1) * PIECE).min(w.len), (d as *mut u8).add(offs[p])).map(|_| ())
                    }).map_err(|_| k)?;
                }
                *dst.add(l.iter().sum::<usize>()) = 0;
            }
            Ok(())
        })
    });
    if let Err(k) = res {
        for &q in &bufs {
            // SAFETY: allocated above, not published.
            unsafe { ffi::PyObject_Free(q as *mut std::ffi::c_void) };
        }
        // Python's own conversion raises the canonical error for this string
        return Err(match wides[k]._obj.bind(py).extract::<PyBackedStr>() {
            Err(e) => e,
            Ok(_) => PyValueError::new_err("str is not encodable as UTF-8"),
        });
    }
    // 4. publish into each str's UTF-8 cache, as PyUnicode_AsUTF8AndSize does
    let mut spans_u: Vec<(usize, usize)> = Vec::with_capacity(wides.len());
    for (k, w) in wides.iter().enumerate() {
        let n: usize = lens[k].iter().sum();
        // SAFETY: GIL held; a compact non-ASCII str whose cache was empty at
        // extraction (checked in PyText::extract) and still is: nothing ran
        // Python code since. If it was filled meanwhile, keep that one.
        unsafe {
            let c = w._obj.as_ptr() as *mut ffi::PyCompactUnicodeObject;
            if (*c).utf8.is_null() {
                (*c).utf8 = bufs[k] as *mut std::ffi::c_char;
                (*c).utf8_length = n as ffi::Py_ssize_t;
            } else {
                ffi::PyObject_Free(bufs[k] as *mut std::ffi::c_void);
            }
            spans_u.push(((*c).utf8 as usize, (*c).utf8_length as usize));
        }
    }
    Ok(WideCopies { spans: uniq_of.iter().map(|&u| spans_u[u as usize]).collect() })
}

/// Run `f` on the pool for `num_threads` (0 = the global pool).
fn in_pool<R: Send>(num_threads: usize, f: impl FnOnce() -> R + Send) -> R {
    match num_threads {
        n if n > 1 => match fixed_pool(n.min(1024)) {
            Some(pool) => pool.install(f),
            None => f(),
        },
        _ => f(),
    }
}

/// `transcode_wide` for `str` items (GIL held).
fn wide_texts(py: Python<'_>, texts: &[PyText], num_threads: usize) -> PyResult<WideCopies> {
    transcode_wide(py, texts.iter().map(PyText::wide), num_threads)
}

/// As `wide_texts`, for `str`-or-`bytes` items.
fn wide_args(py: Python<'_>, texts: &[TextArg], num_threads: usize) -> PyResult<WideCopies> {
    transcode_wide(py, texts.iter().map(TextArg::wide), num_threads)
}

/// `&str` views of `PyText` items: borrowed, or from `wide` (the
/// `transcode_wide` output for the same items, in order).
fn text_views<'a>(items: &'a [PyText], wide: &'a WideCopies) -> Vec<&'a str> {
    let mut w = wide.iter();
    items
        .iter()
        .map(|t| match t {
            PyText::Utf8(s) => &**s,
            PyText::Wide(_) => w.next().expect("one transcoded copy per wide item"),
        })
        .collect()
}

/// Borrow every item as `&str`. `bytes` items must be valid UTF-8 (the same
/// requirement a `str` has); large ones are validated in parallel, split at
/// character boundaries (a byte that is not 10xxxxxx starts a char).
/// A batch input: UTF-8 `&str`, unvalidated `bytes`, or a compact non-ASCII
/// Python `str` (`Wide`) transcoded on the fly in per-worker L1/L2 scratch.
#[derive(Clone, Copy)]
enum Src<'a> {
    Str(&'a str),
    Raw(&'a [u8]),
    Wide(&'a WideStr),
}

#[derive(Clone, Copy, Debug)]
enum BadInput {
    Bytes(usize),
    Wide(usize),
}

/// Element type of the batch engine's final output buffer. Workers always
/// encode into `u32` scratch (cache-resident); only the copy into the final
/// buffer narrows, so `u16` output halves the bytes written to fresh memory
/// (and its page faults).
trait OutTok: Copy + Send + Sync + 'static {
    /// Write `src` to `dst[..src.len()]`.
    /// SAFETY: `dst` valid for `src.len()` writes.
    unsafe fn put(src: &[u32], dst: *mut Self);
    /// Convert an owned `Vec<u32>` into `Vec<Self>` (zero-copy for `u32`).
    fn from_u32(src: Vec<u32>) -> Vec<Self>;
}
impl OutTok for u32 {
    #[inline]
    unsafe fn put(src: &[u32], dst: *mut u32) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
    }
    #[inline]
    fn from_u32(src: Vec<u32>) -> Vec<u32> {
        src
    }
}
impl OutTok for u16 {
    #[inline]
    unsafe fn put(src: &[u32], dst: *mut u16) {
        // ids < vocab_size <= 65536 (checked by the caller); vectorizes to
        // narrowing moves
        let d = std::slice::from_raw_parts_mut(dst, src.len());
        for (x, &y) in d.iter_mut().zip(src) {
            *x = y as u16;
        }
    }
    #[inline]
    fn from_u32(src: Vec<u32>) -> Vec<u16> {
        let mut o = Vec::with_capacity(src.len());
        // SAFETY: capacity src.len(); every element written by put.
        unsafe {
            Self::put(&src, o.as_mut_ptr());
            o.set_len(src.len());
        }
        o
    }
}

/// Document ranges of `b` split on `sep` (non-empty): exactly `bytes.split`
/// (leftmost, non-overlapping matches; separators dropped; empty pieces kept).
/// Large inputs are searched in parallel pieces (SIMD substring search); all
/// occurrences are collected, then the leftmost non-overlapping ones kept.
fn split_on(b: &[u8], sep: &[u8], num_threads: usize) -> Vec<(usize, usize)> {
    let n = b.len();
    let k = sep.len();
    let finder = memchr::memmem::Finder::new(sep);
    let find_all = |lo: usize, hi: usize| -> Vec<usize> {
        // occurrences starting in [lo, hi), overlapping ones included
        let end = (hi + k - 1).min(n);
        let mut v = Vec::new();
        let mut p = lo;
        while p < hi {
            match finder.find(&b[p..end]) {
                Some(i) if p + i < hi => {
                    v.push(p + i);
                    p += i + 1;
                }
                _ => break,
            }
        }
        v
    };
    const PIECE: usize = 8 << 20;
    let occ: Vec<usize> = if n >= 2 * PIECE && num_threads != 1 && parallel_ok() {
        let cuts: Vec<usize> = (0..n.div_ceil(PIECE)).map(|i| i * PIECE).collect();
        cuts.par_iter().map(|&lo| find_all(lo, (lo + PIECE).min(n))).collect::<Vec<_>>().concat()
    } else {
        find_all(0, n)
    };
    let mut docs = Vec::with_capacity(occ.len() + 1);
    let mut start = 0usize;
    for o in occ {
        if o >= start {
            docs.push((start, o));
            start = o + k;
        }
    }
    docs.push((start, n));
    docs
}

/// Inputs for the batch engine. With `lazy` (models without a normalizer),
/// `bytes` items are NOT validated here: each worker validates the slice it
/// is about to encode, so the input is read once instead of twice (an
/// up-front pass over a 2 GB `bytes` was ~10% of a one-call encode).
/// `wide`: `transcode_wide` of the items' wide strs (see `TextArg::wide`).
fn texts_as_src<'a>(items: &'a [TextArg], wide: &'a WideCopies, lazy: bool) -> PyResult<Vec<Src<'a>>> {
    if lazy {
        let mut w = wide.iter();
        return Ok(items
            .iter()
            .map(|t| match t {
                TextArg::Str(PyText::Utf8(s)) => Src::Str(s),
                TextArg::Str(PyText::Wide(_)) => Src::Str(w.next().expect("one copy per wide item")),
                TextArg::Bytes(b) => Src::Raw(b),
            })
            .collect());
    }
    Ok(texts_as_str(items, wide)?.into_iter().map(Src::Str).collect())
}

/// Zero-allocation `Src` view for streaming `WideStr` transcoding (no GIL
/// `transcode_wide` pass, no `PyObject_Malloc`, no full-corpus UTF-8 copy).
fn texts_as_src_stream<'a>(items: &'a [TextArg]) -> Vec<Src<'a>> {
    items
        .iter()
        .map(|t| match t {
            TextArg::Str(PyText::Utf8(s)) => Src::Str(s),
            TextArg::Str(PyText::Wide(w)) => Src::Wide(w),
            TextArg::Bytes(b) => Src::Raw(b),
        })
        .collect()
}

/// Compact 24-byte owned handle to one Python `str` or `bytes` batch item.
/// Avoids PyO3's 48-byte `TextArg`, generic `PyIter_Next`, and per-item C-API
/// overhead on compact ASCII / cached UTF-8 `str` and `bytes` lists.
const FAST_TAG_SHIFT: u32 = 61;
const FAST_LEN_MASK: usize = (1usize << FAST_TAG_SHIFT) - 1;
const FAST_ASCII: usize = 0; // compact PyASCIIObject — guaranteed pure ASCII
const FAST_UTF8: usize = 1;  // compact PyCompactUnicodeObject with cached UTF-8
const FAST_RAW: usize = 2;   // PyBytesObject (unvalidated UTF-8)
const FAST_WIDE1: usize = 3;
const FAST_WIDE2: usize = 4;
const FAST_WIDE4: usize = 5;

#[derive(Clone, Copy)]
struct FastItem {
    obj: *mut pyo3::ffi::PyObject,
    ptr: *const u8,
    len_tag: usize,
}
// SAFETY: `ptr` points into immutable Python `str`/`bytes` kept alive by `obj`
// whose refcount is owned by `FastBatch` on the calling thread while detached.
unsafe impl Send for FastItem {}
unsafe impl Sync for FastItem {}

impl FastItem {
    #[inline(always)]
    fn len(self) -> usize {
        self.len_tag & FAST_LEN_MASK
    }
    #[inline(always)]
    fn tag(self) -> usize {
        self.len_tag >> FAST_TAG_SHIFT
    }
    #[inline(always)]
    fn append_wide_utf8(self, dst: &mut Vec<u8>) -> Result<(), usize> {
        let count = self.len();
        if count == 0 {
            return Ok(());
        }
        unsafe {
            match self.tag() {
                FAST_WIDE1 => {
                    latin1_to_utf8_append(std::slice::from_raw_parts(self.ptr, count), dst);
                    Ok(())
                }
                FAST_WIDE2 => {
                    ucs2_to_utf8_append(std::slice::from_raw_parts(self.ptr as *const u16, count), dst)
                }
                _ => {
                    ucs4_to_utf8_append(std::slice::from_raw_parts(self.ptr as *const u32, count), dst)
                }
            }
        }
    }

    /// Borrow this `FastItem` as `&str`, using `wide_buf` only for Python wide
    /// strings (`FAST_WIDE1/2/4`). Returns `Err(BadInput)` on invalid UTF-8 bytes
    /// or lone UTF-16 surrogates.
    #[inline(always)]
    fn as_str(self, idx: usize, wide_buf: &mut Vec<u8>) -> Result<&str, BadInput> {
        unsafe {
            match self.tag() {
                FAST_ASCII | FAST_UTF8 => Ok(std::str::from_utf8_unchecked(
                    std::slice::from_raw_parts(self.ptr, self.len()),
                )),
                FAST_RAW => {
                    let sl = std::slice::from_raw_parts(self.ptr, self.len());
                    simdutf8::basic::from_utf8(sl).map_err(|_| BadInput::Bytes(idx))
                }
                _ => {
                    wide_buf.clear();
                    self.append_wide_utf8(wide_buf).map_err(|_| BadInput::Wide(idx))?;
                    Ok(std::str::from_utf8_unchecked(wide_buf.as_slice()))
                }
            }
        }
    }
}

/// GIL-bound container of `FastItem`s; `Drop` safely decrefs every owned item
/// while the GIL token `_py` is still alive.
struct FastBatch<'py> {
    items: Vec<FastItem>,
    total_len: usize,
    max_len: usize,
    has_bytes: bool,
    _py: Python<'py>,
}

impl Drop for FastBatch<'_> {
    fn drop(&mut self) {
        // SAFETY: `_py: Python<'py>` proves the GIL is held when FastBatch drops.
        unsafe {
            for it in &self.items {
                pyo3::ffi::Py_DECREF(it.obj);
            }
        }
    }
}

/// Direct CPython `PyList`/`PyTuple` fast extractor: inspects `PyASCIIObject` /
/// `PyCompactUnicodeObject` / `PyBytesObject` inline with zero `PyIter_Next` and
/// zero C-API calls per item. Returns `None` on any non-exact type or unsupported
/// Python runtime so the caller falls back cleanly to `extract::<Vec<TextArg>>()`.
fn try_extract_fast_batch<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> Option<FastBatch<'py>> {
    #[cfg(all(Py_3_12, not(any(PyPy, GraalPy, Py_LIMITED_API, Py_GIL_DISABLED))))]
    unsafe {
        use pyo3::ffi;
        let lp = obj.as_ptr();
        let (is_list, n) = if ffi::PyList_CheckExact(lp) != 0 {
            (true, ffi::PyList_GET_SIZE(lp) as usize)
        } else if ffi::PyTuple_CheckExact(lp) != 0 {
            (false, ffi::PyTuple_GET_SIZE(lp) as usize)
        } else {
            return None;
        };
        let mut batch = FastBatch {
            items: Vec::with_capacity(n),
            total_len: 0,
            max_len: 0,
            has_bytes: false,
            _py: py,
        };
        let ascii_off = std::mem::size_of::<ffi::PyASCIIObject>();
        let compact_off = std::mem::size_of::<ffi::PyCompactUnicodeObject>();
        for i in 0..n {
            let p = if is_list {
                ffi::PyList_GET_ITEM(lp, i as ffi::Py_ssize_t)
            } else {
                ffi::PyTuple_GET_ITEM(lp, i as ffi::Py_ssize_t)
            };
            if ffi::PyUnicode_CheckExact(p) != 0 {
                let data = ffi::PyUnicode_DATA(p) as *const u8;
                let diff = (data as usize).wrapping_sub(p as usize);
                if diff == ascii_off {
                    let len = ffi::PyUnicode_GET_LENGTH(p) as usize;
                    if len > FAST_LEN_MASK {
                        return None;
                    }
                    ffi::Py_INCREF(p);
                    batch.items.push(FastItem { obj: p, ptr: data, len_tag: len });
                    batch.total_len += len;
                    if len > batch.max_len {
                        batch.max_len = len;
                    }
                } else if diff == compact_off {
                    let c = p as *const ffi::PyCompactUnicodeObject;
                    if !(*c).utf8.is_null() {
                        let len = (*c).utf8_length as usize;
                        if len > FAST_LEN_MASK {
                            return None;
                        }
                        ffi::Py_INCREF(p);
                        batch.items.push(FastItem {
                            obj: p,
                            ptr: (*c).utf8 as *const u8,
                            len_tag: len | (FAST_UTF8 << FAST_TAG_SHIFT),
                        });
                        batch.total_len += len;
                        if len > batch.max_len {
                            batch.max_len = len;
                        }
                    } else {
                        let len = ffi::PyUnicode_GET_LENGTH(p) as usize;
                        if len > FAST_LEN_MASK {
                            return None;
                        }
                        let tag = match ffi::PyUnicode_KIND(p) {
                            1 => FAST_WIDE1,
                            2 => FAST_WIDE2,
                            4 => FAST_WIDE4,
                            _ => return None,
                        };
                        ffi::Py_INCREF(p);
                        batch.items.push(FastItem {
                            obj: p,
                            ptr: data,
                            len_tag: len | (tag << FAST_TAG_SHIFT),
                        });
                        batch.total_len += len;
                        if len > batch.max_len {
                            batch.max_len = len;
                        }
                    }
                } else {
                    return None;
                }
            } else if ffi::PyBytes_CheckExact(p) != 0 {
                let len = ffi::Py_SIZE(p) as usize;
                if len > FAST_LEN_MASK {
                    return None;
                }
                let ptr = ffi::PyBytes_AS_STRING(p) as *const u8;
                ffi::Py_INCREF(p);
                batch.has_bytes = true;
                batch.items.push(FastItem {
                    obj: p,
                    ptr,
                    len_tag: len | (FAST_RAW << FAST_TAG_SHIFT),
                });
                batch.total_len += len;
                if len > batch.max_len {
                    batch.max_len = len;
                }
            } else {
                return None;
            }
        }
        Some(batch)
    }
    #[cfg(not(all(Py_3_12, not(any(PyPy, GraalPy, Py_LIMITED_API, Py_GIL_DISABLED)))))]
    {
        let _ = (py, obj);
        None
    }
}

fn bad_fast_to_pyerr(py: Python<'_>, batch: &FastBatch<'_>, err: BadInput) -> PyErr {
    match err {
        BadInput::Bytes(i) => PyValueError::new_err(format!("item {i}: bytes are not valid UTF-8")),
        BadInput::Wide(i) => {
            if let Some(it) = batch.items.get(i) {
                let mut n: pyo3::ffi::Py_ssize_t = 0;
                // SAFETY: `it.obj` is a live Python str owned by `batch`; GIL is held.
                let p = unsafe { pyo3::ffi::PyUnicode_AsUTF8AndSize(it.obj, &mut n) };
                if p.is_null() {
                    return PyErr::fetch(py);
                }
            }
            PyValueError::new_err("str is not encodable as UTF-8")
        }
    }
}

/// Convert a `BadInput` from the detached Rust engine into the canonical Python
/// exception (holding the GIL).
fn bad_input_to_pyerr(py: Python<'_>, texts: &[TextArg], err: BadInput) -> PyErr {
    match err {
        BadInput::Bytes(i) => PyValueError::new_err(format!("item {i}: bytes are not valid UTF-8")),
        BadInput::Wide(i) => match texts.get(i).and_then(TextArg::wide) {
            Some(w) => match w._obj.bind(py).extract::<PyBackedStr>() {
                Err(e) => e,
                Ok(_) => PyValueError::new_err("str is not encodable as UTF-8"),
            },
            None => PyValueError::new_err("str is not encodable as UTF-8"),
        },
    }
}

impl TextArg {
    fn wide(&self) -> Option<&WideStr> {
        match self {
            TextArg::Str(t) => t.wide(),
            TextArg::Bytes(_) => None,
        }
    }
}

fn texts_as_str<'a>(items: &'a [TextArg], wide: &'a WideCopies) -> PyResult<Vec<&'a str>> {
    fn validate(b: &[u8]) -> bool {
        const PIECE: usize = 1 << 20;
        if b.len() < 4 * PIECE || !parallel_ok() {
            return simdutf8::basic::from_utf8(b).is_ok();
        }
        let mut cuts = vec![0usize];
        let mut p = PIECE;
        while p < b.len() {
            while p < b.len() && b[p] & 0xC0 == 0x80 {
                p += 1;
            }
            cuts.push(p);
            p += PIECE;
        }
        cuts.push(b.len());
        cuts.dedup();
        cuts.par_windows(2).all(|w| simdutf8::basic::from_utf8(&b[w[0]..w[1]]).is_ok())
    }
    // wide strs are already UTF-8 (in `wide`): mark their slots, fill after
    fn one((i, t): (usize, &TextArg)) -> PyResult<&str> {
        match t {
            TextArg::Str(PyText::Utf8(s)) => Ok(&**s),
            TextArg::Str(PyText::Wide(_)) => Ok(""),
            TextArg::Bytes(b) => {
                let bytes: &[u8] = b;
                if validate(bytes) {
                    // SAFETY: validated as UTF-8 just above.
                    Ok(unsafe { std::str::from_utf8_unchecked(bytes) })
                } else {
                    Err(PyValueError::new_err(format!("item {i}: bytes are not valid UTF-8")))
                }
            }
        }
    }
    // many items: validate them in parallel too (a 200 MB batch of small docs
    // would otherwise be checked serially on the calling thread)
    let mut out: Vec<&str> = if items.len() >= 64 && parallel_ok() {
        items.par_iter().enumerate().with_min_len(32).map(one).collect::<PyResult<_>>()?
    } else {
        items.iter().enumerate().map(one).collect::<PyResult<_>>()?
    };
    let mut w = wide.iter();
    for (o, t) in out.iter_mut().zip(items) {
        if t.wide().is_some() {
            *o = w.next().expect("one copy per wide item");
        }
    }
    Ok(out)
}

/// A token buffer handed to numpy WITHOUT a copy: it owns the `Vec<u32>` the
/// encoder filled (in parallel) and exposes it through the buffer protocol, so
/// `numpy.frombuffer(buf, "uint32")` views that memory directly and keeps the
/// owner alive via `array.base`. (A `bytearray` would first be zero-filled on
/// one thread with the GIL held — ~40 MB of memset for a 50 MB batch.)
/// Held as a raw pointer so writes through the (writable) numpy array never
/// alias a live Rust reference.
#[pyclass(module = "tuetoken", frozen)]
struct TokenBuffer {
    ptr: *mut u8,
    bytes: usize,
    owned: Owned,
}

/// Raw parts of the Vec a TokenBuffer owns (element type decides the layout
/// it is freed with).
enum Owned {
    U8(*mut u8, usize, usize),
    U16(*mut u16, usize, usize),
    U32(*mut u32, usize, usize),
    I64(*mut i64, usize, usize),
}

// SAFETY: TokenBuffer exclusively owns its allocation (like a Vec); Rust never
// reads or writes it after construction except to free it in Drop.
unsafe impl Send for TokenBuffer {}
unsafe impl Sync for TokenBuffer {}

impl TokenBuffer {
    fn new(v: Vec<u32>) -> Self {
        let mut v = std::mem::ManuallyDrop::new(v);
        let (p, l, c) = (v.as_mut_ptr(), v.len(), v.capacity());
        TokenBuffer { ptr: p as *mut u8, bytes: l * 4, owned: Owned::U32(p, l, c) }
    }

    fn new_u8(v: Vec<u8>) -> Self {
        let mut v = std::mem::ManuallyDrop::new(v);
        let (p, l, c) = (v.as_mut_ptr(), v.len(), v.capacity());
        TokenBuffer { ptr: p, bytes: l, owned: Owned::U8(p, l, c) }
    }

    fn new_u16(v: Vec<u16>) -> Self {
        let mut v = std::mem::ManuallyDrop::new(v);
        let (p, l, c) = (v.as_mut_ptr(), v.len(), v.capacity());
        TokenBuffer { ptr: p as *mut u8, bytes: l * 2, owned: Owned::U16(p, l, c) }
    }

    fn new_i64(v: Vec<i64>) -> Self {
        let mut v = std::mem::ManuallyDrop::new(v);
        let (p, l, c) = (v.as_mut_ptr(), v.len(), v.capacity());
        TokenBuffer { ptr: p as *mut u8, bytes: l * 8, owned: Owned::I64(p, l, c) }
    }
}

impl Drop for TokenBuffer {
    fn drop(&mut self) {
        // SAFETY: the raw parts came from a Vec of that element type we own.
        unsafe {
            match self.owned {
                Owned::U8(p, l, c) => drop(Vec::from_raw_parts(p, l, c)),
                Owned::U16(p, l, c) => drop(Vec::from_raw_parts(p, l, c)),
                Owned::U32(p, l, c) => drop(Vec::from_raw_parts(p, l, c)),
                Owned::I64(p, l, c) => drop(Vec::from_raw_parts(p, l, c)),
            }
        }
    }
}

#[pymethods]
impl TokenBuffer {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(pyo3::exceptions::PyBufferError::new_err("View is null"));
        }
        let (ptr, len) = {
            let me = slf.get();
            (me.ptr, me.bytes)
        };
        static FORMAT: &[u8] = b"B\0";
        // SAFETY: `view` is a valid Py_buffer to fill; `ptr` stays valid while the
        // view holds a reference to `slf` (stored in view.obj).
        unsafe {
            (*view).obj = slf.into_any().into_ptr();
            (*view).buf = ptr as *mut std::os::raw::c_void;
            (*view).len = len as isize;
            (*view).readonly = 0;
            (*view).itemsize = 1;
            (*view).format = if flags & pyo3::ffi::PyBUF_FORMAT == pyo3::ffi::PyBUF_FORMAT {
                FORMAT.as_ptr() as *mut std::os::raw::c_char
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if flags & pyo3::ffi::PyBUF_ND == pyo3::ffi::PyBUF_ND {
                &mut (*view).len
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if flags & pyo3::ffi::PyBUF_STRIDES == pyo3::ffi::PyBUF_STRIDES {
                &mut (*view).itemsize
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut pyo3::ffi::Py_buffer) {}

    fn __len__(&self) -> usize {
        self.bytes
    }
}

/// Unique id per Tokenizer instance (never reused), so a thread-local cache can
/// tell whose results it holds even if a dropped tokenizer's address is reused.
static NEXT_UID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

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
        const PAR_MIN_BYTES: usize = 65_536;
        let nt = if num_threads == 0 { 0 } else { num_threads.min(1024) };
        if nt == 1 || items.len() <= 1 || !parallel_ok() {
            return items.iter().map(f).collect();
        }
        let total: usize = items.iter().map(&weight).sum();
        if total < PAR_MIN_BYTES {
            return items.iter().map(f).collect();
        }
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

    /// `par_over` for `str` batch items; `wide` = `wide_texts(texts)`.
    fn par_texts<R: Send>(&self, num_threads: usize, texts: &[PyText], wide: &WideCopies, f: impl Fn(&str) -> R + Sync) -> Vec<R> {
        let views = text_views(texts, wide);
        self.par_over(num_threads, &views, |t| t.len(), |t| f(t))
    }

    /// Inputs shorter than this are encoded without releasing the GIL: for a
    /// short string the detach/re-attach round trip costs more than the encode.
    const DETACH_MIN: usize = 2048;

    fn ints<'py>(&self, py: Python<'py>) -> &Vec<Py<PyAny>> {
        self.int_table.get_or_init(|| {
            (0..self.vocab_size as u64)
                .map(|i| i.into_pyobject(py).unwrap().into_any().unbind())
                .collect()
        })
    }

    /// Cold fallback for out-of-table token IDs in `fill_pylist_prefetched`.
    /// Kept out-of-line so the hot loop preserves all caller-saved registers
    /// (`ip`, `tp`, `ob_item`) with zero stack spills.
    #[cold]
    #[inline(never)]
    unsafe fn alloc_pylong_tagged(t: u32) -> usize {
        (pyo3::ffi::PyLong_FromUnsignedLong(t as _) as usize) | 1
    }

    /// Fill `list[base_idx .. base_idx + ids.len()]` with owned `PyLong` refs
    /// from `table` using a 2-stage 16-slot software-prefetched ring:
    /// - Stage 0 (`i + 32`): prefetch `&table[ids[i + 32]]`
    /// - Stage 1 (`i + 16`): resolve pointer `p = table[ids[i + 16]]` & issue
    ///   write-intent prefetch `prefetch_write(p)` (`prefetchw` on x86_64 /
    ///   `prfm pstl1keep` on ARM64) for `p->ob_refcnt`
    /// - Stage 2 (`i`): `Py_INCREF(p)` (now L1d-resident in Exclusive state!)
    ///   + direct store into hoisted `ob_item[i]`.
    #[inline]
    unsafe fn fill_pylist_prefetched(
        table: &[Py<PyAny>],
        ids: &[u32],
        list: *mut pyo3::ffi::PyObject,
        base_idx: usize,
    ) {
        use pyo3::ffi;
        let n = ids.len();
        if n == 0 {
            return;
        }
        let tlen = table.len();
        let tp = table.as_ptr();
        let ip = ids.as_ptr();
        #[cfg(not(any(PyPy, GraalPy)))]
        let ob_item = (*(list as *mut ffi::PyListObject)).ob_item.add(base_idx);

        #[inline(always)]
        unsafe fn write_item(
            #[cfg(not(any(PyPy, GraalPy)))] ob_item: *mut *mut ffi::PyObject,
            #[cfg(any(PyPy, GraalPy))] list: *mut ffi::PyObject,
            #[cfg(any(PyPy, GraalPy))] base_idx: usize,
            i: usize,
            p: *mut ffi::PyObject,
        ) {
            #[cfg(not(any(PyPy, GraalPy)))]
            {
                *ob_item.add(i) = p;
            }
            #[cfg(any(PyPy, GraalPy))]
            {
                ffi::PyList_SET_ITEM(list, (base_idx + i) as ffi::Py_ssize_t, p);
            }
        }

        // Short slice: straight loop with hoisted ob_item.
        if n < 24 {
            for i in 0..n {
                let t = *ip.add(i) as usize;
                let obj = if t < tlen {
                    let p = (*tp.add(t)).as_ptr();
                    ffi::Py_INCREF(p);
                    p
                } else {
                    ffi::PyLong_FromUnsignedLong(t as _)
                };
                #[cfg(not(any(PyPy, GraalPy)))]
                write_item(ob_item, i, obj);
                #[cfg(any(PyPy, GraalPy))]
                write_item(list, base_idx, i, obj);
            }
            return;
        }

        // Bit 0 of a tagged usize pointer is 1 iff freshly allocated via
        // `PyLong_FromUnsignedLong` (already owns refcount 1, no Py_INCREF).
        #[inline(always)]
        unsafe fn resolve_and_pf(tp: *const Py<PyAny>, tlen: usize, t: u32) -> usize {
            let ti = t as usize;
            if ti < tlen {
                let p = (*tp.add(ti)).as_ptr();
                chunk_table::prefetch_write(p as *const u8);
                p as usize
            } else {
                Tokenizer::alloc_pylong_tagged(t)
            }
        }

        const RING: usize = 16;
        let mut ring = [std::mem::MaybeUninit::<usize>::uninit(); RING];
        for (j, slot) in ring.iter_mut().enumerate() {
            let pt = *ip.add(j + RING) as usize;
            if pt < tlen {
                chunk_table::prefetch(tp.add(pt) as *const u8);
            }
            slot.write(resolve_and_pf(tp, tlen, *ip.add(j)));
        }

        let ring_p = ring.as_mut_ptr() as *mut usize;
        let steady = n.saturating_sub(2 * RING);
        for i in 0..steady {
            let slot = i & (RING - 1);
            let tagged = *ring_p.add(slot);
            let pt = *ip.add(i + 2 * RING) as usize;
            if pt < tlen {
                chunk_table::prefetch(tp.add(pt) as *const u8);
            }
            *ring_p.add(slot) = resolve_and_pf(tp, tlen, *ip.add(i + RING));
            let p = (tagged & !1usize) as *mut ffi::PyObject;
            if (tagged & 1) == 0 {
                ffi::Py_INCREF(p);
            }
            #[cfg(not(any(PyPy, GraalPy)))]
            write_item(ob_item, i, p);
            #[cfg(any(PyPy, GraalPy))]
            write_item(list, base_idx, i, p);
        }
        for i in steady..n {
            let slot = i & (RING - 1);
            let tagged = *ring_p.add(slot);
            if i + RING < n {
                *ring_p.add(slot) = resolve_and_pf(tp, tlen, *ip.add(i + RING));
            }
            let p = (tagged & !1usize) as *mut ffi::PyObject;
            if (tagged & 1) == 0 {
                ffi::Py_INCREF(p);
            }
            #[cfg(not(any(PyPy, GraalPy)))]
            write_item(ob_item, i, p);
            #[cfg(any(PyPy, GraalPy))]
            write_item(list, base_idx, i, p);
        }
    }

    /// `ids` -> Python list[int], reusing the pre-built int objects.
    fn to_pylist<'py>(&self, py: Python<'py>, ids: &[u32]) -> Bound<'py, PyList> {
        let table = self.ints(py);
        // SAFETY: PyList_New(n) returns a new list with n NULL slots, each of which
        // we fill exactly once with an owned reference (PyList_SET_ITEM steals it).
        unsafe {
            let list = pyo3::ffi::PyList_New(ids.len() as pyo3::ffi::Py_ssize_t);
            if list.is_null() {
                return PyList::new(py, ids).expect("list alloc");
            }
            Self::fill_pylist_prefetched(table, ids, list, 0);
            Bound::from_owned_ptr(py, list).cast_into_unchecked()
        }
    }

    /// Python list-of-lists from one contiguous token buffer + per-doc counts.
    fn data_to_pylists<'py>(&self, py: Python<'py>, data: &[u32], doc_counts: &[usize]) -> Bound<'py, PyList> {
        let table = self.ints(py);
        let tlen = table.len();
        let tp = table.as_ptr();
        let dlen = data.len();
        let mut off = 0usize;
        // SAFETY: each new list's slots are filled exactly once with owned refs.
        unsafe {
            let outer = pyo3::ffi::PyList_New(doc_counts.len() as pyo3::ffi::Py_ssize_t);
            assert!(!outer.is_null(), "list alloc");
            for (d, &c) in doc_counts.iter().enumerate() {
                let inner = pyo3::ffi::PyList_New(c as pyo3::ffi::Py_ssize_t);
                assert!(!inner.is_null(), "list alloc");
                if c >= 24 {
                    Self::fill_pylist_prefetched(table, &data[off..off + c], inner, 0);
                } else {
                    for k in 0..c {
                        let gi = off + k;
                        if gi + 16 < dlen {
                            let pt = *data.get_unchecked(gi + 16) as usize;
                            if pt < tlen {
                                chunk_table::prefetch(tp.add(pt) as *const u8);
                            }
                        }
                        let t = *data.get_unchecked(gi);
                        let obj = if (t as usize) < tlen {
                            let p = (*tp.add(t as usize)).as_ptr();
                            pyo3::ffi::Py_INCREF(p);
                            p
                        } else {
                            pyo3::ffi::PyLong_FromUnsignedLong(t as _)
                        };
                        pyo3::ffi::PyList_SET_ITEM(inner, k as pyo3::ffi::Py_ssize_t, obj);
                    }
                }
                off += c;
                pyo3::ffi::PyList_SET_ITEM(outer, d as pyo3::ffi::Py_ssize_t, inner);
            }
            Bound::from_owned_ptr(py, outer).cast_into_unchecked()
        }
    }

    /// Python list-of-lists directly from flat per-chunk `(buf, ends)` pairs
    /// (zero intermediate flat concatenation copy).
    fn chunks_to_pylists<'py>(
        &self,
        py: Python<'py>,
        chunks: &[(Vec<u32>, Vec<usize>)],
    ) -> Bound<'py, PyList> {
        let nrows: usize = chunks.iter().map(|(_, e)| e.len()).sum();
        let table = self.ints(py);
        let tlen = table.len();
        let tp = table.as_ptr();
        // SAFETY: each new list's slots are filled exactly once with owned refs.
        unsafe {
            let outer = pyo3::ffi::PyList_New(nrows as pyo3::ffi::Py_ssize_t);
            assert!(!outer.is_null(), "list alloc");
            let mut row = 0usize;
            for (data, ends) in chunks {
                let dlen = data.len();
                let mut off = 0usize;
                for &end in ends {
                    let c = end - off;
                    let inner = pyo3::ffi::PyList_New(c as pyo3::ffi::Py_ssize_t);
                    assert!(!inner.is_null(), "list alloc");
                    if c >= 24 {
                        Self::fill_pylist_prefetched(table, &data[off..end], inner, 0);
                    } else {
                        for k in 0..c {
                            let gi = off + k;
                            if gi + 16 < dlen {
                                let pt = *data.get_unchecked(gi + 16) as usize;
                                if pt < tlen {
                                    chunk_table::prefetch(tp.add(pt) as *const u8);
                                }
                            }
                            let t = *data.get_unchecked(gi);
                            let obj = if (t as usize) < tlen {
                                let p = (*tp.add(t as usize)).as_ptr();
                                pyo3::ffi::Py_INCREF(p);
                                p
                            } else {
                                pyo3::ffi::PyLong_FromUnsignedLong(t as _)
                            };
                            pyo3::ffi::PyList_SET_ITEM(inner, k as pyo3::ffi::Py_ssize_t, obj);
                        }
                    }
                    off = end;
                    pyo3::ffi::PyList_SET_ITEM(outer, row as pyo3::ffi::Py_ssize_t, inner);
                    row += 1;
                }
            }
            Bound::from_owned_ptr(py, outer).cast_into_unchecked()
        }
    }

    /// Run `f` with the GIL released only when the input is big enough to pay off.
    #[inline]
    fn maybe_detach<T: Send>(&self, py: Python<'_>, len: usize, f: impl FnOnce() -> T + Send) -> T {
        if len < Self::DETACH_MIN {
            f()
        } else {
            py.detach(f)
        }
    }


    /// Build the whole-chunk table: for every vocab token whose bytes are a
    /// plausible pre-token (1..=15 bytes), verify whether those bytes encode to
    /// ONE token under this tokenizer's exact BPE rules.
    ///
    /// For monotonic ByteLevel BPE (`byte_fast && monotonic && ignore_lookup.is_none()`),
    /// we prove self-encoding directly on the merge DAG in parallel (~3.7 boundary-spine
    /// probes per token instead of full BPE merges): token `tg = (t1, t2)` at rank `r`
    /// self-encodes iff `t1` and `t2` self-encode, `bytes(t1) ++ bytes(t2) == bytes(tg)`,
    /// `map.lookup_rank(t1, t2) == r`, and walking the right spine of `t1` & left spine
    /// of `t2` bottom-up in exact canonical rank order finds no preempting cross-boundary
    /// merge (`< lu` against left-spine merges, `<= rv` against right-spine merges).
    /// Any token not proven by the fast DAG check falls back to exact `self.emit`,
    /// guaranteeing 100% identical output for every possible vocabulary.
    fn build_chunk_table(&mut self) {
        if self.singleton_eow.is_some() || self.byte_fallback {
            // - Models with `end_of_word_suffix` (e.g. CLIP `</w>`) store the suffix
            //   in `byte_span`, whereas pre-token chunks arrive without it.
            // - `byte_fallback` / SentencePiece models (Gemma-2/3, Mistral, Phi-3.5, Yi-1.5)
            //   have 50-60K non-monotonic / multi-prod merges; empirically skipping the
            //   12.6 MB static `ChunkTable` cuts startup ~1.8-2.0x (-16..17 ms on Gemma-2/3),
            //   saves 12.6 MB RAM, AND improves cold 1T encode +11-21% by keeping our
            //   2 MiB frequency-ranked `dense` matrix + `SharedTable` hot in L2 cache!
            self.chunk_table = chunk_table::ChunkTable::empty();
            return;
        }
        let this = &*self;
        let n_vocab = this.byte_span.len();
        let emit_one = |s: &mut Scratch, id: u32| -> Option<(&[u8], u32)> {
            let (off, len) = this.byte_span[id as usize];
            let b = &this.byte_arena[off as usize..(off + len) as usize];
            s.out.clear();
            this.emit(b, s, Merge::Auto);
            if s.out.len() != 1 {
                return None;
            }
            Some((b, s.out[0]))
        };

        let pairs: Vec<(&[u8], u32)> = if this.ignore_merges {
            // Under `ignore_merges: true`, every vocab token `1..=15B` wins directly
            // over BPE merges (first-ID-wins on duplicates, which `ChunkTable` preserves).
            let mut pairs: Vec<(&[u8], u32)> = Vec::with_capacity(n_vocab);
            for (id, &(off, len)) in this.byte_span.iter().enumerate() {
                if len >= 1 && (len as usize) <= chunk_table::MAX_KEY {
                    pairs.push((&this.byte_arena[off as usize..(off + len) as usize], id as u32));
                }
            }
            pairs
        } else {
            // Generalized O(N) local-monotonic DAG self-encoding verifier for ALL ByteLevel
            // models (both globally monotonic models and locally-monotonic models with rare
            // rank inversions such as Qwen3.6-35B with 23 inverted tokens out of 248K).
            //
            // Strong-induction proof:
            //   1. Mark the 256 byte singletons as atomic leaves (`is_leaf[sid] = true`,
            //      `prod[sid] = LEAF`).
            //   2. For every non-leaf `tg`, record its FIRST (lowest-rank) merge `(t1, t2, r)`.
            //   3. `verify_self_enc_range` checks in parallel (with SIMD MergeMap probing) that:
            //      - `bytes(t1) + bytes(t2) == bytes(tg)` and `lookup(t1, t2) == (r, tg)`,
            //      - every edge along `right_spine(t1)` and `left_spine(t2)` has strictly
            //        decreasing rank `cr < nxt_r` down to a `LEAF`, and
            //      - no cross-seam merge `(u, v)` fires earlier than its enclosing internal merge.
            //   4. Propagate `self_enc` in ascending merge-rank order `r`: since `local_ok[tg]`
            //      requires `prod[t1].2 < r` and `prod[t2].2 < r` (or `LEAF`), both children are
            //      already finalized at rank `r`. Any unproven or rank-inverted token falls back
            //      to exact `emit_one`, guaranteeing 100% bit-for-bit identical output.
            const LEAF: u32 = u32::MAX;
            let mut is_leaf = vec![false; n_vocab];
            for b in 0..256usize {
                if this.singleton[b] >= 0 {
                    let sid = this.singleton_u32[b] as usize;
                    if sid < n_vocab {
                        let (off, len) = this.byte_span[sid];
                        if len == 1 && this.byte_arena[off as usize] == b as u8 {
                            is_leaf[sid] = true;
                        }
                    }
                }
            }

            let mut prod: Vec<(u32, u32, u32)> = vec![(0, 0, LEAF); n_vocab];
            for (r, &(t1, t2, tg)) in this.raw_merges.iter().enumerate() {
                let tgi = tg as usize;
                if (t1 as usize) < n_vocab
                    && (t2 as usize) < n_vocab
                    && tgi < n_vocab
                    && !is_leaf[tgi]
                    && prod[tgi].2 == LEAF
                {
                    prod[tgi] = (t1, t2, r as u32);
                }
            }
            let mut local_ok = vec![false; n_vocab];
            if parallel_ok() {
                local_ok
                    .par_chunks_mut(2048)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        this.map.verify_self_enc_range(
                            &this.byte_arena,
                            &this.byte_span,
                            &prod,
                            ci * 2048,
                            chunk,
                        );
                    });
            } else {
                this.map.verify_self_enc_range(&this.byte_arena, &this.byte_span, &prod, 0, &mut local_ok);
            }

            // Topological propagation in ascending merge-rank order.
            let mut self_enc = is_leaf;
            for (r, &(t1, t2, tg)) in this.raw_merges.iter().enumerate() {
                let tgi = tg as usize;
                if tgi < n_vocab
                    && prod[tgi].2 == r as u32
                    && local_ok[tgi]
                    && self_enc[t1 as usize]
                    && self_enc[t2 as usize]
                {
                    self_enc[tgi] = true;
                }
            }

            let mut pairs: Vec<(&[u8], u32)> = Vec::with_capacity(n_vocab);
            let mut fallback_ids: Vec<u32> = Vec::new();
            for id in 0..n_vocab as u32 {
                let (off, len) = this.byte_span[id as usize];
                if len >= 1 && (len as usize) <= chunk_table::MAX_KEY {
                    if self_enc[id as usize] {
                        pairs.push((&this.byte_arena[off as usize..(off + len) as usize], id));
                    } else {
                        fallback_ids.push(id);
                    }
                }
            }
            if !fallback_ids.is_empty() {
                if parallel_ok() && fallback_ids.len() >= 1024 {
                    let extra: Vec<(&[u8], u32)> = fallback_ids
                        .par_iter()
                        .with_min_len(256)
                        .map_init(Scratch::default, |s, &id| emit_one(s, id))
                        .flatten()
                        .collect();
                    pairs.extend(extra);
                } else {
                    let mut s = Scratch::default();
                    for id in fallback_ids {
                        if let Some(p) = emit_one(&mut s, id) {
                            pairs.push(p);
                        }
                    }
                }
            }
            pairs
        };
        let table = chunk_table::ChunkTable::build_with_parallel(&pairs, parallel_ok());
        // Demand-paged active working-set SharedTable: keep `self.learned`
        // initially empty (512 KiB) so it projects a 200K multilingual vocab
        // down to the exact L2-resident active working set of the user's text
        // (ablated +5-7% faster than blind low-ID pre-seeding). Call `warmup()`
        // with representative domain text to pre-populate for production.
        drop(pairs);
        self.chunk_table = table;
    }

    /// Lazily construct the 16-byte inline decode table (zero startup cost when
    /// decode is not called). Every token `<= 12` bytes stores its payload
    /// inline with precomputed ASCII / special / leading-space flags.
    #[inline]
    fn decode_table(&self) -> &[DecodeEntry] {
        self.decode_table.get_or_init(|| {
            const {
                assert!(std::mem::size_of::<DecodeEntry>() == 16);
                assert!(std::mem::align_of::<DecodeEntry>() == 16);
            }
            let n = self.byte_span.len();
            let mut tbl = Vec::with_capacity(n);
            let arena = self.byte_arena.as_slice();
            let skip = self.special_skip_mask.as_slice();
            for (ti, &(off, len)) in self.byte_span.iter().enumerate() {
                let (off_u, len_u) = (off as usize, len as usize);
                let bytes = &arena[off_u..off_u + len_u];
                let mut flags = 0u8;
                if bytes.iter().any(|&b| b >= 0x80) {
                    flags |= DEC_FLAG_NON_ASCII;
                }
                if ti < skip.len() && skip[ti] {
                    flags |= DEC_FLAG_SPECIAL;
                }
                let mut data = [0u8; 12];
                let elen = if len_u <= 12 {
                    data[..len_u].copy_from_slice(bytes);
                    len_u as u8
                } else {
                    data[0..4].copy_from_slice(&off.to_ne_bytes());
                    data[4..8].copy_from_slice(&len.to_ne_bytes());
                    DEC_OUTLINE_LEN
                };
                tbl.push(DecodeEntry { data, len: elen, flags, _pad: 0 });
            }
            tbl
        })
    }

    /// Single-pass fast decode of `&[u32]` into `buf` (cleared first), returning
    /// `is_ascii` (true iff every emitted byte is `< 0x80`). Halves `DecodeEntry`
    /// L2/L3 cache misses vs two-pass measure+copy by pre-reserving `ids.len()*5+32`
    /// and emitting 16-byte inline SIMD stores directly with 16-ahead prefetching.
    #[inline]
    fn gather_bytes_single_pass_into(
        &self,
        ids: &[u32],
        skip_special: bool,
        buf: &mut Vec<u8>,
    ) -> bool {
        buf.clear();
        let ids = if skip_special
            && self.whisper_sop_id != u32::MAX
            && ids.first() == Some(&self.whisper_sop_id)
        {
            match ids.iter().position(|&t| t == self.whisper_sot_id) {
                Some(pos) => &ids[pos..],
                None => return true,
            }
        } else {
            ids
        };
        let n = ids.len();
        if n == 0 {
            return true;
        }
        let dt = self.decode_table();
        let ndt = dt.len();
        let dtp = dt.as_ptr();
        let arena = self.byte_arena.as_ptr();
        let ts_begin = self.whisper_ts_begin;
        let unfiltered = !skip_special && ts_begin == u32::MAX;

        // Typical LLM token averages ~4 bytes; reserving 5*n + 32 avoids realloc
        // on >99% of inputs while keeping +16 slack for unconditional 16B stores.
        buf.reserve(n.saturating_mul(5).saturating_add(32));
        let mut base = buf.as_mut_ptr();
        let mut dst = base;
        let mut cap_end = unsafe { base.add(buf.capacity()) };
        let mut non_ascii = 0u8;

        #[inline(always)]
        unsafe fn ensure_cap(buf: &mut Vec<u8>, base: &mut *mut u8, dst: &mut *mut u8, cap_end: &mut *mut u8, need: usize, rem_toks: usize) {
            let cur = (*dst).offset_from(*base) as usize;
            buf.set_len(cur);
            buf.reserve(need.saturating_add(rem_toks.saturating_mul(4)).saturating_add(32));
            *base = buf.as_mut_ptr();
            *dst = (*base).add(cur);
            *cap_end = (*base).add(buf.capacity());
        }

        unsafe {
            if unfiltered {
                for i in 0..n {
                    if i + 16 < n {
                        let pt = *ids.get_unchecked(i + 16) as usize;
                        if pt < ndt {
                            chunk_table::prefetch(dtp.add(pt) as *const u8);
                        }
                    }
                    let ti = *ids.get_unchecked(i) as usize;
                    if ti >= ndt {
                        continue;
                    }
                    let e = *dtp.add(ti);
                    non_ascii |= e.flags;
                    if e.len != DEC_OUTLINE_LEN {
                        if dst.add(16) > cap_end {
                            ensure_cap(buf, &mut base, &mut dst, &mut cap_end, 16, n - i);
                        }
                        std::ptr::copy_nonoverlapping(&e as *const DecodeEntry as *const u8, dst, 16);
                        dst = dst.add(e.len as usize);
                    } else {
                        let (off, len) = e.outline_span();
                        if dst.add(len + 16) > cap_end {
                            ensure_cap(buf, &mut base, &mut dst, &mut cap_end, len + 16, n - i);
                        }
                        std::ptr::copy_nonoverlapping(arena.add(off), dst, len);
                        dst = dst.add(len);
                    }
                }
            } else {
                for i in 0..n {
                    if i + 16 < n {
                        let pt = *ids.get_unchecked(i + 16) as usize;
                        if pt < ndt {
                            chunk_table::prefetch(dtp.add(pt) as *const u8);
                        }
                    }
                    let t = *ids.get_unchecked(i);
                    if t >= ts_begin {
                        continue;
                    }
                    let ti = t as usize;
                    if ti >= ndt {
                        continue;
                    }
                    let e = *dtp.add(ti);
                    if skip_special && (e.flags & DEC_FLAG_SPECIAL) != 0 {
                        continue;
                    }
                    non_ascii |= e.flags;
                    if e.len != DEC_OUTLINE_LEN {
                        if dst.add(16) > cap_end {
                            ensure_cap(buf, &mut base, &mut dst, &mut cap_end, 16, n - i);
                        }
                        std::ptr::copy_nonoverlapping(&e as *const DecodeEntry as *const u8, dst, 16);
                        dst = dst.add(e.len as usize);
                    } else {
                        let (off, len) = e.outline_span();
                        if dst.add(len + 16) > cap_end {
                            ensure_cap(buf, &mut base, &mut dst, &mut cap_end, len + 16, n - i);
                        }
                        std::ptr::copy_nonoverlapping(arena.add(off), dst, len);
                        dst = dst.add(len);
                    }
                }
            }
            buf.set_len(dst.offset_from(base) as usize);
        }
        (non_ascii & DEC_FLAG_NON_ASCII) == 0
    }

    /// Fused single-pass `PyList[int]` -> `buf` decode with 2-stage software
    /// prefetching (Stage 0 prefetches `PyLongObject` 24 ahead; Stage 1 extracts
    /// `u32` + prefetches `DecodeEntry` 12 ahead; Stage 2 copies 16B inline
    /// bytes straight into `buf`). Zero intermediate `Vec<u32>` allocation!
    /// Returns `Some(is_ascii)` on success or `None` if any element is non-u32
    /// (caller clears PyErr and falls back to generic extractor).
    #[inline]
    unsafe fn gather_pylist_fused_into(
        &self,
        list_ptr: *mut pyo3::ffi::PyObject,
        n: usize,
        inline_long: bool,
        skip_special: bool,
        buf: &mut Vec<u8>,
    ) -> Option<bool> {
        use pyo3::ffi;
        buf.clear();
        if n == 0 {
            return Some(true);
        }
        // Fast bail for Whisper `<|startofprev|>` prompt strip (rare; let slice path handle it).
        let ts_begin = self.whisper_ts_begin;
        if skip_special && self.whisper_sop_id != u32::MAX {
            return None;
        }
        let dt = self.decode_table();
        let ndt = dt.len();
        let dtp = dt.as_ptr();
        let arena = self.byte_arena.as_ptr();
        let unfiltered = !skip_special && ts_begin == u32::MAX;

        buf.reserve(n.saturating_mul(5).saturating_add(32));
        let mut base = buf.as_mut_ptr();
        let mut dst = base;
        let mut cap_end = base.add(buf.capacity());
        let mut non_ascii = 0u8;

        #[inline(always)]
        unsafe fn read_one_id(item: *mut ffi::PyObject, inline_long: bool) -> Option<u32> {
            #[cfg(all(Py_3_12, target_pointer_width = "64", not(any(PyPy, GraalPy, Py_LIMITED_API, Py_GIL_DISABLED))))]
            if inline_long {
                let lt = &raw mut ffi::PyLong_Type;
                if ffi::Py_TYPE(item) == lt {
                    let tag = *(item as *const usize).add(2) & !4usize;
                    if tag == 8 {
                        return Some(*(item as *const u32).add(6));
                    }
                    if tag == 1 {
                        return Some(0);
                    }
                }
            }
            let _ = inline_long;
            let val = ffi::PyLong_AsLongLong(item);
            if val < 0 || val > u32::MAX as i64 || !ffi::PyErr_Occurred().is_null() {
                return None;
            }
            Some(val as u32)
        }

        #[inline(always)]
        unsafe fn ensure_cap(buf: &mut Vec<u8>, base: &mut *mut u8, dst: &mut *mut u8, cap_end: &mut *mut u8, need: usize, rem_toks: usize) {
            let cur = (*dst).offset_from(*base) as usize;
            buf.set_len(cur);
            buf.reserve(need.saturating_add(rem_toks.saturating_mul(4)).saturating_add(32));
            *base = buf.as_mut_ptr();
            *dst = (*base).add(cur);
            *cap_end = (*base).add(buf.capacity());
        }

        // Split into zero-branch steady-state loop (`0..steady`) + short tail
        // (`steady..n`) and hoist `unfiltered` + `inline_long` outside.
        const RING: usize = 16;
        let mut ring = [0u32; RING];
        let prime = n.min(RING);
        for (j, slot) in ring.iter_mut().enumerate().take(prime) {
            if j + RING < n {
                chunk_table::prefetch(ffi::PyList_GET_ITEM(list_ptr, (j + RING) as ffi::Py_ssize_t) as *const u8);
            }
            let item = ffi::PyList_GET_ITEM(list_ptr, j as ffi::Py_ssize_t);
            let t = read_one_id(item, inline_long)?;
            *slot = t;
            if (t as usize) < ndt {
                chunk_table::prefetch(dtp.add(t as usize) as *const u8);
            }
        }

        macro_rules! emit_tok {
            ($t:expr, $rem_toks:expr, $unfilt:expr) => {{
                let t: u32 = $t;
                if $unfilt || t < ts_begin {
                    let ti = t as usize;
                    if ti < ndt {
                        let e = *dtp.add(ti);
                        if $unfilt || !skip_special || (e.flags & DEC_FLAG_SPECIAL) == 0 {
                            non_ascii |= e.flags;
                            if e.len != DEC_OUTLINE_LEN {
                                if dst.add(16) > cap_end {
                                    ensure_cap(buf, &mut base, &mut dst, &mut cap_end, 16, $rem_toks);
                                }
                                std::ptr::copy_nonoverlapping(&e as *const DecodeEntry as *const u8, dst, 16);
                                dst = dst.add(e.len as usize);
                            } else {
                                let (off, len) = e.outline_span();
                                if dst.add(len + 16) > cap_end {
                                    ensure_cap(buf, &mut base, &mut dst, &mut cap_end, len + 16, $rem_toks);
                                }
                                std::ptr::copy_nonoverlapping(arena.add(off), dst, len);
                                dst = dst.add(len);
                            }
                        }
                    }
                }
            }};
        }

        let steady = n.saturating_sub(2 * RING);
        if unfiltered {
            for i in 0..steady {
                let slot = i & (RING - 1);
                let t = *ring.get_unchecked(slot);
                let pf_item = ffi::PyList_GET_ITEM(list_ptr, (i + 2 * RING) as ffi::Py_ssize_t);
                chunk_table::prefetch(pf_item as *const u8);
                let next_item = ffi::PyList_GET_ITEM(list_ptr, (i + RING) as ffi::Py_ssize_t);
                let tn = read_one_id(next_item, inline_long)?;
                *ring.get_unchecked_mut(slot) = tn;
                if (tn as usize) < ndt {
                    chunk_table::prefetch(dtp.add(tn as usize) as *const u8);
                }
                emit_tok!(t, n - i, true);
            }
            for i in steady..n {
                let slot = i & (RING - 1);
                let t = *ring.get_unchecked(slot);
                let next_idx = i + RING;
                if next_idx < n {
                    let next_item = ffi::PyList_GET_ITEM(list_ptr, next_idx as ffi::Py_ssize_t);
                    let tn = read_one_id(next_item, inline_long)?;
                    *ring.get_unchecked_mut(slot) = tn;
                    if (tn as usize) < ndt {
                        chunk_table::prefetch(dtp.add(tn as usize) as *const u8);
                    }
                }
                emit_tok!(t, n - i, true);
            }
        } else {
            for i in 0..steady {
                let slot = i & (RING - 1);
                let t = *ring.get_unchecked(slot);
                let pf_item = ffi::PyList_GET_ITEM(list_ptr, (i + 2 * RING) as ffi::Py_ssize_t);
                chunk_table::prefetch(pf_item as *const u8);
                let next_item = ffi::PyList_GET_ITEM(list_ptr, (i + RING) as ffi::Py_ssize_t);
                let tn = read_one_id(next_item, inline_long)?;
                *ring.get_unchecked_mut(slot) = tn;
                if (tn as usize) < ndt {
                    chunk_table::prefetch(dtp.add(tn as usize) as *const u8);
                }
                emit_tok!(t, n - i, false);
            }
            for i in steady..n {
                let slot = i & (RING - 1);
                let t = *ring.get_unchecked(slot);
                let next_idx = i + RING;
                if next_idx < n {
                    let next_item = ffi::PyList_GET_ITEM(list_ptr, next_idx as ffi::Py_ssize_t);
                    let tn = read_one_id(next_item, inline_long)?;
                    *ring.get_unchecked_mut(slot) = tn;
                    if (tn as usize) < ndt {
                        chunk_table::prefetch(dtp.add(tn as usize) as *const u8);
                    }
                }
                emit_tok!(t, n - i, false);
            }
        }

        buf.set_len(dst.offset_from(base) as usize);
        Some((non_ascii & DEC_FLAG_NON_ASCII) == 0)
    }

    /// Fast decode of `&[u32]` into `(Vec<u8>, is_ascii)` (already leading-space
    /// stripped) for parallel `decode_batch` workers.
    fn gather_bytes_fast_opts(
        &self,
        ids: &[u32],
        skip_special: bool,
        decode_with_timestamps: bool,
        time_precision: f64,
        segment_size: usize,
    ) -> (Vec<u8>, bool) {
        let mut buf = Vec::new();
        let is_ascii = if decode_with_timestamps && self.whisper_ts_begin != u32::MAX {
            self.gather_bytes_into_opts(ids, skip_special, true, time_precision, segment_size, &mut buf)
        } else {
            self.gather_bytes_single_pass_into(ids, skip_special, &mut buf)
        };
        if self.decode_strip_space && buf.first() == Some(&b' ') {
            buf.remove(0);
        }
        (buf, is_ascii)
    }

    /// Decode `&[u32]` directly to a Python `str` using a single-pass gather
    /// into reusable `scratch_buf`, then direct `PyUnicode_New(len, 127)` for
    /// pure ASCII (skipping UTF-8 validation) or 1-pass `PyUnicode_DecodeUTF8`.
    #[allow(clippy::too_many_arguments)]
    fn decode_slice_to_pystr<'py>(
        &self,
        py: Python<'py>,
        ids: &[u32],
        skip_special: bool,
        decode_with_timestamps: bool,
        time_precision: f64,
        segment_size: usize,
        scratch_buf: &mut Vec<u8>,
    ) -> Bound<'py, PyString> {
        let is_ascii = if decode_with_timestamps && self.whisper_ts_begin != u32::MAX {
            self.gather_bytes_into_opts(ids, skip_special, true, time_precision, segment_size, scratch_buf)
        } else {
            self.gather_bytes_single_pass_into(ids, skip_special, scratch_buf)
        };
        let slice = self.strip_decoded(scratch_buf);
        if is_ascii {
            ascii_bytes_to_pystr(py, slice)
        } else {
            bytes_to_pystr(py, slice)
        }
    }

    /// Gather the raw bytes for `ids` into `out` (cleared first), supporting
    /// `skip_special` and Whisper prompt stripping / timestamp stripping /
    /// multi-segment timestamp accumulation (`decode_with_timestamps=true`).
    /// Returns `true` iff every emitted byte is guaranteed ASCII (`< 0x80`).
    fn gather_bytes_into_opts(
        &self,
        ids: &[u32],
        skip_special: bool,
        decode_with_timestamps: bool,
        time_precision: f64,
        segment_size: usize,
        out: &mut Vec<u8>,
    ) -> bool {
        out.clear();
        let ts_begin = self.whisper_ts_begin;
        if !skip_special && ts_begin == u32::MAX {
            return self.gather_bytes_single_pass_into(ids, false, out);
        }
        // Whisper `_strip_prompt`: when `skip_special=true` and the sequence begins
        // with `<|startofprev|>`, drop everything before `<|startoftranscript|>`
        // (or yield empty if absent).
        let ids = if skip_special
            && self.whisper_sop_id != u32::MAX
            && ids.first() == Some(&self.whisper_sop_id)
        {
            match ids.iter().position(|&t| t == self.whisper_sot_id) {
                Some(pos) => &ids[pos..],
                None => return true,
            }
        } else {
            ids
        };

        if decode_with_timestamps && ts_begin != u32::MAX {
            // Exact single-pass port of HF `WhisperTokenizerFast._decode_with_timestamps`:
            // accumulate segment rollovers (>30s audio) and rewind consecutive
            // boundary timestamps using 16B inline `DecodeEntry` stores, 16-ahead
            // prefetching, zero-alloc stack `marks`, and pure-ASCII tracking.
            use std::io::Write;
            let n = ids.len();
            if n == 0 {
                return true;
            }
            let dt = self.decode_table();
            let ndt = dt.len();
            let dtp = dt.as_ptr();
            let arena = self.byte_arena.as_ptr();

            out.reserve(n.saturating_mul(5).saturating_add(32));
            let mut base = out.as_mut_ptr();
            let mut dst = base;
            let mut cap_end = unsafe { base.add(out.capacity()) };
            let mut non_ascii = 0u8;

            #[inline(always)]
            unsafe fn ensure_cap(
                out: &mut Vec<u8>,
                base: &mut *mut u8,
                dst: &mut *mut u8,
                cap_end: &mut *mut u8,
                need: usize,
                rem_toks: usize,
            ) {
                let cur = (*dst).offset_from(*base) as usize;
                out.set_len(cur);
                out.reserve(need.saturating_add(rem_toks.saturating_mul(4)).saturating_add(32));
                *base = out.as_mut_ptr();
                *dst = (*base).add(cur);
                *cap_end = (*base).add(out.capacity());
            }

            // Stack-backed mark buffer (zero heap alloc for <= 31 timestamps/transcript).
            let mut marks_arr = [0usize; 64];
            let mut marks_len = 1usize;
            let mut marks_vec: Vec<usize> = Vec::new();
            let push_mark = |val: usize, arr: &mut [usize; 64], mlen: &mut usize, mv: &mut Vec<usize>| {
                if mv.is_empty() {
                    if *mlen < 64 {
                        arr[*mlen] = val;
                        *mlen += 1;
                    } else {
                        mv.reserve(128);
                        mv.extend_from_slice(&arr[..*mlen]);
                        mv.push(val);
                    }
                } else {
                    mv.push(val);
                }
            };

            let mut cur_max_ts = 0.0f64;
            let mut prev_seg_len = 0.0f64;
            let mut penult_ts = 0.0f64;

            unsafe {
                for i in 0..n {
                    if i + 16 < n {
                        let pt = *ids.get_unchecked(i + 16) as usize;
                        if pt < ndt {
                            chunk_table::prefetch(dtp.add(pt) as *const u8);
                        }
                    }
                    let t = *ids.get_unchecked(i);
                    if t >= ts_begin {
                        let ts = (t - ts_begin) as f64 * time_precision;
                        if ts < cur_max_ts {
                            let last_was_single_ending =
                                i >= 2 && !(*ids.get_unchecked(i - 1) >= ts_begin && *ids.get_unchecked(i - 2) >= ts_begin);
                            if last_was_single_ending {
                                prev_seg_len += time_precision * (segment_size as f64);
                            } else {
                                cur_max_ts = penult_ts;
                                prev_seg_len += penult_ts;
                                if marks_vec.is_empty() {
                                    if marks_len >= 2 {
                                        marks_len -= 2;
                                        dst = base.add(marks_arr[marks_len]);
                                    } else {
                                        marks_len = 0;
                                        dst = base;
                                    }
                                } else if marks_vec.len() >= 2 {
                                    let new_len = marks_vec.len() - 2;
                                    let cut = marks_vec[new_len];
                                    marks_vec.truncate(new_len);
                                    dst = base.add(cut);
                                } else {
                                    marks_vec.clear();
                                    dst = base;
                                }
                            }
                        }
                        penult_ts = cur_max_ts;
                        cur_max_ts = ts;
                        let cur_len = dst.offset_from(base) as usize;
                        push_mark(cur_len, &mut marks_arr, &mut marks_len, &mut marks_vec);
                        // Fast fixed-point `<|S.CC|>` formatting for standard 0.02s Whisper
                        // precision (exact match to `format!("<|{:.2}|>", val)` without std::fmt
                        // float Dragonbox overhead); falls back to `write!` for non-standard precisions.
                        let val = ts + prev_seg_len;
                        if time_precision == 0.02 && (0.0..1e9).contains(&val) {
                            if dst.add(24) > cap_end {
                                ensure_cap(out, &mut base, &mut dst, &mut cap_end, 24, n - i);
                            }
                            let cents = (val * 100.0 + 0.5) as u64;
                            let secs = cents / 100;
                            let frac = (cents % 100) as u8;
                            *dst = b'<';
                            *dst.add(1) = b'|';
                            dst = dst.add(2);
                            if secs == 0 {
                                *dst = b'0';
                                dst = dst.add(1);
                            } else {
                                let mut d = [0u8; 12];
                                let mut dn = 0usize;
                                let mut s = secs;
                                while s > 0 {
                                    *d.get_unchecked_mut(dn) = b'0' + (s % 10) as u8;
                                    s /= 10;
                                    dn += 1;
                                }
                                while dn > 0 {
                                    dn -= 1;
                                    *dst = *d.get_unchecked(dn);
                                    dst = dst.add(1);
                                }
                            }
                            *dst = b'.';
                            *dst.add(1) = b'0' + (frac / 10);
                            *dst.add(2) = b'0' + (frac % 10);
                            *dst.add(3) = b'|';
                            *dst.add(4) = b'>';
                            dst = dst.add(5);
                        } else {
                            out.set_len(cur_len);
                            let _ = write!(out, "<|{val:.2}|>");
                            base = out.as_mut_ptr();
                            dst = base.add(out.len());
                            cap_end = base.add(out.capacity());
                        }
                        push_mark(dst.offset_from(base) as usize, &mut marks_arr, &mut marks_len, &mut marks_vec);
                    } else {
                        let ti = t as usize;
                        if ti < ndt {
                            let e = *dtp.add(ti);
                            if skip_special && (e.flags & DEC_FLAG_SPECIAL) != 0 {
                                continue;
                            }
                            non_ascii |= e.flags;
                            if e.len != DEC_OUTLINE_LEN {
                                if dst.add(16) > cap_end {
                                    ensure_cap(out, &mut base, &mut dst, &mut cap_end, 16, n - i);
                                }
                                std::ptr::copy_nonoverlapping(&e as *const DecodeEntry as *const u8, dst, 16);
                                dst = dst.add(e.len as usize);
                            } else {
                                let (off, len) = e.outline_span();
                                if dst.add(len + 16) > cap_end {
                                    ensure_cap(out, &mut base, &mut dst, &mut cap_end, len + 16, n - i);
                                }
                                std::ptr::copy_nonoverlapping(arena.add(off), dst, len);
                                dst = dst.add(len);
                            }
                        }
                    }
                }
                out.set_len(dst.offset_from(base) as usize);
            }
            return (non_ascii & DEC_FLAG_NON_ASCII) == 0;
        }

        self.gather_bytes_single_pass_into(ids, skip_special, out)
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
        // Explicit streaming parity-test path (built lazily on demand).
        if mode == Merge::Stream && !self.byte_fallback && self.monotonic {
            let st = self
                .stream
                .get_or_init(|| bpe::StreamTables::new(&self.raw_merges, self.vocab_size));
            st.merge_into(&s.chunk, &mut s.out, &mut s.state, &mut s.todo);
            return;
        }
        // For chunks > 64 tokens (or non-monotonic vocabs), use our O(n log n)
        // zero-lookup-pop canonical heap merger (20-100x faster than StreamTables'
        // `next_token1_dest` chain walk on long runs and exact for all vocabs).
        if !self.monotonic || s.chunk.len() > 64 {
            bpe::merge_canonical(&mut s.chunk, &self.map, &mut s.canon);
            s.out.extend_from_slice(&s.chunk);
            return;
        }
        let ok = bpe::merge(&mut s.chunk, &self.map, &mut s.ranks, &mut s.ntok,
                            &mut s.src, &mut s.nrank);
        debug_assert!(ok, "monotonic vocab hit a rank inversion");
        let _ = ok;
        s.out.extend_from_slice(&s.chunk);
    }

    /// Emit one pre-token chunk: byte_fallback (char lookup) or byte-level.
    #[inline]
    fn emit(&self, chunk: &[u8], s: &mut Scratch, mode: Merge) {
        let n = chunk.len();
        if mode == Merge::Auto
            && n.wrapping_sub(1) < chunk_table::MAX_KEY
            && (self.sp_fast_meta == SpFastMeta::None || !chunk.contains(&b' '))
        {
            let (lo, hi) = chunk_table::pack(chunk);
            if let Some(t) = self.chunk_table.get(lo, hi) {
                s.out.push(t);
                return;
            }
            if s.dyn_cache.owned_by(self.uid) {
                if let Some(ts) = s.dyn_cache.get(lo, hi) {
                    s.out.extend_from_slice(ts); // disjoint field borrows
                    return;
                }
                let start = s.out.len();
                if self.byte_fallback {
                    self.process_chunk_bf(chunk, s, mode);
                } else {
                    self.process_chunk(chunk, s, mode);
                }
                let (out, dc) = (&s.out, &mut s.dyn_cache);
                dc.put(lo, hi, &out[start..]);
                return;
            }
        }
        if self.byte_fallback {
            self.process_chunk_bf(chunk, s, mode);
        } else {
            self.process_chunk(chunk, s, mode);
        }
    }

    /// ByteLevel chunk: each byte -> its singleton token, then merge.
    #[inline]
    fn process_chunk(&self, chunk: &[u8], s: &mut Scratch, mode: Merge) {
        self.process_chunk_impl::<false>(chunk, s, mode);
    }

    /// `CHECKED_SHORT == true`: caller already confirmed `chunk.len() <= MAX_KEY`
    /// missed `self.chunk_table` (avoids redundant `chunk_table` re-lookup under
    /// `ignore_merges: true`).
    #[inline]
    fn process_chunk_impl<const CHECKED_SHORT: bool>(&self, chunk: &[u8], s: &mut Scratch, mode: Merge) {
        if chunk.is_empty() {
            return;
        }
        if self.ignore_merges {
            if !CHECKED_SHORT
                && !self.byte_fallback
                && self.singleton_eow.is_none()
                && chunk.len() <= chunk_table::MAX_KEY
            {
                let (lo, hi) = chunk_table::pack(chunk);
                if let Some(id) = self.chunk_table.get(lo, hi) {
                    s.out.push(id);
                    return;
                }
            } else if (!CHECKED_SHORT || chunk.len() > chunk_table::MAX_KEY)
                && self.ignore_lookup.is_some()
            {
                if let Some(&id) = self.ignore_lookup.as_ref().unwrap().get(chunk) {
                    s.out.push(id);
                    return;
                }
            }
        }
        if self.byte_fast && mode != Merge::Stream && chunk.len() <= 64 {
            bpe::merge_bytes_into(chunk, &self.singleton_u32, &self.map, &mut s.out);
            return;
        }
        s.chunk.clear();
        if let Some(eow) = &self.singleton_eow {
            let last = chunk.len() - 1;
            for &b in &chunk[..last] {
                let t = self.singleton[b as usize];
                if t >= 0 {
                    s.chunk.push(t as u32);
                }
            }
            let b = chunk[last] as usize;
            let t = if eow[b] >= 0 { eow[b] } else { self.singleton[b] };
            if t >= 0 {
                s.chunk.push(t as u32);
            }
        } else {
            for &b in chunk {
                let t = self.singleton[b as usize];
                if t >= 0 {
                    s.chunk.push(t as u32);
                }
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

    /// Direct SentencePiece chunk BPE using 1-byte `' '` as a lossless surrogate
    /// for 3-byte `'▁'` (zero UTF-8 expansion, zero `char_map` lookup for `'▁'`).
    #[inline]
    fn process_chunk_sp_direct(&self, chunk: &[u8], s: &mut Scratch) {
        s.chunk.clear();
        let n = chunk.len();
        let mut i = 0usize;
        while i < n {
            // SAFETY: `i < n`.
            let b = unsafe { *chunk.get_unchecked(i) };
            if b < 128 {
                let tid = if b == b' ' {
                    self.sp_rep_id
                } else {
                    unsafe { *self.char_ascii.get_unchecked(b as usize) }
                };
                if tid >= 0 {
                    s.chunk.push(tid as u32);
                } else if b == b' ' {
                    for &eb in "\u{2581}".as_bytes() {
                        let t = self.singleton[eb as usize];
                        if t >= 0 {
                            s.chunk.push(t as u32);
                        }
                    }
                } else {
                    let t = self.singleton[b as usize];
                    if t >= 0 {
                        s.chunk.push(t as u32);
                    }
                }
                i += 1;
            } else {
                let (cp, cl) = unicode::decode(chunk, i);
                let tid = self.char_map.get(&cp).map(|&x| x as i32).unwrap_or(-1);
                if tid >= 0 {
                    s.chunk.push(tid as u32);
                } else {
                    let end = (i + cl).min(n);
                    for &eb in &chunk[i..end] {
                        let t = self.singleton[eb as usize];
                        if t >= 0 {
                            s.chunk.push(t as u32);
                        }
                    }
                }
                i += cl;
            }
        }
        self.finish_chunk(s, Merge::Auto);
    }

    /// Cold-miss resolver for SentencePiece `<= 15B` space-surrogate chunks:
    /// runs `process_chunk_sp_direct` directly (avoiding a 12+ MB static `ChunkTable`
    /// that would thrash L2) and learns into `self.learned` / `self.spill`.
    #[inline(never)]
    fn emit_sp_chunk_fresh(&self, chunk: &[u8], lo: u64, hi: u64, s: &mut Scratch) {
        let h = chunk_table::hash(lo, hi);
        let st = s.out.len();
        self.process_chunk_sp_direct(chunk, s);
        let toks = &s.out[st..];
        if !self.learned.put_6(lo, hi, h, toks) {
            self.spill.put(chunk, toks);
            self.learned.put_spilled(lo, hi, h);
        }
    }

    /// Cold-miss resolver for SentencePiece `16..=31B` chunks via `SharedLongTable`.
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn emit_sp_long_at(
        &self,
        bytes: &[u8],
        pos: usize,
        end: usize,
        lp: u64,
        lh: u64,
        lview: chunk_table::LongView<'_>,
        s: &mut Scratch,
    ) {
        let len = end - pos;
        let lk = chunk_table::pack32_at(bytes, pos, len);
        match unsafe { lview.get_into_at(lk, lp, &mut s.out) } {
            chunk_table::LongHit::Hit => return,
            chunk_table::LongHit::Empty => {
                if !lview.is_current(&self.long_table)
                    && self.long_table.get_into(lk, lh, &mut s.out) == chunk_table::LongHit::Hit
                {
                    return;
                }
                let key = &bytes[pos..end];
                let st = s.out.len();
                self.process_chunk_sp_direct(key, s);
                let toks = &s.out[st..];
                if !self.long_table.put(lk, lh, toks) {
                    let sh = chunk_table::hash_bytes(key);
                    let _ = self.spill.put_h(key, sh, toks);
                    s.long_cache.put_h(self.uid, key, sh, toks);
                }
                return;
            }
            chunk_table::LongHit::Spilled => {}
        }
        self.emit_sp_chunk_slice(&bytes[pos..end], s);
    }

    /// Single-chunk SentencePiece fast resolver (`ChunkTable` + `SharedTable` +
    /// `SharedLongTable` + `SpillMap`) for chunks with 1-byte space surrogate.
    #[inline]
    fn emit_sp_chunk_slice(&self, chunk: &[u8], s: &mut Scratch) {
        let len = chunk.len();
        if len == 0 {
            return;
        }
        if len <= chunk_table::MAX_KEY {
            let (lo, hi) = chunk_table::pack(chunk);
            let h = chunk_table::hash(lo, hi);
            if let Some((toks, cnt)) = self.learned.get_6(lo, hi, h) {
                if cnt != chunk_table::SPILLED {
                    s.out.extend_from_slice(&toks[..cnt]);
                    return;
                }
                let sh = chunk_table::hash_bytes(chunk);
                if self.spill.get_into_h(chunk, sh, &mut s.out) {
                    return;
                }
            }
            let st = s.out.len();
            self.process_chunk_sp_direct(chunk, s);
            let toks = &s.out[st..];
            if !self.learned.put_6(lo, hi, h, toks) {
                self.spill.put(chunk, toks);
                self.learned.put_spilled(lo, hi, h);
            }
            return;
        }
        if len <= chunk_table::MAX_LONG_KEY {
            let lk = chunk_table::pack32_at(chunk, 0, len);
            let lh = chunk_table::hash32(lk);
            if self.long_table.get_into(lk, lh, &mut s.out) == chunk_table::LongHit::Hit {
                return;
            }
            let sh = chunk_table::hash_bytes(chunk);
            if let Some(ts) = s.long_cache.get_h(self.uid, chunk, sh) {
                s.out.extend_from_slice(ts);
                return;
            }
            if self.spill.get_into_h(chunk, sh, &mut s.out) {
                return;
            }
            let st = s.out.len();
            self.process_chunk_sp_direct(chunk, s);
            let toks = &s.out[st..];
            if !self.long_table.put(lk, lh, toks) {
                let _ = self.spill.put_h(chunk, sh, toks);
                s.long_cache.put_h(self.uid, chunk, sh, toks);
            }
            return;
        }
        let sh = chunk_table::hash_bytes(chunk);
        if let Some(ts) = s.long_cache.get_h(self.uid, chunk, sh) {
            s.out.extend_from_slice(ts);
            return;
        }
        if self.spill.get_into_h(chunk, sh, &mut s.out) {
            return;
        }
        let st = s.out.len();
        self.process_chunk_sp_direct(chunk, s);
        let (out, lc) = (&s.out, &mut s.long_cache);
        lc.put_h(self.uid, chunk, sh, &out[st..]);
        let _ = self.spill.put_h(chunk, sh, &out[st..]);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn flush_recs_sp(
        &self,
        bytes: &[u8],
        base_rp: *const PipeRec,
        len: usize,
        ends: &[u32],
        first_a: usize,
        s: &mut Scratch,
        view: chunk_table::View<'_, 3>,
        lview: chunk_table::LongView<'_>,
    ) {
        s.out.reserve(6 * len + 16);
        let mut base_out = s.out.as_mut_ptr();
        let mut wp = unsafe { base_out.add(s.out.len()) };
        let mut rp = base_rp;
        let end_rp = unsafe { base_rp.add(len) };
        while rp < end_rp {
            #[cfg(target_arch = "x86_64")]
            {
                let pf = unsafe { (*rp.add(X86_PF_DIST)).p };
                chunk_table::prefetch(pf as *const u8);
            }
            let r = unsafe { *rp };
            let empty_slot = if r.p != 0 {
                match unsafe { view.emit_inline_at(r.lo, r.hi, r.p, wp) } {
                    chunk_table::InlineHit::Hit(c) => {
                        wp = unsafe { wp.add(c) };
                        rp = unsafe { rp.add(1) };
                        continue;
                    }
                    chunk_table::InlineHit::Empty => true,
                    chunk_table::InlineHit::Spilled => false,
                }
            } else {
                false
            };
            let k = unsafe { rp.offset_from(base_rp) as usize };
            let w = unsafe { wp.offset_from(base_out) as usize };
            unsafe { s.out.set_len(w) };
            let a = if k == 0 { first_a } else { ends[k - 1] as usize };
            let e = ends[k] as usize;
            if empty_slot {
                self.emit_sp_chunk_fresh(&bytes[a..e], r.lo, r.hi, s);
            } else if r.p == 0 && r.lo != 0 {
                self.emit_sp_long_at(bytes, a, e, r.lo, r.hi, lview, s);
            } else {
                self.emit_sp_chunk_slice(&bytes[a..e], s);
            }
            s.out.reserve(6 * (len - k) + 16);
            base_out = s.out.as_mut_ptr();
            wp = unsafe { base_out.add(s.out.len()) };
            rp = unsafe { rp.add(1) };
        }
        unsafe { s.out.set_len(wp.offset_from(base_out) as usize) };
    }

    /// Find the next proven-exact SentencePiece chunk end starting at `pos`:
    /// skips leading `' '` run, then scans to the first `' '` whose preceding byte
    /// is NOT in `cross` (so rare cross-space merges like Gemma-3 `">▁</"` stay intact).
    #[inline(always)]
    fn next_sp_end(bytes: &[u8], pos: usize, cross: &[u64; 4], has_cross: bool) -> usize {
        let n = bytes.len();
        let mut q = pos;
        while q < n && unsafe { *bytes.get_unchecked(q) } == b' ' {
            q += 1;
        }
        if !has_cross {
            return if q < n {
                match memchr::memchr(b' ', &bytes[q..]) {
                    Some(rel) => q + rel,
                    None => n,
                }
            } else {
                n
            };
        }
        loop {
            if q >= n {
                return n;
            }
            match memchr::memchr(b' ', &bytes[q..]) {
                Some(rel) => {
                    let p = q + rel;
                    let prev = unsafe { *bytes.get_unchecked(p - 1) } as usize;
                    if ((cross[prev >> 6] >> (prev & 63)) & 1) == 0 {
                        return p;
                    }
                    q = p + 1;
                    while q < n && unsafe { *bytes.get_unchecked(q) } == b' ' {
                        q += 1;
                    }
                }
                None => return n,
            }
        }
    }

    /// Zero-allocation SIMD-chunked, `SharedTable`-pipelined SentencePiece encoder.
    /// Clean spans containing no literal `'▁'` run through the full SIMD pipeline;
    /// rare literal `'▁'` words are isolated to their exact surrounding safe space
    /// boundaries and routed to the exact normalizer/metaspace fallback so one
    /// literal `'▁'` in a 5 MB corpus never degrades the entire document.
    fn encode_sp_fast(&self, text: &str, prepend: bool, is_doc_start: bool, s: &mut Scratch) -> bool {
        if text.is_empty() {
            return true;
        }
        let bytes = text.as_bytes();
        let Some(first_u) = memchr::memmem::find(bytes, "\u{2581}".as_bytes()) else {
            return self.encode_sp_fast_clean(bytes, prepend, is_doc_start, s);
        };
        let n = bytes.len();
        let cross = &self.sp_cross_prev;
        let is_safe_split = |p: usize| -> bool {
            if p == 0 || p >= n || bytes[p] != b' ' {
                return false;
            }
            let prev = bytes[p - 1];
            prev < 0x80 && prev != b' ' && ((cross[(prev as usize) >> 6] >> (prev & 63)) & 1) == 0
        };
        let mut start = 0usize;
        let mut cur_u = first_u;
        loop {
            // Walk back from `cur_u` to the nearest proven-safe `' '` boundary > `start`.
            let mut safe_a = start;
            let mut search_end = cur_u;
            while search_end > start {
                match memchr::memrchr(b' ', &bytes[start..search_end]) {
                    Some(rel) => {
                        let p = start + rel;
                        if p > start && is_safe_split(p) {
                            safe_a = p;
                            break;
                        }
                        search_end = p;
                    }
                    None => break,
                }
            }
            if safe_a > start {
                self.encode_sp_fast_clean(
                    &bytes[start..safe_a],
                    prepend && start == 0,
                    is_doc_start && start == 0,
                    s,
                );
            }
            // Walk forward past `cur_u + 3` to the nearest proven-safe `' '` boundary,
            // merging any adjacent literal `'▁'`s within the same fallback window.
            let mut scan = cur_u + 3;
            let safe_b = loop {
                let cand = match memchr::memchr(b' ', &bytes[scan..]) {
                    Some(rel) => {
                        let p = scan + rel;
                        if is_safe_split(p) {
                            p
                        } else {
                            scan = p + 1;
                            continue;
                        }
                    }
                    None => n,
                };
                if let Some(rel_u) = memchr::memmem::find(&bytes[cur_u + 3..cand], "\u{2581}".as_bytes()) {
                    cur_u = cur_u + 3 + rel_u;
                    scan = cur_u + 3;
                    continue;
                }
                break cand;
            };
            let sub = &text[safe_a..safe_b];
            if safe_a == 0 && is_doc_start {
                let norm = self.normalize(sub);
                self.encode_into_norm(norm.as_ref(), Merge::Auto, prepend, s);
            } else {
                match self.sp_fast_meta {
                    SpFastMeta::NormPrependAlways | SpFastMeta::NormNoPrepend => {
                        let replaced = sub.replace(' ', "\u{2581}");
                        self.encode_into_norm(&replaced, Merge::Auto, false, s);
                    }
                    SpFastMeta::Metaspace { .. } => {
                        self.encode_into_norm(sub, Merge::Auto, false, s);
                    }
                    SpFastMeta::None => return false,
                }
            }
            start = safe_b;
            if start >= n {
                break;
            }
            match memchr::memmem::find(&bytes[start..], "\u{2581}".as_bytes()) {
                Some(rel) => cur_u = start + rel,
                None => {
                    self.encode_sp_fast_clean(&bytes[start..], false, false, s);
                    break;
                }
            }
        }
        true
    }

    /// SIMD-chunked SentencePiece fast path for a `'▁'`-free byte slice.
    fn encode_sp_fast_clean(&self, bytes: &[u8], prepend: bool, is_doc_start: bool, s: &mut Scratch) -> bool {
        let n = bytes.len();
        if n == 0 {
            return true;
        }
        let need_prepend = match self.sp_fast_meta {
            SpFastMeta::NormPrependAlways => is_doc_start,
            SpFastMeta::NormNoPrepend => false,
            SpFastMeta::Metaspace { prepend_scheme } => {
                let allow = match prepend_scheme {
                    1 => is_doc_start,
                    2 => is_doc_start && prepend,
                    _ => false,
                };
                allow && bytes[0] != b' '
            }
            SpFastMeta::None => return false,
        };
        let cross = &self.sp_cross_prev;
        let has_cross = self.sp_has_cross;

        // Chunk 0: `' '* non-' '*` starting at 0 (preserving any cross-space prev bytes),
        // with optional virtual leading `' '`.
        let e0 = Self::next_sp_end(bytes, 0, cross, has_cross);
        if need_prepend {
            if e0 < 63 {
                let mut stack = [0u8; 64];
                stack[0] = b' ';
                stack[1..1 + e0].copy_from_slice(&bytes[..e0]);
                self.emit_sp_chunk_slice(&stack[..1 + e0], s);
            } else {
                let mut v = Vec::with_capacity(1 + e0);
                v.push(b' ');
                v.extend_from_slice(&bytes[..e0]);
                self.emit_sp_chunk_slice(&v, s);
            }
        } else {
            self.emit_sp_chunk_slice(&bytes[..e0], s);
        }

        let mut pos = e0;
        if pos >= n {
            return true;
        }

        // Remaining chunks all start with `b' '` inside `bytes` (100% zero-copy!).
        if n - pos >= 1024 && n <= u32::MAX as usize {
            const CAP: usize = 256;
            let mut ends = [0u32; CAP];
            let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); CAP + X86_PF_DIST];
            while pos < n {
                let first_a = pos;
                let mut k = 0usize;
                while pos < n && k < CAP {
                    let next = Self::next_sp_end(bytes, pos + 1, cross, has_cross);
                    ends[k] = next as u32;
                    k += 1;
                    pos = next;
                }
                let view = self.learned.view();
                let lview = self.long_table.view();
                Self::fill_recs(bytes, &ends, 0, k, first_a, buf.as_mut_ptr(), view, lview);
                Self::prime_pipe_prefetch(&mut buf, k);
                self.flush_recs_sp(bytes, buf.as_ptr() as *const PipeRec, k, &ends[..k], first_a, s, view, lview);
            }
        } else {
            while pos < n {
                let start = pos;
                let next = Self::next_sp_end(bytes, pos + 1, cross, has_cross);
                self.emit_sp_chunk_slice(&bytes[start..next], s);
                pos = next;
            }
        }
        true
    }

    /// Apply the model's composable normalizer pipeline (`Cow::Borrowed` when
    /// identity or already normalized).
    #[inline]
    fn normalize<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        self.normalizer.apply(text)
    }

    /// Split metaspace text on `rep` boundaries (`rep* non-rep*`) and emit each.
    /// Used only when load-time verification proved no merge crosses `(non-rep | rep)`.
    fn encode_meta_char(&self, meta: &str, rep: char, s: &mut Scratch, mode: Merge) {
        let b = meta.as_bytes();
        let n = b.len();
        let mut rbuf = [0u8; 4];
        let rb = rep.encode_utf8(&mut rbuf).as_bytes();
        let rl = rb.len();
        let is_ms = |p: usize| p + rl <= n && &b[p..p + rl] == rb;
        let mut pos = 0usize;
        while pos < n {
            let start = pos;
            while is_ms(pos) {
                pos += rl; // leading rep run
            }
            if pos > start && pos >= n {
                self.emit(&b[start..pos], s, mode);
                break;
            }
            while pos < n {
                if is_ms(pos) {
                    if pos == start {
                        pos += rl;
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
        if mode == Merge::Auto {
            self.encode_range(bytes, 0, bytes.len(), s);
            return;
        }
        let n = bytes.len();
        let mut pos = 0usize;
        while pos < n {
            let end = pretok::next_chunk_end(self.kind, bytes, pos);
            self.emit(&bytes[pos..end], s, mode);
            pos = end;
        }
    }

    /// Hot loop over `bytes[start..stop]` (Merge::Auto). `stop` must be a chunk
    /// boundary of the full-buffer scan; scanning always sees the WHOLE buffer, so
    /// look-ahead rules (`\s+(?!\S)`) behave exactly as in an unsplit encode.
    #[inline]
    fn encode_range(&self, bytes: &[u8], start: usize, stop: usize, s: &mut Scratch) {
        const PIPE_MIN_BYTES: usize = 1024;
        let n = bytes.len();
        // 64-byte block scanner (identical boundaries, see blockscan.rs).
        #[cfg(test)]
        let pipe = !perf::NO_PIPE.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let pipe = true;
        // Short ranges resolve immediately: the pipeline's batch setup costs
        // more than its prefetch lead saves below ~1 KB.
        if pipe && stop - start >= PIPE_MIN_BYTES {
            const CAP: usize = 256;
            #[cfg(test)]
            let batched = !perf::NO_SCAN.load(std::sync::atomic::Ordering::Relaxed);
            #[cfg(not(test))]
            let batched = true;
            if batched && n <= u32::MAX as usize {
                let mut ends = [0u32; CAP + blockscan::SCAN_SLACK];
                let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); CAP + blockscan::SCAN_SLACK + X86_PF_DIST];
                let mut scan = blockscan::Scan::new(start);
                let mut a = start;
                loop {
                    let k = scan.fill(self.kind, bytes, stop, &mut ends, 0, CAP);
                    if k == 0 {
                        break;
                    }
                    let view = self.learned.view();
                    let lview = self.long_table.view();
                    let bufp = buf.as_mut_ptr();
                    let first_a = a;
                    a = Self::fill_recs(bytes, &ends, 0, k, a, bufp, view, lview);
                    Self::prime_pipe_prefetch(&mut buf, k);
                    self.flush_recs(bytes, n, buf.as_ptr() as *const PipeRec, k, &ends[..k], first_a, s, view, lview);
                }
                return;
            }
            let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); CAP + blockscan::SCAN_SLACK];
            let mut spans = [(0usize, 0usize); CAP + blockscan::SCAN_SLACK];
            let mut cnt = 0usize;
            let mut bview = self.learned.view();
            let bp = bytes.as_ptr();
            blockscan::chunks_batched(self.kind, bytes, start, stop, |first, ends| {
                if cnt == 0 {
                    bview = self.learned.view();
                }
                let view = bview;
                let mut c = cnt;
                let mut a = first;
                let bufp = buf.as_mut_ptr();
                let (n, bp) = (n, bp);
                for &e in ends {
                    let len = e - a;
                    let v = if len.wrapping_sub(1) < chunk_table::MAX_KEY && a + 16 <= n {
                        let (lo, hi) = unsafe { chunk_table::pack_wide(bp.add(a), len) };
                        let p = view.slot_addr(chunk_table::hash(lo, hi));
                        chunk_table::prefetch(p as *const u8);
                        PipeRec { lo, hi, p }
                    } else {
                        PipeRec { lo: 0, hi: 0, p: 0 }
                    };
                    unsafe { (*bufp.add(c)).write(v) };
                    spans[c] = (a, e);
                    c += 1;
                    a = e;
                }
                cnt = c;
                if cnt >= CAP {
                    let sl = unsafe { &*(&buf[..cnt] as *const _ as *const [PipeRec]) };
                    self.flush_recs_spans(bytes, n, sl, &spans[..cnt], s, bview);
                    cnt = 0;
                }
            });
            let sl = unsafe { &*(&buf[..cnt] as *const _ as *const [PipeRec]) };
            self.flush_recs_spans(bytes, n, sl, &spans[..cnt], s, bview);
            return;
        }
        blockscan::chunks_batched(self.kind, bytes, start, stop, |first, ends| {
            let mut a = first;
            for &e in ends {
                self.emit_chunk(bytes, a, e, n, s);
                a = e;
            }
        });
    }

    /// Out-of-line prefetch builder for `> 15B` chunks: prefetches the
    /// `SharedLongTable` 64-byte slot 256 chunks ahead (storing `(lp, h)` in
    /// the unused `(lo, hi)` fields of `PipeRec { p: 0 }`) instead of polluting
    /// L1 with a bogus `SharedTable` prefetch.
    #[cold]
    #[inline(never)]
    fn fill_rec_long(
        bytes: &[u8],
        a: usize,
        len: usize,
        lview: chunk_table::LongView<'_>,
    ) -> PipeRec {
        if (16..=chunk_table::MAX_LONG_KEY).contains(&len) && lview.is_valid() {
            let lk = chunk_table::pack32_at(bytes, a, len);
            let lh = chunk_table::hash32(lk);
            let lp = lview.slot_addr(lh);
            chunk_table::prefetch(lp as *const u8);
            PipeRec { lo: lp, hi: lh, p: 0 }
        } else {
            PipeRec { lo: 0, hi: 0, p: 0 }
        }
    }

    /// On x86_64 (32 KiB L1d / 24 LFBs), issuing 256 back-to-back `prefetcht0`s in
    /// `fill_recs` saturates LFBs and evicts earlier prefetched lines before Phase 2
    /// consumes them. Instead, write only the `.p` field of `buf[total..total + X86_PF_DIST]`
    /// with an L1-hot dummy address (halving tail padding stores), prime the first
    /// `X86_PF_DIST` slots, and prefetch `rp + X86_PF_DIST` in steady state during
    /// Phase 2. On ARM64 (Apple M4 Max 128 KiB L1d), `fill_recs` batch prefetching
    /// fits in L1d without eviction, so `X86_PF_DIST = 0`.
    #[inline(always)]
    fn prime_pipe_prefetch(buf: &mut [std::mem::MaybeUninit<PipeRec>], total: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            debug_assert!(total + X86_PF_DIST <= buf.len());
            let dummy_p = buf.as_ptr() as u64;
            let tail = unsafe { buf.as_mut_ptr().add(total) as *mut PipeRec };
            for j in 0..X86_PF_DIST {
                unsafe { std::ptr::addr_of_mut!((*tail.add(j)).p).write(dummy_p) };
            }
            let p0 = buf.as_ptr() as *const PipeRec;
            for j in 0..total.min(X86_PF_DIST) {
                let p = unsafe { (*p0.add(j)).p };
                if p != 0 {
                    chunk_table::prefetch(p as *const u8);
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (buf, total);
        }
    }

    /// Cold tail loop of `fill_recs` for the last 16 bytes of a document.
    #[inline(never)]
    fn fill_recs_tail(
        bytes: &[u8],
        ends: &[u32],
        mut a: usize,
        mut out: *mut std::mem::MaybeUninit<PipeRec>,
        view: chunk_table::View<'_, 3>,
        lview: chunk_table::LongView<'_>,
    ) -> usize {
        for &e_u32 in ends {
            let e = e_u32 as usize;
            let len = e - a;
            let v = if len.wrapping_sub(1) < chunk_table::MAX_KEY {
                let (klo, khi) = chunk_table::pack_at(bytes, a, len);
                let p = view.slot_addr(chunk_table::hash(klo, khi));
                #[cfg(not(target_arch = "x86_64"))]
                chunk_table::prefetch(p as *const u8);
                PipeRec { lo: klo, hi: khi, p }
            } else {
                Self::fill_rec_long(bytes, a, len, lview)
            };
            unsafe { (*out).write(v) };
            out = unsafe { out.add(1) };
            a = e;
        }
        a
    }

    #[cfg(target_arch = "x86_64")]
    #[cold]
    #[inline(never)]
    fn fill_rec_long_cold(
        cold: &(&[u8], chunk_table::LongView<'_>),
        a: usize,
        len: usize,
    ) -> PipeRec {
        Self::fill_rec_long(cold.0, a, len, cold.1)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "bmi2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn fill_recs_bmi2(
        bp: *const u8,
        mut ep: *const u32,
        end_ep: *const u32,
        mut a: usize,
        mut out: *mut std::mem::MaybeUninit<PipeRec>,
        vbase: u64,
        vshift: u32,
        cold: &(&[u8], chunk_table::LongView<'_>),
    ) -> usize {
        while ep < end_ep {
            let e = *ep as usize;
            let len = e - a;
            if len.wrapping_sub(1) < chunk_table::MAX_KEY {
                let (klo, khi) = chunk_table::pack_wide_bmi2(bp.add(a), len);
                let h = chunk_table::hash(klo, khi);
                let p = vbase.wrapping_add((h >> vshift) << 5);
                (*out).write(PipeRec { lo: klo, hi: khi, p });
            } else {
                let v = Self::fill_rec_long_cold(cold, a, len);
                (*out).write(v);
            }
            out = out.add(1);
            ep = ep.add(1);
            a = e;
        }
        a
    }

    /// Phase 1 records for chunk ends `ends[lo..hi]` (the first chunk starts at
    /// `a`): key, home-slot address (prefetched on ARM64) per chunk; returns the
    /// last end. Straight-line wide loads for `<= 15B` chunks; `16..=31B` chunks
    /// prefetch their `SharedLongTable` 64B slot out-of-line.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn fill_recs(
        bytes: &[u8],
        ends: &[u32],
        lo: usize,
        hi: usize,
        a: usize,
        bufp: *mut std::mem::MaybeUninit<PipeRec>,
        view: chunk_table::View<'_, 3>,
        lview: chunk_table::LongView<'_>,
    ) -> usize {
        let sl = &ends[lo..hi];
        let out = unsafe { bufp.add(lo) };
        let n = bytes.len();
        let mut j = sl.len();
        while j > 0 && sl[j - 1] as usize + 16 > n {
            j -= 1;
        }
        let a = if j > 0 {
            let head = &sl[..j];
            #[cfg(target_arch = "x86_64")]
            {
                if has_avx2_bmi1() {
                    let cold = (bytes, lview);
                    let ep = head.as_ptr();
                    unsafe {
                        Self::fill_recs_bmi2(
                            bytes.as_ptr(),
                            ep,
                            ep.add(head.len()),
                            a,
                            out,
                            view.base_addr(),
                            view.shift(),
                            &cold,
                        )
                    }
                } else {
                    Self::fill_recs_generic(bytes, head, a, out, view, &lview)
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                Self::fill_recs_generic(bytes, head, a, out, view, &lview)
            }
        } else {
            a
        };
        if j < sl.len() {
            Self::fill_recs_tail(bytes, &sl[j..], a, unsafe { out.add(j) }, view, lview)
        } else {
            a
        }
    }

    #[inline(never)]
    fn fill_recs_generic(
        bytes: &[u8],
        ends: &[u32],
        mut a: usize,
        mut out: *mut std::mem::MaybeUninit<PipeRec>,
        view: chunk_table::View<'_, 3>,
        lview: &chunk_table::LongView<'_>,
    ) -> usize {
        let bp = bytes.as_ptr();
        for &e_u32 in ends {
            let e = e_u32 as usize;
            let len = e - a;
            if len.wrapping_sub(1) < chunk_table::MAX_KEY {
                let (klo, khi) = unsafe { chunk_table::pack_wide(bp.add(a), len) };
                let p = view.slot_addr(chunk_table::hash(klo, khi));
                #[cfg(not(target_arch = "x86_64"))]
                chunk_table::prefetch(p as *const u8);
                unsafe { (*out).write(PipeRec { lo: klo, hi: khi, p }) };
            } else {
                let v = Self::fill_rec_long(bytes, a, len, *lview);
                unsafe { (*out).write(v) };
            }
            out = unsafe { out.add(1) };
            a = e;
        }
        a
    }

    /// Whether batch items take the plain byte-level path (`encode_bytes` ->
    /// `encode_range`), so `encode_items_piped` applies.
    #[inline]
    fn items_pipe_ok(&self) -> bool {
        #[cfg(test)]
        {
            if perf::NO_PIPE.load(std::sync::atomic::Ordering::Relaxed) || perf::NO_SCAN.load(std::sync::atomic::Ordering::Relaxed) {
                return false;
            }
        }
        self.fast_pretok == FastPretok::BlockScan
    }

    /// Phase 2 of the pipeline for records of one buffer `bytes` (len `n`).
    /// `ends` has the same length as `buf` and `first_a` is the start offset of
    /// `buf[0]`; neither is touched on the 99%+ inline hit path.
    /// Uses pointer-only loop induction (`rp`, `end_rp`, `wp`) so zero `k` or
    /// `dst`-base stack reloads remain in the hot hit path.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn flush_recs(
        &self,
        bytes: &[u8],
        n: usize,
        base_rp: *const PipeRec,
        len: usize,
        ends: &[u32],
        first_a: usize,
        s: &mut Scratch,
        view: chunk_table::View<'_, 3>,
        lview: chunk_table::LongView<'_>,
    ) {
        s.out.reserve(6 * len + 16);
        let mut base_out = s.out.as_mut_ptr();
        let mut wp = unsafe { base_out.add(s.out.len()) };
        let mut rp = base_rp;
        let end_rp = unsafe { base_rp.add(len) };
        while rp < end_rp {
            #[cfg(target_arch = "x86_64")]
            {
                let pf = unsafe { (*rp.add(X86_PF_DIST)).p };
                chunk_table::prefetch(pf as *const u8);
            }
            let r = unsafe { *rp };
            let empty_slot = if r.p != 0 {
                // SAFETY: capacity >= current_len + 6 * remaining entries (reserved).
                match unsafe { view.emit_inline_at(r.lo, r.hi, r.p, wp) } {
                    chunk_table::InlineHit::Hit(c) => {
                        wp = unsafe { wp.add(c) };
                        rp = unsafe { rp.add(1) };
                        continue;
                    }
                    chunk_table::InlineHit::Empty => true,
                    chunk_table::InlineHit::Spilled => false,
                }
            } else {
                false
            };
            let k = unsafe { rp.offset_from(base_rp) as usize };
            let w = unsafe { wp.offset_from(base_out) as usize };
            // SAFETY: [0, w) initialized.
            unsafe { s.out.set_len(w) };
            let a = if k == 0 { first_a } else { ends[k - 1] as usize };
            let e = ends[k] as usize;
            if empty_slot {
                self.emit_chunk_fresh(&bytes[a..e], r.lo, r.hi, s);
            } else if r.p == 0 && r.lo != 0 {
                self.emit_long_at(bytes, a, e, n, r.lo, r.hi, lview, s);
            } else {
                self.emit_chunk_cold(bytes, a, e, n, s);
            }
            s.out.reserve(6 * (len - k) + 16);
            base_out = s.out.as_mut_ptr();
            wp = unsafe { base_out.add(s.out.len()) };
            rp = unsafe { rp.add(1) };
        }
        // SAFETY: [0, wp - base_out) initialized.
        unsafe { s.out.set_len(wp.offset_from(base_out) as usize) };
    }

    /// Zero-write Phase 2 token counter for `count_tokens*` paths: probes
    /// `view.count_inline_at` in registers with zero writes to `s.out` on hits.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn flush_recs_count(
        &self,
        bytes: &[u8],
        n: usize,
        base_rp: *const PipeRec,
        len: usize,
        ends: &[u32],
        first_a: usize,
        s: &mut Scratch,
        view: chunk_table::View<'_, 3>,
        lview: chunk_table::LongView<'_>,
    ) -> usize {
        let mut total = 0usize;
        let mut rp = base_rp;
        let end_rp = unsafe { base_rp.add(len) };
        while rp < end_rp {
            #[cfg(target_arch = "x86_64")]
            {
                let pf = unsafe { (*rp.add(X86_PF_DIST)).p };
                chunk_table::prefetch(pf as *const u8);
            }
            let r = unsafe { *rp };
            let empty_slot = if r.p != 0 {
                match unsafe { view.count_inline_at(r.lo, r.hi, r.p) } {
                    chunk_table::InlineHit::Hit(c) => {
                        total += c;
                        rp = unsafe { rp.add(1) };
                        continue;
                    }
                    chunk_table::InlineHit::Empty => true,
                    chunk_table::InlineHit::Spilled => false,
                }
            } else {
                false
            };
            let k = unsafe { rp.offset_from(base_rp) as usize };
            let a = if k == 0 { first_a } else { ends[k - 1] as usize };
            let e = ends[k] as usize;
            s.out.clear();
            if empty_slot {
                self.emit_chunk_fresh(&bytes[a..e], r.lo, r.hi, s);
            } else if r.p == 0 && r.lo != 0 {
                self.emit_long_at(bytes, a, e, n, r.lo, r.hi, lview, s);
            } else {
                self.emit_chunk_cold(bytes, a, e, n, s);
            }
            total += s.out.len();
            rp = unsafe { rp.add(1) };
        }
        total
    }

    /// Zero-write Phase 2 segment counter for `encode_sep_macro` and
    /// `encode_fast_slice_piped` when `want_output == false`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn flush_seg_count<'a>(
        &self,
        buf_ptr: *const PipeRec,
        ends: &[u32],
        slo: usize,
        shi: usize,
        first_a: usize,
        get_doc: impl Fn() -> &'a [u8],
        s: &mut Scratch,
        view: chunk_table::View<'_, 3>,
        lview: chunk_table::LongView<'_>,
    ) -> usize {
        let mut total = 0usize;
        let base_rp = unsafe { buf_ptr.add(slo) };
        let mut rp = base_rp;
        let end_rp = unsafe { buf_ptr.add(shi) };
        while rp < end_rp {
            #[cfg(target_arch = "x86_64")]
            {
                let pf = unsafe { (*rp.add(X86_PF_DIST)).p };
                chunk_table::prefetch(pf as *const u8);
            }
            let r = unsafe { *rp };
            let empty_slot = if r.p != 0 {
                match unsafe { view.count_inline_at(r.lo, r.hi, r.p) } {
                    chunk_table::InlineHit::Hit(c) => {
                        total += c;
                        rp = unsafe { rp.add(1) };
                        continue;
                    }
                    chunk_table::InlineHit::Empty => true,
                    chunk_table::InlineHit::Spilled => false,
                }
            } else {
                false
            };
            let k = unsafe { rp.offset_from(base_rp) as usize };
            let d = get_doc();
            let a = if k == 0 { first_a } else { ends[slo + k - 1] as usize };
            let e = ends[slo + k] as usize;
            s.out.clear();
            if empty_slot {
                self.emit_chunk_fresh(&d[a..e], r.lo, r.hi, s);
            } else if r.p == 0 && r.lo != 0 {
                self.emit_long_at(d, a, e, d.len(), r.lo, r.hi, lview, s);
            } else {
                self.emit_chunk_cold(d, a, e, d.len(), s);
            }
            total += s.out.len();
            rp = unsafe { rp.add(1) };
        }
        total
    }

    /// Phase 2 span-based flush (used by >4 GiB buffers and pipelined `PretokPipe`).
    fn flush_recs_spans(
        &self,
        bytes: &[u8],
        n: usize,
        buf: &[PipeRec],
        spans: &[(usize, usize)],
        s: &mut Scratch,
        view: chunk_table::View<'_, 3>,
    ) {
        s.out.reserve(6 * buf.len() + 16);
        let mut w = s.out.len();
        let mut dst = s.out.as_mut_ptr();
        for (k, r) in buf.iter().enumerate() {
            let empty_slot = if r.p != 0 {
                match unsafe { view.emit_inline_at(r.lo, r.hi, r.p, dst.add(w)) } {
                    chunk_table::InlineHit::Hit(c) => {
                        w += c;
                        continue;
                    }
                    chunk_table::InlineHit::Empty => true,
                    chunk_table::InlineHit::Spilled => false,
                }
            } else {
                false
            };
            unsafe { s.out.set_len(w) };
            let (a, e) = spans[k];
            if empty_slot {
                self.emit_chunk_fresh(&bytes[a..e], r.lo, r.hi, s);
            } else {
                self.emit_chunk_cold(bytes, a, e, n, s);
            }
            s.out.reserve(6 * (buf.len() - k) + 16);
            w = s.out.len();
            dst = s.out.as_mut_ptr();
        }
        unsafe { s.out.set_len(w) };
    }

    /// Flush one segment `[slo..shi)` inside a pre-reserved 256-record batch.
    /// Keeps `base_out` and `wp` live across documents so short-line batches pay
    /// ONE `s.out.reserve` + ONE `s.out.set_len` per 256 chunks instead of once
    /// per document, and never even loads the document slice on warm hits.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn flush_seg_into<'a>(
        &self,
        buf_ptr: *const PipeRec,
        ends: &[u32],
        slo: usize,
        shi: usize,
        batch_cnt: usize,
        first_a: usize,
        get_doc: impl Fn() -> &'a [u8],
        s: &mut Scratch,
        base_out: &mut *mut u32,
        wp: &mut *mut u32,
        view: chunk_table::View<'_, 3>,
        lview: chunk_table::LongView<'_>,
    ) -> usize {
        let mut emitted = 0usize;
        let mut seg_start_wp = *wp;
        let base_rp = unsafe { buf_ptr.add(slo) };
        let mut rp = base_rp;
        let end_rp = unsafe { buf_ptr.add(shi) };
        while rp < end_rp {
            #[cfg(target_arch = "x86_64")]
            {
                let pf = unsafe { (*rp.add(X86_PF_DIST)).p };
                chunk_table::prefetch(pf as *const u8);
            }
            let r = unsafe { *rp };
            let empty_slot = if r.p != 0 {
                // SAFETY: caller reserved `6 * remaining_batch + 16` slots.
                match unsafe { view.emit_inline_at(r.lo, r.hi, r.p, *wp) } {
                    chunk_table::InlineHit::Hit(c) => {
                        *wp = unsafe { (*wp).add(c) };
                        rp = unsafe { rp.add(1) };
                        continue;
                    }
                    chunk_table::InlineHit::Empty => true,
                    chunk_table::InlineHit::Spilled => false,
                }
            } else {
                false
            };
            let k = unsafe { rp.offset_from(base_rp) as usize };
            let before_miss = unsafe { (*wp).offset_from(*base_out) as usize };
            emitted += unsafe { (*wp).offset_from(seg_start_wp) as usize };
            // SAFETY: [0, before_miss) initialized.
            unsafe { s.out.set_len(before_miss) };
            let d = get_doc();
            let a = if k == 0 { first_a } else { ends[slo + k - 1] as usize };
            let e = ends[slo + k] as usize;
            if empty_slot {
                self.emit_chunk_fresh(&d[a..e], r.lo, r.hi, s);
            } else if r.p == 0 && r.lo != 0 {
                self.emit_long_at(d, a, e, d.len(), r.lo, r.hi, lview, s);
            } else {
                self.emit_chunk_cold(d, a, e, d.len(), s);
            }
            let after_miss = s.out.len();
            emitted += after_miss - before_miss;
            s.out.reserve(6 * (batch_cnt - (slo + k)) + 16);
            *base_out = s.out.as_mut_ptr();
            *wp = unsafe { (*base_out).add(after_miss) };
            seg_start_wp = *wp;
            rp = unsafe { rp.add(1) };
        }
        emitted + unsafe { (*wp).offset_from(seg_start_wp) as usize }
    }

    /// Zero-`plan()` pipelined worker over a contiguous slice of `FastItem`s
    /// (every doc `< SPLIT_MIN`, `normalizer.is_ascii_identity()`, `items_pipe_ok()`).
    /// Pure-ASCII `PyASCIIObject` items (`FAST_ASCII`) bypass normalization
    /// completely even on NFC/NFD/NFKC/NFKD models (Qwen2.5/3, MiniMax, EXAONE);
    /// non-ASCII items normalize per-worker in `s.utf8` with zero upfront copy.
    #[allow(clippy::too_many_arguments)]
    fn encode_fast_slice_piped<const SPECIAL: bool, const WANT_OUTPUT: bool>(
        &self,
        items: &[FastItem],
        base_doc: usize,
        s: &mut Scratch,
        doc_counts: &mut [usize],
        bad_bytes: &mut dyn FnMut(usize),
        bad_wide: &mut dyn FnMut(usize),
    ) {
        const CAP: usize = 256;
        let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); CAP + blockscan::SCAN_SLACK + X86_PF_DIST];
        let mut ends = [0u32; CAP + blockscan::SCAN_SLACK];
        // (slo, shi, local_di, first_a, uoff, ulen) — ulen == 0 => raw/utf8 FastItem
        let mut segs: Vec<(usize, usize, usize, usize, usize, usize)> = Vec::with_capacity(64);
        let mut cnt = 0usize;
        let mut view = self.learned.view();
        let mut lview = self.long_table.view();
        let mut ubuf = std::mem::take(&mut s.utf8);
        let need_norm = !self.normalizer.is_identity();
        let has_special = SPECIAL && self.special_ac.is_some();
        let flush = |buf: &mut [std::mem::MaybeUninit<PipeRec>],
                     total: usize,
                     ends: &[u32],
                     segs: &[(usize, usize, usize, usize, usize, usize)],
                     ubuf: &[u8],
                     s: &mut Scratch,
                     doc_counts: &mut [usize],
                     view: chunk_table::View<'_, 3>,
                     lview: chunk_table::LongView<'_>| {
            if segs.is_empty() {
                return;
            }
            Self::prime_pipe_prefetch(buf, total);
            let buf = buf.as_ptr() as *const PipeRec;
            if !WANT_OUTPUT {
                for &(slo, shi, ldi, first_a, uoff, ulen) in segs {
                    let n = self.flush_seg_count(
                        buf,
                        ends,
                        slo,
                        shi,
                        first_a,
                        || {
                            if ulen != 0 {
                                &ubuf[uoff..uoff + ulen]
                            } else {
                                let it = items[ldi];
                                unsafe { std::slice::from_raw_parts(it.ptr, it.len()) }
                            }
                        },
                        s,
                        view,
                        lview,
                    );
                    doc_counts[ldi] += n;
                }
                return;
            }
            s.out.reserve(6 * total + 16);
            let mut base_out = s.out.as_mut_ptr();
            let mut wp = unsafe { base_out.add(s.out.len()) };
            for &(slo, shi, ldi, first_a, uoff, ulen) in segs {
                let n = self.flush_seg_into(
                    buf,
                    ends,
                    slo,
                    shi,
                    total,
                    first_a,
                    || {
                        if ulen != 0 {
                            &ubuf[uoff..uoff + ulen]
                        } else {
                            let it = items[ldi];
                            unsafe { std::slice::from_raw_parts(it.ptr, it.len()) }
                        }
                    },
                    s,
                    &mut base_out,
                    &mut wp,
                    view,
                    lview,
                );
                doc_counts[ldi] += n;
            }
            unsafe { s.out.set_len(wp.offset_from(base_out) as usize) };
        };

        for (ldi, &it) in items.iter().enumerate() {
            let tag = it.tag();
            let raw_len = it.len();
            if raw_len == 0 {
                continue;
            }
            if tag >= FAST_WIDE1 {
                if cnt == 0 {
                    ubuf.clear();
                } else if raw_len > 16384 {
                    flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, doc_counts, view, lview);
                    cnt = 0;
                    segs.clear();
                    ubuf.clear();
                }
                let uoff = ubuf.len();
                if it.append_wide_utf8(&mut ubuf).is_err() {
                    ubuf.truncate(uoff);
                    bad_wide(base_doc + ldi);
                    continue;
                }
                if has_special && self.special_prefilter.find_first(&ubuf[uoff..]).is_some() {
                    if cnt > 0 {
                        flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, doc_counts, view, lview);
                        cnt = 0;
                        segs.clear();
                    }
                    let txt = unsafe { std::str::from_utf8_unchecked(&ubuf[uoff..]) };
                    let before = s.out.len();
                    self.encode_special_into(txt, s);
                    doc_counts[ldi] += s.out.len() - before;
                    if !WANT_OUTPUT {
                        s.out.clear();
                    }
                    ubuf.truncate(uoff);
                    continue;
                }
                if need_norm {
                    // SAFETY: `append_wide_utf8` succeeded so `ubuf[uoff..]` is valid UTF-8.
                    let norm_owned = {
                        let txt = unsafe { std::str::from_utf8_unchecked(&ubuf[uoff..]) };
                        match self.normalizer.apply(txt) {
                            std::borrow::Cow::Borrowed(_) => None,
                            std::borrow::Cow::Owned(ns) => Some(ns),
                        }
                    };
                    if let Some(ns) = norm_owned {
                        ubuf.truncate(uoff);
                        ubuf.extend_from_slice(ns.as_bytes());
                    }
                }
                let ulen = ubuf.len() - uoff;
                if ulen == 0 {
                    continue;
                }
                let mut scan = blockscan::Scan::new(0);
                let mut a = 0usize;
                loop {
                    if cnt == 0 {
                        view = self.learned.view();
                        lview = self.long_table.view();
                    }
                    let d = &ubuf[uoff..uoff + ulen];
                    let k1 = scan.fill(self.kind, d, ulen, &mut ends, cnt, CAP);
                    if k1 == cnt {
                        break;
                    }
                    let first_a = a;
                    a = Self::fill_recs(d, &ends, cnt, k1, a, buf.as_mut_ptr(), view, lview);
                    segs.push((cnt, k1, ldi, first_a, uoff, ulen));
                    cnt = k1;
                    if cnt >= CAP {
                        flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, doc_counts, view, lview);
                        cnt = 0;
                        segs.clear();
                    }
                    if a >= ulen {
                        break;
                    }
                }
                continue;
            }
            let d = unsafe { std::slice::from_raw_parts(it.ptr, raw_len) };
            if has_special && tag != FAST_RAW && self.special_prefilter.find_first(d).is_some() {
                if cnt > 0 {
                    flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, doc_counts, view, lview);
                    cnt = 0;
                    segs.clear();
                }
                let txt = unsafe { std::str::from_utf8_unchecked(d) };
                let before = s.out.len();
                self.encode_special_into(txt, s);
                doc_counts[ldi] += s.out.len() - before;
                if !WANT_OUTPUT {
                    s.out.clear();
                }
                continue;
            }
            let mut check_utf8 = tag == FAST_RAW;
            if need_norm && tag != FAST_ASCII {
                let txt = if tag == FAST_RAW {
                    check_utf8 = false;
                    match simdutf8::basic::from_utf8(d) {
                        Ok(t) => t,
                        Err(_) => {
                            bad_bytes(base_doc + ldi);
                            continue;
                        }
                    }
                } else {
                    // SAFETY: `FAST_UTF8` comes from CPython's cached UTF-8 buffer.
                    unsafe { std::str::from_utf8_unchecked(d) }
                };
                if let std::borrow::Cow::Owned(ns) = self.normalizer.apply(txt) {
                    if cnt == 0 {
                        ubuf.clear();
                    } else if ns.len() > 16384 {
                        flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, doc_counts, view, lview);
                        cnt = 0;
                        segs.clear();
                        ubuf.clear();
                    }
                    let uoff = ubuf.len();
                    ubuf.extend_from_slice(ns.as_bytes());
                    let ulen = ubuf.len() - uoff;
                    if ulen == 0 {
                        continue;
                    }
                    let mut scan = blockscan::Scan::new(0);
                    let mut a = 0usize;
                    loop {
                        if cnt == 0 {
                            view = self.learned.view();
                            lview = self.long_table.view();
                        }
                        let nd = &ubuf[uoff..uoff + ulen];
                        let k1 = scan.fill(self.kind, nd, ulen, &mut ends, cnt, CAP);
                        if k1 == cnt {
                            break;
                        }
                        let first_a = a;
                        a = Self::fill_recs(nd, &ends, cnt, k1, a, buf.as_mut_ptr(), view, lview);
                        segs.push((cnt, k1, ldi, first_a, uoff, ulen));
                        cnt = k1;
                        if cnt >= CAP {
                            flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, doc_counts, view, lview);
                            cnt = 0;
                            segs.clear();
                        }
                        if a >= ulen {
                            break;
                        }
                    }
                    continue;
                }
            }
            let mut scan = blockscan::Scan::new_checked(0, check_utf8);
            let mut a = 0usize;
            loop {
                if cnt == 0 {
                    view = self.learned.view();
                    lview = self.long_table.view();
                }
                let k1 = scan.fill(self.kind, d, raw_len, &mut ends, cnt, CAP);
                if k1 == cnt {
                    break;
                }
                let first_a = a;
                a = Self::fill_recs(d, &ends, cnt, k1, a, buf.as_mut_ptr(), view, lview);
                segs.push((cnt, k1, ldi, first_a, 0, 0));
                cnt = k1;
                if cnt >= CAP {
                    flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, doc_counts, view, lview);
                    cnt = 0;
                    segs.clear();
                }
                if a >= raw_len {
                    break;
                }
            }
            if !scan.finish_utf8(d, raw_len) {
                bad_bytes(base_doc + ldi);
            }
        }
        flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, doc_counts, view, lview);
        ubuf.clear();
        s.utf8 = ubuf;
    }

    /// Shared decoupled look-back parallel task executor & scatter engine.
    ///
    /// Workers dynamically claim tasks `0..ntask`, encode each into `s.out` via
    /// `encode_task(k, s, &mut local)`, publish `counts[k]`, and either copy
    /// straight into `out[base..base+cnt]` when every predecessor is done or
    /// park the buffer in `stash[k]` for a helper / final parallel scatter.
    /// Used by `encode_fast_short_t`, `encode_fused_sep_t`, and `encode_src_t`.
    #[allow(clippy::too_many_arguments)]
    fn run_decoupled_tasks<O: OutTok, W>(
        &self,
        ntask: usize,
        cap: usize,
        num_threads: usize,
        nt: usize,
        want_output: bool,
        init_worker: impl Fn() -> W + Sync,
        encode_task: impl Fn(usize, &mut Scratch, &mut W) + Sync,
    ) -> Vec<O> {
        use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*};
        const UNK: u64 = u64::MAX;
        let cap = if want_output { cap } else { 0 };
        let mut out: Vec<O> = Vec::with_capacity(cap);
        if want_output {
            chunk_table::advise_hugepage(out.as_mut_ptr() as *mut u8, cap * std::mem::size_of::<O>());
        }
        struct SendPtr<P>(*mut P);
        // SAFETY: tasks write disjoint index ranges through these pointers.
        unsafe impl<P> Sync for SendPtr<P> {}
        impl<P> SendPtr<P> {
            fn get(&self) -> *mut P {
                self.0
            }
        }
        let out_ptr = SendPtr(out.as_mut_ptr());
        let counts: Vec<AtomicU64> = (0..ntask).map(|_| AtomicU64::new(UNK)).collect();
        let incl: Vec<AtomicU64> = (0..ntask).map(|_| AtomicU64::new(UNK)).collect();
        let placed: Vec<AtomicBool> = (0..ntask).map(|_| AtomicBool::new(false)).collect();
        let stash: Vec<std::sync::Mutex<Option<Vec<u32>>>> =
            (0..ntask).map(|_| std::sync::Mutex::new(None)).collect();
        let pool: std::sync::Mutex<Vec<Vec<u32>>> = std::sync::Mutex::new(Vec::new());
        fn lock(m: &std::sync::Mutex<Option<Vec<u32>>>) -> std::sync::MutexGuard<'_, Option<Vec<u32>>> {
            m.lock().unwrap_or_else(|e| e.into_inner())
        }
        let next = AtomicUsize::new(0);

        let look_back = |k: usize| -> Option<u64> {
            let mut sum = 0u64;
            let mut j = k;
            while j > 0 {
                let i = j - 1;
                let p = incl[i].load(Acquire);
                if p != UNK {
                    return Some(p + sum);
                }
                let c = counts[i].load(Acquire);
                if c == UNK {
                    return None;
                }
                sum += c;
                j = i;
            }
            Some(sum)
        };
        let place = |k: usize, base: u64| -> bool {
            let mut g = lock(&stash[k]);
            let len = g.as_ref().expect("parked").len();
            if base == UNK || base as usize + len > cap {
                placed[k].store(false, Release);
                return false;
            }
            let v = g.take().expect("parked");
            drop(g);
            incl[k].store(base + len as u64, Release);
            // SAFETY: [base, base+len) is task k's exclusive range within cap;
            // the claim on placed[k] makes this the only writer.
            unsafe { O::put(&v, out_ptr.get().add(base as usize)) };
            pool.lock().unwrap_or_else(|e| e.into_inner()).push(v);
            true
        };

        let worker = || {
            self.with_scratch(|s| {
                let saved = std::mem::take(&mut s.out);
                let mut local = init_worker();
                loop {
                    let k = next.fetch_add(1, Relaxed);
                    if k >= ntask {
                        break;
                    }
                    s.out.clear();
                    encode_task(k, s, &mut local);
                    let cnt = s.out.len() as u64;
                    counts[k].store(cnt, Release);
                    if !want_output {
                        continue;
                    }
                    let mut ready = false;
                    if let Some(base) = look_back(k) {
                        if (base + cnt) as usize <= cap {
                            incl[k].store(base + cnt, Release);
                            unsafe { O::put(&s.out, out_ptr.get().add(base as usize)) };
                            placed[k].store(true, Release);
                            ready = true;
                        }
                    }
                    if !ready {
                        let c = s.out.capacity();
                        let spare = pool.lock().unwrap_or_else(|e| e.into_inner()).pop();
                        let mut spare = spare.unwrap_or_else(|| Vec::with_capacity(c));
                        spare.clear();
                        *lock(&stash[k]) = Some(std::mem::replace(&mut s.out, spare));
                        match look_back(k) {
                            Some(base) if !placed[k].swap(true, AcqRel) && place(k, base) => {}
                            _ => continue,
                        }
                    }
                    // Help forward: place parked successors whose offset is now known.
                    let mut j = k + 1;
                    while j < ntask && !placed[j].load(Acquire) && lock(&stash[j]).is_some() {
                        let base = incl[j - 1].load(Acquire);
                        if base == UNK || placed[j].swap(true, AcqRel) {
                            break;
                        }
                        if !place(j, base) {
                            break;
                        }
                        j += 1;
                    }
                }
                s.out = saved;
            })
        };
        if num_threads == 0 {
            rayon::broadcast(|_| worker());
        } else {
            match fixed_pool(nt) {
                Some(pool) => pool.install(|| { rayon::broadcast(|_| worker()); }),
                None => { rayon::broadcast(|_| worker()); }
            }
        }
        if !want_output {
            return Vec::new();
        }
        let mut pref = Vec::with_capacity(ntask + 1);
        pref.push(0usize);
        for c in &counts {
            let last = *pref.last().unwrap();
            pref.push(last + c.load(Relaxed) as usize);
        }
        let total = pref[ntask];
        let stashed: Vec<(usize, Vec<u32>)> = stash
            .into_iter()
            .enumerate()
            .filter_map(|(k, v)| v.into_inner().unwrap_or_else(|e| e.into_inner()).map(|v| (k, v)))
            .collect();
        #[cfg(test)]
        if std::env::var("TT_DIAG").is_ok() {
            let fin = stashed.iter().filter(|(k, _)| !placed[*k].load(Relaxed)).count();
            eprintln!("diag: tasks {ntask} | parked {} | final-pass {fin} | total {total} cap {cap}", stashed.len());
        }
        if total <= cap {
            let dst = SendPtr(out.as_mut_ptr());
            stashed.par_iter().filter(|(k, _)| !placed[*k].load(Relaxed)).for_each(|(k, v)| {
                unsafe { O::put(v, dst.get().add(pref[*k])) };
            });
            unsafe { out.set_len(total) };
            return out;
        }
        let mut exact: Vec<O> = Vec::with_capacity(total);
        let mut by_task: Vec<Option<&Vec<u32>>> = vec![None; ntask];
        for (k, v) in &stashed {
            by_task[*k] = Some(v);
        }
        let edst = SendPtr(exact.as_mut_ptr());
        let osrc = SendPtr(out.as_ptr() as *mut O);
        (0..ntask).into_par_iter().for_each(|k| {
            let len = pref[k + 1] - pref[k];
            let dst = unsafe { edst.get().add(pref[k]) };
            match by_task[k] {
                Some(v) => unsafe { O::put(v, dst) },
                None => unsafe {
                    std::ptr::copy_nonoverlapping(osrc.get().add(pref[k]) as *const O, dst, len)
                },
            }
        });
        unsafe { exact.set_len(total) };
        exact
    }

    /// Zero-`plan()` batch engine for `list[str]` / `list[bytes]` where every
    /// document is `< SPLIT_MIN` (`256 KiB`), `normalizer.is_ascii_identity()`,
    /// `items_pipe_ok()`, and `sep.is_none()`. Eliminates `Vec<Src>`, `norm`,
    /// `items`, `trusted`, and `item_counts` (~24 MB on 400K-line corpora).
    fn encode_fast_short_t<O: OutTok, const SPECIAL: bool>(
        &self,
        items: &[FastItem],
        total_len: usize,
        num_threads: usize,
        want_output: bool,
    ) -> Result<(Vec<O>, Vec<usize>), BadInput> {
        use std::sync::atomic::{AtomicUsize, Ordering::*};
        let n = items.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut doc_counts = vec![0usize; n];
        let bad_doc = AtomicUsize::new(usize::MAX);
        let bad_wide = AtomicUsize::new(usize::MAX);
        let nt = if num_threads == 0 {
            rayon::current_num_threads()
        } else {
            num_threads.min(1024)
        }
        .max(1);

        let tasks: Vec<(usize, usize)> = if nt > 1 && total_len >= (1 << 16) && parallel_ok() {
            let unit = (total_len / (nt * 32)).clamp(1 << 16, 1 << 20);
            let mut v = Vec::with_capacity(total_len / unit + 4);
            let (mut st, mut acc) = (0usize, 0usize);
            for (i, it) in items.iter().enumerate() {
                acc += it.len();
                if acc >= unit {
                    v.push((st, i + 1));
                    st = i + 1;
                    acc = 0;
                }
            }
            if st < n {
                v.push((st, n));
            }
            v
        } else {
            Vec::new()
        };

        let out = if tasks.len() <= 1 {
            let raw = self.with_scratch(|s| {
                let saved = std::mem::take(&mut s.out);
                if want_output {
                    let cap = total_len / 2 + 16;
                    s.out.reserve(cap);
                    chunk_table::advise_hugepage(s.out.as_mut_ptr() as *mut u8, cap * 4);
                }
                if want_output {
                    self.encode_fast_slice_piped::<SPECIAL, true>(
                        items,
                        0,
                        s,
                        &mut doc_counts,
                        &mut |d| { bad_doc.fetch_min(d, Relaxed); },
                        &mut |d| { bad_wide.fetch_min(d, Relaxed); },
                    );
                } else {
                    self.encode_fast_slice_piped::<SPECIAL, false>(
                        items,
                        0,
                        s,
                        &mut doc_counts,
                        &mut |d| { bad_doc.fetch_min(d, Relaxed); },
                        &mut |d| { bad_wide.fetch_min(d, Relaxed); },
                    );
                }
                std::mem::replace(&mut s.out, saved)
            });
            if want_output { O::from_u32(raw) } else { Vec::new() }
        } else {
            struct DcPtr(*mut usize);
            unsafe impl Sync for DcPtr {}
            impl DcPtr {
                fn get(&self) -> *mut usize { self.0 }
            }
            let dc_ptr = DcPtr(doc_counts.as_mut_ptr());
            self.run_decoupled_tasks::<O, ()>(
                tasks.len(),
                total_len * 2 / 3 + 4096,
                num_threads,
                nt,
                want_output,
                || (),
                |k, s, _| {
                    let (lo, hi) = tasks[k];
                    // SAFETY: tasks partition `0..n` into disjoint `[lo..hi)` slices.
                    let dc_slice = unsafe { std::slice::from_raw_parts_mut(dc_ptr.get().add(lo), hi - lo) };
                    if want_output {
                        self.encode_fast_slice_piped::<SPECIAL, true>(
                            &items[lo..hi],
                            lo,
                            s,
                            dc_slice,
                            &mut |d| { bad_doc.fetch_min(d, Relaxed); },
                            &mut |d| { bad_wide.fetch_min(d, Relaxed); },
                        );
                    } else {
                        self.encode_fast_slice_piped::<SPECIAL, false>(
                            &items[lo..hi],
                            lo,
                            s,
                            dc_slice,
                            &mut |d| { bad_doc.fetch_min(d, Relaxed); },
                            &mut |d| { bad_wide.fetch_min(d, Relaxed); },
                        );
                    }
                },
            )
        };
        let bw = bad_wide.load(Relaxed);
        if bw != usize::MAX {
            return Err(BadInput::Wide(bw));
        }
        let bad = bad_doc.load(Relaxed);
        if bad != usize::MAX {
            return Err(BadInput::Bytes(bad));
        }
        Ok((out, doc_counts))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_items_piped<const WANT_OUTPUT: bool>(
        &self,
        texts: &[Src<'_>],
        norm: &[std::borrow::Cow<'_, [u8]>],
        trusted: &[bool],
        items: &[(u32, usize, usize, bool)],
        s: &mut Scratch,
        counts: &mut [u32],
        bad_bytes: &mut dyn FnMut(u32),
        bad_wide: &mut dyn FnMut(u32),
    ) {
        const CAP: usize = 256;
        let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); CAP + blockscan::SCAN_SLACK + X86_PF_DIST];
        let mut ends = [0u32; CAP + blockscan::SCAN_SLACK];
        // (seg_lo, seg_hi, item_idx, first_a, ubuf_off, ubuf_len) — ubuf_len == 0 => norm[doc]
        let mut segs: Vec<(usize, usize, usize, usize, usize, usize)> = Vec::with_capacity(64);
        let mut cnt = 0usize;
        let mut view = self.learned.view();
        let mut lview = self.long_table.view();
        let mut ubuf = std::mem::take(&mut s.utf8);
        let flush = |buf: &mut [std::mem::MaybeUninit<PipeRec>],
                     total: usize,
                     ends: &[u32],
                     segs: &[(usize, usize, usize, usize, usize, usize)],
                     ubuf: &[u8],
                     s: &mut Scratch,
                     counts: &mut [u32],
                     view: chunk_table::View<'_, 3>,
                     lview: chunk_table::LongView<'_>| {
            if segs.is_empty() {
                return;
            }
            Self::prime_pipe_prefetch(buf, total);
            let buf = buf.as_ptr() as *const PipeRec;
            if !WANT_OUTPUT {
                for &(slo, shi, ii, first_a, uoff, ulen) in segs {
                    let n = self.flush_seg_count(
                        buf,
                        ends,
                        slo,
                        shi,
                        first_a,
                        || {
                            if ulen != 0 {
                                &ubuf[uoff..uoff + ulen]
                            } else {
                                &norm[items[ii].0 as usize]
                            }
                        },
                        s,
                        view,
                        lview,
                    );
                    counts[ii] += n as u32;
                }
                return;
            }
            s.out.reserve(6 * total + 16);
            let mut base_out = s.out.as_mut_ptr();
            let mut wp = unsafe { base_out.add(s.out.len()) };
            for &(slo, shi, ii, first_a, uoff, ulen) in segs {
                let n = self.flush_seg_into(
                    buf,
                    ends,
                    slo,
                    shi,
                    total,
                    first_a,
                    || {
                        if ulen != 0 {
                            &ubuf[uoff..uoff + ulen]
                        } else {
                            &norm[items[ii].0 as usize]
                        }
                    },
                    s,
                    &mut base_out,
                    &mut wp,
                    view,
                    lview,
                );
                counts[ii] += n as u32;
            }
            unsafe { s.out.set_len(wp.offset_from(base_out) as usize) };
        };
        let defer_norm = self.parallel_unicode_norm();
        for (ii, &(doc, a0, b0, seg)) in items.iter().enumerate() {
            if let Src::Wide(w) = texts[doc as usize] {
                if cnt == 0 {
                    ubuf.clear();
                } else if b0 - a0 > 16384 {
                    flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, counts, view, lview);
                    cnt = 0;
                    segs.clear();
                    ubuf.clear();
                }
                let uoff = ubuf.len();
                if w.append_utf8(a0, b0, &mut ubuf).is_err() {
                    ubuf.truncate(uoff);
                    bad_wide(doc);
                    continue;
                }
                let ustop = ubuf.len() - uoff;
                if ustop == 0 {
                    continue;
                }
                // Preserve exact look-ahead for split segments (`\s+(?!\S)` in GPT-2):
                // `split_points` guarantees `w[b0]` is an ASCII letter when `b0 < w.len`.
                if seg && b0 < w.len {
                    ubuf.push(b'a');
                }
                let ulen = ubuf.len() - uoff;
                if ulen > u32::MAX as usize {
                    flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, counts, view, lview);
                    cnt = 0;
                    segs.clear();
                    let before = s.out.len();
                    self.encode_range(&ubuf[uoff..uoff + ulen], 0, ustop, s);
                    counts[ii] += (s.out.len() - before) as u32;
                    ubuf.clear();
                    continue;
                }
                let mut scan = blockscan::Scan::new(0);
                let mut a = 0usize;
                loop {
                    if cnt == 0 {
                        view = self.learned.view();
                        lview = self.long_table.view();
                    }
                    let d = &ubuf[uoff..uoff + ulen];
                    let k1 = scan.fill(self.kind, d, ustop, &mut ends, cnt, CAP);
                    if k1 == cnt {
                        break;
                    }
                    let first_a = a;
                    a = Self::fill_recs(d, &ends, cnt, k1, a, buf.as_mut_ptr(), view, lview);
                    segs.push((cnt, k1, ii, first_a, uoff, ulen));
                    cnt = k1;
                    if cnt >= CAP {
                        flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, counts, view, lview);
                        cnt = 0;
                        segs.clear();
                    }
                    if a >= ustop {
                        break;
                    }
                }
                continue;
            }
            let d: &[u8] = &norm[doc as usize];
            let n = d.len();
            let (start, stop) = if seg { (a0, b0) } else { (0, n) };
            let mut need_utf8_check = !trusted[doc as usize];
            if defer_norm {
                if need_utf8_check && simdutf8::basic::from_utf8(&d[start..stop]).is_err() {
                    bad_bytes(doc);
                    continue;
                }
                need_utf8_check = false;
                // SAFETY: `trusted` or validated immediately above; `split_points_unicode`
                // cuts strictly on ASCII char boundaries.
                let sub = unsafe { std::str::from_utf8_unchecked(&d[start..stop]) };
                if let std::borrow::Cow::Owned(norm_s) = self.normalizer.apply(sub) {
                    if cnt == 0 {
                        ubuf.clear();
                    } else if norm_s.len() > 16384 {
                        flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, counts, view, lview);
                        cnt = 0;
                        segs.clear();
                        ubuf.clear();
                    }
                    let uoff = ubuf.len();
                    ubuf.extend_from_slice(norm_s.as_bytes());
                    let ustop = ubuf.len() - uoff;
                    if ustop == 0 {
                        continue;
                    }
                    if seg && stop < n {
                        ubuf.push(b'a');
                    }
                    let ulen = ubuf.len() - uoff;
                    if ulen > u32::MAX as usize {
                        flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, counts, view, lview);
                        cnt = 0;
                        segs.clear();
                        let before = s.out.len();
                        self.encode_range(&ubuf[uoff..uoff + ulen], 0, ustop, s);
                        counts[ii] += (s.out.len() - before) as u32;
                        ubuf.clear();
                        continue;
                    }
                    let mut scan = blockscan::Scan::new(0);
                    let mut a = 0usize;
                    loop {
                        if cnt == 0 {
                            view = self.learned.view();
                            lview = self.long_table.view();
                        }
                        let ud = &ubuf[uoff..uoff + ulen];
                        let k1 = scan.fill(self.kind, ud, ustop, &mut ends, cnt, CAP);
                        if k1 == cnt {
                            break;
                        }
                        let first_a = a;
                        a = Self::fill_recs(ud, &ends, cnt, k1, a, buf.as_mut_ptr(), view, lview);
                        segs.push((cnt, k1, ii, first_a, uoff, ulen));
                        cnt = k1;
                        if cnt >= CAP {
                            flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, counts, view, lview);
                            cnt = 0;
                            segs.clear();
                        }
                        if a >= ustop {
                            break;
                        }
                    }
                    continue;
                }
            }
            if n > u32::MAX as usize {
                if need_utf8_check && simdutf8::basic::from_utf8(&d[start..stop]).is_err() {
                    bad_bytes(doc);
                    continue;
                }
                flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, counts, view, lview);
                cnt = 0;
                segs.clear();
                let before = s.out.len();
                self.encode_range(d, start, stop, s);
                counts[ii] += (s.out.len() - before) as u32;
                continue;
            }
            let mut scan = blockscan::Scan::new_checked(start, need_utf8_check);
            let mut a = start;
            loop {
                if cnt == 0 {
                    view = self.learned.view();
                    lview = self.long_table.view();
                }
                let k1 = scan.fill(self.kind, d, stop, &mut ends, cnt, CAP);
                if k1 == cnt {
                    break;
                }
                let first_a = a;
                a = Self::fill_recs(d, &ends, cnt, k1, a, buf.as_mut_ptr(), view, lview);
                segs.push((cnt, k1, ii, first_a, 0, 0));
                cnt = k1;
                if cnt >= CAP {
                    flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, counts, view, lview);
                    cnt = 0;
                    segs.clear();
                }
                if a >= stop {
                    break;
                }
            }
            if !scan.finish_utf8(d, stop) {
                bad_bytes(doc);
            }
        }
        flush(&mut buf, cnt, &ends[..cnt], &segs, &ubuf, s, counts, view, lview);
        ubuf.clear();
        s.utf8 = ubuf;
    }

    /// Out-of-line `emit_chunk` for the hot loop's miss path, so the hot loop
    /// doesn't carry the merge/cache code (and its register pressure) inline.
    #[cold]
    #[inline(never)]
    fn emit_chunk_cold(&self, bytes: &[u8], pos: usize, end: usize, n: usize, s: &mut Scratch) {
        self.emit_chunk_impl::<true>(bytes, pos, end, n, s);
    }

    /// Encode one pre-token chunk [pos, end) (Merge::Auto): load-time table,
    /// then the per-thread cache, then the exact merge (memoized).
    #[inline(always)]
    fn emit_chunk(&self, bytes: &[u8], pos: usize, end: usize, n: usize, s: &mut Scratch) {
        self.emit_chunk_impl::<true>(bytes, pos, end, n, s);
    }

    /// Fast path for a prefetched 16..=31 byte chunk (`r.p == 0 && r.lo != 0`):
    /// probes the already-in-L1 `LSlot` directly before falling back on cold miss.
    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn emit_long_at(
        &self,
        bytes: &[u8],
        pos: usize,
        end: usize,
        n: usize,
        addr: u64,
        h: u64,
        lview: chunk_table::LongView<'_>,
        s: &mut Scratch,
    ) {
        let len = end - pos;
        let k = chunk_table::pack32_at(&bytes[..n], pos, len);
        match unsafe { lview.get_into_at(k, addr, &mut s.out) } {
            chunk_table::LongHit::Hit => return,
            chunk_table::LongHit::Empty => {
                // Only re-check current table if `long_table` grew during this 256-chunk batch
                // (avoids a redundant L3/DRAM probe on 99.99% of cold 16..=31B chunks).
                if !lview.is_current(&self.long_table) {
                    if let chunk_table::LongHit::Hit = self.long_table.get_into(k, h, &mut s.out) {
                        return;
                    }
                }
                let key = &bytes[pos..end];
                let st = s.out.len();
                #[cfg(test)]
                perf::LONG_MERGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.process_chunk(key, s, Merge::Auto);
                let (out, lc) = (&s.out, &mut s.long_cache);
                let res = &out[st..];
                if !self.long_table.put(k, h, res) {
                    let bh = chunk_table::hash_bytes(key);
                    lc.put_h(self.uid, key, bh, res);
                    let _ = self.spill.put_h(key, bh, res);
                }
                return;
            }
            chunk_table::LongHit::Spilled => {}
        }
        self.emit_chunk_cold(bytes, pos, end, n, s);
    }

    /// Cold-miss fast path when `emit_inline_at` just saw an empty slot (`m == 0`):
    /// checks the read-only load-time `chunk_table` first (demand-paging any
    /// unseeded vocab token in ~20 ns with zero BPE merges), else merges once.
    #[cold]
    #[inline(never)]
    fn emit_chunk_fresh(&self, chunk: &[u8], lo: u64, hi: u64, s: &mut Scratch) {
        let h = chunk_table::hash(lo, hi);
        if let Some(t) = self.chunk_table.get_h(lo, hi, h) {
            s.out.push(t);
            self.learned.put(lo, hi, h, &[t]);
            return;
        }
        if !s.dyn_cache.owned_by(self.uid) {
            self.process_chunk_impl::<true>(chunk, s, Merge::Auto);
            return;
        }
        let st = s.out.len();
        #[cfg(test)]
        perf::MERGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.process_chunk_impl::<true>(chunk, s, Merge::Auto);
        let (out, dc) = (&s.out, &mut s.dyn_cache);
        let res = &out[st..];
        if !self.learned.put_6(lo, hi, h, res) {
            dc.put(lo, hi, res);
            let _ = self.spill.put(chunk, res);
        }
    }

    #[inline(always)]
    fn emit_chunk_impl<const VOCAB: bool>(&self, bytes: &[u8], pos: usize, end: usize, n: usize, s: &mut Scratch) {
        let len = end - pos;
        if len.wrapping_sub(1) < chunk_table::MAX_KEY {
            let (lo, hi) = chunk_table::pack_at(&bytes[..n], pos, len);
            let h = chunk_table::hash(lo, hi);
            if VOCAB {
                if let Some(t) = self.chunk_table.get_h(lo, hi, h) {
                    s.out.push(t);
                    self.learned.put(lo, hi, h, &[t]);
                    return;
                }
            }
            let mut spilled = false;
            if let Some((ts, k)) = self.learned.get_6(lo, hi, h) {
                #[cfg(test)]
                perf::DYN_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if k <= 6 {
                    s.out.extend_from_slice(&ts[..k]);
                    return;
                }
                spilled = k == chunk_table::SPILLED;
            }
            if s.dyn_cache.owned_by(self.uid) {
                let st = s.out.len();
                if spilled {
                    if let Some(ts) = s.dyn_cache.get(lo, hi) {
                        #[cfg(test)]
                        perf::DYN_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        s.out.extend_from_slice(ts);
                        return;
                    }
                    if self.spill.get_into(&bytes[pos..end], &mut s.out) {
                        let (out, dc) = (&s.out, &mut s.dyn_cache);
                        dc.put(lo, hi, &out[st..]);
                        return;
                    }
                }
                #[cfg(test)]
                perf::MERGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.process_chunk(&bytes[pos..end], s, Merge::Auto);
                let (out, dc) = (&s.out, &mut s.dyn_cache);
                let res = &out[st..];
                if !self.learned.put_6(lo, hi, h, res) {
                    dc.put(lo, hi, res);
                    let _ = self.spill.put(&bytes[pos..end], res);
                }
                return;
            }
        } else if len > chunk_table::MAX_KEY && !self.byte_fallback {
            // 16..=31 byte chunks (99.8% of long words): lock-free lazy 64B SharedLongTable
            // (zero RwLock, zero per-thread wipes, up to 11 inline tokens).
            if len <= chunk_table::MAX_LONG_KEY {
                let k = chunk_table::pack32_at(&bytes[..n], pos, len);
                let h = chunk_table::hash32(k);
                match self.long_table.get_into(k, h, &mut s.out) {
                    chunk_table::LongHit::Hit => return,
                    chunk_table::LongHit::Empty => {
                        let key = &bytes[pos..end];
                        let st = s.out.len();
                        #[cfg(test)]
                        perf::LONG_MERGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        self.process_chunk(key, s, Merge::Auto);
                        let (out, lc) = (&s.out, &mut s.long_cache);
                        let res = &out[st..];
                        if !self.long_table.put(k, h, res) {
                            let bh = chunk_table::hash_bytes(key);
                            lc.put_h(self.uid, key, bh, res);
                            let _ = self.spill.put_h(key, bh, res);
                        }
                        return;
                    }
                    chunk_table::LongHit::Spilled => {}
                }
            }
            // >31B or >11-token spilled long chunk: single hash across LongCache and SpillMap
            let key = &bytes[pos..end];
            let h = chunk_table::hash_bytes(key);
            if let Some(ts) = s.long_cache.get_h(self.uid, key, h) {
                s.out.extend_from_slice(ts);
                return;
            }
            let st = s.out.len();
            if self.spill.get_into_h(key, h, &mut s.out) {
                let (out, lc) = (&s.out, &mut s.long_cache);
                lc.put_h(self.uid, key, h, &out[st..]);
                return;
            }
            #[cfg(test)]
            perf::LONG_MERGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.process_chunk(key, s, Merge::Auto);
            let (out, lc) = (&s.out, &mut s.long_cache);
            lc.put_h(self.uid, key, h, &out[st..]);
            let _ = self.spill.put_h(key, h, &out[st..]);
            return;
        }
        self.emit(&bytes[pos..end], s, Merge::Auto);
    }

    /// Can one long document be split across threads? Hand-written SIMD BlockScan,
    /// streaming DeepSeek CJK, and zero-alloc SIMD SentencePiece all have proven
    /// exact boundary rules (`split_points` / `split_points_sp`).
    fn splittable(&self) -> bool {
        self.fast_pretok == FastPretok::BlockScan
            || (self.fast_pretok == FastPretok::Cjk && self.singleton_eow.is_none())
            || self.sp_fast_meta != SpFastMeta::None
            || (self.fast_pretok == FastPretok::Pipe
                && !self.byte_fallback
                && self.singleton_eow.is_none()
                && self.pretok_pipe.as_ref().is_some_and(|p| p.is_nl_letter_splittable()))
    }

    /// True when whole-document serial normalization can be deferred to parallel
    /// per-segment normalization inside workers (sound for pure NFC/NFD/NFKC/NFKD
    /// cut at `split_points_unicode` ASCII boundaries).
    #[inline(always)]
    fn parallel_unicode_norm(&self) -> bool {
        self.normalizer.is_unicode_only() && self.splittable() && self.sp_fast_meta == SpFastMeta::None
    }

    /// Proven-exact chunk boundaries for `SpFastMeta`: any `b[p] == b' '` preceded
    /// by `b[p - 1] != b' '` and not in `cross` starts a new chunk uncrossable by BPE.
    fn split_points_sp(b: &[u8], seg: usize, cross: &[u64; 4]) -> Vec<usize> {
        let n = b.len();
        let mut pts = vec![0usize];
        let mut target = seg;
        while target < n {
            let lim = (target + seg).min(n);
            let mut found = None;
            let mut i = target.max(1);
            while i < lim {
                if let Some(rel) = memchr::memchr(b' ', &b[i..lim]) {
                    let p = i + rel;
                    let prev = b[p - 1];
                    // Require ASCII `prev < 0x80` so a split never lands directly after a
                    // literal `'▁'` (`0xE2 0x96 0x81`) followed by `' '` (which becomes `"▁▁"`).
                    if prev < 0x80 && prev != b' ' && ((cross[(prev as usize) >> 6] >> (prev & 63)) & 1) == 0 {
                        found = Some(p);
                        break;
                    }
                    i = p + 1;
                } else {
                    break;
                }
            }
            match found {
                Some(p) => {
                    pts.push(p);
                    target = p + seg;
                }
                None => target = lim + 1,
            }
        }
        pts.push(n);
        pts
    }

    /// Positions `p` that are GUARANTEED chunk boundaries for every grammar
    /// (Word / Generic / CamelCase): `b[p-1] == '\n'` and `b[p]` is an ASCII
    /// letter. Proof sketch: the only chunks that can contain a '\n' are
    /// whitespace chunks (`\s*[\r\n]+` ends at the last newline of the run;
    /// `\s+(?!\S)` leaves the last ws char — here the '\n' — to a 1-char `\s+`
    /// chunk) and symbol runs with a `[\r\n]*`/`[\r\n/]*` tail, which stop at a
    /// letter. No letter-run prefix (`[^\r\n\p{L}\p{N}]?`, ` ?`) can be '\n'.
    /// So the chunk containing `p-1` ends exactly at `p`, and since every chunk
    /// scan only looks forward, the scan from `p` is independent of what came
    /// before. Returns [0, p1, p2, …, n], segments ~`seg` bytes.
    /// Proven-exact split points for DeepSeek CJK: `b[p-1] == '\n'` followed by
    /// ASCII alphanumeric OR a 3-byte CJK ideograph (`0xE4..=0xE9`, U+4000..U+9FFF),
    /// so pure Chinese/Japanese documents parallelize across all cores too!
    fn split_points_cjk(b: &[u8], seg: usize) -> Vec<usize> {
        let n = b.len();
        let mut pts = vec![0usize];
        let mut target = seg;
        while target < n {
            let lim = (target + seg).min(n);
            let mut found = None;
            let mut i = target.max(1) - 1;
            while i + 1 < lim {
                if let Some(rel) = memchr::memchr(b'\n', &b[i..lim - 1]) {
                    let p = i + rel + 1;
                    let c = b[p];
                    if c.is_ascii_alphanumeric() || (0xE4..=0xE9).contains(&c) {
                        found = Some(p);
                        break;
                    }
                    i = p;
                } else {
                    break;
                }
            }
            match found {
                Some(p) => {
                    pts.push(p);
                    target = p + seg;
                }
                None => target = lim + 1,
            }
        }
        pts.push(n);
        pts
    }

    fn split_points(b: &[u8], seg: usize) -> Vec<usize> {
        let n = b.len();
        let mut pts = vec![0usize];
        let mut target = seg;
        while target < n {
            let lim = (target + seg).min(n);
            let mut found = None;
            let mut i = target.max(1) - 1;
            while i + 1 < lim {
                if let Some(rel) = memchr::memchr(b'\n', &b[i..lim - 1]) {
                    let p = i + rel + 1;
                    if b[p].is_ascii_alphabetic() {
                        found = Some(p);
                        break;
                    }
                    i = p;
                } else {
                    break;
                }
            }
            match found {
                Some(p) => {
                    pts.push(p);
                    target = p + seg;
                }
                None => target = lim + 1,
            }
        }
        pts.push(n);
        pts
    }

    /// Split points guaranteed invariant under ALL four Unicode normalization forms
    /// (NFC/NFD/NFKC/NFKD) AND exact chunk boundaries: `b[p-1] == '\n'`, `b[p]` is
    /// ASCII alphabetic, and `b[p+1]` (if present) is ASCII (so no following
    /// combining mark can compose with `b[p]`).
    fn split_points_unicode(b: &[u8], seg: usize) -> Vec<usize> {
        let n = b.len();
        let mut pts = vec![0usize];
        let mut target = seg;
        while target < n {
            let lim = (target + seg).min(n);
            let mut found = None;
            let mut i = target.max(1) - 1;
            while i + 1 < lim {
                if let Some(rel) = memchr::memchr(b'\n', &b[i..lim - 1]) {
                    let p = i + rel + 1;
                    if b[p].is_ascii_alphabetic() && (p + 1 == n || b[p + 1] < 0x80) {
                        found = Some(p);
                        break;
                    }
                    i = p;
                } else {
                    break;
                }
            }
            match found {
                Some(p) => {
                    pts.push(p);
                    target = p + seg;
                }
                None => target = lim + 1,
            }
        }
        pts.push(n);
        pts
    }

    /// Encode one long document in parallel segments (state-machine path only).
    fn encode_split(&self, text: &str) -> Vec<u32> {
        self.encode_direct(&[text], 0, true).0
    }

    /// Run `f` on this thread's warm Scratch (claimed for this tokenizer), or on a
    /// fresh one if the thread-local is already borrowed (re-entrant rayon steal).
    #[inline]
    fn with_scratch<R>(&self, f: impl FnOnce(&mut Scratch) -> R) -> R {
        SCRATCH.with(|cell| match cell.try_borrow_mut() {
            Ok(mut g) => {
                g.dyn_cache.claim(self.uid);
                g.long_cache.claim(self.uid);
                f(&mut g)
            }
            Err(_) => f(&mut Scratch::default()),
        })
    }

    /// The batch engine behind every multi-document / large-document path.
    ///
    /// Work items are whole documents, or — for long documents on the
    /// state-machine path — segments cut at PROVEN chunk boundaries (see
    /// `split_points`). Consecutive items are grouped into tasks of at least
    /// TASK_BYTES of input, and each task encodes into ONE contiguous buffer
    /// (no per-document Vec, no per-document free). Tasks run on rayon (GIL
    /// released by the caller); the caller then scatters the task buffers into
    /// its destination in parallel. Output is identical to per-document
    /// `encode_one`, in document order.
    /// Split a batch into work items (whole documents, or proven-boundary
    /// segments of long ones) grouped into tasks of >= TASK_BYTES of input.
    /// Returns (normalized docs, items (doc, start, end, is_segment), tasks
    /// (first item, end item, bytes)).
    #[allow(clippy::type_complexity)]
    fn plan<'t>(
        &self,
        texts: &[Src<'t>],
        num_threads: usize,
    ) -> (Vec<std::borrow::Cow<'t, [u8]>>, Vec<(u32, usize, usize, bool)>, Vec<(usize, usize, usize)>) {
        use std::borrow::Cow;
        let splittable = self.splittable();
        let to_bytes = |c: Cow<'t, str>| -> Cow<'t, [u8]> {
            match c {
                Cow::Borrowed(s) => Cow::Borrowed(s.as_bytes()),
                Cow::Owned(s) => Cow::Owned(s.into_bytes()),
            }
        };
        let as_str = |t: &Src<'t>| -> &'t str {
            match *t {
                Src::Str(s) => s,
                Src::Raw(_) | Src::Wide(_) => unreachable!("unvalidated bytes/wide reach plan() only without a normalizer"),
            }
        };
        let defer_norm = self.parallel_unicode_norm();
        let norm: Vec<Cow<'t, [u8]>> = if self.normalizer.is_identity() || self.sp_fast_meta != SpFastMeta::None || defer_norm {
            texts
                .iter()
                .map(|t| {
                    Cow::Borrowed(match *t {
                        Src::Str(s) => s.as_bytes(),
                        Src::Raw(b) => b,
                        Src::Wide(_) => &[],
                    })
                })
                .collect()
        } else {
            // quick-check is ~free on ASCII; only rare runs allocate
            if num_threads == 1 || !parallel_ok() {
                texts.iter().map(|t| to_bytes(self.normalize(as_str(t)))).collect()
            } else {
                texts.par_iter().with_min_len(64).map(|t| to_bytes(self.normalize(as_str(t)))).collect()
            }
        };
        // Work-unit size: ~64 tasks per thread, between 64 KiB and 1 MiB. Bigger
        // units cut per-task overhead (hand-off, offset look-back, parking):
        // 2 GB on 14 threads, 64 KiB -> 1 MiB units: +9%; smaller inputs keep
        // enough tasks to balance.
        let total: usize = texts
            .iter()
            .zip(&norm)
            .map(|(t, d)| match *t {
                Src::Wide(w) => w.len,
                _ => d.len(),
            })
            .sum();
        let nt = if num_threads == 0 { rayon::current_num_threads() } else { num_threads }.max(1);
        let default_unit = (total / (nt * 64)).clamp(1 << 16, 1 << 20);
        #[cfg(test)]
        let unit = std::env::var("TT_SEG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default_unit);
        #[cfg(not(test))]
        let unit = default_unit;
        let (seg_bytes, task_bytes) = (unit, unit);
        let mut items: Vec<(u32, usize, usize, bool)> = Vec::with_capacity(texts.len());
        for (i, (t, d)) in texts.iter().zip(&norm).enumerate() {
            if let Src::Wide(w) = *t {
                if splittable && w.len >= SPLIT_MIN {
                    let pts = w.split_points(seg_bytes);
                    for win in pts.windows(2) {
                        items.push((i as u32, win[0], win[1], true));
                    }
                } else {
                    items.push((i as u32, 0, w.len, false));
                }
                continue;
            }
            let b: &[u8] = d;
            if splittable && b.len() >= SPLIT_MIN {
                if self.sp_fast_meta != SpFastMeta::None {
                    let pts = Self::split_points_sp(b, seg_bytes, &self.sp_cross_prev);
                    for w in pts.windows(2) {
                        items.push((i as u32, w[0], w[1], true));
                    }
                } else if self.fast_pretok == FastPretok::Cjk && !defer_norm {
                    let pts = Self::split_points_cjk(b, seg_bytes);
                    for w in pts.windows(2) {
                        items.push((i as u32, w[0], w[1], true));
                    }
                } else if defer_norm {
                    let pts = Self::split_points_unicode(b, seg_bytes);
                    for w in pts.windows(2) {
                        items.push((i as u32, w[0], w[1], true));
                    }
                } else {
                    let pts = Self::split_points(b, seg_bytes);
                    for w in pts.windows(2) {
                        items.push((i as u32, w[0], w[1], true));
                    }
                }
            } else {
                items.push((i as u32, 0, b.len(), false));
            }
        }
        let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
        let (mut st, mut acc) = (0usize, 0usize);
        for (k, it) in items.iter().enumerate() {
            acc += it.2 - it.1;
            if acc >= task_bytes {
                tasks.push((st, k + 1, acc));
                st = k + 1;
                acc = 0;
            }
        }
        if st < items.len() {
            tasks.push((st, items.len(), acc));
        }
        (norm, items, tasks)
    }

    /// Encode one item into `s.out` (appends). An item of an unvalidated
    /// document is validated first (it is about to be read anyway); false if it
    /// is not valid UTF-8. Items split a document only before an ASCII byte, so
    /// the document is valid iff every item is.
    #[inline]
    fn encode_item(&self, norm: &[std::borrow::Cow<'_, [u8]>], trusted: &[bool], item: (u32, usize, usize, bool), s: &mut Scratch) -> bool {
        let (doc, a, b, seg) = item;
        let d: &[u8] = &norm[doc as usize];
        if !trusted[doc as usize] && simdutf8::basic::from_utf8(&d[a..b]).is_err() {
            return false;
        }
        if self.sp_fast_meta != SpFastMeta::None {
            if seg {
                // SAFETY: split_points_sp splits strictly before ASCII `' '`, and
                // non-trusted bytes were validated above; `b` has no literal `'▁'`.
                let sub = unsafe { std::str::from_utf8_unchecked(&d[a..b]) };
                let ok = self.encode_sp_fast(sub, a == 0, a == 0, s);
                debug_assert!(ok);
            } else {
                // SAFETY: trusted (from a str) or validated just above (whole doc).
                let whole = unsafe { std::str::from_utf8_unchecked(d) };
                self.encode_into(whole, Merge::Auto, true, s);
            }
            return true;
        }
        if self.fast_pretok == FastPretok::Cjk && self.singleton_eow.is_none() {
            if self.parallel_unicode_norm() {
                let sub = unsafe { std::str::from_utf8_unchecked(&d[a..b]) };
                let norm_sub = self.normalizer.apply(sub);
                match norm_sub {
                    std::borrow::Cow::Borrowed(_) => self.encode_cjk_range(d, a, b, s),
                    std::borrow::Cow::Owned(ns) => self.encode_cjk_range(ns.as_bytes(), 0, ns.len(), s),
                }
            } else if seg {
                self.encode_cjk_range(d, a, b, s);
            } else {
                self.encode_cjk_range(d, 0, d.len(), s);
            }
            return true;
        }
        if self.fast_pretok == FastPretok::Pipe
            && !self.byte_fallback
            && self.singleton_eow.is_none()
            && self.pretok_pipe.as_ref().is_some_and(|p| p.is_slice_only())
        {
            let pipe = self.pretok_pipe.as_ref().unwrap();
            let sub = unsafe { std::str::from_utf8_unchecked(&d[a..b]) };
            if self.parallel_unicode_norm() {
                let norm_sub = self.normalizer.apply(sub);
                self.encode_pipe_piped(norm_sub.as_ref(), pipe, s);
            } else {
                self.encode_pipe_piped(sub, pipe, s);
            }
            return true;
        }
        if self.parallel_unicode_norm() {
            // SAFETY: trusted or validated immediately above; `split_points_unicode`
            // cuts strictly on ASCII char boundaries.
            let sub = unsafe { std::str::from_utf8_unchecked(&d[a..b]) };
            let norm_sub = self.normalizer.apply(sub);
            if seg {
                match norm_sub {
                    std::borrow::Cow::Borrowed(_) => self.encode_range(d, a, b, s),
                    std::borrow::Cow::Owned(ns) => self.encode_range(ns.as_bytes(), 0, ns.len(), s),
                }
            } else {
                self.encode_into_norm(norm_sub.as_ref(), Merge::Auto, true, s);
            }
            return true;
        }
        if seg {
            self.encode_range(d, a, b, s);
        } else {
            // SAFETY: trusted (from a str) or validated just above (whole doc).
            self.encode_into_norm(unsafe { std::str::from_utf8_unchecked(d) }, Merge::Auto, true, s);
        }
        true
    }

    /// The batch engine: one flat output in document order, plus tokens per
    /// document. Written directly into the final buffer with a single-pass
    /// "decoupled look-back" scheme (as in GPU prefix sums):
    ///
    /// * worker threads take tasks IN ORDER from an atomic counter and encode
    ///   each into their own reused scratch buffer (warm cache, no new pages);
    /// * a finished task publishes its token count, then finds its output
    ///   offset by walking back over its predecessors (started earlier, so
    ///   normally already done): the nearest inclusive prefix plus the counts in
    ///   between; it publishes its own inclusive prefix and copies its tokens
    ///   straight into the final buffer;
    /// * if a predecessor is still running past a short spin, the task stashes
    ///   its tokens and moves on; stashed tasks are placed at the end.
    ///
    /// No per-task allocations, no second full-size buffer, no gather pass. The
    /// final buffer is sized by an estimate (tokens <= bytes / 2 + slack); if a
    /// pathological input exceeds it, the output is assembled exactly anyway.
    /// `want_output = false` only counts (no output buffer at all).
    /// Whether non-ASCII Python `str` items can be stream-transcoded in per-worker
    /// L1/L2 scratch without touching `PyObject_Malloc` or mutating the strings.
    #[inline]
    fn stream_wide_ok(&self, sep: Option<&[u8]>) -> bool {
        sep.is_none() && self.normalizer.is_identity() && self.items_pipe_ok()
    }

    /// Encode one macro-chunk `chunk` split on `sep` in a single streaming pass
    /// through the 256-chunk pipeline (zero up-front `split_on` or `plan()`).
    /// Appends tokens to `s.out`, pushes one token count per doc to `doc_counts`,
    /// and returns `Some(first_bad_local_doc)` if any unvalidated doc is not UTF-8.
    #[allow(clippy::too_many_arguments)]
    fn encode_sep_macro<const WANT_OUTPUT: bool>(
        &self,
        chunk: &[u8],
        sep: &[u8],
        finder: &memchr::memmem::Finder<'_>,
        trusted: bool,
        s: &mut Scratch,
        segs: &mut Vec<(usize, usize, usize, usize, usize, usize)>,
        doc_counts: &mut Vec<usize>,
    ) -> Option<usize> {
        const CAP: usize = 256;
        let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); CAP + blockscan::SCAN_SLACK + X86_PF_DIST];
        let mut ends = [0u32; CAP + blockscan::SCAN_SLACK];
        segs.clear();
        let mut cnt = 0usize;
        let mut view = self.learned.view();
        let mut lview = self.long_table.view();
        let flush = |buf: &mut [std::mem::MaybeUninit<PipeRec>],
                     total: usize,
                     ends: &[u32],
                     segs: &[(usize, usize, usize, usize, usize, usize)],
                     s: &mut Scratch,
                     doc_counts: &mut [usize],
                     view: chunk_table::View<'_, 3>,
                     lview: chunk_table::LongView<'_>| {
            if segs.is_empty() {
                return;
            }
            Self::prime_pipe_prefetch(buf, total);
            let buf = buf.as_ptr() as *const PipeRec;
            if !WANT_OUTPUT {
                for &(slo, shi, di, first_a, ds, de) in segs {
                    let n = self.flush_seg_count(
                        buf,
                        ends,
                        slo,
                        shi,
                        first_a,
                        || &chunk[ds..de],
                        s,
                        view,
                        lview,
                    );
                    doc_counts[di] += n;
                }
                return;
            }
            s.out.reserve(6 * total + 16);
            let mut base_out = s.out.as_mut_ptr();
            let mut wp = unsafe { base_out.add(s.out.len()) };
            for &(slo, shi, di, first_a, ds, de) in segs {
                let n = self.flush_seg_into(
                    buf,
                    ends,
                    slo,
                    shi,
                    total,
                    first_a,
                    || &chunk[ds..de],
                    s,
                    &mut base_out,
                    &mut wp,
                    view,
                    lview,
                );
                doc_counts[di] += n;
            }
            unsafe { s.out.set_len(wp.offset_from(base_out) as usize) };
        };
        let mut p = 0usize;
        let mut bad_local: Option<usize> = None;
        let n = chunk.len();
        let k = sep.len();
        let sep1 = if k == 1 { Some(sep[0]) } else { None };
        loop {
            let m = match sep1 {
                Some(b0) => memchr::memchr(b0, &chunk[p..]),
                None => finder.find(&chunk[p..]),
            };
            let doc_end = match m {
                Some(rel) => p + rel,
                None => n,
            };
            let di = doc_counts.len();
            doc_counts.push(0);
            let doc_len = doc_end - p;
            if doc_len != 0 {
                let avail = &chunk[p..];
                let doc = &avail[..doc_len];
                if doc_len > u32::MAX as usize {
                    if !trusted && simdutf8::basic::from_utf8(doc).is_err() {
                        if bad_local.is_none() {
                            bad_local = Some(di);
                        }
                    } else {
                        flush(&mut buf, cnt, &ends[..cnt], segs, s, doc_counts, view, lview);
                        cnt = 0;
                        segs.clear();
                        let before = s.out.len();
                        self.encode_range(doc, 0, doc_len, s);
                        doc_counts[di] += s.out.len() - before;
                    }
                } else {
                    let mut scan = blockscan::Scan::new_checked(0, !trusted);
                    let mut a = 0usize;
                    loop {
                        if cnt == 0 {
                            view = self.learned.view();
                            lview = self.long_table.view();
                        }
                        let k1 = scan.fill_in(self.kind, avail, doc_len, doc_len, &mut ends, cnt, CAP);
                        if k1 == cnt {
                            break;
                        }
                        let first_a = a;
                        // Pass `avail` (the full macro-chunk tail) so `pack_wide_any`
                        // reads 16 bytes directly without calling `pack_tail` on the
                        // last 16 bytes of every short document!
                        a = Self::fill_recs(avail, &ends, cnt, k1, a, buf.as_mut_ptr(), view, lview);
                        segs.push((cnt, k1, di, first_a, p, doc_end));
                        cnt = k1;
                        if cnt >= CAP {
                            flush(&mut buf, cnt, &ends[..cnt], segs, s, doc_counts, view, lview);
                            cnt = 0;
                            segs.clear();
                        }
                        if a >= doc_len {
                            break;
                        }
                    }
                    if !scan.finish_utf8(doc, doc_len) {
                        if bad_local.is_none() {
                            bad_local = Some(di);
                        }
                        break;
                    }
                }
            }
            match m {
                Some(_) => p = doc_end + k,
                None => break,
            }
        }
        flush(&mut buf, cnt, &ends[..cnt], segs, s, doc_counts, view, lview);
        bad_local
    }

    /// Single-pass fused separator split + encode for one large buffer `raw`.
    /// Cuts `raw` into ~1 MiB macro-chunks at proven non-overlapping separator
    /// positions in ~0.05 ms, then each worker splits and encodes its macro-chunk
    /// in one L1/L2 pass (no up-front full-buffer `split_on`, no 250K `plan()`).
    fn encode_fused_sep_t<O: OutTok>(
        &self,
        raw: &[u8],
        sep: &[u8],
        trusted: bool,
        num_threads: usize,
        want_output: bool,
    ) -> Result<(Vec<O>, Vec<usize>), BadInput> {
        use std::sync::atomic::{AtomicUsize, Ordering::*};
        let n = raw.len();
        let k = sep.len();
        let nt = if num_threads == 0 { rayon::current_num_threads() } else { num_threads.min(1024) }.max(1);
        if nt <= 1 || !parallel_ok() {
            let mut doc_counts = Vec::new();
            let mut bad_local = None;
            let out = self.with_scratch(|s| {
                let saved = std::mem::take(&mut s.out);
                if want_output {
                    let cap = n / 2 + 16;
                    s.out.reserve(cap);
                    chunk_table::advise_hugepage(s.out.as_mut_ptr() as *mut u8, cap * 4);
                }
                let finder = memchr::memmem::Finder::new(sep);
                let mut segs = Vec::with_capacity(64);
                bad_local = if want_output {
                    self.encode_sep_macro::<true>(raw, sep, &finder, trusted, s, &mut segs, &mut doc_counts)
                } else {
                    self.encode_sep_macro::<false>(raw, sep, &finder, trusted, s, &mut segs, &mut doc_counts)
                };
                std::mem::replace(&mut s.out, saved)
            });
            if let Some(di) = bad_local {
                return Err(BadInput::Bytes(di));
            }
            if !want_output {
                return Ok((Vec::new(), doc_counts));
            }
            return Ok((O::from_u32(out), doc_counts));
        }

        // Partition `raw` into coarse macro-chunks at proven `bytes.split` cuts.
        let border_free = (1..k).all(|j| !sep[j..].starts_with(&sep[..k - j]));
        let finder = memchr::memmem::Finder::new(sep);
        let unit = (n / (nt * 16)).clamp(256 << 10, 2 << 20);
        let mut macros: Vec<(usize, usize)> = Vec::with_capacity(n / unit + 2);
        let mut cur = 0usize;
        while cur + unit < n {
            let mut search = cur + unit;
            let search_lim = (cur + 2 * unit).min(n);
            let mut cut = None;
            while search < search_lim {
                match finder.find(&raw[search..]) {
                    Some(rel) => {
                        let m = search + rel;
                        if border_free || m == 0 || !sep.contains(&raw[m - 1]) {
                            cut = Some(m);
                            break;
                        }
                        search = m + 1;
                    }
                    None => break,
                }
            }
            match cut {
                Some(m) => {
                    macros.push((cur, m));
                    cur = m + k;
                }
                None => break,
            }
        }
        macros.push((cur, n));
        let ntask = macros.len();
        if ntask <= 1 {
            // Very rare: single huge document with no safe separator cuts -> fall back
            // to `encode_src_t` so `split_points` parallelizes across `\n[A-Za-z]`.
            let docs: Vec<Src<'_>> = split_on(raw, sep, num_threads)
                .into_iter()
                .map(|(a, b)| {
                    if trusted {
                        Src::Str(unsafe { std::str::from_utf8_unchecked(&raw[a..b]) })
                    } else {
                        Src::Raw(&raw[a..b])
                    }
                })
                .collect();
            return self.encode_src_t::<O>(&docs, num_threads, want_output);
        }

        let macro_doc_counts: Vec<std::sync::Mutex<Vec<usize>>> =
            (0..ntask).map(|_| std::sync::Mutex::new(Vec::new())).collect();
        let macro_bad: Vec<AtomicUsize> = (0..ntask).map(|_| AtomicUsize::new(usize::MAX)).collect();
        let out = self.run_decoupled_tasks::<O, Vec<(usize, usize, usize, usize, usize, usize)>>(
            ntask,
            n * 2 / 3 + 4096,
            num_threads,
            nt,
            want_output,
            || Vec::with_capacity(64),
            |k, s, segs| {
                let (lo, hi) = macros[k];
                let mut lcounts = Vec::with_capacity(((hi - lo) >> 9).clamp(16, 4096));
                let bad_opt = if want_output {
                    self.encode_sep_macro::<true>(&raw[lo..hi], sep, &finder, trusted, s, segs, &mut lcounts)
                } else {
                    self.encode_sep_macro::<false>(&raw[lo..hi], sep, &finder, trusted, s, segs, &mut lcounts)
                };
                if let Some(bl) = bad_opt {
                    macro_bad[k].store(bl, Relaxed);
                }
                *macro_doc_counts[k].lock().unwrap_or_else(|e| e.into_inner()) = lcounts;
            },
        );
        // Stitch exact per-doc counts & check earliest UTF-8 error across macro-chunks.
        let mut total_docs = 0usize;
        let mut first_bad: Option<usize> = None;
        let parts: Vec<Vec<usize>> = macro_doc_counts
            .into_iter()
            .enumerate()
            .map(|(k, m)| {
                let v = m.into_inner().unwrap_or_else(|e| e.into_inner());
                let b = macro_bad[k].load(Relaxed);
                if b != usize::MAX && first_bad.is_none() {
                    first_bad = Some(total_docs + b);
                }
                total_docs += v.len();
                v
            })
            .collect();
        if let Some(di) = first_bad {
            return Err(BadInput::Bytes(di));
        }
        let mut doc_counts = Vec::with_capacity(total_docs);
        for v in parts {
            doc_counts.extend_from_slice(&v);
        }
        Ok((out, doc_counts))
    }

    /// `wide`: `wide_args(texts)` (UTF-8 copies of the non-ASCII `str` items),
    /// or empty when `stream_wide` is true (streaming L1/L2 WideStr transcoding).
    fn encode_batch_sep_t<O: OutTok>(
        &self,
        texts: &[TextArg],
        wide: &WideCopies,
        stream_wide: bool,
        num_threads: usize,
        want_output: bool,
        sep: Option<&[u8]>,
    ) -> Result<(Vec<O>, Vec<usize>), Result<BadInput, PyErr>> {
        if stream_wide {
            let src = texts_as_src_stream(texts);
            return self.encode_src_t::<O>(&src, num_threads, want_output).map_err(Ok);
        }
        let allow_raw = self.normalizer.is_identity() || self.parallel_unicode_norm();
        let src = texts_as_src(texts, wide, allow_raw).map_err(Err)?;
        // Single-pass fused separator split + encode for one large buffer.
        if let Some(sep) = sep {
            if sep.is_empty() {
                return Err(Err(PyValueError::new_err("separator must not be empty")));
            }
            if src.len() == 1 && self.normalizer.is_identity() && self.items_pipe_ok() {
                let (raw, trusted) = match src[0] {
                    Src::Str(s) => {
                        if std::str::from_utf8(sep).is_err() {
                            return Err(Err(PyValueError::new_err("separator must be valid UTF-8 for str inputs")));
                        }
                        (s.as_bytes(), true)
                    }
                    Src::Raw(b) => (b, false),
                    Src::Wide(_) => unreachable!(),
                };
                if raw.len() >= 1 << 18 {
                    return self.encode_fused_sep_t::<O>(raw, sep, trusted, num_threads, want_output).map_err(Ok);
                }
            }
        }
        let src = match sep {
            Some(sep) => {
                let mut docs = Vec::new();
                for t in &src {
                    match *t {
                        Src::Str(s) => {
                            if std::str::from_utf8(sep).is_err() {
                                return Err(Err(PyValueError::new_err("separator must be valid UTF-8 for str inputs")));
                            }
                            docs.extend(split_on(s.as_bytes(), sep, num_threads).into_iter().map(|(a, b)| Src::Str(unsafe { std::str::from_utf8_unchecked(&s.as_bytes()[a..b]) })));
                        }
                        Src::Raw(b) => docs.extend(split_on(b, sep, num_threads).into_iter().map(|(a, e)| Src::Raw(&b[a..e]))),
                        Src::Wide(_) => unreachable!(),
                    }
                }
                docs
            }
            None => src,
        };
        self.encode_src_t::<O>(&src, num_threads, want_output).map_err(Ok)
    }

    fn encode_direct<T: AsRef<str> + Sync>(&self, texts: &[T], num_threads: usize, want_output: bool) -> (Vec<u32>, Vec<usize>) {
        let src: Vec<Src<'_>> = texts.iter().map(|t| Src::Str(t.as_ref())).collect();
        self.encode_src(&src, num_threads, want_output).expect("str inputs are valid UTF-8")
    }

    /// The batch engine over `Src` inputs.
    fn encode_src(&self, texts: &[Src<'_>], num_threads: usize, want_output: bool) -> Result<(Vec<u32>, Vec<usize>), BadInput> {
        self.encode_src_t::<u32>(texts, num_threads, want_output)
    }

    fn encode_src_t<O: OutTok>(&self, texts: &[Src<'_>], num_threads: usize, want_output: bool) -> Result<(Vec<O>, Vec<usize>), BadInput> {
        use std::sync::atomic::{AtomicUsize, Ordering::*};
        let defer_norm = self.parallel_unicode_norm();
        let (norm, items, tasks) = self.plan(texts, num_threads);
        let trusted: Vec<bool> = texts
            .iter()
            .map(|t| matches!(t, Src::Str(_) | Src::Wide(_)) || (!self.normalizer.is_identity() && !defer_norm))
            .collect();
        let bad_doc = AtomicUsize::new(usize::MAX);
        let bad_wide = AtomicUsize::new(usize::MAX);
        let ntask = tasks.len();
        let mut doc_counts = vec![0usize; texts.len()];
        let nt = if num_threads == 0 { rayon::current_num_threads() } else { num_threads.min(1024) };
        let piped = self.items_pipe_ok();

        if nt <= 1 || ntask <= 1 || !parallel_ok() {
            // serial: encode straight into the output (zero-copy for u32!)
            let out = self.with_scratch(|s| {
                let saved = std::mem::take(&mut s.out);
                let total_bytes: usize = tasks.iter().map(|t| t.2).sum();
                if want_output {
                    let cap = total_bytes / 2 + 16;
                    s.out.reserve(cap);
                    chunk_table::advise_hugepage(s.out.as_mut_ptr() as *mut u8, cap * 4);
                }
                if piped {
                    let mut ic: Vec<u32> = Vec::new();
                    for &(x, y, _) in &tasks {
                        ic.clear();
                        ic.resize(y - x, 0);
                        if want_output {
                            self.encode_items_piped::<true>(
                                texts,
                                &norm,
                                &trusted,
                                &items[x..y],
                                s,
                                &mut ic,
                                &mut |d| { bad_doc.fetch_min(d as usize, Relaxed); },
                                &mut |d| { bad_wide.fetch_min(d as usize, Relaxed); },
                            );
                        } else {
                            self.encode_items_piped::<false>(
                                texts,
                                &norm,
                                &trusted,
                                &items[x..y],
                                s,
                                &mut ic,
                                &mut |d| { bad_doc.fetch_min(d as usize, Relaxed); },
                                &mut |d| { bad_wide.fetch_min(d as usize, Relaxed); },
                            );
                        }
                        for (ii, &c) in ic.iter().enumerate() {
                            doc_counts[items[x + ii].0 as usize] += c as usize;
                        }
                        if !want_output {
                            s.out.clear();
                        }
                    }
                } else {
                    for &it in &items {
                        let before = s.out.len();
                        if !self.encode_item(&norm, &trusted, it, s) {
                            bad_doc.fetch_min(it.0 as usize, Relaxed);
                        }
                        doc_counts[it.0 as usize] += s.out.len() - before;
                        if !want_output {
                            s.out.clear();
                        }
                    }
                }
                std::mem::replace(&mut s.out, saved)
            });
            let bw = bad_wide.load(Relaxed);
            if bw != usize::MAX {
                return Err(BadInput::Wide(bw));
            }
            let bad = bad_doc.load(Relaxed);
            if bad != usize::MAX {
                return Err(BadInput::Bytes(bad));
            }
            if !want_output {
                return Ok((Vec::new(), doc_counts));
            }
            return Ok((O::from_u32(out), doc_counts));
        }

        let total_bytes: usize = tasks.iter().map(|t| t.2).sum();
        struct IcPtr(*mut u32);
        // SAFETY: tasks write disjoint `x..y` item ranges through `ic_ptr`.
        unsafe impl Sync for IcPtr {}
        impl IcPtr {
            fn get(&self) -> *mut u32 { self.0 }
        }
        let mut item_counts = vec![0u32; items.len()];
        let ic_ptr = IcPtr(item_counts.as_mut_ptr());
        let out = self.run_decoupled_tasks::<O, ()>(
            ntask,
            total_bytes * 2 / 3 + 4096,
            num_threads,
            nt,
            want_output,
            || (),
            |k, s, _| {
                let (x, y, _) = tasks[k];
                if piped {
                    // SAFETY: items x..y belong to task k only.
                    let ic = unsafe { std::slice::from_raw_parts_mut(ic_ptr.get().add(x), y - x) };
                    if want_output {
                        self.encode_items_piped::<true>(
                            texts,
                            &norm,
                            &trusted,
                            &items[x..y],
                            s,
                            ic,
                            &mut |d| { bad_doc.fetch_min(d as usize, Relaxed); },
                            &mut |d| { bad_wide.fetch_min(d as usize, Relaxed); },
                        );
                    } else {
                        self.encode_items_piped::<false>(
                            texts,
                            &norm,
                            &trusted,
                            &items[x..y],
                            s,
                            ic,
                            &mut |d| { bad_doc.fetch_min(d as usize, Relaxed); },
                            &mut |d| { bad_wide.fetch_min(d as usize, Relaxed); },
                        );
                    }
                } else {
                    for (ii, &it) in items[x..y].iter().enumerate() {
                        let before = s.out.len();
                        if !self.encode_item(&norm, &trusted, it, s) {
                            bad_doc.fetch_min(it.0 as usize, Relaxed);
                        }
                        // SAFETY: item x+ii belongs to task k only.
                        unsafe { *ic_ptr.get().add(x + ii) = (s.out.len() - before) as u32 };
                    }
                }
            },
        );
        for (k, it) in items.iter().enumerate() {
            doc_counts[it.0 as usize] += item_counts[k] as usize;
        }
        let bw = bad_wide.load(Relaxed);
        if bw != usize::MAX {
            return Err(BadInput::Wide(bw));
        }
        let bad = bad_doc.load(Relaxed);
        if bad != usize::MAX {
            return Err(BadInput::Bytes(bad));
        }
        Ok((out, doc_counts))
    }

    /// DeepSeek CJK streaming 256-chunk pipelined encoder over `bytes[start..stop]`
    /// (Merge::Auto, no EOW suffix). Single-pass zero-alloc `cjk::CjkScan` feeding
    /// the same `fill_recs` / `flush_recs` L1-prefetched `SharedTable`+`SharedLongTable`
    /// pipeline as `BlockScan`.
    fn encode_cjk_range(&self, bytes: &[u8], start: usize, stop: usize, s: &mut Scratch) {
        let n = bytes.len();
        if stop <= start {
            return;
        }
        if stop - start >= 512 && n <= u32::MAX as usize {
            const CAP: usize = 256;
            let mut ends = [0u32; CAP + blockscan::SCAN_SLACK];
            let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); CAP + blockscan::SCAN_SLACK + X86_PF_DIST];
            let mut scan = cjk::CjkScan::new(start);
            let mut a = start;
            loop {
                let k = scan.fill(bytes, stop, &mut ends, 0, CAP);
                if k == 0 {
                    break;
                }
                let view = self.learned.view();
                let lview = self.long_table.view();
                let first_a = a;
                a = Self::fill_recs(bytes, &ends, 0, k, a, buf.as_mut_ptr(), view, lview);
                Self::prime_pipe_prefetch(&mut buf, k);
                self.flush_recs(bytes, n, buf.as_ptr() as *const PipeRec, k, &ends[..k], first_a, s, view, lview);
            }
            return;
        }
        if n <= u32::MAX as usize {
            let mut ends = [0u32; 64];
            let mut scan = cjk::CjkScan::new(start);
            let mut a = start;
            loop {
                let k = scan.fill(bytes, stop, &mut ends, 0, 64);
                if k == 0 {
                    break;
                }
                for &e in &ends[..k] {
                    let eu = e as usize;
                    self.emit_chunk(bytes, a, eu, n, s);
                    a = eu;
                }
            }
            return;
        }
        // > 4 GiB fallback
        let sub = unsafe { std::str::from_utf8_unchecked(&bytes[start..stop]) };
        let mut cur = std::mem::take(&mut s.cjk_cur);
        let mut nxt = std::mem::take(&mut s.cjk_nxt);
        cjk::chunk_ranges(sub, &mut cur, &mut nxt);
        for &(ps, pe) in &cur {
            self.emit_chunk(bytes, start + ps, start + pe, n, s);
        }
        s.cjk_cur = cur;
        s.cjk_nxt = nxt;
    }

    /// DeepSeek path: 1-pass streaming chunker + 256-deep pipelined `SharedTable`
    /// under `Merge::Auto`, falling back to `chunk_ranges` for explicit merge modes.
    fn encode_cjk(&self, text: &str, s: &mut Scratch, mode: Merge) {
        let bytes = text.as_bytes();
        if mode == Merge::Auto && self.singleton_eow.is_none() {
            self.encode_cjk_range(bytes, 0, bytes.len(), s);
            return;
        }
        let mut cur = std::mem::take(&mut s.cjk_cur);
        let mut nxt = std::mem::take(&mut s.cjk_nxt);
        cjk::chunk_ranges(text, &mut cur, &mut nxt);
        for &(ps, pe) in &cur {
            self.emit(&bytes[ps..pe], s, mode);
        }
        s.cjk_cur = cur;
        s.cjk_nxt = nxt;
    }

    /// BPE one ByteLevel chunk while tracking each token's byte span in the input.
    #[allow(clippy::too_many_arguments)]
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
        if self.ignore_merges {
            if !self.byte_fallback
                && self.singleton_eow.is_none()
                && chunk.len() <= chunk_table::MAX_KEY
            {
                let (lo, hi) = chunk_table::pack(chunk);
                if let Some(id) = self.chunk_table.get(lo, hi) {
                    ids.push(id);
                    spans.push((cs, ce));
                    return;
                }
            } else if let Some(map) = &self.ignore_lookup {
                if let Some(&id) = map.get(chunk) {
                    ids.push(id);
                    spans.push((cs, ce));
                    return;
                }
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
        // Long single documents: split at proven chunk boundaries and encode the
        // segments in parallel (nests fine inside a rayon batch — work stealing).
        if mode == Merge::Auto && text.len() >= SPLIT_MIN && self.splittable() {
            return self.encode_split(text);
        }
        SCRATCH.with(|cell| match cell.try_borrow_mut() {
            Ok(mut s) => {
                s.dyn_cache.claim(self.uid);
                s.long_cache.claim(self.uid);
                let need = text.len() / 3 + 8;
                if s.out.capacity() < need {
                    s.out = Vec::with_capacity(need);
                } else {
                    s.out.clear();
                }
                self.encode_into(text, mode, prepend, &mut s);
                std::mem::take(&mut s.out)
            }
            // re-entrant use on this thread (not expected): fall back to fresh scratch
            Err(_) => {
                let mut s = Scratch::default();
                self.encode_into(text, mode, prepend, &mut s);
                s.out
            }
        })
    }

    fn encode_into(&self, text: &str, mode: Merge, prepend: bool, s: &mut Scratch) {
        if mode == Merge::Auto && self.sp_fast_meta != SpFastMeta::None && self.encode_sp_fast(text, prepend, true, s) {
            return;
        }
        // The normalizer runs first in the HF pipeline (normalizer -> pre_tokenizer
        // -> model), for ByteLevel models too — not just SentencePiece. Skipping it
        // silently mistokenizes decomposed/combining Unicode (NFD text, jamo, Å, …)
        // on every model that declares one (a large share of popular models).
        let norm = self.normalize(text);
        self.encode_into_norm(norm.as_ref(), mode, prepend, s);
    }

    /// Pipelined 256-deep L1-prefetched `SharedTable`+`SharedLongTable` encoder
    /// for any `slice_only` `PretokPipe` model (e.g. Falcon3, Falcon-H1). Streams
    /// L1-hot windows of byte spans through `pack_wide` + `prefetch` + `flush_recs_spans`.
    fn encode_pipe_piped(&self, text: &str, pipe: &pretok_pipe::PretokPipe, s: &mut Scratch) {
        const CAP: usize = 256;
        let bytes = text.as_bytes();
        let n = bytes.len();
        if n == 0 {
            return;
        }
        let bp = bytes.as_ptr();
        let mut ps = std::mem::take(&mut s.pipe);
        let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); CAP];
        let mut spans_buf = [(0usize, 0usize); CAP];
        let mut cnt = 0usize;
        let mut view = self.learned.view();
        pipe.for_each_slice_spans(text, &mut ps, |spans| {
            for &(a, e) in spans {
                if cnt == 0 {
                    view = self.learned.view();
                }
                let len = e - a;
                let v = if len.wrapping_sub(1) < chunk_table::MAX_KEY {
                    let (lo, hi) = if a + 16 <= n {
                        unsafe { chunk_table::pack_wide(bp.add(a), len) }
                    } else {
                        chunk_table::pack(&bytes[a..e])
                    };
                    let p = view.slot_addr(chunk_table::hash(lo, hi));
                    chunk_table::prefetch(p as *const u8);
                    PipeRec { lo, hi, p }
                } else {
                    PipeRec { lo: 0, hi: 0, p: 0 }
                };
                buf[cnt].write(v);
                spans_buf[cnt] = (a, e);
                cnt += 1;
                if cnt == CAP {
                    // SAFETY: buf[..CAP] initialized.
                    let sl = unsafe { &*(&buf[..CAP] as *const _ as *const [PipeRec]) };
                    self.flush_recs_spans(bytes, n, sl, &spans_buf[..CAP], s, view);
                    cnt = 0;
                }
            }
        });
        if cnt > 0 {
            let sl = unsafe { &*(&buf[..cnt] as *const _ as *const [PipeRec]) };
            self.flush_recs_spans(bytes, n, sl, &spans_buf[..cnt], s, view);
        }
        s.pipe = ps;
    }

    /// `encode_into` for text that is ALREADY normalized. Appends to `s.out`.
    fn encode_into_norm(&self, text: &str, mode: Merge, prepend: bool, s: &mut Scratch) {
        match self.fast_pretok {
            FastPretok::BlockScan => {
                self.encode_bytes(text.as_bytes(), s, mode);
            }
            FastPretok::Cjk => {
                self.encode_cjk(text, s, mode);
            }
            FastPretok::Pipe => {
                let pipe = self.pretok_pipe.as_ref().expect("Pipe mode has pretok_pipe");
                let use_shared =
                    mode == Merge::Auto && !self.byte_fallback && self.singleton_eow.is_none();
                let split_ms = self.safe_meta_split;
                if use_shared && split_ms.is_none() && pipe.is_slice_only() {
                    self.encode_pipe_piped(text, pipe, s);
                    return;
                }
                let mut ps = std::mem::take(&mut s.pipe);
                pipe.for_each_chunk(text, prepend, &self.c2b, &mut ps, |chunk| {
                    if let Some(rep) = split_ms {
                        if let Ok(cs) = std::str::from_utf8(chunk) {
                            self.encode_meta_char(cs, rep, s, mode);
                            return;
                        }
                    }
                    if use_shared {
                        self.emit_chunk(chunk, 0, chunk.len(), chunk.len(), s);
                    } else {
                        self.emit(chunk, s, mode);
                    }
                });
                s.pipe = ps;
            }
            FastPretok::WholeText => {
                if let Some(rep) = self.safe_meta_split {
                    self.encode_meta_char(text, rep, s, mode);
                } else {
                    self.emit(text.as_bytes(), s, mode);
                }
            }
        }
    }

    /// BPE one inter-special fragment directly into `s.out` (zero extra `Vec`
    /// allocation or `SCRATCH` re-borrow). `at_start` => the fragment begins at
    /// byte 0 (gets the metaspace prefix under scheme "first").
    #[inline]
    fn encode_fragment_into(&self, frag: &str, at_start: bool, s: &mut Scratch) {
        if frag.is_empty() {
            return;
        }
        let prepend = self.prepend_scheme != 2 || at_start;
        if frag.len() >= SPLIT_MIN && self.splittable() && (at_start || self.sp_fast_meta == SpFastMeta::None) {
            s.out.extend_from_slice(&self.encode_split(frag));
        } else {
            self.encode_into(frag, Merge::Auto, prepend, s);
        }
    }

    /// Special-token-aware encode of one string appended directly into `s.out`.
    #[inline]
    fn encode_special_into(&self, text: &str, s: &mut Scratch) {
        let Some(ac) = &self.special_ac else {
            self.encode_fragment_into(text, true, s);
            return;
        };
        let Some(first) = self.special_prefilter.find_first(text.as_bytes()) else {
            self.encode_fragment_into(text, true, s);
            return;
        };
        debug_assert!(text.is_char_boundary(first));
        let n = text.len();
        let mut last = 0usize;
        for m in ac.find_iter(&text[first..]) {
            let ms = first + m.start();
            let me = first + m.end();
            if ms < last {
                continue;
            }
            let pat = m.pattern().as_usize();
            let (mut start, mut end) = (ms, me);
            if self.special_lstrip[pat] {
                start = last + text[last..start].trim_end().len();
            }
            if self.special_rstrip[pat] {
                end = n - text[end..].trim_start().len();
            }
            if start > last {
                self.encode_fragment_into(&text[last..start], last == 0, s);
            }
            s.out.push(self.special_ids[pat]);
            last = end;
        }
        if last < n {
            self.encode_fragment_into(&text[last..], last == 0, s);
        }
    }

    /// Special-token-aware encode of one string (no GIL handling) — shared by the
    /// `encode_special` and `encode_special_batch` entry points. Uses `SpecialPrefilter`
    /// to skip Aho-Corasick completely on plain text, and borrows `SCRATCH` once to
    /// write all fragments + special IDs directly into one output `Vec`, honoring
    /// Unicode `char::is_whitespace()` for HF `AddedToken` `lstrip`/`rstrip`.
    fn encode_special_one(&self, text: &str) -> Vec<u32> {
        if self.special_ac.is_none() || self.special_prefilter.find_first(text.as_bytes()).is_none()
        {
            return self.encode_one(text, Merge::Auto);
        }
        let n = text.len();
        let run = |s: &mut Scratch| -> Vec<u32> {
            let saved = std::mem::take(&mut s.out);
            s.out = Vec::with_capacity(n / 3 + 8);
            self.encode_special_into(text, s);
            std::mem::replace(&mut s.out, saved)
        };
        SCRATCH.with(|cell| match cell.try_borrow_mut() {
            Ok(mut s) => {
                s.dyn_cache.claim(self.uid);
                s.long_cache.claim(self.uid);
                run(&mut s)
            }
            Err(_) => {
                let mut s = Scratch::default();
                run(&mut s)
            }
        })
    }

    /// Try to encode the EXACT first `min(full_len, max_tokens)` ordinary tokens
    /// of `bytes` in O(`max_tokens`) time without touching the rest of the input.
    ///
    /// Returns `Some(consumed_byte_end)` iff `s.out[start_out..]` is mathematically
    /// proven 100% identical to `full_ordinary_encode(bytes)[..emitted]` and either
    /// `consumed_byte_end == bytes.len()` or `emitted >= max_tokens`. On any edge
    /// case that cannot be proven locally (e.g. non-ASCII inside the prefix of an
    /// NFC model, or a literal `'▁'` inside the prefix of a SentencePiece model),
    /// restores `s.out` and returns `None` so the caller falls back to full encode.
    fn encode_ordinary_prefix_try(
        &self,
        bytes: &[u8],
        max_tokens: usize,
        prepend: bool,
        is_doc_start: bool,
        s: &mut Scratch,
    ) -> Option<usize> {
        let n = bytes.len();
        if n == 0 || max_tokens == 0 {
            return Some(0);
        }
        if n > u32::MAX as usize {
            return None;
        }
        let start_out = s.out.len();

        // 1. SentencePiece fast path (`SpFastMeta`, e.g. Mistral, Gemma-2/3, Phi-3.5, Yi-1.5).
        if self.sp_fast_meta != SpFastMeta::None {
            let need_prepend = match self.sp_fast_meta {
                SpFastMeta::NormPrependAlways => is_doc_start,
                SpFastMeta::NormNoPrepend => false,
                SpFastMeta::Metaspace { prepend_scheme } => {
                    let allow = match prepend_scheme {
                        1 => is_doc_start,
                        2 => is_doc_start && prepend,
                        _ => false,
                    };
                    allow && bytes[0] != b' '
                }
                SpFastMeta::None => unreachable!(),
            };
            let cross = &self.sp_cross_prev;
            let has_cross = self.sp_has_cross;
            let is_safe_split = |p: usize| -> bool {
                if p == 0 || p >= n || bytes[p] != b' ' {
                    return false;
                }
                let prev = bytes[p - 1];
                prev < 0x80 && prev != b' ' && ((cross[(prev as usize) >> 6] >> (prev & 63)) & 1) == 0
            };

            let e0 = Self::next_sp_end(bytes, 0, cross, has_cross);
            if bytes[..e0].contains(&0xE2) && memchr::memmem::find(&bytes[..e0], "\u{2581}".as_bytes()).is_some() {
                return None;
            }
            if need_prepend {
                if e0 < 63 {
                    let mut stack = [0u8; 64];
                    stack[0] = b' ';
                    stack[1..1 + e0].copy_from_slice(&bytes[..e0]);
                    self.emit_sp_chunk_slice(&stack[..1 + e0], s);
                } else {
                    let mut v = Vec::with_capacity(1 + e0);
                    v.push(b' ');
                    v.extend_from_slice(&bytes[..e0]);
                    self.emit_sp_chunk_slice(&v, s);
                }
            } else {
                self.emit_sp_chunk_slice(&bytes[..e0], s);
            }
            let mut pos = e0;
            // Continue until we have >= max_tokens AND sit at a proven-safe boundary
            // (so any literal `'▁'` later in the document cannot walk back past `pos`).
            const PCAP: usize = 64;
            let mut ends = [0u32; PCAP];
            let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); PCAP + X86_PF_DIST];
            while pos < n && (s.out.len() - start_out < max_tokens || !is_safe_split(pos)) {
                let first_a = pos;
                let mut k = 0usize;
                let want_chunks = if s.out.len() - start_out < max_tokens {
                    (max_tokens - (s.out.len() - start_out)).clamp(8, PCAP)
                } else {
                    1
                };
                while pos < n && k < want_chunks {
                    let next = Self::next_sp_end(bytes, pos + 1, cross, has_cross);
                    ends[k] = next as u32;
                    k += 1;
                    pos = next;
                }
                if bytes[first_a..pos].contains(&0xE2)
                    && memchr::memmem::find(&bytes[first_a..pos], "\u{2581}".as_bytes()).is_some()
                {
                    s.out.truncate(start_out);
                    return None;
                }
                if k >= 16 {
                    let view = self.learned.view();
                    let lview = self.long_table.view();
                    Self::fill_recs(bytes, &ends, 0, k, first_a, buf.as_mut_ptr(), view, lview);
                    Self::prime_pipe_prefetch(&mut buf, k);
                    self.flush_recs_sp(bytes, buf.as_ptr() as *const PipeRec, k, &ends[..k], first_a, s, view, lview);
                } else {
                    let mut a = first_a;
                    for &e in &ends[..k] {
                        let eu = e as usize;
                        self.emit_sp_chunk_slice(&bytes[a..eu], s);
                        a = eu;
                    }
                }
            }
            return Some(pos);
        }

        // 2. ByteLevel `BlockScan` (GPT-2, cl100k, o200k, Llama-3/4, Qwen2.5/3, SmolLM3, gpt-oss, ...).
        if self.items_pipe_ok() && (self.normalizer.is_identity() || self.normalizer.is_ascii_identity()) {
            const PCAP: usize = 64;
            let mut ends = [0u32; PCAP + blockscan::SCAN_SLACK];
            let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); PCAP + blockscan::SCAN_SLACK + X86_PF_DIST];
            // For Unicode normalizers (NFC/NFD/NFKC/NFKD, e.g. Qwen2.5/3/3.6, MiniMax),
            // check if a small prefix is non-ASCII up front to avoid scanning twice.
            // Any ASCII byte `< 0x80` is a starter (ccc=0) that never composes backward,
            // so `Norm(bytes[..n]) == Norm(bytes[..cut]) + Norm(bytes[cut..])` exactly.
            let pre_est = max_tokens.saturating_mul(6).saturating_add(256).min(n);
            if !self.normalizer.is_identity()
                && self.normalizer.is_unicode_only()
                && !bytes[..pre_est].is_ascii()
            {
                let mut cut = pre_est;
                let search_lim = pre_est.saturating_add(4096).min(n);
                while cut < search_lim && bytes[cut] >= 0x80 {
                    cut += 1;
                }
                if cut < n && bytes[cut] >= 0x80 {
                    return None;
                }
                let prefix_str = std::str::from_utf8(&bytes[..cut]).ok()?;
                let norm = self.normalizer.apply(prefix_str);
                let nb = norm.as_bytes();
                let nn = nb.len();
                let mut scan = blockscan::Scan::new(0);
                let mut na = 0usize;
                while na < nn && s.out.len() - start_out < max_tokens {
                    let rem = max_tokens - (s.out.len() - start_out);
                    let lim = rem.clamp(16, PCAP);
                    let k = scan.fill(self.kind, nb, nn, &mut ends, 0, lim);
                    if k == 0 {
                        break;
                    }
                    let view = self.learned.view();
                    let lview = self.long_table.view();
                    let first_a = na;
                    na = Self::fill_recs(nb, &ends, 0, k, na, buf.as_mut_ptr(), view, lview);
                    Self::prime_pipe_prefetch(&mut buf, k);
                    self.flush_recs(nb, nn, buf.as_ptr() as *const PipeRec, k, &ends[..k], first_a, s, view, lview);
                }
                if cut == n {
                    return Some(n);
                }
                if s.out.len() - start_out >= max_tokens {
                    let mut nq = na;
                    while nq < nn && nb[nq].is_ascii_whitespace() {
                        nq += 1;
                    }
                    if nq + 4 <= nn {
                        return Some(cut);
                    }
                }
                s.out.truncate(start_out);
                return None;
            }
            let mut scan = blockscan::Scan::new(0);
            let mut a = 0usize;
            while a < n && s.out.len() - start_out < max_tokens {
                let rem = max_tokens - (s.out.len() - start_out);
                let lim = rem.clamp(16, PCAP);
                let k = scan.fill(self.kind, bytes, n, &mut ends, 0, lim);
                if k == 0 {
                    break;
                }
                let view = self.learned.view();
                let lview = self.long_table.view();
                let first_a = a;
                a = Self::fill_recs(bytes, &ends, 0, k, a, buf.as_mut_ptr(), view, lview);
                Self::prime_pipe_prefetch(&mut buf, k);
                self.flush_recs(bytes, n, buf.as_ptr() as *const PipeRec, k, &ends[..k], first_a, s, view, lview);
            }
            if !self.normalizer.is_identity() {
                // Prove `normalizer(text)` starts identically through `a` + whitespace/contraction
                // lookahead: pure ASCII through 4 bytes past any whitespace run after `a`.
                let mut q = a;
                while q < n && bytes[q].is_ascii_whitespace() {
                    q += 1;
                }
                let guard_end = (q + 4).min(n);
                if !bytes[..guard_end].is_ascii() {
                    s.out.truncate(start_out);
                    return None;
                }
            }
            return Some(a);
        }

        // 3. DeepSeek-V3/R1 CJK streaming scanner (`FastPretok::Cjk`, identity normalizer).
        if self.fast_pretok == FastPretok::Cjk
            && self.normalizer.is_identity()
            && !self.byte_fallback
            && self.singleton_eow.is_none()
        {
            const PCAP: usize = 64;
            let mut ends = [0u32; PCAP + blockscan::SCAN_SLACK];
            let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); PCAP + blockscan::SCAN_SLACK + X86_PF_DIST];
            let mut scan = cjk::CjkScan::new(0);
            let mut a = 0usize;
            while a < n && s.out.len() - start_out < max_tokens {
                let rem = max_tokens - (s.out.len() - start_out);
                let lim = rem.clamp(16, PCAP);
                let k = scan.fill(bytes, n, &mut ends, 0, lim);
                if k == 0 {
                    break;
                }
                let view = self.learned.view();
                let lview = self.long_table.view();
                let first_a = a;
                a = Self::fill_recs(bytes, &ends, 0, k, a, buf.as_mut_ptr(), view, lview);
                Self::prime_pipe_prefetch(&mut buf, k);
                self.flush_recs(bytes, n, buf.as_ptr() as *const PipeRec, k, &ends[..k], first_a, s, view, lview);
            }
            return Some(a);
        }

        None
    }

    /// Encode at most the first `max_tokens` ordinary tokens of `text` into `s.out`,
    /// using O(`max_tokens`) early exit when `text` is much longer than `max_tokens`.
    #[inline]
    fn encode_ordinary_prefix_into(&self, text: &str, max_tokens: usize, s: &mut Scratch) {
        if max_tokens == 0 || text.is_empty() {
            return;
        }
        let start_out = s.out.len();
        if text.len() > max_tokens.saturating_mul(6).saturating_add(128)
            && self.encode_ordinary_prefix_try(text.as_bytes(), max_tokens, true, true, s).is_some()
        {
            if s.out.len() > start_out + max_tokens {
                s.out.truncate(start_out + max_tokens);
            }
            return;
        }
        if text.len() >= SPLIT_MIN && self.splittable() {
            s.out.extend_from_slice(&self.encode_split(text));
        } else {
            self.encode_into(text, Merge::Auto, true, s);
        }
        if s.out.len() > start_out + max_tokens {
            s.out.truncate(start_out + max_tokens);
        }
    }

    /// Exact special-token-aware encode of at most the first `max_tokens` tokens
    /// of `text` into `s.out` in O(`max_tokens`) time (even on 100 MB documents
    /// prefixed with chat special tokens like `<|im_start|>user\n...`).
    fn encode_special_prefix_into(&self, text: &str, max_tokens: usize, s: &mut Scratch) {
        if max_tokens == 0 || text.is_empty() {
            return;
        }
        let start_out = s.out.len();
        if text.len() <= max_tokens.saturating_mul(6).saturating_add(128) {
            self.encode_special_into(text, s);
            if s.out.len() > start_out + max_tokens {
                s.out.truncate(start_out + max_tokens);
            }
            return;
        }
        let Some(ac) = &self.special_ac else {
            self.encode_ordinary_prefix_into(text, max_tokens, s);
            return;
        };
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut last = 0usize;
        while last < n && s.out.len() - start_out < max_tokens {
            let rem = max_tokens - (s.out.len() - start_out);
            let before_gap = s.out.len();
            let at_start = last == 0;
            let prepend = self.prepend_scheme != 2 || at_start;
            if let Some(rel_a) = self.encode_ordinary_prefix_try(&bytes[last..], rem, prepend, true, s) {
                let abs_a = last + rel_a;
                // Advance past any whitespace run after `abs_a`: no special token starting
                // strictly after `abs_q` can `lstrip` across the non-whitespace char at `abs_q`.
                let ws_len = text[abs_a..].len() - text[abs_a..].trim_start().len();
                let abs_q = abs_a + ws_len;
                // `+ 2` so 2-byte prefilters (`Prefix2` / `Prefix1Mask`) see a needle starting at `abs_q`.
                let pf_end = (abs_q + 2).min(n);
                if self.special_prefilter.find_first(&bytes[last..pf_end]).is_none() {
                    if s.out.len() > start_out + max_tokens {
                        s.out.truncate(start_out + max_tokens);
                    }
                    return;
                }
                // Check Aho-Corasick ONLY on the local window `last..win_end` covering
                // any match starting at or before `abs_q` (preserving LeftmostLongest).
                let mut win_end = abs_q.saturating_add(self.special_max_len).saturating_add(1).min(n);
                while win_end < n && !text.is_char_boundary(win_end) {
                    win_end += 1;
                }
                match ac.find(&text[last..win_end]) {
                    None => {
                        if s.out.len() > start_out + max_tokens {
                            s.out.truncate(start_out + max_tokens);
                        }
                        return;
                    }
                    Some(m) if last + m.start() > abs_q => {
                        if s.out.len() > start_out + max_tokens {
                            s.out.truncate(start_out + max_tokens);
                        }
                        return;
                    }
                    Some(m) => {
                        s.out.truncate(before_gap);
                        let pat = m.pattern().as_usize();
                        let mut start = last + m.start();
                        let mut end = last + m.end();
                        if self.special_lstrip[pat] {
                            start = last + text[last..start].trim_end().len();
                        }
                        if self.special_rstrip[pat] {
                            end = n - text[end..].trim_start().len();
                        }
                        if start > last {
                            self.encode_fragment_into(&text[last..start], at_start, s);
                        }
                        if s.out.len() - start_out >= max_tokens {
                            s.out.truncate(start_out + max_tokens);
                            return;
                        }
                        s.out.push(self.special_ids[pat]);
                        last = end;
                        continue;
                    }
                }
            }
            // Fallback for the remaining tail when `encode_ordinary_prefix_try` declined
            // (e.g. non-ASCII inside NFC prefix, literal `'▁'` in SP prefix, or Pipe model).
            if last == 0 {
                self.encode_special_into(text, s);
            } else {
                for m in ac.find_iter(&text[last..]) {
                    let ms = last + m.start();
                    let me = last + m.end();
                    if ms < last {
                        continue;
                    }
                    let pat = m.pattern().as_usize();
                    let (mut start, mut end) = (ms, me);
                    if self.special_lstrip[pat] {
                        start = last + text[last..start].trim_end().len();
                    }
                    if self.special_rstrip[pat] {
                        end = n - text[end..].trim_start().len();
                    }
                    if start > last {
                        self.encode_fragment_into(&text[last..start], false, s);
                    }
                    if s.out.len() - start_out >= max_tokens {
                        break;
                    }
                    s.out.push(self.special_ids[pat]);
                    last = end;
                    if s.out.len() - start_out >= max_tokens {
                        break;
                    }
                }
                if last < n && s.out.len() - start_out < max_tokens {
                    self.encode_fragment_into(&text[last..], false, s);
                }
            }
            break;
        }
        if s.out.len() > start_out + max_tokens {
            s.out.truncate(start_out + max_tokens);
        }
    }

    /// Search backward in `lo..=hint` (`0 < p < b.len()`) for the rightmost byte
    /// offset `p` proven to be an exact ordinary chunk & normalizer boundary:
    /// `encode(text) == encode(text[..p]) || encode_nonfirst(text[p..])`.
    fn find_safe_ordinary_split_before(&self, b: &[u8], hint: usize, lo: usize) -> Option<usize> {
        let n = b.len();
        if n < 4 || hint == 0 || lo >= hint || hint >= n {
            return None;
        }
        // 1. SentencePiece (`SpFastMeta`, e.g. Gemma-2/3, Mistral, Phi-3.5, Yi-1.5).
        if self.sp_fast_meta != SpFastMeta::None {
            let cross = &self.sp_cross_prev;
            let mut hi = hint + 1;
            let min_p = lo.max(1);
            while hi > min_p {
                let rel = memchr::memrchr(b' ', &b[min_p..hi])?;
                let p = min_p + rel;
                let prev = b[p - 1];
                if prev < 0x80 && prev != b' ' && ((cross[(prev as usize) >> 6] >> (prev & 63)) & 1) == 0 {
                    return Some(p);
                }
                hi = p;
            }
            return None;
        }
        if self.singleton_eow.is_some()
            || !(self.normalizer.is_identity() || self.normalizer.is_unicode_only())
        {
            return None;
        }
        let id_norm = self.normalizer.is_identity();
        // 2. DeepSeek-V3/R1 CJK (`\n` + alphanumeric or CJK lead byte).
        if self.fast_pretok == FastPretok::Cjk {
            let mut hi = hint;
            let min_nl = lo.saturating_sub(1);
            while hi > min_nl {
                let rel = memchr::memrchr(b'\n', &b[min_nl..hi])?;
                let nl = min_nl + rel;
                let p = nl + 1;
                if p < n {
                    let c = b[p];
                    let ok = if id_norm {
                        c.is_ascii_alphanumeric() || (0xE4..=0xE9).contains(&c)
                    } else {
                        c.is_ascii_alphabetic() && (p + 1 == n || b[p + 1] < 0x80)
                    };
                    if ok {
                        return Some(p);
                    }
                }
                hi = nl;
            }
            return None;
        }
        // 3. ByteLevel `BlockScan` (GPT-2, cl100k, o200k, Qwen2.5/3/3.6, Llama-3/4,
        //    gpt-oss, SmolLM3, Seed-Coder):
        //    First try exact single-space boundary `[a-z] [A-Za-z]` (works even on
        //    10 MB single-line docs with ZERO newlines!), then fall back to `\n[A-Za-z]`.
        if self.fast_pretok == FastPretok::BlockScan {
            let mut hi = (hint + 1).min(n - 1);
            let min_p = lo.max(1);
            while hi > min_p {
                let Some(rel) = memchr::memrchr(b' ', &b[min_p..hi]) else {
                    break;
                };
                let p = min_p + rel;
                if b[p - 1].is_ascii_lowercase()
                    && b[p + 1].is_ascii_alphabetic()
                    && (id_norm || p + 2 == n || b[p + 2] < 0x80)
                {
                    return Some(p);
                }
                hi = p;
            }
            let mut hi = hint;
            let min_nl = lo.saturating_sub(1);
            while hi > min_nl {
                let rel = memchr::memrchr(b'\n', &b[min_nl..hi])?;
                let nl = min_nl + rel;
                let p = nl + 1;
                if p < n && b[p].is_ascii_alphabetic() && (id_norm || p + 1 == n || b[p + 1] < 0x80) {
                    return Some(p);
                }
                hi = nl;
            }
            return None;
        }
        // 4. Slice-only `Pipe` (`Falcon-H1`, `Falcon3`) when splittable (`\n[A-Za-z]`).
        if self.fast_pretok == FastPretok::Pipe && self.splittable() {
            let mut hi = hint;
            let min_nl = lo.saturating_sub(1);
            while hi > min_nl {
                let rel = memchr::memrchr(b'\n', &b[min_nl..hi])?;
                let nl = min_nl + rel;
                let p = nl + 1;
                if p < n && b[p].is_ascii_alphabetic() && (id_norm || p + 1 == n || b[p + 1] < 0x80) {
                    return Some(p);
                }
                hi = nl;
            }
        }
        None
    }

    /// Encode a proven non-first interior suffix fragment `sub` (starting at `p > 0`
    /// inside an ordinary text span) into `s.out`.
    #[inline]
    fn encode_ordinary_nonfirst_into(&self, sub: &str, s: &mut Scratch) {
        if sub.is_empty() {
            return;
        }
        if self.sp_fast_meta != SpFastMeta::None {
            let ok = self.encode_sp_fast(sub, false, false, s);
            debug_assert!(ok);
            return;
        }
        self.encode_into(sub, Merge::Auto, false, s);
    }

    /// Encode at most the LAST `max_tokens` ordinary tokens of `text` into `s.out`,
    /// using O(`max_tokens`) backward-window suffix splitting when `text` is much
    /// longer than `max_tokens`.
    fn encode_ordinary_suffix_into(&self, text: &str, max_tokens: usize, s: &mut Scratch) {
        if max_tokens == 0 || text.is_empty() {
            return;
        }
        let start_out = s.out.len();
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut win = max_tokens.saturating_mul(8).saturating_add(512);
        for _ in 0..2 {
            if n <= win.saturating_add(256) {
                break;
            }
            let hint = n - win;
            let lo = hint.saturating_sub(4096).max(1);
            if let Some(p) = self.find_safe_ordinary_split_before(bytes, hint, lo) {
                self.encode_ordinary_nonfirst_into(&text[p..], s);
                let got = s.out.len() - start_out;
                if got >= max_tokens {
                    if got > max_tokens {
                        s.out.drain(start_out..s.out.len() - max_tokens);
                    }
                    return;
                }
                s.out.truncate(start_out);
            }
            win = win.saturating_mul(4);
        }
        if n >= SPLIT_MIN && self.splittable() {
            s.out.extend_from_slice(&self.encode_split(text));
        } else {
            self.encode_into(text, Merge::Auto, true, s);
        }
        let got = s.out.len() - start_out;
        if got > max_tokens {
            s.out.drain(start_out..s.out.len() - max_tokens);
        }
    }

    /// Exact special-token-aware encode of at most the LAST `max_tokens` tokens of
    /// `text` into `s.out` in O(`max_tokens`) time (even on 100 MB chat logs!).
    fn encode_special_suffix_into(&self, text: &str, max_tokens: usize, s: &mut Scratch) {
        if max_tokens == 0 || text.is_empty() {
            return;
        }
        let Some(ac) = &self.special_ac else {
            self.encode_ordinary_suffix_into(text, max_tokens, s);
            return;
        };
        let start_out = s.out.len();
        let bytes = text.as_bytes();
        let n = bytes.len();
        let mut win = max_tokens.saturating_mul(8).saturating_add(512);
        for _ in 0..2 {
            if n <= win.saturating_add(256) {
                break;
            }
            let mut hint = n - win;
            let min_lo = hint.saturating_sub(8192).max(1);
            let mut found_p = None;
            // Try up to 4 candidate ordinary split points, verifying none lies inside
            // or adjacent to any special token (including lstrip/rstrip whitespace reach).
            for _ in 0..4 {
                if hint <= min_lo {
                    break;
                }
                let Some(p) = self.find_safe_ordinary_split_before(bytes, hint, min_lo) else {
                    break;
                };
                let ws_lo = text[..p].trim_end().len();
                let ws_after = text[p..].len() - text[p..].trim_start().len();
                if ws_lo > 0 && p + ws_after < n {
                    let mut guard_lo = ws_lo.saturating_sub(self.special_max_len + 2);
                    while guard_lo > 0 && !text.is_char_boundary(guard_lo) {
                        guard_lo -= 1;
                    }
                    let mut guard_hi = (p + ws_after + self.special_max_len + 2).min(n);
                    while guard_hi < n && !text.is_char_boundary(guard_hi) {
                        guard_hi += 1;
                    }
                    let clear = if self.special_prefilter.find_first(&bytes[guard_lo..guard_hi]).is_none() {
                        true
                    } else {
                        // Scan every special-token match starting in `guard_lo..=p+ws_after`
                        // using standard unanchored LeftmostLongest `ac.find` (zero StartKind::Both
                        // build overhead): LeftmostLongest skips non-matching spans in one jump and
                        // returns the longest match at `ms`, mathematically ruling out ANY special
                        // token crossing or reaching `p` (including via lstrip/rstrip) even when
                        // unrelated specials like `\n` appear nearby in Gemma-3.
                        let scan_hi = (p + ws_after + 1).min(n);
                        let mut ok = true;
                        let mut s_idx = guard_lo;
                        while s_idx < scan_hi {
                            let Some(m) = ac.find(&text[s_idx..guard_hi]) else {
                                break;
                            };
                            let ms = s_idx + m.start();
                            if ms >= scan_hi {
                                break;
                            }
                            let me = s_idx + m.end();
                            let pat = m.pattern().as_usize();
                            let start = if self.special_lstrip[pat] {
                                text[..ms].trim_end().len()
                            } else {
                                ms
                            };
                            let end = if self.special_rstrip[pat] {
                                n - text[me..].trim_start().len()
                            } else {
                                me
                            };
                            if start <= p && end >= p {
                                ok = false;
                                break;
                            }
                            // Advance past `ms` to the next UTF-8 char boundary so any overlapping
                            // special starting after `ms` is still checked.
                            s_idx = ms + text[ms..].chars().next().map_or(1, char::len_utf8);
                        }
                        ok
                    };
                    if clear {
                        found_p = Some(p);
                        break;
                    }
                    hint = guard_lo;
                } else {
                    hint = p.saturating_sub(1);
                }
            }
            if let Some(p) = found_p {
                let sub = &text[p..];
                let sb = sub.as_bytes();
                let sn = sb.len();
                if let Some(first) = self.special_prefilter.find_first(sb) {
                    let mut last = 0usize;
                    let mut first_frag = true;
                    for m in ac.find_iter(&sub[first..]) {
                        let ms = first + m.start();
                        let me = first + m.end();
                        if ms < last {
                            continue;
                        }
                        let pat = m.pattern().as_usize();
                        let (mut start, mut end) = (ms, me);
                        if self.special_lstrip[pat] {
                            start = last + sub[last..start].trim_end().len();
                        }
                        if self.special_rstrip[pat] {
                            end = sn - sub[end..].trim_start().len();
                        }
                        if start > last {
                            if first_frag {
                                self.encode_ordinary_nonfirst_into(&sub[last..start], s);
                            } else {
                                self.encode_fragment_into(&sub[last..start], false, s);
                            }
                        }
                        s.out.push(self.special_ids[pat]);
                        first_frag = false;
                        last = end;
                    }
                    if last < sn {
                        if first_frag {
                            self.encode_ordinary_nonfirst_into(&sub[last..], s);
                        } else {
                            self.encode_fragment_into(&sub[last..], false, s);
                        }
                    }
                } else {
                    self.encode_ordinary_nonfirst_into(sub, s);
                }
                let got = s.out.len() - start_out;
                if got >= max_tokens {
                    if got > max_tokens {
                        s.out.drain(start_out..s.out.len() - max_tokens);
                    }
                    return;
                }
                s.out.truncate(start_out);
            }
            win = win.saturating_mul(4);
        }
        self.encode_special_into(text, s);
        let got = s.out.len() - start_out;
        if got > max_tokens {
            s.out.drain(start_out..s.out.len() - max_tokens);
        }
    }

    /// Encode a batch of text views serially into ONE flat `Vec<u32>` + row ends
    /// (single `SCRATCH` borrow, zero per-row `Vec` allocations, zero rayon wakeup).
    /// When `trunc_cap` is `Some((cap, left))`, uses O(`cap`) early-exit prefix
    /// (`left == false`) or suffix (`left == true`) encoding.
    fn encode_special_serial_flat_cap(
        &self,
        views: &[&str],
        total_bytes: usize,
        trunc_cap: Option<(usize, bool)>,
    ) -> (Vec<u32>, Vec<usize>) {
        let est = match trunc_cap {
            Some((cap, _)) => views.len().saturating_mul(cap.min(4096)).min(total_bytes / 3 + views.len() * 2) + 8,
            None => total_bytes / 3 + views.len() * 2 + 8,
        };
        let run = |s: &mut Scratch| -> (Vec<u32>, Vec<usize>) {
            let saved = std::mem::take(&mut s.out);
            s.out = Vec::with_capacity(est);
            let mut ends = Vec::with_capacity(views.len());
            match trunc_cap {
                Some((cap, false)) => {
                    for &t in views {
                        self.encode_special_prefix_into(t, cap, s);
                        ends.push(s.out.len());
                    }
                }
                Some((cap, true)) => {
                    for &t in views {
                        self.encode_special_suffix_into(t, cap, s);
                        ends.push(s.out.len());
                    }
                }
                None => {
                    for &t in views {
                        self.encode_special_into(t, s);
                        ends.push(s.out.len());
                    }
                }
            }
            let flat = std::mem::replace(&mut s.out, saved);
            (flat, ends)
        };
        SCRATCH.with(|cell| match cell.try_borrow_mut() {
            Ok(mut s) => {
                s.dyn_cache.claim(self.uid);
                s.long_cache.claim(self.uid);
                run(&mut s)
            }
            Err(_) => {
                let mut s = Scratch::default();
                run(&mut s)
            }
        })
    }

    /// Encode a batch of text views in parallel into a small number of flat
    /// per-chunk `(Vec<u32>, Vec<usize>)` buffers (avoiding N per-row heap allocs).
    fn encode_special_par_chunks_cap(
        &self,
        views: &[&str],
        num_threads: usize,
        trunc_cap: Option<(usize, bool)>,
    ) -> Vec<(Vec<u32>, Vec<usize>)> {
        let n = views.len();
        let target_chunks = if num_threads == 0 {
            rayon::current_num_threads().saturating_mul(2)
        } else {
            num_threads.min(1024).saturating_mul(2)
        }
        .max(1);
        let chunk_len = n.div_ceil(target_chunks).max(16);
        let run = || {
            views
                .par_chunks(chunk_len)
                .map(|slice| {
                    let bytes: usize = slice.iter().map(|t| t.len()).sum();
                    self.encode_special_serial_flat_cap(slice, bytes, trunc_cap)
                })
                .collect::<Vec<_>>()
        };
        if num_threads == 0 {
            run()
        } else {
            match fixed_pool(num_threads.min(1024)) {
                Some(pool) => pool.install(run),
                None => run(),
            }
        }
    }

    /// Build byte-balanced `(lo, hi)` task spans over `FastItem`s so uniform
    /// short-prompt batches, small batches of large docs, and skewed chat batches
    /// all scale cleanly across cores with one `SCRATCH` borrow per task.
    fn fast_item_tasks(items: &[FastItem], eff_work: usize, lim: usize, nt: usize) -> Vec<(usize, usize)> {
        let n = items.len();
        let unit = (eff_work / (nt * 4)).clamp(4096, 1 << 20);
        let mut tasks = Vec::with_capacity((eff_work / unit).min(n) + 4);
        let (mut st, mut acc) = (0usize, 0usize);
        for (i, it) in items.iter().enumerate() {
            acc += it.len().min(lim).max(1);
            if acc >= unit {
                tasks.push((st, i + 1));
                st = i + 1;
                acc = 0;
            }
        }
        if st < n {
            tasks.push((st, n));
        }
        tasks
    }

    /// Universal zero-`PyText` / zero-`Vec<Vec<u32>>` flat batch encoder over
    /// `FastItem`s for ALL model families (Cjk/DeepSeek, SpFastMeta/Gemma/Mistral,
    /// Pipe/Falcon, NFC/Qwen3.6, and early-exit right/left truncation on any model).
    #[allow(clippy::type_complexity)]
    fn encode_fast_items_flat_cap(
        &self,
        items: &[FastItem],
        total_bytes: usize,
        num_threads: usize,
        special: bool,
        trunc_cap: Option<(usize, bool)>,
    ) -> Result<Vec<(Vec<u32>, Vec<usize>)>, BadInput> {
        use std::sync::atomic::{AtomicUsize, Ordering::*};
        let n = items.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let lim = match trunc_cap {
            Some((cap, true)) => cap.saturating_mul(8).saturating_add(512),
            Some((cap, false)) => cap.saturating_mul(6).saturating_add(128),
            None => usize::MAX,
        };
        let eff_work: usize = if trunc_cap.is_some() {
            items.iter().map(|it| it.len().min(lim)).sum()
        } else {
            total_bytes
        };
        let nt = if num_threads == 0 {
            rayon::current_num_threads()
        } else {
            num_threads.min(1024)
        }
        .max(1);
        let bad_wide = AtomicUsize::new(usize::MAX);
        let bad_doc = AtomicUsize::new(usize::MAX);

        let encode_span = |lo: usize, hi: usize| -> (Vec<u32>, Vec<usize>) {
            let span = &items[lo..hi];
            let span_bytes: usize = span.iter().map(|it| it.len().min(lim)).sum();
            let est = match trunc_cap {
                Some((cap, _)) => {
                    span.len().saturating_mul(cap.min(4096)).min(span_bytes / 3 + span.len() * 2) + 8
                }
                None => span_bytes / 3 + span.len() * 2 + 8,
            };
            let run = |s: &mut Scratch| -> (Vec<u32>, Vec<usize>) {
                let saved = std::mem::take(&mut s.out);
                s.out = Vec::with_capacity(est);
                let mut ends = Vec::with_capacity(span.len());
                let mut wide_buf = Vec::new();
                for (off, &it) in span.iter().enumerate() {
                    let idx = lo + off;
                    let t = match it.as_str(idx, &mut wide_buf) {
                        Ok(t) => t,
                        Err(BadInput::Wide(d)) => {
                            bad_wide.fetch_min(d, Relaxed);
                            ends.push(s.out.len());
                            continue;
                        }
                        Err(BadInput::Bytes(d)) => {
                            bad_doc.fetch_min(d, Relaxed);
                            ends.push(s.out.len());
                            continue;
                        }
                    };
                    if !t.is_empty() {
                        if special {
                            match trunc_cap {
                                Some((cap, false)) => self.encode_special_prefix_into(t, cap, s),
                                Some((cap, true)) => self.encode_special_suffix_into(t, cap, s),
                                None => self.encode_special_into(t, s),
                            }
                        } else {
                            match trunc_cap {
                                Some((cap, false)) => self.encode_ordinary_prefix_into(t, cap, s),
                                Some((cap, true)) => self.encode_ordinary_suffix_into(t, cap, s),
                                None => self.encode_into(t, Merge::Auto, true, s),
                            }
                        }
                    }
                    ends.push(s.out.len());
                }
                let flat = std::mem::replace(&mut s.out, saved);
                (flat, ends)
            };
            SCRATCH.with(|cell| match cell.try_borrow_mut() {
                Ok(mut s) => {
                    s.dyn_cache.claim(self.uid);
                    s.long_cache.claim(self.uid);
                    run(&mut s)
                }
                Err(_) => {
                    let mut s = Scratch::default();
                    run(&mut s)
                }
            })
        };

        const FLAT_SERIAL_MAX: usize = 65_536;
        let chunks = if nt == 1 || n <= 1 || !parallel_ok() || eff_work < FLAT_SERIAL_MAX {
            vec![encode_span(0, n)]
        } else {
            let tasks = Self::fast_item_tasks(items, eff_work, lim, nt);
            if tasks.len() <= 1 {
                vec![encode_span(0, n)]
            } else {
                let run = || tasks.par_iter().map(|&(lo, hi)| encode_span(lo, hi)).collect();
                if num_threads == 0 {
                    run()
                } else {
                    match fixed_pool(nt) {
                        Some(pool) => pool.install(run),
                        None => run(),
                    }
                }
            }
        };
        let bw = bad_wide.load(Relaxed);
        if bw != usize::MAX {
            return Err(BadInput::Wide(bw));
        }
        let bd = bad_doc.load(Relaxed);
        if bd != usize::MAX {
            return Err(BadInput::Bytes(bd));
        }
        Ok(chunks)
    }

    /// Universal zero-`PyText` / zero-output-alloc token counter over `FastItem`s
    /// for ALL model families (Cjk, SpFastMeta, Pipe, NFC, etc.).
    fn count_fast_items(
        &self,
        items: &[FastItem],
        total_bytes: usize,
        num_threads: usize,
        special: bool,
    ) -> Result<Vec<usize>, BadInput> {
        use std::sync::atomic::{AtomicUsize, Ordering::*};
        let n = items.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut counts = vec![0usize; n];
        let nt = if num_threads == 0 {
            rayon::current_num_threads()
        } else {
            num_threads.min(1024)
        }
        .max(1);
        let bad_wide = AtomicUsize::new(usize::MAX);
        let bad_doc = AtomicUsize::new(usize::MAX);

        let count_span = |lo: usize, dst: &mut [usize]| {
            let span = &items[lo..lo + dst.len()];
            let mut run = |s: &mut Scratch| {
                let mut wide_buf = Vec::new();
                for (off, (&it, out_c)) in span.iter().zip(dst.iter_mut()).enumerate() {
                    let idx = lo + off;
                    let t = match it.as_str(idx, &mut wide_buf) {
                        Ok(t) => t,
                        Err(BadInput::Wide(d)) => {
                            bad_wide.fetch_min(d, Relaxed);
                            continue;
                        }
                        Err(BadInput::Bytes(d)) => {
                            bad_doc.fetch_min(d, Relaxed);
                            continue;
                        }
                    };
                    if t.is_empty() {
                        *out_c = 0;
                    } else {
                        s.out.clear();
                        if special {
                            self.encode_special_into(t, s);
                        } else {
                            self.encode_into(t, Merge::Auto, true, s);
                        }
                        *out_c = s.out.len();
                    }
                }
            };
            SCRATCH.with(|cell| match cell.try_borrow_mut() {
                Ok(mut s) => {
                    s.dyn_cache.claim(self.uid);
                    s.long_cache.claim(self.uid);
                    run(&mut s);
                }
                Err(_) => {
                    let mut s = Scratch::default();
                    run(&mut s);
                }
            });
        };

        if nt == 1 || n <= 1 || !parallel_ok() || total_bytes < 65_536 {
            count_span(0, &mut counts);
        } else {
            let tasks = Self::fast_item_tasks(items, total_bytes, usize::MAX, nt);
            if tasks.len() <= 1 {
                count_span(0, &mut counts);
            } else {
                struct CpPtr(*mut usize);
                unsafe impl Sync for CpPtr {}
                impl CpPtr {
                    fn get(&self) -> *mut usize { self.0 }
                }
                let cp = CpPtr(counts.as_mut_ptr());
                let run = || {
                    tasks.par_iter().for_each(|&(lo, hi)| {
                        // SAFETY: `tasks` partitions `0..n` into disjoint `[lo..hi)` slices.
                        let dst = unsafe { std::slice::from_raw_parts_mut(cp.get().add(lo), hi - lo) };
                        count_span(lo, dst);
                    });
                };
                if num_threads == 0 {
                    run();
                } else {
                    match fixed_pool(nt) {
                        Some(pool) => pool.install(run),
                        None => run(),
                    }
                }
            }
        }
        let bw = bad_wide.load(Relaxed);
        if bw != usize::MAX {
            return Err(BadInput::Wide(bw));
        }
        let bd = bad_doc.load(Relaxed);
        if bd != usize::MAX {
            return Err(BadInput::Bytes(bd));
        }
        Ok(counts)
    }

    /// Build Python `list[list[int]]` rows (and optionally all-1s `attention_mask`
    /// rows) from special-token-aware encoded texts in one Rust pass.
    #[allow(clippy::too_many_arguments)]
    fn build_special_post_lists<'py>(
        &self,
        py: Python<'py>,
        texts_obj: &Bound<'py, PyAny>,
        prefix: &[u32],
        suffix: &[u32],
        budget: Option<usize>,
        trunc_left: bool,
        want_mask: bool,
        num_threads: usize,
    ) -> PyResult<(Bound<'py, PyList>, Option<Bound<'py, PyList>>)> {
        // Universal FastBatch path: extracts list/tuple of str ONCE (zero PyIter_Next,
        // zero Vec<PyText>, zero GIL wide transcoding). Uses the super-pipelined
        // `encode_fast_short_t` on BlockScan + ASCII-identity short docs, and the
        // flat chunk engine on all other models (Cjk, SpFastMeta, Pipe, NFC) and
        // O(cap) early-exit truncation batches!
        let trunc_cap = budget.map(|cap| (cap, trunc_left));
        type FlatChunks = Vec<(Vec<u32>, Vec<usize>)>;
        let fast_res: Option<(usize, FlatChunks)> =
            if let Some(batch) = try_extract_fast_batch(py, texts_obj) {
                if !batch.has_bytes {
                    let early_exit_worthy = trunc_cap.is_some_and(|(cap, left)| {
                        let thresh = if left {
                            cap.saturating_mul(8).saturating_add(768)
                        } else {
                            cap.saturating_mul(6).saturating_add(128)
                        };
                        batch.max_len > thresh
                    });
                    let slice = &batch.items;
                    let total_len = batch.total_len;
                    let nrows = slice.len();
                    if self.normalizer.is_ascii_identity()
                        && self.items_pipe_ok()
                        && batch.max_len < SPLIT_MIN
                        && !early_exit_worthy
                    {
                        let res = self.maybe_detach(py, total_len, || {
                            self.encode_fast_short_t::<u32, true>(slice, total_len, num_threads, true)
                        });
                        match res {
                            Ok((data, counts)) => {
                                drop(batch);
                                let mut ends = Vec::with_capacity(counts.len());
                                let mut acc = 0usize;
                                for c in counts {
                                    acc += c;
                                    ends.push(acc);
                                }
                                Some((nrows, vec![(data, ends)]))
                            }
                            Err(b) => return Err(bad_fast_to_pyerr(py, &batch, b)),
                        }
                    } else {
                        let detach_work = trunc_cap.map_or(total_len, |(cap, left)| {
                            let lim = if left {
                                cap.saturating_mul(8).saturating_add(512)
                            } else {
                                cap.saturating_mul(6).saturating_add(128)
                            };
                            total_len.min(nrows.saturating_mul(lim))
                        });
                        let res = self.maybe_detach(py, detach_work, || {
                            self.encode_fast_items_flat_cap(slice, total_len, num_threads, true, trunc_cap)
                        });
                        match res {
                            Ok(chunks) => {
                                drop(batch);
                                Some((nrows, chunks))
                            }
                            Err(b) => return Err(bad_fast_to_pyerr(py, &batch, b)),
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

        let (nrows, chunks): (usize, FlatChunks) = if let Some(v) = fast_res {
            v
        } else {
            const FLAT_SERIAL_MAX: usize = 131_072;
            let texts: Vec<PyText> = texts_obj.extract()?;
            let wide = wide_texts(py, &texts, num_threads)?;
            let views = text_views(&texts, &wide);
            let total: usize = views.iter().map(|t| t.len()).sum();
            let eff_work: usize = match trunc_cap {
                Some((cap, left)) => {
                    let lim = if left {
                        cap.saturating_mul(8).saturating_add(512)
                    } else {
                        cap.saturating_mul(6).saturating_add(128)
                    };
                    views.iter().map(|t| t.len().min(lim)).sum()
                }
                None => total,
            };
            let nrows = views.len();
            let chunks =
                if num_threads == 1 || nrows <= 1 || !parallel_ok() || eff_work < FLAT_SERIAL_MAX {
                    let one = self.maybe_detach(py, eff_work, || {
                        self.encode_special_serial_flat_cap(&views, total, trunc_cap)
                    });
                    vec![one]
                } else {
                    py.detach(|| self.encode_special_par_chunks_cap(&views, num_threads, trunc_cap))
                };
            (nrows, chunks)
        };

        let table = self.ints(py);
        // Build a 1-element template list `[1]` once when `want_mask` is set:
        // `PySequence_Repeat([1], n)` hits CPython's C `list_repeat` fast path,
        // which allocates via `PyMem_Malloc` (avoiding `PyList_New`'s redundant
        // `calloc` zero-fill), updates `1`'s refcount at most once (or zero
        // times when immortal on 3.12+), and fills the buffer in a tight loop.
        let one_list = if want_mask {
            unsafe {
                let l = pyo3::ffi::PyList_New(1);
                assert!(!l.is_null(), "list alloc");
                let one_ptr = if let Some(o) = table.get(1) {
                    let p = o.as_ptr();
                    pyo3::ffi::Py_INCREF(p);
                    p
                } else {
                    pyo3::ffi::PyLong_FromLong(1)
                };
                pyo3::ffi::PyList_SET_ITEM(l, 0, one_ptr);
                l
            }
        } else {
            std::ptr::null_mut()
        };
        // SAFETY: each new list's slots are filled exactly once with owned refs.
        unsafe {
            let outer = pyo3::ffi::PyList_New(nrows as pyo3::ffi::Py_ssize_t);
            assert!(!outer.is_null(), "list alloc");
            let m_outer = if want_mask {
                let p = pyo3::ffi::PyList_New(nrows as pyo3::ffi::Py_ssize_t);
                assert!(!p.is_null(), "list alloc");
                p
            } else {
                std::ptr::null_mut()
            };
            let mut r = 0usize;
            for (flat_buf, flat_ends) in &chunks {
                let mut start = 0usize;
                for &end in flat_ends {
                    let c = &flat_buf[start..end];
                    start = end;
                    let keep = budget.map_or(c.len(), |b| c.len().min(b));
                    let content: &[u32] =
                        if trunc_left { &c[c.len() - keep..] } else { &c[..keep] };
                    let n = prefix.len() + content.len() + suffix.len();
                    let inner = pyo3::ffi::PyList_New(n as pyo3::ffi::Py_ssize_t);
                    assert!(!inner.is_null(), "list alloc");
                    if !prefix.is_empty() {
                        Self::fill_pylist_prefetched(table, prefix, inner, 0);
                    }
                    Self::fill_pylist_prefetched(table, content, inner, prefix.len());
                    if !suffix.is_empty() {
                        Self::fill_pylist_prefetched(table, suffix, inner, prefix.len() + content.len());
                    }
                    pyo3::ffi::PyList_SET_ITEM(outer, r as pyo3::ffi::Py_ssize_t, inner);
                    if !m_outer.is_null() {
                        let m_inner = pyo3::ffi::PySequence_Repeat(one_list, n as pyo3::ffi::Py_ssize_t);
                        assert!(!m_inner.is_null(), "list alloc");
                        pyo3::ffi::PyList_SET_ITEM(m_outer, r as pyo3::ffi::Py_ssize_t, m_inner);
                    }
                    r += 1;
                }
            }
            if !one_list.is_null() {
                pyo3::ffi::Py_DECREF(one_list);
            }
            let seqs = Bound::from_owned_ptr(py, outer).cast_into_unchecked();
            let masks = if m_outer.is_null() {
                None
            } else {
                Some(Bound::from_owned_ptr(py, m_outer).cast_into_unchecked())
            };
            Ok((seqs, masks))
        }
    }

    fn apply_warmup_arg(&self, py: Python<'_>, warmup: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if let Some(w) = warmup {
            if w.is_instance_of::<pyo3::types::PyBool>() {
                if w.extract::<bool>()? {
                    self.warmup(py, None, 0)?;
                }
            } else if !w.is_none() {
                self.warmup(py, Some(w), 0)?;
            }
        }
        Ok(())
    }

    fn encode_to_token_buffer<'py, O: OutTok>(
        &self,
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        num_threads: usize,
        separator: Option<&[u8]>,
        wrap: impl FnOnce(Vec<O>) -> TokenBuffer,
    ) -> PyResult<(Bound<'py, TokenBuffer>, Vec<usize>)> {
        if separator.is_none() && self.normalizer.is_ascii_identity() && self.items_pipe_ok() {
            if let Some(batch) = try_extract_fast_batch(py, texts) {
                if batch.max_len < SPLIT_MIN {
                    let slice = &batch.items;
                    let total_len = batch.total_len;
                    let res = py.detach(|| self.encode_fast_short_t::<O, false>(slice, total_len, num_threads, true));
                    let (d, c) = match res {
                        Ok(v) => v,
                        Err(b) => return Err(bad_fast_to_pyerr(py, &batch, b)),
                    };
                    drop(batch);
                    return Ok((Bound::new(py, wrap(d))?, c));
                }
            }
        }
        let texts: Vec<TextArg> = texts.extract()?;
        let stream_wide = self.stream_wide_ok(separator);
        let wide = if stream_wide { WideCopies::default() } else { wide_args(py, &texts, num_threads)? };
        let res = py.detach(|| self.encode_batch_sep_t::<O>(&texts, &wide, stream_wide, num_threads, true, separator));
        let (d, c) = match res {
            Ok(v) => v,
            Err(Ok(b)) => return Err(bad_input_to_pyerr(py, &texts, b)),
            Err(Err(e)) => return Err(e),
        };
        Ok((Bound::new(py, wrap(d))?, c))
    }

    /// Encode or count raw UTF-8 bytes (e.g. an mmap'd file) with optional
    /// document separator, releasing the GIL in the caller.
    fn encode_raw_bytes_sep_t<O: OutTok>(
        &self,
        raw: &[u8],
        num_threads: usize,
        want_output: bool,
        sep: Option<&[u8]>,
    ) -> Result<(Vec<O>, Vec<usize>), Result<BadInput, String>> {
        if let Some(sep) = sep {
            if sep.is_empty() {
                return Err(Err("separator must not be empty".to_string()));
            }
            if self.normalizer.is_identity() && self.items_pipe_ok() && raw.len() >= (1 << 18) {
                return self.encode_fused_sep_t::<O>(raw, sep, false, num_threads, want_output).map_err(Ok);
            }
            let allow_raw = self.normalizer.is_identity() || self.parallel_unicode_norm();
            let spans = split_on(raw, sep, num_threads);
            if allow_raw {
                let src: Vec<Src<'_>> = spans.into_iter().map(|(a, b)| Src::Raw(&raw[a..b])).collect();
                return self.encode_src_t::<O>(&src, num_threads, want_output).map_err(Ok);
            }
            let mut src = Vec::with_capacity(spans.len());
            for (idx, (a, b)) in spans.into_iter().enumerate() {
                match simdutf8::basic::from_utf8(&raw[a..b]) {
                    Ok(s) => src.push(Src::Str(s)),
                    Err(_) => return Err(Ok(BadInput::Bytes(idx))),
                }
            }
            return self.encode_src_t::<O>(&src, num_threads, want_output).map_err(Ok);
        }
        let allow_raw = self.normalizer.is_identity() || self.parallel_unicode_norm();
        if allow_raw {
            return self.encode_src_t::<O>(&[Src::Raw(raw)], num_threads, want_output).map_err(Ok);
        }
        match simdutf8::basic::from_utf8(raw) {
            Ok(s) => self.encode_src_t::<O>(&[Src::Str(s)], num_threads, want_output).map_err(Ok),
            Err(_) => Err(Ok(BadInput::Bytes(0))),
        }
    }

    #[inline]
    fn count_ordinary_one(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        if text.len() >= SPLIT_MIN && self.splittable() {
            return self.encode_direct(&[text], 0, false).1[0];
        }
        if text.len() >= 1024 && self.items_pipe_ok() && self.normalizer.is_identity() && text.len() <= u32::MAX as usize {
            let bytes = text.as_bytes();
            let n = bytes.len();
            return self.with_scratch(|s| {
                const CAP: usize = 256;
                let mut ends = [0u32; CAP + blockscan::SCAN_SLACK];
                let mut buf = [std::mem::MaybeUninit::<PipeRec>::uninit(); CAP + blockscan::SCAN_SLACK + X86_PF_DIST];
                let mut scan = blockscan::Scan::new(0);
                let mut a = 0usize;
                let mut total = 0usize;
                loop {
                    let k = scan.fill(self.kind, bytes, n, &mut ends, 0, CAP);
                    if k == 0 {
                        break;
                    }
                    let view = self.learned.view();
                    let lview = self.long_table.view();
                    let first_a = a;
                    a = Self::fill_recs(bytes, &ends, 0, k, a, buf.as_mut_ptr(), view, lview);
                    Self::prime_pipe_prefetch(&mut buf, k);
                    total += self.flush_recs_count(bytes, n, buf.as_ptr() as *const PipeRec, k, &ends[..k], first_a, s, view, lview);
                }
                total
            });
        }
        self.with_scratch(|s| {
            s.out.clear();
            self.encode_into(text, Merge::Auto, true, s);
            s.out.len()
        })
    }

    #[inline]
    fn count_special_one(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        if self.special_ac.is_none() || self.special_prefilter.find_first(text.as_bytes()).is_none() {
            return self.count_ordinary_one(text);
        }
        self.with_scratch(|s| {
            s.out.clear();
            self.encode_special_into(text, s);
            s.out.len()
        })
    }
    /// Construct `Tokenizer` from the parallel zero-malloc fast JSON loader.
    fn from_fast_loaded(
        path: &str,
        c2b: [u16; 324],
        allow_par: bool,
        fl: fast_load::FastLoadedTokenizer,
    ) -> PyResult<Self> {
        let norm_val = &fl.normalizer;
        let pre = &fl.pre_tokenizer;
        let dec_val = &fl.decoder;

        if let Some(ref mt) = fl.model_type {
            if mt != "BPE" {
                return Err(PyValueError::new_err(format!(
                    "unsupported model type {mt:?}: tuetoken only supports BPE models"
                )));
            }
        } else if !fl.has_merges {
            return Err(PyValueError::new_err(
                "unsupported non-BPE model (missing merges): tuetoken only supports BPE models",
            ));
        }
        if let Some(ref pfx) = fl.continuing_subword_prefix {
            if !pfx.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "unsupported BPE continuing_subword_prefix {pfx:?}"
                )));
            }
        }
        let eow_suffix = fl.end_of_word_suffix.filter(|s| !s.is_empty());

        let normalizer = norm::Normalizer::from_json(norm_val).map_err(|t| {
            PyValueError::new_err(format!(
                "unsupported normalizer stage {t:?}: this model rewrites text in a \
                 way tuetoken does not reproduce"
            ))
        })?;

        let byte_fallback = fl.byte_fallback;
        let decode_strip_space = detect_decode_strip(dec_val);
        let prepend_scheme_str = detect_prepend_scheme_str(pre, norm_val);
        let post_processor_json = fl
            .post_processor
            .as_ref()
            .filter(|v| !v.is_null())
            .and_then(|v| serde_json::to_string(v).ok());

        let (kind, fast_pretok, pretok_pipe) = if !byte_fallback && is_cjk_pretok(pre) {
            (Pretok::Word(false), FastPretok::Cjk, None)
        } else if let Some(k) = (!byte_fallback).then(|| detect_pretok(pre)).flatten() {
            (k, FastPretok::BlockScan, None)
        } else {
            if !byte_fallback
                && !contains_bytelevel(pre, "pretokenizers")
                && !contains_bytelevel(dec_val, "decoders")
            {
                return Err(PyValueError::new_err(
                    "unsupported pre_tokenizer (expected ByteLevel or byte_fallback)",
                ));
            }
            match pretok_pipe::PretokPipe::from_json(pre)
                .map_err(|e| PyValueError::new_err(format!("{e} (refusing to mistokenize)")))?
            {
                Some(pipe) => (Pretok::Word(false), FastPretok::Pipe, Some(pipe)),
                None => (Pretok::Word(false), FastPretok::WholeText, None),
            }
        };
        let ignore_merges = fl.ignore_merges;

        let meta_split_cand: Option<char> = if byte_fallback && !ignore_merges && eow_suffix.is_none()
        {
            if let Some(ref pipe) = pretok_pipe {
                if let Some(rep) = pipe.single_unsplit_metaspace() {
                    Some(rep)
                } else if normalizer.replaces_space_with('\u{2581}')
                    && pre.get("type").and_then(|v| v.as_str()) == Some("Split")
                    && pre.get("pattern").and_then(|p| p.get("String")).and_then(|v| v.as_str())
                        == Some(" ")
                    && pre.get("behavior").and_then(|v| v.as_str()) == Some("MergedWithPrevious")
                    && !pre.get("invert").and_then(|v| v.as_bool()).unwrap_or(false)
                {
                    Some('\u{2581}')
                } else {
                    None
                }
            } else if normalizer.replaces_space_with('\u{2581}') {
                Some('\u{2581}')
            } else {
                None
            }
        } else {
            None
        };

        let vocab_size = fl.vocab_size;
        let mut added_meta: Vec<(String, u32, bool, bool, bool)> = Vec::with_capacity(fl.added_tokens.len());
        for at in fl.added_tokens {
            if let Some(id) = at.id {
                if let Some(content) = at.content {
                    added_meta.push((content, id as u32, at.special, at.lstrip, at.rstrip));
                }
            }
        }

        let mut byte_arena = fl.byte_arena;
        let mut byte_span = fl.byte_span;
        let canonical = fl.canonical;
        let has_internal_ms = fl.has_internal_ms;
        let mut special_skip_mask = vec![false; vocab_size];
        for (content, id, sp, _, _) in &added_meta {
            let id = *id as usize;
            if id < vocab_size {
                if *sp {
                    special_skip_mask[id] = true;
                }
                if byte_span[id].1 == 0 {
                    let start = byte_arena.len() as u32;
                    byte_arena.extend_from_slice(content.as_bytes());
                    byte_span[id] = (start, content.len() as u32);
                }
            }
        }

        let mut singleton = [-1i32; 256];
        let mut singleton_canon = [false; 256];
        for id in 0..vocab_size {
            let (off, len) = byte_span[id];
            if len == 1 {
                let bv = byte_arena[off as usize] as usize;
                let canon = canonical[id];
                if singleton[bv] < 0
                    || (canon && !singleton_canon[bv])
                    || (canon == singleton_canon[bv] && (id as i32) < singleton[bv])
                {
                    singleton[bv] = id as i32;
                    singleton_canon[bv] = canon;
                }
            }
        }

        let merges = fl.merges;
        let check_meta_and_mono = || {
            let sp_info = analyze_sp_cross_prev(
                meta_split_cand,
                &merges,
                &has_internal_ms,
                &byte_arena,
                &byte_span,
            );
            let mut produce_max: Vec<i64> = vec![-1; vocab_size];
            let mut consume_min: Vec<i64> = vec![i64::MAX; vocab_size];
            let mut len_ok = !byte_fallback
                && fl.singleton_eow.is_none()
                && singleton.iter().enumerate().all(|(b, &sid)| {
                    // Bytes 0xC0, 0xC1, 0xF5..=0xFF can never appear in valid UTF-8 `&str`
                    // (RFC 3629); models like Falcon-H1 omit those 13 impossible bytes.
                    matches!(b, 0xC0 | 0xC1 | 0xF5..=0xFF)
                        || (sid >= 0 && (sid as usize) < vocab_size && byte_span[sid as usize].1 == 1)
                });
            for (i, &(a, b, t)) in merges.iter().enumerate() {
                let i = i as i64;
                let (au, bu, tu) = (a as usize, b as usize, t as usize);
                if tu < vocab_size {
                    produce_max[tu] = produce_max[tu].max(i);
                }
                if au < vocab_size {
                    consume_min[au] = consume_min[au].min(i);
                }
                if bu < vocab_size {
                    consume_min[bu] = consume_min[bu].min(i);
                }
                if len_ok
                    && (au >= vocab_size
                        || bu >= vocab_size
                        || tu >= vocab_size
                        || byte_span[tu].1 != byte_span[au].1 + byte_span[bu].1)
                {
                    len_ok = false;
                }
            }
            let mono = (0..vocab_size).all(|t| produce_max[t] < 0 || consume_min[t] > produce_max[t]);
            (sp_info, mono, len_ok)
        };
        let (mut map, ((safe_meta_split, sp_split_ok, sp_cross_prev), monotonic, byte_len_exact)) = if allow_par {
            rayon::join(
                || bpe::MergeMap::new_with_opts(&merges, true, byte_fallback),
                check_meta_and_mono,
            )
        } else {
            (bpe::MergeMap::new_with_opts(&merges, false, byte_fallback), check_meta_and_mono())
        };

        let short_in_ct = !byte_fallback && fl.singleton_eow.is_none();
        let ignore_lookup = if ignore_merges {
            let mut m: FxHashMap<Box<[u8]>, u32> = FxHashMap::default();
            for (id, &(off, len)) in byte_span.iter().enumerate() {
                let lu = len as usize;
                if lu > 0 && (!short_in_ct || lu > chunk_table::MAX_KEY) {
                    let b = &byte_arena[off as usize..(off + len) as usize];
                    m.entry(b.to_vec().into_boxed_slice()).or_insert(id as u32);
                }
            }
            if m.is_empty() { None } else { Some(m) }
        } else {
            None
        };

        let mut singleton_u32 = [0u32; 256];
        for b in 0..256 {
            if singleton[b] >= 0 {
                singleton_u32[b] = singleton[b] as u32;
            }
        }
        let byte_fast = !byte_fallback
            && fl.singleton_eow.is_none()
            && map.init_byte_tables(&singleton, &merges, monotonic);

        let sp_fast_meta = detect_sp_fast_meta(
            byte_fallback,
            ignore_merges,
            fl.singleton_eow.is_some(),
            sp_split_ok,
            &normalizer,
            pretok_pipe.as_ref(),
            pre,
        );
        let sp_has_cross = sp_cross_prev != [0u64; 4];
        let sp_rep_id = fl.char_map.get(&('\u{2581}' as u32)).map(|&x| x as i32).unwrap_or(-1);

        let mut tk = Tokenizer {
            kind,
            fast_pretok,
            singleton,
            singleton_u32,
            singleton_eow: fl.singleton_eow,
            byte_fast,
            map,
            raw_merges: merges,
            stream: OnceLock::new(),
            byte_arena,
            byte_span,
            c2b,
            ignore_merges,
            ignore_lookup,
            vocab_size,
            normalizer,
            pretok_pipe,
            byte_fallback,
            safe_meta_split,
            sp_fast_meta,
            sp_cross_prev,
            sp_has_cross,
            sp_rep_id,
            char_ascii: fl.char_ascii,
            char_map: fl.char_map,
            decode_strip_space,
            source_path: path.to_string(),
            is_byte_level: !byte_fallback,
            byte_len_exact,
            monotonic,
            special_ac: None,
            special_prefilter: SpecialPrefilter::None,
            special_ids: Vec::new(),
            special_lstrip: Vec::new(),
            special_rstrip: Vec::new(),
            special_max_len: 0,
            special_skip_mask,
            whisper_ts_begin: u32::MAX,
            whisper_sop_id: u32::MAX,
            whisper_sot_id: u32::MAX,
            prepend_scheme: 0,
            chunk_table: chunk_table::ChunkTable::empty(),
            uid: NEXT_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            learned: chunk_table::SharedTable::new(DEFAULT_CACHE_MIB),
            long_table: chunk_table::SharedLongTable::new(DEFAULT_CACHE_MIB / 4),
            spill: chunk_table::SpillMap::new((DEFAULT_CACHE_MIB << 20) / 4),
            int_table: OnceLock::new(),
            decode_table: OnceLock::new(),
            auto_meta: AutoMeta {
                added_tokens: added_meta,
                post_processor_json,
                prepend_scheme: prepend_scheme_str,
                known_token_ids: fl.known_token_ids,
            },
        };
        tk.build_chunk_table();
        Ok(tk)
    }

    /// Build Python `(list[int], list[tuple[int, int]])` directly via CPython C-API:
    /// - `ids_list`: 2-stage prefetched cached `int_table` (`to_pylist`)
    /// - `offsets_list`: contiguous byte spans reconstructed from `self.byte_span[t].1`
    ///   with adjacent `PyLong` sharing (`prev_obj` reused via `Py_INCREF` as the next
    ///   tuple's start element — halving `PyLong` allocations and avoiding any
    ///   intermediate `Vec<(usize, usize)>`).
    fn build_offsets_pytuple<'py>(
        &self,
        py: Python<'py>,
        ids: &[u32],
    ) -> PyResult<Bound<'py, PyTuple>> {
        use pyo3::ffi;
        let ids_list = self.to_pylist(py, ids);
        let n = ids.len();
        let spans_ptr = unsafe { ffi::PyList_New(n as ffi::Py_ssize_t) };
        if spans_ptr.is_null() {
            return Err(PyErr::fetch(py));
        }
        let span_tbl = self.byte_span.as_ptr();
        let tbl_len = self.byte_span.len();
        let mut pos = 0usize;
        let mut prev_obj = if n > 0 {
            unsafe { ffi::PyLong_FromSize_t(0) }
        } else {
            std::ptr::null_mut()
        };
        if n > 0 && prev_obj.is_null() {
            unsafe { ffi::Py_DECREF(spans_ptr) };
            return Err(PyErr::fetch(py));
        }
        unsafe {
            let items = (*(spans_ptr as *mut ffi::PyListObject)).ob_item;
            for (i, &tid) in ids.iter().enumerate() {
                let iu = tid as usize;
                let len = if iu < tbl_len { (*span_tbl.add(iu)).1 as usize } else { 0 };
                pos += len;
                let next_obj = ffi::PyLong_FromSize_t(pos);
                let tup = ffi::PyTuple_New(2);
                if next_obj.is_null() || tup.is_null() {
                    if !next_obj.is_null() {
                        ffi::Py_DECREF(next_obj);
                    }
                    if !tup.is_null() {
                        ffi::Py_DECREF(tup);
                    }
                    ffi::Py_DECREF(prev_obj);
                    ffi::Py_DECREF(spans_ptr);
                    return Err(PyErr::fetch(py));
                }
                ffi::PyTuple_SET_ITEM(tup, 0, prev_obj); // steals prev_obj
                ffi::Py_INCREF(next_obj);
                ffi::PyTuple_SET_ITEM(tup, 1, next_obj); // steals 1 ref; we retain 1 ref
                *items.add(i) = tup;
                prev_obj = next_obj;
            }
            if !prev_obj.is_null() {
                ffi::Py_DECREF(prev_obj);
            }
        }
        let spans_list: Bound<'py, PyList> =
            unsafe { Bound::from_owned_ptr(py, spans_ptr).cast_into_unchecked() };
        PyTuple::new(py, [ids_list.into_any(), spans_list.into_any()])
    }
}

type AutoMetaTuple = (Vec<(String, u32, bool, bool, bool)>, Option<String>, String, Vec<(String, u32)>);
type FillRowItem<'a> = (usize, (&'a mut [std::mem::MaybeUninit<i64>], &'a mut [std::mem::MaybeUninit<i64>]));

#[pymethods]
impl Tokenizer {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        // Zero-copy mmap from OS page cache (falls back to fs::read for pipes/special files),
        // eliminating a 12–33 MB heap alloc + memcpy + page reclaim on every load.
        let file_bytes = FileBytes::open(std::path::Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("cannot read {path}: {e}")))?;
        let raw_bytes = file_bytes.as_slice();
        let allow_par = parallel_ok();
        let utf8_ok = if allow_par && raw_bytes.len() >= (1 << 20) {
            const PIECE: usize = 1 << 20;
            let mut cuts = Vec::with_capacity(raw_bytes.len() / PIECE + 2);
            cuts.push(0usize);
            let mut p = PIECE;
            while p < raw_bytes.len() {
                while p < raw_bytes.len() && (raw_bytes[p] & 0xC0) == 0x80 {
                    p += 1;
                }
                cuts.push(p);
                p += PIECE;
            }
            cuts.push(raw_bytes.len());
            cuts.dedup();
            cuts.par_windows(2)
                .all(|w| simdutf8::basic::from_utf8(&raw_bytes[w[0]..w[1]]).is_ok())
        } else {
            simdutf8::basic::from_utf8(raw_bytes).is_ok()
        };
        if !utf8_ok {
            simdutf8::basic::from_utf8(raw_bytes)
                .map_err(|e| PyValueError::new_err(format!("cannot read {path}: invalid UTF-8: {e}")))?;
        }
        // SAFETY: UTF-8 validity verified immediately above across the entire buffer.
        let data = unsafe { std::str::from_utf8_unchecked(raw_bytes) };

        let c2b = bytes_to_unicode_table();
        let detect_meta_cand = |pre: &Value, norm_val: &Value, byte_fallback: bool, ignore_merges: bool, has_eow: bool| -> Option<char> {
            if !byte_fallback || ignore_merges || has_eow {
                return None;
            }
            let normalizer = norm::Normalizer::from_json(norm_val).ok()?;
            let pipe = pretok_pipe::PretokPipe::from_json(pre).ok()?;
            if let Some(ref p) = pipe {
                if let Some(rep) = p.single_unsplit_metaspace() {
                    return Some(rep);
                }
                if normalizer.replaces_space_with('\u{2581}')
                    && pre.get("type").and_then(|v| v.as_str()) == Some("Split")
                    && pre.get("pattern").and_then(|p| p.get("String")).and_then(|v| v.as_str()) == Some(" ")
                    && pre.get("behavior").and_then(|v| v.as_str()) == Some("MergedWithPrevious")
                    && !pre.get("invert").and_then(|v| v.as_bool()).unwrap_or(false)
                {
                    return Some('\u{2581}');
                }
                None
            } else if normalizer.replaces_space_with('\u{2581}') {
                Some('\u{2581}')
            } else {
                None
            }
        };

        if let Some(fl) = fast_load::try_load_fast(
            raw_bytes,
            &c2b,
            allow_par,
            decode_bytelevel_into,
            decode_byte_fallback_into,
            detect_meta_cand,
        ) {
            return Self::from_fast_loaded(path, c2b, allow_par, fl);
        }

        let raw: RawTokenizerJson<'_> = serde_json::from_str(data)
            .map_err(|e| PyValueError::new_err(format!("bad JSON: {e}")))?;
        let model = raw.model.ok_or_else(|| PyValueError::new_err("no model"))?;
        let norm_val = raw.normalizer.as_ref().unwrap_or(&Value::Null);
        let pre = raw.pre_tokenizer.as_ref().unwrap_or(&Value::Null);
        let dec_val = raw.decoder.as_ref().unwrap_or(&Value::Null);

        // Fail closed on non-BPE models (WordPiece, Unigram, WordLevel, ...).
        if let Some(ref mt) = model.model_type {
            if mt != "BPE" {
                return Err(PyValueError::new_err(format!(
                    "unsupported model type {mt:?}: tuetoken only supports BPE models"
                )));
            }
        } else if !model.has_merges {
            return Err(PyValueError::new_err(
                "unsupported non-BPE model (missing merges): tuetoken only supports BPE models",
            ));
        }
        if let Some(ref pfx) = model.continuing_subword_prefix {
            if !pfx.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "unsupported BPE continuing_subword_prefix {pfx:?}"
                )));
            }
        }
        let eow_suffix = model.end_of_word_suffix.filter(|s| !s.is_empty());

        // Composable HF normalizer pipeline — fails closed on unknown normalizer types
        // (e.g. BertNormalizer, Precompiled) rather than silently mistokenizing.
        let normalizer = norm::Normalizer::from_json(norm_val).map_err(|t| {
            PyValueError::new_err(format!(
                "unsupported normalizer stage {t:?}: this model rewrites text in a \
                 way tuetoken does not reproduce"
            ))
        })?;

        let byte_fallback = model.byte_fallback;
        let decode_strip_space = detect_decode_strip(dec_val);
        let prepend_scheme_str = detect_prepend_scheme_str(pre, norm_val);
        let post_processor_json = raw
            .post_processor
            .as_ref()
            .filter(|v| !v.is_null())
            .and_then(|v| serde_json::to_string(v).ok());

        // Pre-tokenizer dispatch:
        //   1. Fast hand-written SIMD `blockscan` / `cjk` machines (exact structural match).
        //   2. Universal composable `PretokPipe` for any other HF pre_tokenizer
        //      (Split, Digits, Punctuation, ByteLevel, Metaspace, Whitespace, ...).
        //   3. WholeText for `pre_tokenizer: null` / empty Sequence.
        //   4. Fail closed on any unrecognized pre_tokenizer stage.
        let (kind, fast_pretok, pretok_pipe) = if !byte_fallback && is_cjk_pretok(pre) {
            (Pretok::Word(false), FastPretok::Cjk, None)
        } else if let Some(k) = (!byte_fallback).then(|| detect_pretok(pre)).flatten() {
            (k, FastPretok::BlockScan, None)
        } else {
            if !byte_fallback
                && !contains_bytelevel(pre, "pretokenizers")
                && !contains_bytelevel(dec_val, "decoders")
            {
                return Err(PyValueError::new_err(
                    "unsupported pre_tokenizer (expected ByteLevel or byte_fallback)",
                ));
            }
            match pretok_pipe::PretokPipe::from_json(pre)
                .map_err(|e| PyValueError::new_err(format!("{e} (refusing to mistokenize)")))?
            {
                Some(pipe) => (Pretok::Word(false), FastPretok::Pipe, Some(pipe)),
                None => (Pretok::Word(false), FastPretok::WholeText, None),
            }
        };
        let ignore_merges = model.ignore_merges;

        // Candidate metaspace chunk-split character for unsplit byte_fallback models
        // (e.g. Mistral-7B-v0.3, Phi-3.5, Yi-1.5, Gemma-2/3, ERNIE-4.5). Verified
        // mathematically against `merges` below before enabling.
        let meta_split_cand: Option<char> = if byte_fallback && !ignore_merges && eow_suffix.is_none()
        {
            if let Some(ref pipe) = pretok_pipe {
                if let Some(rep) = pipe.single_unsplit_metaspace() {
                    Some(rep)
                } else if normalizer.replaces_space_with('\u{2581}')
                    && pre.get("type").and_then(|v| v.as_str()) == Some("Split")
                    && pre.get("pattern").and_then(|p| p.get("String")).and_then(|v| v.as_str())
                        == Some(" ")
                    && pre.get("behavior").and_then(|v| v.as_str()) == Some("MergedWithPrevious")
                    && !pre.get("invert").and_then(|v| v.as_bool()).unwrap_or(false)
                {
                    Some('\u{2581}')
                } else {
                    None
                }
            } else if normalizer.replaces_space_with('\u{2581}') {
                Some('\u{2581}')
            } else {
                None
            }
        } else {
            None
        };

        let c2b = bytes_to_unicode_table();
        let vocab = model.vocab.ok_or_else(|| PyValueError::new_err("no vocab"))?;
        let mut vocab_size = (model.max_id as usize) + 1;

        // Collect added_tokens metadata + extend id space
        let mut added_meta: Vec<(String, u32, bool, bool, bool)> = Vec::new();
        if let Some(arr) = raw.added_tokens {
            added_meta.reserve(arr.len());
            for at in arr {
                if let Some(id) = at.id {
                    vocab_size = vocab_size.max(id as usize + 1);
                    if let Some(content) = at.content {
                        added_meta.push((content, id as u32, at.special, at.lstrip, at.rstrip));
                    }
                }
            }
        }

        // Snapshot common special-token ids from `vocab` so AutoTokenizer never
        // needs to re-parse the 12+ MB tokenizer.json in Python on construction.
        const KNOWN_TOKS: &[&str] = &[
            "<pad>", "[PAD]", "<|pad|>", "<|endoftext|>", "<s>", "</s>",
            "[CLS]", "[SEP]", "<unk>", "[UNK]", "<mask>", "[MASK]",
            "<|begin_of_text|>", "<|end_of_text|>", "<|im_start|>", "<|im_end|>",
        ];
        let mut known_token_ids: Vec<(String, u32)> = Vec::new();
        for &kt in KNOWN_TOKS {
            if let Some(&id) = vocab.get(kt) {
                known_token_ids.push((kt.to_string(), id));
            }
        }

        // Optional end-of-word singleton lookup table (e.g. CLIP's `</w>`).
        let singleton_eow: Option<[i32; 256]> = if let Some(ref suf) = eow_suffix {
            if byte_fallback {
                return Err(PyValueError::new_err(format!(
                    "unsupported end_of_word_suffix {suf:?} with byte_fallback"
                )));
            }
            let mut b2u = ['\0'; 256];
            for (cp, &bv) in c2b.iter().enumerate() {
                if bv != u16::MAX {
                    b2u[bv as usize] = char::from_u32(cp as u32).unwrap();
                }
            }
            let mut tbl = [-1i32; 256];
            let mut key = String::with_capacity(suf.len() + 4);
            for b in 0..256 {
                key.clear();
                key.push(b2u[b]);
                key.push_str(suf);
                if let Some(&id) = vocab.get(key.as_str()) {
                    tbl[b] = id as i32;
                }
            }
            Some(tbl)
        } else {
            None
        };

        // Stream decoded token bytes directly into ONE contiguous byte_arena
        // (eliminating 200K Vec<u8> heap allocations and SipHash lookups).
        let mut byte_arena: Vec<u8> = Vec::with_capacity(vocab.len() * 8);
        let mut byte_span: Vec<(u32, u32)> = vec![(0, 0); vocab_size];
        let mut canonical = vec![false; vocab_size];
        let mut has_internal_ms = if meta_split_cand.is_some() {
            vec![false; vocab_size]
        } else {
            Vec::new()
        };
        let mut char_ascii = [-1i32; 128];
        let mut char_map: FxHashMap<u32, u32> = FxHashMap::default();
        for (k, &id) in &vocab {
            let ks = k.as_str();
            if let Some(rep) = meta_split_cand {
                if ks.trim_start_matches(rep).contains(rep) {
                    has_internal_ms[id as usize] = true;
                }
            }
            let start = byte_arena.len() as u32;
            if byte_fallback {
                decode_byte_fallback_into(ks, &mut byte_arena);
            } else {
                decode_bytelevel_into(ks, &c2b, &mut byte_arena);
            }
            let len = byte_arena.len() as u32 - start;
            byte_span[id as usize] = (start, len);
            let mut chs = ks.chars();
            if let (Some(c), None) = (chs.next(), chs.next()) {
                if byte_fallback {
                    let cp = c as u32;
                    if cp < 128 {
                        char_ascii[cp as usize] = id as i32;
                    } else {
                        char_map.insert(cp, id);
                    }
                } else {
                    let cp = c as usize;
                    canonical[id as usize] = cp < 324 && c2b[cp] != u16::MAX;
                }
            }
        }
        drop(vocab);
        let mut special_skip_mask = vec![false; vocab_size];
        for (content, id, sp, _, _) in &added_meta {
            let id = *id as usize;
            if id < vocab_size {
                if *sp {
                    special_skip_mask[id] = true;
                }
                if byte_span[id].1 == 0 {
                    let start = byte_arena.len() as u32;
                    byte_arena.extend_from_slice(content.as_bytes());
                    byte_span[id] = (start, content.len() as u32);
                }
            }
        }

        let mut singleton = [-1i32; 256];
        let mut singleton_canon = [false; 256];
        for id in 0..vocab_size {
            let (off, len) = byte_span[id];
            if len == 1 {
                let bv = byte_arena[off as usize] as usize;
                let canon = canonical[id];
                if singleton[bv] < 0
                    || (canon && !singleton_canon[bv])
                    || (canon == singleton_canon[bv] && (id as i32) < singleton[bv])
                {
                    singleton[bv] = id as i32;
                    singleton_canon[bv] = canon;
                }
            }
        }

        let merges = model.merges;
        // Enable metaspace chunk-splitting (and guarded cross-space splitting for
        // models like Gemma-3 with rare cross-space merges such as `">▁</"`),
        // overlapped with MergeMap construction via rayon::join.
        let allow_par = parallel_ok();
        let check_meta_and_mono = || {
            let sp_info = analyze_sp_cross_prev(
                meta_split_cand,
                &merges,
                &has_internal_ms,
                &byte_arena,
                &byte_span,
            );
            let mut produce_max: Vec<i64> = vec![-1; vocab_size];
            let mut consume_min: Vec<i64> = vec![i64::MAX; vocab_size];
            let mut len_ok = !byte_fallback
                && singleton_eow.is_none()
                && singleton.iter().enumerate().all(|(b, &sid)| {
                    matches!(b, 0xC0 | 0xC1 | 0xF5..=0xFF)
                        || (sid >= 0 && (sid as usize) < vocab_size && byte_span[sid as usize].1 == 1)
                });
            for (i, &(a, b, t)) in merges.iter().enumerate() {
                let i = i as i64;
                let (au, bu, tu) = (a as usize, b as usize, t as usize);
                if tu < vocab_size {
                    produce_max[tu] = produce_max[tu].max(i);
                }
                if au < vocab_size {
                    consume_min[au] = consume_min[au].min(i);
                }
                if bu < vocab_size {
                    consume_min[bu] = consume_min[bu].min(i);
                }
                if len_ok
                    && (au >= vocab_size
                        || bu >= vocab_size
                        || tu >= vocab_size
                        || byte_span[tu].1 != byte_span[au].1 + byte_span[bu].1)
                {
                    len_ok = false;
                }
            }
            let mono = (0..vocab_size).all(|t| produce_max[t] < 0 || consume_min[t] > produce_max[t]);
            (sp_info, mono, len_ok)
        };
        let (mut map, ((safe_meta_split, sp_split_ok, sp_cross_prev), monotonic, byte_len_exact)) = if allow_par {
            rayon::join(
                || bpe::MergeMap::new_with_opts(&merges, true, byte_fallback),
                check_meta_and_mono,
            )
        } else {
            (bpe::MergeMap::new_with_opts(&merges, false, byte_fallback), check_meta_and_mono())
        };

        let short_in_ct = !byte_fallback && singleton_eow.is_none();
        let ignore_lookup = if ignore_merges {
            let mut m: FxHashMap<Box<[u8]>, u32> = FxHashMap::default();
            for (id, &(off, len)) in byte_span.iter().enumerate() {
                let lu = len as usize;
                if lu > 0 && (!short_in_ct || lu > chunk_table::MAX_KEY) {
                    let b = &byte_arena[off as usize..(off + len) as usize];
                    m.entry(b.to_vec().into_boxed_slice()).or_insert(id as u32);
                }
            }
            if m.is_empty() { None } else { Some(m) }
        } else {
            None
        };

        let mut singleton_u32 = [0u32; 256];
        for b in 0..256 {
            if singleton[b] >= 0 {
                singleton_u32[b] = singleton[b] as u32;
            }
        }
        let byte_fast = !byte_fallback
            && singleton_eow.is_none()
            && map.init_byte_tables(&singleton, &merges, monotonic);

        let sp_fast_meta = detect_sp_fast_meta(
            byte_fallback,
            ignore_merges,
            singleton_eow.is_some(),
            sp_split_ok,
            &normalizer,
            pretok_pipe.as_ref(),
            pre,
        );
        let sp_has_cross = sp_cross_prev != [0u64; 4];
        let sp_rep_id = char_map.get(&('\u{2581}' as u32)).map(|&x| x as i32).unwrap_or(-1);

        let mut tk = Tokenizer {
            kind,
            fast_pretok,
            singleton,
            singleton_u32,
            singleton_eow,
            byte_fast,
            map,
            raw_merges: merges,
            stream: OnceLock::new(),
            byte_arena,
            byte_span,
            c2b,
            ignore_merges,
            ignore_lookup,
            vocab_size,
            normalizer,
            pretok_pipe,
            byte_fallback,
            safe_meta_split,
            sp_fast_meta,
            sp_cross_prev,
            sp_has_cross,
            sp_rep_id,
            char_ascii,
            char_map,
            decode_strip_space,
            source_path: path.to_string(),
            is_byte_level: !byte_fallback,
            byte_len_exact,
            monotonic,
            special_ac: None,
            special_prefilter: SpecialPrefilter::None,
            special_ids: Vec::new(),
            special_lstrip: Vec::new(),
            special_rstrip: Vec::new(),
            special_max_len: 0,
            special_skip_mask,
            whisper_ts_begin: u32::MAX,
            whisper_sop_id: u32::MAX,
            whisper_sot_id: u32::MAX,
            prepend_scheme: 0,
            chunk_table: chunk_table::ChunkTable::empty(),
            uid: NEXT_UID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            learned: chunk_table::SharedTable::new(DEFAULT_CACHE_MIB),
            long_table: chunk_table::SharedLongTable::new(DEFAULT_CACHE_MIB / 4),
            spill: chunk_table::SpillMap::new((DEFAULT_CACHE_MIB << 20) / 4),
            int_table: OnceLock::new(),
            decode_table: OnceLock::new(),
            auto_meta: AutoMeta {
                added_tokens: added_meta,
                post_processor_json,
                prepend_scheme: prepend_scheme_str,
                known_token_ids,
            },
        };
        tk.build_chunk_table();
        Ok(tk)
    }

    /// Compact AutoTokenizer metadata parsed during `Tokenizer::new` so Python
    /// avoids re-parsing the full tokenizer.json on construction.
    fn _auto_meta(&self) -> AutoMetaTuple {
        (
            self.auto_meta.added_tokens.clone(),
            self.auto_meta.post_processor_json.clone(),
            self.auto_meta.prepend_scheme.clone(),
            self.auto_meta.known_token_ids.clone(),
        )
    }

    /// Optional warmup to eliminate cold-start latency before serving/encoding.
    ///
    /// - `warmup()` (no `texts`): pre-builds the cached Python int table and
    ///   warms the rayon thread pool.
    /// - `warmup(texts, num_threads=0)`: runs a fast parallel learning pass over
    ///   representative domain text(s) (`str`, `bytes`, or an iterable of them)
    ///   without allocating output token buffers, populating the shared chunk
    ///   tables so production calls start at full warm throughput (~3-5x faster).
    ///
    /// Returns the number of cached entries in the shared table.
    #[pyo3(signature = (texts = None, num_threads = 0))]
    fn warmup(
        &self,
        py: Python<'_>,
        texts: Option<&Bound<'_, PyAny>>,
        num_threads: usize,
    ) -> PyResult<usize> {
        let _ = self.ints(py);
        if num_threads > 1 {
            let _ = fixed_pool(num_threads);
        }
        if let Some(obj) = texts {
            let items: Vec<TextArg> = if obj.is_instance_of::<pyo3::types::PyString>()
                || obj.is_instance_of::<pyo3::types::PyBytes>()
                || obj.is_instance_of::<pyo3::types::PyByteArray>()
            {
                vec![obj.extract()?]
            } else {
                obj.extract()?
            };
            if !items.is_empty() {
                let stream_wide = self.stream_wide_ok(None);
                let wide = if stream_wide {
                    WideCopies::default()
                } else {
                    wide_args(py, &items, num_threads)?
                };
                let res = py.detach(|| {
                    self.encode_batch_sep_t::<u32>(&items, &wide, stream_wide, num_threads, false, None)
                });
                if let Err(err) = res {
                    return Err(match err {
                        Ok(b) => bad_input_to_pyerr(py, &items, b),
                        Err(e) => e,
                    });
                }
            }
        }
        Ok(self.learned.len())
    }

    /// Memory cap (MiB) of this tokenizer's shared learned-chunk cache: chunks
    /// the vocabulary table can't answer are merged once and remembered here,
    /// for every thread. Grows on demand from 512 KiB. 0 disables learning.
    #[getter]
    fn cache_limit_mb(&self) -> usize {
        self.learned.limit_mib()
    }

    fn set_cache_limit_mb(&self, mb: usize) {
        self.learned.set_limit_mib(mb);
        self.long_table.set_limit_mib(mb / 4);
        self.spill.set_limit((mb << 20) / 4);
    }

    /// Entries in the shared chunk table: the vocabulary's single-token chunks
    /// (seeded at load) plus every encoding learned since (approximate under
    /// concurrency).
    #[getter]
    fn learned_entries(&self) -> usize {
        self.learned.len()
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

    /// Load by HuggingFace repo id (downloads `tokenizer.json`, or transparently
    /// converts `tekken.json`, `tiktoken.model`, or legacy `vocab.json`+`merges.txt`
    /// when a repo omits `tokenizer.json`). Requires `huggingface_hub`.
    #[staticmethod]
    #[pyo3(signature = (repo_id, revision = None, warmup = None))]
    fn from_pretrained(
        py: Python<'_>,
        repo_id: &str,
        revision: Option<&str>,
        warmup: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let hub = py.import("huggingface_hub").map_err(|_| {
            PyValueError::new_err(
                "from_pretrained requires huggingface_hub (pip install huggingface_hub)",
            )
        })?;
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(rev) = revision {
            kwargs.set_item("revision", rev)?;
        }
        let path: String = match hub.call_method("hf_hub_download", (repo_id, "tokenizer.json"), Some(&kwargs)) {
            Ok(p) => p.extract()?,
            Err(orig_err) => {
                let fmt = py.import("tuetoken._formats")?;
                let res = fmt.call_method1("resolve_non_json_repo", (repo_id, revision))?;
                if res.is_none() {
                    return Err(orig_err);
                }
                res.extract()?
            }
        };
        let tk = Self::new(&path)?;
        tk.apply_warmup_arg(py, warmup)?;
        Ok(tk)
    }

    /// Load a Mistral `tekken.json` tokenizer from a local path or HF repo id.
    #[staticmethod]
    #[pyo3(signature = (source, revision = None, warmup = None))]
    fn from_tekken(
        py: Python<'_>,
        source: &str,
        revision: Option<&str>,
        warmup: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let fmt = py.import("tuetoken._formats")?;
        let res = fmt.call_method1("try_load_tekken_repo", (source, revision))?;
        if res.is_none() {
            return Err(PyValueError::new_err(format!("no tekken.json found for {source:?}")));
        }
        let (path, _cfg): (String, Py<PyAny>) = res.extract()?;
        let tk = Self::new(&path)?;
        tk.apply_warmup_arg(py, warmup)?;
        Ok(tk)
    }

    /// Load an OpenAI tiktoken encoding by name (e.g. "cl100k_base", "o200k_base").
    /// Downloads the rank file and builds a tokenizer.json (cached).
    #[staticmethod]
    #[pyo3(signature = (name, warmup = None))]
    fn from_tiktoken(py: Python<'_>, name: &str, warmup: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let helper = py.import("tuetoken._tiktoken")?;
        let path: String = helper
            .call_method1("get_cached_tokenizer_path", (name,))?
            .extract()?;
        let tk = Self::new(&path)?;
        tk.apply_warmup_arg(py, warmup)?;
        Ok(tk)
    }

    fn encode_ordinary<'py>(&self, py: Python<'py>, text: &str) -> Bound<'py, PyList> {
        if text.len() < Self::DETACH_MIN {
            return self.with_scratch(|s| {
                s.out.clear();
                self.encode_into(text, Merge::Auto, true, s);
                self.to_pylist(py, &s.out)
            });
        }
        let ids = py.detach(|| self.encode_one(text, Merge::Auto));
        let list = self.to_pylist(py, &ids);
        // Recycle `ids` allocation back into thread-local scratch so repeated
        // single-prompt calls (4KB–128KB) avoid per-call Vec malloc/free.
        if ids.capacity() <= (1 << 20) {
            let _ = SCRATCH.try_with(|cell| {
                if let Ok(mut s) = cell.try_borrow_mut() {
                    if ids.capacity() > s.out.capacity() {
                        s.out = ids;
                    }
                }
            });
        }
        list
    }

    /// Like `encode_ordinary` but WITHOUT the SentencePiece metaspace prepend —
    /// used by the AutoTokenizer wrapper for fragments after a special token under
    /// `prepend_scheme="first"`. Identical to `encode_ordinary` for non-metaspace
    /// models.
    fn encode_ordinary_no_prefix<'py>(&self, py: Python<'py>, text: &str) -> Bound<'py, PyList> {
        if text.len() < Self::DETACH_MIN {
            return self.with_scratch(|s| {
                s.out.clear();
                self.encode_into(text, Merge::Auto, false, s);
                self.to_pylist(py, &s.out)
            });
        }
        let ids = py.detach(|| self.encode_one_p(text, Merge::Auto, false));
        self.to_pylist(py, &ids)
    }

    /// Register the AutoTokenizer's special tokens so `encode_special` can split on
    /// them in Rust. Parallel arrays `contents`/`ids`/`lstrip`/`rstrip`; `scheme` is
    /// the metaspace prepend rule for fragments (0 none, 1 always, 2 first).
    fn set_special_tokens(&mut self, contents: Vec<String>, ids: Vec<u32>,
                          lstrip: Vec<bool>, rstrip: Vec<bool>, scheme: u8) -> PyResult<()> {
        use aho_corasick::{AhoCorasick, MatchKind};
        self.special_prefilter = SpecialPrefilter::from_patterns(&contents);
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
        self.special_max_len = contents.iter().map(|s| s.len()).max().unwrap_or(0);
        self.special_ids = ids;
        self.special_lstrip = lstrip;
        self.special_rstrip = rstrip;
        self.prepend_scheme = scheme;
        Ok(())
    }

    /// Configure Whisper-specific decode handling in Rust (timestamp token range
    /// starting at `ts_begin`, plus `<|startofprev|>` / `<|startoftranscript|>` ids
    /// for prompt stripping when `skip_special_tokens=True`).
    fn set_whisper_decode(&mut self, ts_begin: u32, sop_id: u32, sot_id: u32) {
        self.whisper_ts_begin = ts_begin;
        self.whisper_sop_id = sop_id;
        self.whisper_sot_id = sot_id;
    }

    /// Special-token-aware encode (WITHOUT the post-processor's BOS/EOS): split on
    /// the registered added tokens (leftmost-longest, honouring lstrip/rstrip), BPE
    /// each gap, splice the special ids — all in one Rust pass. Same ids as the
    /// Python fragment loop, but no per-fragment Python↔Rust crossing.
    fn encode_special<'py>(&self, py: Python<'py>, text: &str) -> Bound<'py, PyList> {
        if text.len() < Self::DETACH_MIN {
            return self.with_scratch(|s| {
                s.out.clear();
                self.encode_special_into(text, s);
                self.to_pylist(py, &s.out)
            });
        }
        let ids = py.detach(|| self.encode_special_one(text));
        self.to_pylist(py, &ids)
    }

    /// Batch of `encode_special`, parallel over rayon (GIL released) — so the
    /// AutoTokenizer's `__call__` on a list is one Rust pass instead of a Python
    /// loop. `num_threads` 0 = all cores.
    #[pyo3(signature = (texts, num_threads = 0))]
    fn encode_special_batch<'py>(&self, py: Python<'py>, texts: &Bound<'py, PyAny>, num_threads: usize) -> PyResult<Bound<'py, PyList>> {
        let (seqs, _) = self.build_special_post_lists(py, texts, &[], &[], None, false, false, num_threads)?;
        Ok(seqs)
    }

    /// AutoTokenizer `__call__` fast path (single sequences, no tensors): special
    /// -aware encode of every text in parallel, then build each row's Python list
    /// directly as `prefix + content + suffix` (content truncated to `budget`,
    /// from the left if `trunc_left`) from the pre-built int table — no Python
    /// per-row list surgery.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (texts, prefix, suffix, budget = None, trunc_left = false, num_threads = 0))]
    fn encode_special_post<'py>(
        &self,
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        prefix: Vec<u32>,
        suffix: Vec<u32>,
        budget: Option<usize>,
        trunc_left: bool,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let (seqs, _) = self.build_special_post_lists(
            py,
            texts,
            &prefix,
            &suffix,
            budget,
            trunc_left,
            false,
            num_threads,
        )?;
        Ok(seqs)
    }

    /// Same as `encode_special_post`, optionally also constructing the all-1s
    /// `attention_mask` Python lists in the same Rust FFI pass (avoiding a
    /// Python `[[1] * len(q) for q in seqs]` loop over 10K+ rows).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (texts, prefix, suffix, budget = None, trunc_left = false, want_mask = true, num_threads = 0))]
    fn encode_special_post_masked<'py>(
        &self,
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        prefix: Vec<u32>,
        suffix: Vec<u32>,
        budget: Option<usize>,
        trunc_left: bool,
        want_mask: bool,
        num_threads: usize,
    ) -> PyResult<(Bound<'py, PyList>, Option<Bound<'py, PyList>>)> {
        self.build_special_post_lists(
            py,
            texts,
            &prefix,
            &suffix,
            budget,
            trunc_left,
            want_mask,
            num_threads,
        )
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

    /// Count tokens for one string without constructing a Python list or (on
    /// long splittable documents) allocating an output token buffer.
    fn count_tokens(&self, py: Python<'_>, text: &str) -> usize {
        if text.len() < Self::DETACH_MIN {
            return self.count_ordinary_one(text);
        }
        py.detach(|| self.count_ordinary_one(text))
    }

    /// Count tokens for one string with registered added/special tokens (without
    /// post-processor BOS/EOS or Python list allocation).
    fn count_special_tokens(&self, py: Python<'_>, text: &str) -> usize {
        if text.len() < Self::DETACH_MIN {
            return self.count_special_one(text);
        }
        py.detach(|| self.count_special_one(text))
    }

    /// Parallel token counts with registered added/special tokens (zero output
    /// token buffer allocation on FastBatch; GIL released).
    #[pyo3(signature = (texts, num_threads = 0))]
    fn count_special_tokens_batch(
        &self,
        py: Python<'_>,
        texts: &Bound<'_, PyAny>,
        num_threads: usize,
    ) -> PyResult<Vec<usize>> {
        if let Some(batch) = try_extract_fast_batch(py, texts) {
            if !batch.has_bytes {
                let slice = &batch.items;
                let total_len = batch.total_len;
                if self.normalizer.is_ascii_identity() && self.items_pipe_ok() && batch.max_len < SPLIT_MIN {
                    let res = self.maybe_detach(py, total_len, || {
                        self.encode_fast_short_t::<u32, true>(slice, total_len, num_threads, false)
                    });
                    return match res {
                        Ok((_, counts)) => Ok(counts),
                        Err(b) => Err(bad_fast_to_pyerr(py, &batch, b)),
                    };
                }
                let res = self.maybe_detach(py, total_len, || {
                    self.count_fast_items(slice, total_len, num_threads, true)
                });
                return match res {
                    Ok(counts) => Ok(counts),
                    Err(b) => Err(bad_fast_to_pyerr(py, &batch, b)),
                };
            }
        }
        let texts: Vec<PyText> = texts.extract()?;
        let wide = wide_texts(py, &texts, num_threads)?;
        let views = text_views(&texts, &wide);
        let total: usize = views.iter().map(|t| t.len()).sum();
        if num_threads == 1 || views.len() <= 1 || !parallel_ok() || total < 65_536 {
            return Ok(self.maybe_detach(py, total, || {
                views.iter().map(|t| self.count_special_one(t)).collect()
            }));
        }
        Ok(py.detach(|| {
            self.par_over(num_threads, &views, |t| t.len(), |t| self.count_special_one(t))
        }))
    }

    /// Zero-copy mmap file tokenization: maps `path` straight from the OS page
    /// cache (with `MADV_SEQUENTIAL` on Linux/macOS, falling back to `fs::read`
    /// for pipes/non-regular files) and tokenizes in parallel with the GIL
    /// released into `{"data": np.ndarray (uint32/uint16), "offsets": np.ndarray (int64)}`.
    #[pyo3(signature = (path, num_threads = 0, separator = None, dtype = None))]
    fn encode_file_to_buffer<'py>(
        &self,
        py: Python<'py>,
        path: std::path::PathBuf,
        num_threads: usize,
        separator: Option<&Bound<'py, PyAny>>,
        dtype: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sep = extract_sep_arg(separator)?;
        let np_mod = || {
            py.import("numpy").map_err(|_| {
                PyValueError::new_err("encode_file_to_buffer requires numpy (pip install numpy)")
            })
        };
        let narrow = match &dtype {
            None => false,
            Some(d) => {
                let name: String = np_mod()?.call_method1("dtype", (d,))?.getattr("name")?.extract()?;
                match name.as_str() {
                    "uint32" => false,
                    "uint16" => {
                        if self.vocab_size > 1 << 16 {
                            return Err(PyValueError::new_err(format!(
                                "dtype uint16 needs a vocabulary of at most 65536 ids (this one has {})",
                                self.vocab_size
                            )));
                        }
                        true
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "dtype must be uint16 or uint32, not {other}"
                        )))
                    }
                }
            }
        };
        let (data_ba, counts) = if narrow {
            let res = py.detach(|| -> Result<(Vec<u16>, Vec<usize>), FileEncodeErr> {
                let fb = FileBytes::open(&path).map_err(FileEncodeErr::Io)?;
                self.encode_raw_bytes_sep_t::<u16>(fb.as_slice(), num_threads, true, sep.as_deref())
                    .map_err(FileEncodeErr::Encode)
            });
            let (d, c) = res.map_err(file_err_to_pyerr)?;
            (Bound::new(py, TokenBuffer::new_u16(d))?, c)
        } else {
            let res = py.detach(|| -> Result<(Vec<u32>, Vec<usize>), FileEncodeErr> {
                let fb = FileBytes::open(&path).map_err(FileEncodeErr::Io)?;
                self.encode_raw_bytes_sep_t::<u32>(fb.as_slice(), num_threads, true, sep.as_deref())
                    .map_err(FileEncodeErr::Encode)
            });
            let (d, c) = res.map_err(file_err_to_pyerr)?;
            (Bound::new(py, TokenBuffer::new(d))?, c)
        };
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0i64);
        let mut total = 0usize;
        for &c in &counts {
            total += c;
            offsets.push(total as i64);
        }
        let np = np_mod()?;
        let data_arr = np.call_method1("frombuffer", (data_ba, if narrow { "uint16" } else { "uint32" }))?;
        let off_ba = Bound::new(py, TokenBuffer::new_i64(offsets))?;
        let off_arr = np.call_method1("frombuffer", (off_ba, "int64"))?;
        let out = PyDict::new(py);
        out.set_item("data", data_arr)?;
        out.set_item("offsets", off_arr)?;
        Ok(out)
    }

    /// Zero-copy mmap file token counting with zero output token buffer
    /// allocation. Returns `int` when `separator is None`, or `list[int]` (one
    /// count per separated document) when `separator` is given.
    #[pyo3(signature = (path, num_threads = 0, separator = None))]
    fn count_tokens_file(
        &self,
        py: Python<'_>,
        path: std::path::PathBuf,
        num_threads: usize,
        separator: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let sep = extract_sep_arg(separator)?;
        let has_sep = sep.is_some();
        let res = py.detach(|| -> Result<Vec<usize>, FileEncodeErr> {
            let fb = FileBytes::open(&path).map_err(FileEncodeErr::Io)?;
            self.encode_raw_bytes_sep_t::<u32>(fb.as_slice(), num_threads, false, sep.as_deref())
                .map(|(_, c)| c)
                .map_err(FileEncodeErr::Encode)
        });
        let counts = res.map_err(file_err_to_pyerr)?;
        if has_sep {
            Ok(counts.into_pyobject(py)?.into_any().unbind())
        } else {
            let total: usize = counts.into_iter().sum();
            Ok(total.into_pyobject(py)?.into_any().unbind())
        }
    }

    /// Truncate `text` to at most `max_tokens` tokens without constructing a
    /// Python `list[int]`. Returns `text` unchanged when it already fits within
    /// `max_tokens`. `side` is `"right"` (keep prefix, default) or `"left"`
    /// (keep suffix); `special=True` recognizes registered special tokens.
    #[pyo3(signature = (text, max_tokens, side = "right", special = false))]
    fn truncate<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyString>,
        max_tokens: usize,
        side: &str,
        special: bool,
    ) -> PyResult<Bound<'py, PyString>> {
        let right = match side {
            "right" => true,
            "left" => false,
            other => {
                return Err(PyValueError::new_err(format!(
                    "side must be 'right' or 'left', not {other:?}"
                )))
            }
        };
        if max_tokens == 0 {
            return Ok(PyString::new(py, ""));
        }
        let s_cow = text.to_cow()?;
        let txt: &str = &s_cow;
        if txt.is_empty() {
            return Ok(text.clone());
        }
        // O(max_tokens) early-exit fast path for BOTH right and left truncation:
        // never scans or encodes the discarded portion of multi-MB documents!
        let early_thresh = if right {
            max_tokens.saturating_mul(6).saturating_add(128)
        } else {
            max_tokens.saturating_mul(8).saturating_add(768)
        };
        if txt.len() > early_thresh {
            let need = max_tokens.saturating_add(1);
            let run_into = |sc: &mut Scratch| {
                sc.out.clear();
                if right {
                    if special && self.special_ac.is_some() {
                        self.encode_special_prefix_into(txt, need, sc);
                    } else {
                        self.encode_ordinary_prefix_into(txt, need, sc);
                    }
                } else if special && self.special_ac.is_some() {
                    self.encode_special_suffix_into(txt, need, sc);
                } else {
                    self.encode_ordinary_suffix_into(txt, need, sc);
                }
            };
            if max_tokens < 2048 && self.normalizer.is_identity() {
                return Ok(self.with_scratch(|sc| {
                    run_into(sc);
                    if sc.out.len() <= max_tokens {
                        return text.clone();
                    }
                    let sub = if right {
                        &sc.out[..max_tokens]
                    } else {
                        &sc.out[sc.out.len() - max_tokens..]
                    };
                    self.decode_slice_to_pystr(py, sub, false, false, 0.02, 1500, &mut sc.utf8)
                }));
            }
            let ids: Vec<u32> = py.detach(|| {
                self.with_scratch(|sc| {
                    run_into(sc);
                    sc.out.clone()
                })
            });
            if ids.len() <= max_tokens {
                return Ok(text.clone());
            }
            let sub = if right {
                &ids[..max_tokens]
            } else {
                &ids[ids.len() - max_tokens..]
            };
            return Ok(self.with_scratch(|sc| {
                self.decode_slice_to_pystr(py, sub, false, false, 0.02, 1500, &mut sc.utf8)
            }));
        }
        if txt.len() < Self::DETACH_MIN {
            return Ok(self.with_scratch(|sc| {
                sc.out.clear();
                if special && self.special_ac.is_some() {
                    self.encode_special_into(txt, sc);
                } else {
                    self.encode_into(txt, Merge::Auto, true, sc);
                }
                if sc.out.len() <= max_tokens {
                    return text.clone();
                }
                let sub = if right {
                    &sc.out[..max_tokens]
                } else {
                    &sc.out[sc.out.len() - max_tokens..]
                };
                self.decode_slice_to_pystr(py, sub, false, false, 0.02, 1500, &mut sc.utf8)
            }));
        }
        let ids: Vec<u32> = py.detach(|| {
            if !special && txt.len() >= SPLIT_MIN && self.splittable() {
                self.encode_direct(&[txt], 0, true).0
            } else {
                self.with_scratch(|sc| {
                    sc.out.clear();
                    if special && self.special_ac.is_some() {
                        self.encode_special_into(txt, sc);
                    } else {
                        self.encode_into(txt, Merge::Auto, true, sc);
                    }
                    sc.out.clone()
                })
            }
        });
        if ids.len() <= max_tokens {
            return Ok(text.clone());
        }
        let sub = if right {
            &ids[..max_tokens]
        } else {
            &ids[ids.len() - max_tokens..]
        };
        Ok(self.with_scratch(|sc| {
            self.decode_slice_to_pystr(py, sub, false, false, 0.02, 1500, &mut sc.utf8)
        }))
    }

    /// Split `text` into consecutive text chunks of at most `max_tokens` tokens
    /// each (with `overlap` shared tokens between adjacent chunks) directly in
    /// Rust without Python `list[int]` boxing. Ideal for RAG and long-context
    /// windowing.
    #[pyo3(signature = (text, max_tokens, overlap = 0, special = false))]
    fn chunk_text<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyString>,
        max_tokens: usize,
        overlap: usize,
        special: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        if max_tokens == 0 {
            return Err(PyValueError::new_err("max_tokens must be > 0"));
        }
        if overlap >= max_tokens {
            return Err(PyValueError::new_err("overlap must be < max_tokens"));
        }
        let s_cow = text.to_cow()?;
        let txt: &str = &s_cow;
        if txt.is_empty() {
            return Ok(PyList::empty(py));
        }
        let single_list = || -> Bound<'py, PyList> {
            unsafe {
                let list = pyo3::ffi::PyList_New(1);
                assert!(!list.is_null(), "list alloc");
                pyo3::ffi::PyList_SET_ITEM(list, 0, text.clone().into_ptr());
                Bound::from_owned_ptr(py, list).cast_into_unchecked()
            }
        };
        if txt.len() < Self::DETACH_MIN {
            return Ok(self.with_scratch(|sc| {
                sc.out.clear();
                if special && self.special_ac.is_some() {
                    self.encode_special_into(txt, sc);
                } else {
                    self.encode_into(txt, Merge::Auto, true, sc);
                }
                let n = sc.out.len();
                if n <= max_tokens {
                    return single_list();
                }
                let step = max_tokens - overlap;
                let nchunks = (n - overlap).div_ceil(step);
                unsafe {
                    let list = pyo3::ffi::PyList_New(nchunks as pyo3::ffi::Py_ssize_t);
                    assert!(!list.is_null(), "list alloc");
                    let mut pos = 0usize;
                    for idx in 0..nchunks {
                        let end = (pos + max_tokens).min(n);
                        let piece = self.decode_slice_to_pystr(
                            py,
                            &sc.out[pos..end],
                            false,
                            false,
                            0.02,
                            1500,
                            &mut sc.utf8,
                        );
                        pyo3::ffi::PyList_SET_ITEM(list, idx as pyo3::ffi::Py_ssize_t, piece.into_ptr());
                        pos += step;
                    }
                    Bound::from_owned_ptr(py, list).cast_into_unchecked()
                }
            }));
        }
        let ids: Vec<u32> = py.detach(|| {
            if !special && txt.len() >= SPLIT_MIN && self.splittable() {
                self.encode_direct(&[txt], 0, true).0
            } else {
                self.with_scratch(|sc| {
                    sc.out.clear();
                    if special && self.special_ac.is_some() {
                        self.encode_special_into(txt, sc);
                    } else {
                        self.encode_into(txt, Merge::Auto, true, sc);
                    }
                    sc.out.clone()
                })
            }
        });
        let n = ids.len();
        if n <= max_tokens {
            return Ok(single_list());
        }
        let step = max_tokens - overlap;
        let nchunks = (n - overlap).div_ceil(step);
        Ok(self.with_scratch(|sc| unsafe {
            let list = pyo3::ffi::PyList_New(nchunks as pyo3::ffi::Py_ssize_t);
            assert!(!list.is_null(), "list alloc");
            let mut pos = 0usize;
            for idx in 0..nchunks {
                let end = (pos + max_tokens).min(n);
                let piece = self.decode_slice_to_pystr(
                    py,
                    &ids[pos..end],
                    false,
                    false,
                    0.02,
                    1500,
                    &mut sc.utf8,
                );
                pyo3::ffi::PyList_SET_ITEM(list, idx as pyo3::ffi::Py_ssize_t, piece.into_ptr());
                pos += step;
            }
            Bound::from_owned_ptr(py, list).cast_into_unchecked()
        }))
    }

    /// Encode and return, for each token, its byte span `(start, end)` in `text`
    /// — for highlighting, NER alignment, char-level truncation. Returns
    /// `(ids, offsets)`. ByteLevel models only (byte_fallback/SentencePiece
    /// transform the text so byte offsets wouldn't map back; raises for those).
    fn encode_with_offsets<'py>(
        &self,
        py: Python<'py>,
        text: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if !self.is_byte_level || self.singleton_eow.is_some() {
            return Err(PyValueError::new_err(
                "encode_with_offsets is only available for plain ByteLevel models \
                 (byte_fallback/SentencePiece or end_of_word_suffix models transform \
                 the text, so offsets wouldn't map to the original)",
            ));
        }
        // A normalizer rewrites the text (e.g. NFC composes decomposed chars),
        // which shifts byte positions — offsets into the original would be wrong.
        // Already-normalized input returns `Cow::Borrowed` (zero O(n) memcmp!).
        if !self.normalizer.is_identity()
            && matches!(self.normalize(text), std::borrow::Cow::Owned(ref o) if o.as_str() != text)
        {
            return Err(PyValueError::new_err(
                "encode_with_offsets: input is not already normalized and this \
                 model applies a normalizer; byte offsets into the \
                 original text are undefined (normalize the text first)",
            ));
        }
        if self.fast_pretok == FastPretok::Pipe
            && !self.pretok_pipe.as_ref().is_some_and(|p| p.is_slice_only())
        {
            return Err(PyValueError::new_err(
                "encode_with_offsets: this model's pre_tokenizer mutates piece bytes \
                 (e.g. add_prefix_space or post-ByteLevel splitting); raw input byte \
                 offsets are undefined",
            ));
        }
        // Fast path (100% of real ByteLevel models, including non-monotonic models
        // like Qwen3.6-35B, Llama-4, gpt-oss-20b!): encode via our full SIMD +
        // SharedTable + parallel engine, then reconstruct exact contiguous byte
        // spans from `byte_span[t].1` with adjacent `PyLong` sharing!
        if self.byte_len_exact {
            if text.len() < Self::DETACH_MIN {
                return self.with_scratch(|s| {
                    s.out.clear();
                    self.encode_into(text, Merge::Auto, true, s);
                    self.build_offsets_pytuple(py, &s.out)
                });
            }
            let ids = py.detach(|| self.encode_one(text, Merge::Auto));
            return self.build_offsets_pytuple(py, &ids);
        }
        // Fallback for synthetic ByteLevel vocabs missing some byte singletons:
        // uses the batch span-merger (`merge_with_spans`), which requires monotonic merges.
        if !self.monotonic {
            return Err(PyValueError::new_err(
                "encode_with_offsets: this model's merge table is non-monotonic \
                 and requires the canonical merger, which the offset path does \
                 not implement; offsets would not match encode() (encode/decode \
                 are unaffected)",
            ));
        }
        let bytes = text.as_bytes();
        let (ids, spans) = py.detach(|| {
            let mut ids = Vec::new();
            let mut spans = Vec::new();
            let (mut toks, mut sp, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
            match self.fast_pretok {
                FastPretok::Cjk => {
                    let (mut cur, mut nxt) = (Vec::new(), Vec::new());
                    cjk::chunk_ranges(text, &mut cur, &mut nxt);
                    for &(ps, pe) in &cur {
                        self.emit_chunk_offsets(
                            bytes, ps, pe, &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                        );
                    }
                }
                FastPretok::BlockScan => {
                    let n = bytes.len();
                    let mut pos = 0usize;
                    while pos < n {
                        let end = pretok::next_chunk_end(self.kind, bytes, pos);
                        self.emit_chunk_offsets(
                            bytes, pos, end, &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                        );
                        pos = end;
                    }
                }
                FastPretok::Pipe => {
                    let pipe = self.pretok_pipe.as_ref().expect("Pipe mode has pretok_pipe");
                    for (ps, pe) in pipe.chunk_ranges(text) {
                        self.emit_chunk_offsets(
                            bytes, ps, pe, &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                        );
                    }
                }
                FastPretok::WholeText => {
                    self.emit_chunk_offsets(
                        bytes, 0, bytes.len(), &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                    );
                }
            }
            (ids, spans)
        });
        let ids_list = self.to_pylist(py, &ids);
        let spans_py = spans.into_pyobject(py)?;
        PyTuple::new(py, [ids_list.into_any(), spans_py.into_any()])
    }

    /// Zero-Python-object byte-offset encoding: returns a dict with zero-copy
    /// numpy arrays `{"input_ids": uint32[N], "offsets": int64[N, 2]}` where
    /// `offsets[i] = [start_byte, end_byte]` in `text`. Avoids per-token Python
    /// tuple & int allocation (50x faster than `encode_with_offsets` on 1 MB+).
    /// ByteLevel models only (same preconditions as `encode_with_offsets`).
    fn encode_with_offsets_buffer<'py>(
        &self,
        py: Python<'py>,
        text: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        if !self.is_byte_level || self.singleton_eow.is_some() {
            return Err(PyValueError::new_err(
                "encode_with_offsets_buffer is only available for plain ByteLevel models \
                 (byte_fallback/SentencePiece or end_of_word_suffix models transform \
                 the text, so offsets wouldn't map to the original)",
            ));
        }
        if !self.normalizer.is_identity()
            && matches!(self.normalize(text), std::borrow::Cow::Owned(ref o) if o.as_str() != text)
        {
            return Err(PyValueError::new_err(
                "encode_with_offsets_buffer: input is not already normalized and this \
                 model applies a normalizer; byte offsets into the \
                 original text are undefined (normalize the text first)",
            ));
        }
        if self.fast_pretok == FastPretok::Pipe
            && !self.pretok_pipe.as_ref().is_some_and(|p| p.is_slice_only())
        {
            return Err(PyValueError::new_err(
                "encode_with_offsets_buffer: this model's pre_tokenizer mutates piece bytes \
                 (e.g. add_prefix_space or post-ByteLevel splitting); raw input byte \
                 offsets are undefined",
            ));
        }
        let np = py.import("numpy").map_err(|_| {
            PyValueError::new_err("encode_with_offsets_buffer requires numpy (pip install numpy)")
        })?;
        let (ids, offs_flat): (Vec<u32>, Vec<i64>) = if self.byte_len_exact {
            let build = |ids: Vec<u32>| -> (Vec<u32>, Vec<i64>) {
                let n = ids.len();
                let mut offs: Vec<i64> = Vec::with_capacity(n * 2);
                let span_tbl = self.byte_span.as_slice();
                let tbl_len = span_tbl.len();
                if n >= 500_000 && parallel_ok() {
                    let nt = rayon::current_num_threads().clamp(2, 32);
                    let clen = n.div_ceil(nt);
                    // Pass 1: parallel chunk byte-length totals (zero writes).
                    let sums: Vec<i64> = ids
                        .par_chunks(clen)
                        .map(|ch| {
                            let mut acc = 0i64;
                            for &tid in ch {
                                let iu = tid as usize;
                                if iu < tbl_len {
                                    // SAFETY: `iu < tbl_len`.
                                    acc += unsafe { span_tbl.get_unchecked(iu).1 as i64 };
                                }
                            }
                            acc
                        })
                        .collect();
                    let mut base = Vec::with_capacity(sums.len());
                    let mut acc = 0i64;
                    for s in sums {
                        base.push(acc);
                        acc += s;
                    }
                    // Pass 2: parallel disjoint fill (parallelizes both LUT reads & page faults!).
                    let dst_uninit = &mut offs.spare_capacity_mut()[..n * 2];
                    dst_uninit
                        .par_chunks_mut(clen * 2)
                        .zip(ids.par_chunks(clen))
                        .zip(base.par_iter())
                        .for_each(|((out_ch, id_ch), &start_pos)| {
                            let mut pos = start_pos;
                            let p = out_ch.as_mut_ptr() as *mut i64;
                            for (i, &tid) in id_ch.iter().enumerate() {
                                let iu = tid as usize;
                                let len = if iu < tbl_len {
                                    // SAFETY: `iu < tbl_len`.
                                    unsafe { span_tbl.get_unchecked(iu).1 as i64 }
                                } else {
                                    0
                                };
                                // SAFETY: `out_ch` has length `2 * id_ch.len()`.
                                unsafe {
                                    *p.add(i * 2) = pos;
                                    pos += len;
                                    *p.add(i * 2 + 1) = pos;
                                }
                            }
                        });
                    // SAFETY: Pass 2 initialized all `2 * n` elements.
                    unsafe { offs.set_len(n * 2) };
                } else {
                    let mut pos = 0i64;
                    // SAFETY: `offs` has capacity `2 * n`; we initialize every slot `[0..2*n]`.
                    unsafe {
                        let dst = offs.as_mut_ptr();
                        for (i, &tid) in ids.iter().enumerate() {
                            let iu = tid as usize;
                            let len = if iu < tbl_len { (*span_tbl.as_ptr().add(iu)).1 as i64 } else { 0 };
                            *dst.add(i * 2) = pos;
                            pos += len;
                            *dst.add(i * 2 + 1) = pos;
                        }
                        offs.set_len(n * 2);
                    }
                }
                (ids, offs)
            };
            self.maybe_detach(py, text.len(), || {
                let ids = self.encode_one(text, Merge::Auto);
                build(ids)
            })
        } else {
            if !self.monotonic {
                return Err(PyValueError::new_err(
                    "encode_with_offsets_buffer: this model's merge table is non-monotonic \
                     and requires the canonical merger",
                ));
            }
            let bytes = text.as_bytes();
            py.detach(|| {
                let mut ids = Vec::new();
                let mut spans = Vec::new();
                let (mut toks, mut sp, mut pairs) = (Vec::new(), Vec::new(), Vec::new());
                match self.fast_pretok {
                    FastPretok::Cjk => {
                        let (mut cur, mut nxt) = (Vec::new(), Vec::new());
                        cjk::chunk_ranges(text, &mut cur, &mut nxt);
                        for &(ps, pe) in &cur {
                            self.emit_chunk_offsets(
                                bytes, ps, pe, &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                            );
                        }
                    }
                    FastPretok::BlockScan => {
                        let n = bytes.len();
                        let mut pos = 0usize;
                        while pos < n {
                            let end = pretok::next_chunk_end(self.kind, bytes, pos);
                            self.emit_chunk_offsets(
                                bytes, pos, end, &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                            );
                            pos = end;
                        }
                    }
                    FastPretok::Pipe => {
                        let pipe = self.pretok_pipe.as_ref().expect("Pipe mode has pretok_pipe");
                        for (ps, pe) in pipe.chunk_ranges(text) {
                            self.emit_chunk_offsets(
                                bytes, ps, pe, &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                            );
                        }
                    }
                    FastPretok::WholeText => {
                        self.emit_chunk_offsets(
                            bytes, 0, bytes.len(), &mut ids, &mut spans, &mut toks, &mut sp, &mut pairs,
                        );
                    }
                }
                let mut offs = Vec::with_capacity(spans.len() * 2);
                for (a, b) in spans {
                    offs.push(a as i64);
                    offs.push(b as i64);
                }
                (ids, offs)
            })
        };
        let n = ids.len();
        let id_buf = Py::new(py, TokenBuffer::new(ids))?;
        let off_buf = Py::new(py, TokenBuffer::new_i64(offs_flat))?;
        let id_arr = np.call_method1("frombuffer", (id_buf, "uint32"))?;
        let off_arr = np
            .call_method1("frombuffer", (off_buf, "int64"))?
            .call_method1("reshape", ((n, 2usize),))?;
        let d = PyDict::new(py);
        d.set_item("input_ids", id_arr)?;
        d.set_item("offsets", off_arr)?;
        Ok(d)
    }

    #[pyo3(signature = (texts, num_threads = 0))]
    fn encode_ordinary_batch<'py>(
        &self,
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        if let Some(batch) = try_extract_fast_batch(py, texts) {
            if batch.max_len < SPLIT_MIN {
                let slice = &batch.items;
                let total_len = batch.total_len;
                if self.normalizer.is_ascii_identity() && self.items_pipe_ok() {
                    let res = py.detach(|| self.encode_fast_short_t::<u32, false>(slice, total_len, num_threads, true));
                    let (data, counts) = match res {
                        Ok(v) => v,
                        Err(b) => return Err(bad_fast_to_pyerr(py, &batch, b)),
                    };
                    drop(batch);
                    return Ok(self.data_to_pylists(py, &data, &counts));
                }
                let res = self.maybe_detach(py, total_len, || {
                    self.encode_fast_items_flat_cap(slice, total_len, num_threads, false, None)
                });
                let chunks = match res {
                    Ok(c) => c,
                    Err(b) => return Err(bad_fast_to_pyerr(py, &batch, b)),
                };
                drop(batch);
                return Ok(self.chunks_to_pylists(py, &chunks));
            }
        }
        let texts: Vec<TextArg> = texts.extract()?;
        let stream_wide = self.stream_wide_ok(None);
        let wide = if stream_wide { WideCopies::default() } else { wide_args(py, &texts, num_threads)? };
        let res = py.detach(|| self.encode_batch_sep_t::<u32>(&texts, &wide, stream_wide, num_threads, true, None));
        let (data, counts) = match res {
            Ok(v) => v,
            Err(Ok(b)) => return Err(bad_input_to_pyerr(py, &texts, b)),
            Err(Err(e)) => return Err(e),
        };
        Ok(self.data_to_pylists(py, &data, &counts))
    }


    /// Token counts for many texts, in parallel (no Python list of ids built).
    #[pyo3(signature = (texts, num_threads = 0, separator = None))]
    fn count_tokens_batch(&self, py: Python<'_>, texts: &Bound<'_, PyAny>, num_threads: usize, separator: Option<Vec<u8>>) -> PyResult<Vec<usize>> {
        if separator.is_none() {
            if let Some(batch) = try_extract_fast_batch(py, texts) {
                if batch.max_len < SPLIT_MIN {
                    let slice = &batch.items;
                    let total_len = batch.total_len;
                    if self.normalizer.is_ascii_identity() && self.items_pipe_ok() {
                        let res = py.detach(|| self.encode_fast_short_t::<u32, false>(slice, total_len, num_threads, false));
                        return match res {
                            Ok((_, counts)) => Ok(counts),
                            Err(b) => Err(bad_fast_to_pyerr(py, &batch, b)),
                        };
                    }
                    let res = self.maybe_detach(py, total_len, || {
                        self.count_fast_items(slice, total_len, num_threads, false)
                    });
                    return match res {
                        Ok(counts) => Ok(counts),
                        Err(b) => Err(bad_fast_to_pyerr(py, &batch, b)),
                    };
                }
            }
        }
        let texts: Vec<TextArg> = texts.extract()?;
        let stream_wide = self.stream_wide_ok(separator.as_deref());
        let wide = if stream_wide { WideCopies::default() } else { wide_args(py, &texts, num_threads)? };
        let res = py.detach(|| self.encode_batch_sep_t::<u32>(&texts, &wide, stream_wide, num_threads, false, separator.as_deref()));
        match res {
            Ok((_, counts)) => Ok(counts),
            Err(Ok(b)) => Err(bad_input_to_pyerr(py, &texts, b)),
            Err(Err(e)) => Err(e),
        }
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
        texts: &Bound<'py, PyAny>,
        max_length: Option<usize>,
        pad_id: u32,
        num_threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fast_chunks: Option<Vec<(Vec<u32>, Vec<usize>)>> =
            if let Some(batch) = try_extract_fast_batch(py, texts) {
                if !batch.has_bytes {
                    let early_exit_worthy = max_length
                        .is_some_and(|ml| batch.max_len > ml.saturating_mul(6).saturating_add(128));
                    let slice = &batch.items;
                    let total_len = batch.total_len;
                    if self.normalizer.is_ascii_identity()
                        && self.items_pipe_ok()
                        && batch.max_len < SPLIT_MIN
                        && !early_exit_worthy
                    {
                        let res = self.maybe_detach(py, total_len, || {
                            self.encode_fast_short_t::<u32, false>(slice, total_len, num_threads, true)
                        });
                        match res {
                            Ok((data, counts)) => {
                                drop(batch);
                                let mut ends = Vec::with_capacity(counts.len());
                                let mut acc = 0usize;
                                for c in counts {
                                    acc += c;
                                    ends.push(acc);
                                }
                                Some(vec![(data, ends)])
                            }
                            Err(b) => return Err(bad_fast_to_pyerr(py, &batch, b)),
                        }
                    } else if max_length.is_some() || batch.max_len < SPLIT_MIN {
                        let trunc_cap = max_length.map(|ml| (ml, false));
                        let detach_work = max_length.map_or(total_len, |ml| {
                            total_len.min(slice.len().saturating_mul(ml.saturating_mul(6).saturating_add(128)))
                        });
                        let res = self.maybe_detach(py, detach_work, || {
                            self.encode_fast_items_flat_cap(slice, total_len, num_threads, false, trunc_cap)
                        });
                        match res {
                            Ok(c) => {
                                drop(batch);
                                Some(c)
                            }
                            Err(b) => return Err(bad_fast_to_pyerr(py, &batch, b)),
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

        let chunks: Vec<(Vec<u32>, Vec<usize>)> = if let Some(c) = fast_chunks {
            c
        } else {
            let texts: Vec<PyText> = texts.extract()?;
            let wide = wide_texts(py, &texts, num_threads)?;
            let seqs: Vec<Vec<u32>> = py.detach(|| {
                if let Some(ml) = max_length {
                    self.par_texts(num_threads, &texts, &wide, |t| {
                        self.with_scratch(|sc| {
                            sc.out.clear();
                            self.encode_ordinary_prefix_into(t, ml, sc);
                            sc.out.clone()
                        })
                    })
                } else {
                    self.par_texts(num_threads, &texts, &wide, |t| self.encode_one(t, Merge::Auto))
                }
            });
            let total: usize = seqs.iter().map(|s| s.len()).sum();
            let mut flat = Vec::with_capacity(total);
            let mut ends = Vec::with_capacity(seqs.len());
            for s in seqs {
                flat.extend_from_slice(&s);
                ends.push(flat.len());
            }
            vec![(flat, ends)]
        };

        let row_slices: Vec<&[u32]> = {
            let nrows: usize = chunks.iter().map(|(_, e)| e.len()).sum();
            let mut v = Vec::with_capacity(nrows);
            for (buf, ends) in &chunks {
                let mut start = 0usize;
                for &end in ends {
                    v.push(&buf[start..end]);
                    start = end;
                }
            }
            v
        };
        let rows = row_slices.len();
        let width = max_length.unwrap_or_else(|| row_slices.iter().map(|s| s.len()).max().unwrap_or(0));
        let (ids_flat, mask_flat) = py.detach(|| {
            let total = rows * width;
            let mut ids_flat: Vec<u32> = Vec::with_capacity(total);
            let mut mask_flat: Vec<u8> = Vec::with_capacity(total);
            if width > 0 {
                type FillRowU32<'a> = (
                    usize,
                    (&'a mut [std::mem::MaybeUninit<u32>], &'a mut [std::mem::MaybeUninit<u8>]),
                );
                let fill_row = |(r, (irow, mrow)): FillRowU32<'_>| {
                    let seq = row_slices[r];
                    let len = seq.len().min(width);
                    unsafe {
                        std::ptr::copy_nonoverlapping(seq.as_ptr(), irow.as_mut_ptr() as *mut u32, len);
                        std::ptr::write_bytes(mrow.as_mut_ptr() as *mut u8, 1u8, len);
                        for k in len..width {
                            irow.get_unchecked_mut(k).write(pad_id);
                        }
                        std::ptr::write_bytes((mrow.as_mut_ptr() as *mut u8).add(len), 0u8, width - len);
                    }
                };
                let iu = &mut ids_flat.spare_capacity_mut()[..total];
                let mu = &mut mask_flat.spare_capacity_mut()[..total];
                if parallel_ok() && rows > 1 && total >= 32_768 {
                    iu.par_chunks_mut(width).zip(mu.par_chunks_mut(width)).enumerate().for_each(fill_row);
                } else {
                    iu.chunks_mut(width).zip(mu.chunks_mut(width)).enumerate().for_each(fill_row);
                }
                unsafe {
                    ids_flat.set_len(total);
                    mask_flat.set_len(total);
                }
            }
            (ids_flat, mask_flat)
        });

        // Zero-copy views over owned TokenBuffers (eliminates PyBytes + .copy() double-copy).
        let np = py.import("numpy").map_err(|_| {
            PyValueError::new_err("encode_batch requires numpy (pip install numpy)")
        })?;
        let ids_arr = np
            .call_method1("frombuffer", (Bound::new(py, TokenBuffer::new(ids_flat))?, "uint32"))?
            .call_method1("reshape", ((rows, width),))?;
        let mask_arr = np
            .call_method1("frombuffer", (Bound::new(py, TokenBuffer::new_u8(mask_flat))?, "uint8"))?
            .call_method1("reshape", ((rows, width),))?;

        let out = PyDict::new(py);
        out.set_item("input_ids", ids_arr)?;
        out.set_item("attention_mask", mask_arr)?;
        Ok(out)
    }

    /// Bulk-corpus fast path: encode `texts` in parallel and return ALL token ids
    /// as ONE flat `uint32` numpy array plus an `int64` offsets array (length
    /// rows+1), so document `i`'s tokens are `data[offsets[i]:offsets[i+1]]`.
    ///
    /// Unlike `encode_ordinary_batch` (which builds a Python `list[int]` per doc,
    /// boxing every token into a PyInt) this never crosses the Python-object
    /// boundary per token — the ids go straight into a contiguous buffer, handed
    /// to numpy in a single copy. This is the right path for tokenizing a whole
    /// corpus offline (data prep), where per-token boxing otherwise dominates.
    /// Ragged (no padding), so no wasted work on uneven document lengths.
    #[pyo3(signature = (texts, num_threads = 0, separator = None, dtype = None))]
    fn encode_ordinary_to_buffer<'py>(
        &self,
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        num_threads: usize,
        separator: Option<Vec<u8>>,
        dtype: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        // dtype: None / "uint32" (default) or "uint16" (any spelling numpy
        // accepts, e.g. np.uint16): half the output memory, for vocabularies of
        // at most 65,536 ids.
        let np_mod = || py.import("numpy").map_err(|_| PyValueError::new_err("encode_ordinary_to_buffer requires numpy (pip install numpy)"));
        let narrow = match &dtype {
            None => false,
            Some(d) => {
                let name: String = np_mod()?.call_method1("dtype", (d,))?.getattr("name")?.extract()?;
                match name.as_str() {
                    "uint32" => false,
                    "uint16" => {
                        if self.vocab_size > 1 << 16 {
                            return Err(PyValueError::new_err(format!(
                                "dtype uint16 needs a vocabulary of at most 65536 ids (this one has {})",
                                self.vocab_size
                            )));
                        }
                        true
                    }
                    other => return Err(PyValueError::new_err(format!("dtype must be uint16 or uint32, not {other}"))),
                }
            }
        };
        let (data_ba, counts) = if narrow {
            self.encode_to_token_buffer::<u16>(py, texts, num_threads, separator.as_deref(), TokenBuffer::new_u16)?
        } else {
            self.encode_to_token_buffer::<u32>(py, texts, num_threads, separator.as_deref(), TokenBuffer::new)?
        };
        let mut offsets = Vec::with_capacity(counts.len() + 1);
        offsets.push(0i64);
        let mut total = 0usize;
        for &c in &counts {
            total += c;
            offsets.push(total as i64);
        }
        let np = np_mod()?;
        // Both `data` and `offsets` are zero-copy writable views over owned TokenBuffers.
        let data_arr = np.call_method1("frombuffer", (data_ba, if narrow { "uint16" } else { "uint32" }))?;
        let off_ba = Bound::new(py, TokenBuffer::new_i64(offsets))?;
        let off_arr = np.call_method1("frombuffer", (off_ba, "int64"))?;

        let out = PyDict::new(py);
        out.set_item("data", data_arr)?;
        out.set_item("offsets", off_arr)?;
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
        texts: &Bound<'py, PyAny>,
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
        let trunc_cap = if truncation {
            max_length.map(|ml| (ml.saturating_sub(n_special), trunc_left))
        } else {
            None
        };
        // Universal FastBatch path: extract list/tuple of str ONCE (zero PyIter_Next,
        // zero Vec<PyText>, zero double-extract on truncation batches).
        let fast_chunks: Option<Vec<(Vec<u32>, Vec<usize>)>> =
            if let Some(batch) = try_extract_fast_batch(py, texts) {
                if !batch.has_bytes {
                    let early_exit_worthy = trunc_cap.is_some_and(|(cap, left)| {
                        let thresh = if left {
                            cap.saturating_mul(8).saturating_add(768)
                        } else {
                            cap.saturating_mul(6).saturating_add(128)
                        };
                        batch.max_len > thresh
                    });
                    let slice = &batch.items;
                    let total_len = batch.total_len;
                    if self.normalizer.is_ascii_identity()
                        && self.items_pipe_ok()
                        && batch.max_len < SPLIT_MIN
                        && !early_exit_worthy
                    {
                        let res = self.maybe_detach(py, total_len, || {
                            self.encode_fast_short_t::<u32, true>(slice, total_len, num_threads, true)
                        });
                        match res {
                            Ok((data, counts)) => {
                                drop(batch);
                                let mut ends = Vec::with_capacity(counts.len());
                                let mut acc = 0usize;
                                for c in counts {
                                    acc += c;
                                    ends.push(acc);
                                }
                                Some(vec![(data, ends)])
                            }
                            Err(b) => return Err(bad_fast_to_pyerr(py, &batch, b)),
                        }
                    } else {
                        let detach_work = trunc_cap.map_or(total_len, |(cap, left)| {
                            let lim = if left {
                                cap.saturating_mul(8).saturating_add(512)
                            } else {
                                cap.saturating_mul(6).saturating_add(128)
                            };
                            total_len.min(slice.len().saturating_mul(lim))
                        });
                        let res = self.maybe_detach(py, detach_work, || {
                            self.encode_fast_items_flat_cap(slice, total_len, num_threads, true, trunc_cap)
                        });
                        match res {
                            Ok(c) => {
                                drop(batch);
                                Some(c)
                            }
                            Err(b) => return Err(bad_fast_to_pyerr(py, &batch, b)),
                        }
                    }
                } else {
                    None
                }
            } else {
                None
            };

        let chunks: Vec<(Vec<u32>, Vec<usize>)> = if let Some(c) = fast_chunks {
            c
        } else {
            const FLAT_SERIAL_MAX: usize = 131_072;
            let texts: Vec<PyText> = texts.extract()?;
            let wide = wide_texts(py, &texts, num_threads)?;
            let views = text_views(&texts, &wide);
            let total: usize = views.iter().map(|t| t.len()).sum();
            let eff_work: usize = match trunc_cap {
                Some((cap, left)) => {
                    let lim = if left {
                        cap.saturating_mul(8).saturating_add(512)
                    } else {
                        cap.saturating_mul(6).saturating_add(128)
                    };
                    views.iter().map(|t| t.len().min(lim)).sum()
                }
                None => total,
            };
            let nrows = views.len();
            if num_threads == 1 || nrows <= 1 || !parallel_ok() || eff_work < FLAT_SERIAL_MAX {
                let one = self.maybe_detach(py, eff_work, || {
                    self.encode_special_serial_flat_cap(&views, total, trunc_cap)
                });
                vec![one]
            } else {
                py.detach(|| self.encode_special_par_chunks_cap(&views, num_threads, trunc_cap))
            }
        };

        // Build borrowed per-row slices across the flat chunk buffers (zero per-row Vec allocs).
        let row_slices: Vec<&[u32]> = {
            let nrows: usize = chunks.iter().map(|(_, e)| e.len()).sum();
            let mut v = Vec::with_capacity(nrows);
            for (buf, ends) in &chunks {
                let mut start = 0usize;
                for &end in ends {
                    v.push(&buf[start..end]);
                    start = end;
                }
            }
            v
        };
        let rows = row_slices.len();
        // content length per row after truncation (reserving the specials).
        let trunc_cap = if truncation {
            max_length.map(|ml| ml.saturating_sub(n_special))
        } else {
            None
        };
        let content_len = |c: &[u32]| trunc_cap.map_or(c.len(), |cap| c.len().min(cap));
        let width = if pad_to_max {
            max_length.unwrap_or(0)
        } else {
            row_slices.iter().map(|c| n_special + content_len(c)).max().unwrap_or(0)
        };

        // Fill both [rows, width] int64 matrices in parallel, one row per task,
        // writing every element exactly once (no up-front zero fill).
        let (ids, mask) = py.detach(|| {
            let total = rows * width;
            let mut ids: Vec<i64> = Vec::with_capacity(total);
            let mut mask: Vec<i64> = Vec::with_capacity(total);
            if width > 0 {
                let fill_row = |(r, (irow, mrow)): FillRowItem<'_>| {
                    let c = row_slices[r];
                    // clamp content so prefix+content+suffix never overruns the row.
                    let cl = content_len(c).min(width.saturating_sub(n_special));
                    let cslice: &[u32] = if trunc_left && cl < c.len() { &c[c.len() - cl..] } else { &c[..cl] };
                    let used = (n_special + cl).min(width);
                    let off = if pad_left { width - used } else { 0 };
                    // Contiguous per-array fills so LLVM auto-vectorizes pad/mask/u32->i64.
                    for x in &mut irow[..off] { x.write(pad_id); }
                    for x in &mut mrow[..off] { x.write(0); }
                    let mut w = off;
                    let plen = prefix.len().min(width - w);
                    for (dst, &v) in irow[w..w + plen].iter_mut().zip(&prefix[..plen]) {
                        dst.write(v);
                    }
                    w += plen;
                    let clen = cslice.len().min(width - w);
                    for (dst, &v) in irow[w..w + clen].iter_mut().zip(&cslice[..clen]) {
                        dst.write(v as i64);
                    }
                    w += clen;
                    let slen = suffix.len().min(width - w);
                    for (dst, &v) in irow[w..w + slen].iter_mut().zip(&suffix[..slen]) {
                        dst.write(v);
                    }
                    w += slen;
                    for x in &mut mrow[off..w] { x.write(1); }
                    for x in &mut irow[w..width] { x.write(pad_id); }
                    for x in &mut mrow[w..width] { x.write(0); }
                };
                let iu = &mut ids.spare_capacity_mut()[..total];
                let mu = &mut mask.spare_capacity_mut()[..total];
                if parallel_ok() && rows > 1 && total >= 32_768 {
                    iu.par_chunks_mut(width).zip(mu.par_chunks_mut(width)).enumerate().for_each(fill_row);
                } else {
                    iu.chunks_mut(width).zip(mu.chunks_mut(width)).enumerate().for_each(fill_row);
                }
                // SAFETY: fill_row wrote all `width` elements of every row.
                unsafe {
                    ids.set_len(total);
                    mask.set_len(total);
                }
            }
            (ids, mask)
        });

        let np = py.import("numpy").map_err(|_| {
            PyValueError::new_err("return_tensors requires numpy (pip install numpy)")
        })?;
        // zero-copy: numpy views the Rust-filled buffers (kept alive via .base)
        let to_np = |buf: Vec<i64>| -> PyResult<Bound<'py, PyAny>> {
            let b = Bound::new(py, TokenBuffer::new_i64(buf))?;
            np.call_method1("frombuffer", (b, "int64"))?
                .call_method1("reshape", ((rows, width),))
        };
        let out = PyDict::new(py);
        out.set_item("input_ids", to_np(ids)?)?;
        out.set_item("attention_mask", to_np(mask)?)?;
        Ok(out)
    }

    /// Decode many token sequences, in parallel for large batches or in a single
    /// reused byte buffer for small batches. Supports `skip_special_tokens` and
    /// Whisper timestamp decoding directly in Rust (zero Python per-token loops).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (sequences, num_threads = 0, skip_special_tokens = false, decode_with_timestamps = false, time_precision = 0.02, segment_size = 1500))]
    fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        sequences: &Bound<'py, PyAny>,
        num_threads: usize,
        skip_special_tokens: bool,
        decode_with_timestamps: bool,
        time_precision: f64,
        segment_size: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        // Fast extraction into one flat Vec<u32> + row ends:
        // 1. 2D C-contiguous integer buffer (numpy ndarray, torch CPU tensor):
        //    zero Python int boxing!
        // 2. Plain list[list[int]]: direct inline PyLong digit reads + prefetch.
        // 3. Fallback: generic PyO3 Vec<Vec<u32>> extraction.
        let inline_long = pylong_inline_ok(py);
        let (flat_ids, row_ends): (Vec<u32>, Vec<usize>) = unsafe {
            let ptr = sequences.as_ptr();
            if pyo3::ffi::PyList_Check(ptr) != 0 {
                let nrows = pyo3::ffi::PyList_GET_SIZE(ptr) as usize;
                let mut total_est = 0usize;
                let mut all_lists = true;
                for r in 0..nrows {
                    let row = pyo3::ffi::PyList_GET_ITEM(ptr, r as pyo3::ffi::Py_ssize_t);
                    if pyo3::ffi::PyList_Check(row) == 0 {
                        all_lists = false;
                        break;
                    }
                    total_est += pyo3::ffi::PyList_GET_SIZE(row) as usize;
                }
                if all_lists {
                    let mut flat: Vec<u32> = Vec::with_capacity(total_est);
                    let mut ends: Vec<usize> = Vec::with_capacity(nrows);
                    let mut ok = true;
                    let mut off = 0usize;
                    let dst = flat.as_mut_ptr();
                    for r in 0..nrows {
                        let row = pyo3::ffi::PyList_GET_ITEM(ptr, r as pyo3::ffi::Py_ssize_t);
                        let n = pyo3::ffi::PyList_GET_SIZE(row) as usize;
                        if !extract_pylist_u32_into(row, n, dst.add(off), inline_long) {
                            ok = false;
                            break;
                        }
                        off += n;
                        ends.push(off);
                    }
                    if ok && pyo3::ffi::PyErr_Occurred().is_null() {
                        flat.set_len(off);
                        (flat, ends)
                    } else {
                        pyo3::ffi::PyErr_Clear();
                        let vecs = sequences.extract::<Vec<Vec<u32>>>()?;
                        let total: usize = vecs.iter().map(|v| v.len()).sum();
                        let mut flat = Vec::with_capacity(total);
                        let mut ends = Vec::with_capacity(vecs.len());
                        for v in vecs {
                            flat.extend_from_slice(&v);
                            ends.push(flat.len());
                        }
                        (flat, ends)
                    }
                } else {
                    let vecs = sequences.extract::<Vec<Vec<u32>>>()?;
                    let total: usize = vecs.iter().map(|v| v.len()).sum();
                    let mut flat = Vec::with_capacity(total);
                    let mut ends = Vec::with_capacity(vecs.len());
                    for v in vecs {
                        flat.extend_from_slice(&v);
                        ends.push(flat.len());
                    }
                    (flat, ends)
                }
            } else if pyo3::ffi::PyObject_CheckBuffer(ptr) != 0 {
                let mut guard = PyBufGuard {
                    view: std::mem::zeroed(),
                    active: false,
                };
                let flags = pyo3::ffi::PyBUF_ND | pyo3::ffi::PyBUF_STRIDES | pyo3::ffi::PyBUF_FORMAT;
                let mut got: Option<(Vec<u32>, Vec<usize>)> = None;
                if pyo3::ffi::PyObject_GetBuffer(ptr, &mut guard.view, flags) == 0 {
                    guard.active = true;
                    if guard.view.ndim == 2
                        && pyo3::ffi::PyBuffer_IsContiguous(&guard.view, b'C' as std::os::raw::c_char) != 0
                        && !guard.view.shape.is_null()
                    {
                        if let Some(kind) = classify_int_buffer(&guard.view) {
                            let nrows = *guard.view.shape as usize;
                            let ncols = *guard.view.shape.add(1) as usize;
                            let total = nrows.saturating_mul(ncols);
                            let mut flat = Vec::new();
                            if copy_int_buf_to_u32(guard.view.buf as *const u8, total, kind, &mut flat) {
                                let mut ends = Vec::with_capacity(nrows);
                                for r in 1..=nrows {
                                    ends.push(r * ncols);
                                }
                                got = Some((flat, ends));
                            }
                        }
                    }
                } else {
                    pyo3::ffi::PyErr_Clear();
                }
                drop(guard);
                if let Some(res) = got {
                    res
                } else {
                    let vecs = sequences.extract::<Vec<Vec<u32>>>()?;
                    let total: usize = vecs.iter().map(|v| v.len()).sum();
                    let mut flat = Vec::with_capacity(total);
                    let mut ends = Vec::with_capacity(vecs.len());
                    for v in vecs {
                        flat.extend_from_slice(&v);
                        ends.push(flat.len());
                    }
                    (flat, ends)
                }
            } else {
                let vecs = sequences.extract::<Vec<Vec<u32>>>()?;
                let total: usize = vecs.iter().map(|v| v.len()).sum();
                let mut flat = Vec::with_capacity(total);
                let mut ends = Vec::with_capacity(vecs.len());
                for v in vecs {
                    flat.extend_from_slice(&v);
                    ends.push(flat.len());
                }
                (flat, ends)
            }
        };

        let nrows = row_ends.len();
        const PAR_DECODE_MIN_TOKS: usize = 65_536;
        if num_threads == 1 || nrows <= 1 || !parallel_ok() || flat_ids.len() < PAR_DECODE_MIN_TOKS
        {
            let mut buf = Vec::with_capacity(256);
            unsafe {
                let outer = pyo3::ffi::PyList_New(nrows as pyo3::ffi::Py_ssize_t);
                assert!(!outer.is_null(), "list alloc");
                let mut start = 0usize;
                for (r, &end) in row_ends.iter().enumerate() {
                    let s = self.decode_slice_to_pystr(
                        py,
                        &flat_ids[start..end],
                        skip_special_tokens,
                        decode_with_timestamps,
                        time_precision,
                        segment_size,
                        &mut buf,
                    );
                    start = end;
                    pyo3::ffi::PyList_SET_ITEM(outer, r as pyo3::ffi::Py_ssize_t, s.into_ptr());
                }
                return Ok(Bound::from_owned_ptr(py, outer).cast_into_unchecked());
            }
        }

        // Ensure decode_table is initialized once before fanning out across rayon.
        let _ = self.decode_table();
        let ranges: Vec<(usize, usize)> = {
            let mut v = Vec::with_capacity(nrows);
            let mut start = 0usize;
            for &end in &row_ends {
                v.push((start, end));
                start = end;
            }
            v
        };
        let bufs: Vec<(Vec<u8>, bool)> = py.detach(|| {
            self.par_over(
                num_threads,
                &ranges,
                |&(a, b)| (b - a) * 4,
                |&(a, b)| {
                    self.gather_bytes_fast_opts(
                        &flat_ids[a..b],
                        skip_special_tokens,
                        decode_with_timestamps,
                        time_precision,
                        segment_size,
                    )
                },
            )
        });
        unsafe {
            let outer = pyo3::ffi::PyList_New(nrows as pyo3::ffi::Py_ssize_t);
            assert!(!outer.is_null(), "list alloc");
            for (r, (buf, is_ascii)) in bufs.iter().enumerate() {
                let s = if *is_ascii {
                    ascii_bytes_to_pystr(py, buf)
                } else {
                    bytes_to_pystr(py, buf)
                };
                pyo3::ffi::PyList_SET_ITEM(outer, r as pyo3::ffi::Py_ssize_t, s.into_ptr());
            }
            Ok(Bound::from_owned_ptr(py, outer).cast_into_unchecked())
        }
    }

    #[pyo3(signature = (tokens, skip_special_tokens = false, decode_with_timestamps = false, time_precision = 0.02, segment_size = 1500))]
    fn decode<'py>(
        &self,
        py: Python<'py>,
        tokens: &Bound<'py, PyAny>,
        skip_special_tokens: bool,
        decode_with_timestamps: bool,
        time_precision: f64,
        segment_size: usize,
    ) -> PyResult<Bound<'py, PyString>> {
        // Fast paths:
        // 1. Plain Python `list[int]`: direct inline PyLong digit read into a
        //    stack buffer (for `<= 2048` tokens, zero heap Vec<u32> alloc!) or
        //    single-alloc Vec<u32>, then direct PyUnicode_New for pure ASCII.
        // 2. 1D C-contiguous integer buffer (numpy ndarray, torch CPU tensor,
        //    array.array): zero Python int unboxing!
        // 3. Fallback: generic PyO3 Vec<u32> extraction.
        let ptr = tokens.as_ptr();
        let inline_long = pylong_inline_ok(py);
        // Reuse thread-local `s.utf8` so repeated `decode()` calls perform zero
        // Rust heap allocations (neither `Vec<u32>` nor `Vec<u8>`).
        let fast_res: Option<Bound<'py, PyString>> = self.with_scratch(|s| unsafe {
            if pyo3::ffi::PyList_Check(ptr) != 0 {
                let n = pyo3::ffi::PyList_GET_SIZE(ptr) as usize;
                if n == 0 {
                    return Some(PyString::new(py, ""));
                }
                if !(decode_with_timestamps && self.whisper_ts_begin != u32::MAX) {
                    if let Some(is_ascii) = self.gather_pylist_fused_into(
                        ptr,
                        n,
                        inline_long,
                        skip_special_tokens,
                        &mut s.utf8,
                    ) {
                        let slice = self.strip_decoded(&s.utf8);
                        return Some(if is_ascii {
                            ascii_bytes_to_pystr(py, slice)
                        } else {
                            bytes_to_pystr(py, slice)
                        });
                    }
                    pyo3::ffi::PyErr_Clear();
                }
                if n <= 2048 {
                    let mut stack = [std::mem::MaybeUninit::<u32>::uninit(); 2048];
                    if extract_pylist_u32_into(ptr, n, stack.as_mut_ptr() as *mut u32, inline_long) {
                        let ids = std::slice::from_raw_parts(stack.as_ptr() as *const u32, n);
                        return Some(self.decode_slice_to_pystr(
                            py,
                            ids,
                            skip_special_tokens,
                            decode_with_timestamps,
                            time_precision,
                            segment_size,
                            &mut s.utf8,
                        ));
                    }
                    pyo3::ffi::PyErr_Clear();
                } else {
                    let mut v: Vec<u32> = Vec::with_capacity(n);
                    if extract_pylist_u32_into(ptr, n, v.as_mut_ptr(), inline_long) {
                        v.set_len(n);
                        return Some(self.decode_slice_to_pystr(
                            py,
                            &v,
                            skip_special_tokens,
                            decode_with_timestamps,
                            time_precision,
                            segment_size,
                            &mut s.utf8,
                        ));
                    }
                    pyo3::ffi::PyErr_Clear();
                }
            } else if pyo3::ffi::PyObject_CheckBuffer(ptr) != 0 {
                let mut guard = PyBufGuard {
                    view: std::mem::zeroed(),
                    active: false,
                };
                let flags = pyo3::ffi::PyBUF_ND | pyo3::ffi::PyBUF_STRIDES | pyo3::ffi::PyBUF_FORMAT;
                if pyo3::ffi::PyObject_GetBuffer(ptr, &mut guard.view, flags) == 0 {
                    guard.active = true;
                    if guard.view.ndim == 1
                        && pyo3::ffi::PyBuffer_IsContiguous(&guard.view, b'C' as std::os::raw::c_char) != 0
                        && !guard.view.shape.is_null()
                    {
                        if let Some(kind) = classify_int_buffer(&guard.view) {
                            let count = *guard.view.shape as usize;
                            if count == 0 {
                                return Some(PyString::new(py, ""));
                            }
                            if matches!(kind, IntBufKind::U32) {
                                let ids = std::slice::from_raw_parts(guard.view.buf as *const u32, count);
                                return Some(self.decode_slice_to_pystr(
                                    py,
                                    ids,
                                    skip_special_tokens,
                                    decode_with_timestamps,
                                    time_precision,
                                    segment_size,
                                    &mut s.utf8,
                                ));
                            }
                            if copy_int_buf_to_u32(guard.view.buf as *const u8, count, kind, &mut s.out) {
                                drop(guard);
                                let ids = std::mem::take(&mut s.out);
                                let res = self.decode_slice_to_pystr(
                                    py,
                                    &ids,
                                    skip_special_tokens,
                                    decode_with_timestamps,
                                    time_precision,
                                    segment_size,
                                    &mut s.utf8,
                                );
                                s.out = ids;
                                return Some(res);
                            }
                        }
                    }
                } else {
                    pyo3::ffi::PyErr_Clear();
                }
            }
            None
        });
        if let Some(res) = fast_res {
            return Ok(res);
        }
        let ids: Vec<u32> = tokens.extract()?;
        Ok(self.with_scratch(|s| {
            self.decode_slice_to_pystr(
                py,
                &ids,
                skip_special_tokens,
                decode_with_timestamps,
                time_precision,
                segment_size,
                &mut s.utf8,
            )
        }))
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
        let ids = unsafe {
            std::slice::from_raw_parts(view.buf_ptr() as *const u32, view.item_count())
        };
        Ok(self.with_scratch(|s| {
            self.decode_slice_to_pystr(py, ids, false, false, 0.02, 1500, &mut s.utf8)
        }))
    }

    /// Create an incremental UTF-8-aware streaming decoder (`.step(id)`,
    /// `.step_many(ids)`, `.flush()`, `.reset()`). Holds incomplete trailing
    /// multi-byte UTF-8 sequences across tokens so split emojis and CJK never
    /// emit spurious `\u{FFFD}` mid-stream.
    #[pyo3(signature = (skip_special_tokens = false))]
    fn decode_stream(slf: Bound<'_, Self>, skip_special_tokens: bool) -> DecodeStream {
        let _ = slf.borrow().decode_table();
        DecodeStream {
            tok: slf.unbind(),
            skip_special: skip_special_tokens,
            pending: Vec::with_capacity(16),
            emitted_any: false,
        }
    }

    #[getter]
    fn n_vocab(&self) -> usize {
        self.vocab_size
    }
}

/// Incremental UTF-8-aware streaming token decoder created by
/// `Tokenizer.decode_stream()` or `AutoTokenizer.decode_stream()`.
#[pyclass]
pub struct DecodeStream {
    tok: Py<Tokenizer>,
    skip_special: bool,
    pending: Vec<u8>,
    emitted_any: bool,
}

impl DecodeStream {
    #[inline]
    fn append_token_bytes(tok: &Tokenizer, skip_special: bool, pending: &mut Vec<u8>, token_id: u32) -> bool {
        if token_id >= tok.whisper_ts_begin {
            return true;
        }
        let dt = tok.decode_table();
        let ti = token_id as usize;
        if ti >= dt.len() {
            return true;
        }
        let e = unsafe { *dt.get_unchecked(ti) };
        if skip_special && (e.flags & DEC_FLAG_SPECIAL) != 0 {
            return true;
        }
        if e.len != DEC_OUTLINE_LEN {
            pending.extend_from_slice(&e.data[..e.len as usize]);
        } else {
            let (off, len) = e.outline_span();
            pending.extend_from_slice(&tok.byte_arena[off..off + len]);
        }
        (e.flags & DEC_FLAG_NON_ASCII) == 0
    }

    #[inline]
    fn drain_ready<'py>(&mut self, py: Python<'py>, strip_space: bool) -> Bound<'py, PyString> {
        if !self.emitted_any && !self.pending.is_empty() {
            self.emitted_any = true;
            if strip_space && self.pending[0] == b' ' {
                self.pending.remove(0);
            }
        }
        if self.pending.is_empty() {
            return PyString::new(py, "");
        }
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let res = PyString::new(py, s);
                self.pending.clear();
                res
            }
            Err(err) => {
                if err.error_len().is_none() {
                    let vlen = err.valid_up_to();
                    let res = if vlen == 0 {
                        PyString::new(py, "")
                    } else {
                        // SAFETY: `self.pending[..vlen]` was validated by `from_utf8`.
                        let s = unsafe { std::str::from_utf8_unchecked(&self.pending[..vlen]) };
                        PyString::new(py, s)
                    };
                    if vlen > 0 {
                        self.pending.drain(..vlen);
                    }
                    return res;
                }
                // General case: one or more genuinely ill-formed UTF-8 sequences
                // (maximal-subpart `\u{FFFD}` substitution, holding any trailing
                // incomplete 1..3-byte prefix for the next `.step()`).
                let mut out = String::with_capacity(self.pending.len() + 4);
                let mut pos = 0usize;
                while pos < self.pending.len() {
                    match std::str::from_utf8(&self.pending[pos..]) {
                        Ok(s) => {
                            out.push_str(s);
                            pos = self.pending.len();
                            break;
                        }
                        Err(e) => {
                            let v = e.valid_up_to();
                            if v > 0 {
                                // SAFETY: validated by `from_utf8`.
                                out.push_str(unsafe {
                                    std::str::from_utf8_unchecked(&self.pending[pos..pos + v])
                                });
                                pos += v;
                            }
                            match e.error_len() {
                                Some(bad_len) => {
                                    out.push('\u{FFFD}');
                                    pos += bad_len;
                                }
                                None => break,
                            }
                        }
                    }
                }
                if pos == self.pending.len() {
                    self.pending.clear();
                } else if pos > 0 {
                    self.pending.drain(..pos);
                }
                PyString::new(py, &out)
            }
        }
    }
}

#[pymethods]
impl DecodeStream {
    /// Feed one token id and return any newly completed UTF-8 text (`""` if the
    /// token only contributes to an incomplete multi-byte character or is skipped).
    fn step<'py>(&mut self, py: Python<'py>, token_id: u32) -> Bound<'py, PyString> {
        let strip_space = {
            let tok = self.tok.borrow(py);
            if token_id >= tok.whisper_ts_begin {
                return PyString::new(py, "");
            }
            let dt = tok.decode_table();
            let ti = token_id as usize;
            if ti >= dt.len() {
                return PyString::new(py, "");
            }
            let e = unsafe { *dt.get_unchecked(ti) };
            if self.skip_special && (e.flags & DEC_FLAG_SPECIAL) != 0 {
                return PyString::new(py, "");
            }
            // Zero-alloc pure-ASCII fast path: direct `PyUnicode_New(len, 127)`.
            if self.pending.is_empty() && (e.flags & DEC_FLAG_NON_ASCII) == 0 && e.len != DEC_OUTLINE_LEN {
                let mut slice = &e.data[..e.len as usize];
                if !self.emitted_any && !slice.is_empty() {
                    self.emitted_any = true;
                    if tok.decode_strip_space && slice[0] == b' ' {
                        slice = &slice[1..];
                    }
                }
                return ascii_bytes_to_pystr(py, slice);
            }
            if e.len != DEC_OUTLINE_LEN {
                self.pending.extend_from_slice(&e.data[..e.len as usize]);
            } else {
                let (off, len) = e.outline_span();
                self.pending.extend_from_slice(&tok.byte_arena[off..off + len]);
            }
            tok.decode_strip_space
        };
        self.drain_ready(py, strip_space)
    }

    /// Feed a sequence of token ids (`list[int]`, 1D NumPy/torch/array buffer,
    /// or any iterable of ints) in one Rust call and return newly completed text.
    fn step_many<'py>(&mut self, py: Python<'py>, tokens: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyString>> {
        let ptr = tokens.as_ptr();
        let inline_long = pylong_inline_ok(py);
        let was_empty = self.pending.is_empty();
        let skip_special = self.skip_special;

        let (strip_space, all_ascii) = {
            let tok = self.tok.borrow(py);
            let strip_space = tok.decode_strip_space;
            let mut all_ascii = was_empty;
            let mut handled = false;
            unsafe {
                if pyo3::ffi::PyList_Check(ptr) != 0 {
                    let n = pyo3::ffi::PyList_GET_SIZE(ptr) as usize;
                    if n == 0 {
                        return Ok(PyString::new(py, ""));
                    }
                    if was_empty {
                        if let Some(is_ascii) = tok.gather_pylist_fused_into(
                            ptr,
                            n,
                            inline_long,
                            skip_special,
                            &mut self.pending,
                        ) {
                            all_ascii = is_ascii;
                            handled = true;
                        } else {
                            pyo3::ffi::PyErr_Clear();
                            self.pending.clear();
                        }
                    } else {
                        let save_len = self.pending.len();
                        let mut ok = true;
                        for i in 0..n {
                            let item = pyo3::ffi::PyList_GET_ITEM(ptr, i as pyo3::ffi::Py_ssize_t);
                            let val = pyo3::ffi::PyLong_AsLongLong(item);
                            if val < 0 || val > u32::MAX as i64 || !pyo3::ffi::PyErr_Occurred().is_null() {
                                pyo3::ffi::PyErr_Clear();
                                ok = false;
                                break;
                            }
                            Self::append_token_bytes(&tok, skip_special, &mut self.pending, val as u32);
                        }
                        if ok {
                            handled = true;
                        } else {
                            self.pending.truncate(save_len);
                        }
                    }
                } else if pyo3::ffi::PyObject_CheckBuffer(ptr) != 0 {
                    let mut guard = PyBufGuard {
                        view: std::mem::zeroed(),
                        active: false,
                    };
                    let flags = pyo3::ffi::PyBUF_ND | pyo3::ffi::PyBUF_STRIDES | pyo3::ffi::PyBUF_FORMAT;
                    if pyo3::ffi::PyObject_GetBuffer(ptr, &mut guard.view, flags) == 0 {
                        guard.active = true;
                        if guard.view.ndim == 1
                            && pyo3::ffi::PyBuffer_IsContiguous(&guard.view, b'C' as std::os::raw::c_char) != 0
                            && !guard.view.shape.is_null()
                        {
                            if let Some(kind) = classify_int_buffer(&guard.view) {
                                let count = *guard.view.shape as usize;
                                if count == 0 {
                                    return Ok(PyString::new(py, ""));
                                }
                                if matches!(kind, IntBufKind::U32) {
                                    let ids = std::slice::from_raw_parts(guard.view.buf as *const u32, count);
                                    if was_empty {
                                        all_ascii = tok.gather_bytes_single_pass_into(ids, skip_special, &mut self.pending);
                                    } else {
                                        for &t in ids {
                                            Self::append_token_bytes(&tok, skip_special, &mut self.pending, t);
                                        }
                                    }
                                    handled = true;
                                } else {
                                    let mut tmp = Vec::new();
                                    if copy_int_buf_to_u32(guard.view.buf as *const u8, count, kind, &mut tmp) {
                                        if was_empty {
                                            all_ascii = tok.gather_bytes_single_pass_into(&tmp, skip_special, &mut self.pending);
                                        } else {
                                            for &t in &tmp {
                                                Self::append_token_bytes(&tok, skip_special, &mut self.pending, t);
                                            }
                                        }
                                        handled = true;
                                    }
                                }
                            }
                        }
                    } else {
                        pyo3::ffi::PyErr_Clear();
                    }
                }
            }
            if !handled {
                let ids: Vec<u32> = tokens.extract()?;
                if ids.is_empty() {
                    return Ok(PyString::new(py, ""));
                }
                if was_empty {
                    all_ascii = tok.gather_bytes_single_pass_into(&ids, skip_special, &mut self.pending);
                } else {
                    for &t in &ids {
                        Self::append_token_bytes(&tok, skip_special, &mut self.pending, t);
                    }
                }
            }
            (strip_space, all_ascii)
        };

        if all_ascii && !self.pending.is_empty() {
            let start = if !self.emitted_any {
                self.emitted_any = true;
                usize::from(strip_space && self.pending[0] == b' ')
            } else {
                0
            };
            let res = ascii_bytes_to_pystr(py, &self.pending[start..]);
            self.pending.clear();
            return Ok(res);
        }
        Ok(self.drain_ready(py, strip_space))
    }

    /// Flush any remaining incomplete trailing bytes at end-of-stream (replacing
    /// truncated UTF-8 with `\u{FFFD}` to match `decode()`).
    fn flush<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyString> {
        if self.pending.is_empty() {
            return PyString::new(py, "");
        }
        let tok = self.tok.borrow(py);
        let start = if !self.emitted_any {
            self.emitted_any = true;
            usize::from(tok.decode_strip_space && self.pending[0] == b' ')
        } else {
            0
        };
        let res = bytes_to_pystr(py, &self.pending[start..]);
        self.pending.clear();
        res
    }

    /// Reset internal byte buffer and SentencePiece leading-space state for reuse.
    fn reset(&mut self) {
        self.pending.clear();
        self.emitted_any = false;
    }

    /// Number of incomplete trailing UTF-8 bytes currently buffered.
    #[getter]
    fn pending_bytes(&self) -> usize {
        self.pending.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "DecodeStream(skip_special_tokens={}, pending_bytes={})",
            if self.skip_special { "True" } else { "False" },
            self.pending.len()
        )
    }
}

/// RAII read-only memory-mapped file view (Linux/macOS `mmap` + `MADV_SEQUENTIAL`)
/// with automatic `std::fs::read` fallback for empty files, pipes, or other OSes.
enum FileBytes {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    Mmap { ptr: *mut std::ffi::c_void, len: usize },
    Owned(Vec<u8>),
}

// SAFETY: `Mmap` holds a read-only `MAP_PRIVATE` mapping owned exclusively by this `FileBytes`.
unsafe impl Send for FileBytes {}
unsafe impl Sync for FileBytes {}

impl FileBytes {
    fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let meta = file.metadata()?;
        let len = meta.len() as usize;
        if len == 0 {
            return Ok(Self::Owned(Vec::new()));
        }
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        if meta.is_file() {
            use std::os::unix::io::AsRawFd;
            const PROT_READ: i32 = 1;
            const MAP_PRIVATE: i32 = 2;
            const MADV_SEQUENTIAL: i32 = 2;
            extern "C" {
                fn mmap(
                    addr: *mut std::ffi::c_void,
                    len: usize,
                    prot: i32,
                    flags: i32,
                    fd: i32,
                    offset: i64,
                ) -> *mut std::ffi::c_void;
                fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
            }
            let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
            let map_failed = (!0usize) as *mut std::ffi::c_void;
            if ptr != map_failed && !ptr.is_null() {
                unsafe {
                    let _ = madvise(ptr, len, MADV_SEQUENTIAL);
                }
                return Ok(Self::Mmap { ptr, len });
            }
        }
        drop(file);
        Ok(Self::Owned(std::fs::read(path)?))
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            Self::Mmap { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr as *const u8, *len) },
            Self::Owned(v) => v.as_slice(),
        }
    }
}

impl Drop for FileBytes {
    fn drop(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        if let Self::Mmap { ptr, len } = *self {
            extern "C" {
                fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
            }
            unsafe {
                let _ = munmap(ptr, len);
            }
        }
    }
}

enum FileEncodeErr {
    Io(std::io::Error),
    Encode(Result<BadInput, String>),
}

fn file_err_to_pyerr(e: FileEncodeErr) -> PyErr {
    match e {
        FileEncodeErr::Io(io) => PyErr::from(io),
        FileEncodeErr::Encode(Ok(BadInput::Bytes(i))) => {
            PyValueError::new_err(format!("item {i}: bytes are not valid UTF-8"))
        }
        FileEncodeErr::Encode(Ok(BadInput::Wide(_))) => {
            PyValueError::new_err("file bytes are not valid UTF-8")
        }
        FileEncodeErr::Encode(Err(msg)) => PyValueError::new_err(msg),
    }
}

/// Accept `None`, `bytes`/`bytearray`, or `str` as a document separator.
fn extract_sep_arg(sep: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<u8>>> {
    match sep {
        None => Ok(None),
        Some(obj) => {
            if let Ok(b) = obj.extract::<Vec<u8>>() {
                Ok(Some(b))
            } else if let Ok(s) = obj.extract::<String>() {
                Ok(Some(s.into_bytes()))
            } else {
                Err(PyValueError::new_err("separator must be bytes or str"))
            }
        }
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tokenizer>()?;
    m.add_class::<TokenBuffer>()?;
    m.add_class::<DecodeStream>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod perf;
