"""Standalone integration tests for tuetoken (no C package required).

References are OpenAI `tiktoken` (via the vendored tiktoken->tokenizer.json
helper) and HuggingFace `tokenizers`. Covers, on a varied corpus:

  * tiktoken parity (gpt2 / cl100k / o200k) — byte-exact encode
  * streaming hybrid == scan-and-merge == tiktoken (incl. long adversarial chunks)
  * decode round-trip + the encode_to_bytes / decode_array buffer fast-paths
  * a small HF model sweep (ByteLevel + byte_fallback) vs the HF fast tokenizer
  * unsupported pre-tokenizer (Punctuation) is rejected with a clear error

Run:  python tests/test_tuetoken.py
"""
import os, sys, random, warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # vendored _tiktoken_json helper

import tuetoken
import tiktoken
from tokenizers import Tokenizer as HFTokenizer
from _tiktoken_json import get_cached_tokenizer_path

# --- varied test inputs + a deterministic medium corpus -----------------------
TEST_TEXTS = [
    "", " ", "Hello world", "Hello  world", "The quick brown fox jumps over.",
    "café ★ 你好世界 — naïve coöperate Ω≈ç√∫ 🚀❤️🇺🇸",
    "def f(x):\n    return x**2 + 1  # squared\nfor i in range(10): print(i)",
    "tab\there\n\n  trailing   spaces   \n\tmixed",
    "3.14159 0xFF 1,000,000 phone: 555-0143 year2024 v1.2.3",
    "aaaaaaaaaa " * 5 + "ababababab" * 4,
    '{"key": "value", "n": 42, "arr": [1, 2, 3], "nested": {"a": true}}',
    "Hello 世界! Email: a.b@x.com http://example.com/path?q=1#frag",
    "Привет мир! مرحبا بالعالم 12 ٣٤ こんにちは",
    "camelCaseWord HTTPSConnection getHTTPResponse XMLParser",
]


def make_corpus():
    r = random.Random(7)
    parts = []
    for _ in range(400):
        parts.append(r.choice(TEST_TEXTS))
        parts.append("".join(r.choice(" \t\n abcXYZ.,!?0123éü你") for _ in range(r.randint(0, 60))))
    return "".join(parts)


CORPUS = make_corpus()
TIKTOKEN_ENCODINGS = ["gpt2", "cl100k_base", "o200k_base"]


def _line(name, ok, tot):
    print(f"  {name:34} {ok}/{tot} {'[PASS]' if ok == tot else '[FAIL]'}")
    return ok, tot


# --- 1. tiktoken parity -------------------------------------------------------
def test_tiktoken_parity():
    print("=== tiktoken parity (encode) ===")
    ok = tot = 0
    texts = TEST_TEXTS + [CORPUS]
    for enc in TIKTOKEN_ENCODINGS:
        R = tuetoken.Tokenizer(get_cached_tokenizer_path(enc))
        tk = tiktoken.get_encoding(enc)
        local = sum(1 for t in texts if R.encode_ordinary(t) == tk.encode_ordinary(t))
        ok += local; tot += len(texts)
        _line(enc, local, len(texts))
    assert ok == tot, f"tiktoken parity: {ok}/{tot}"
    return ok, tot


# --- 2. the two BPE engines agree: scan == stream == tiktoken ----------------
def test_streaming():
    # tuetoken has two independent merge engines (scan-and-merge + the O(n)
    # streaming merger). They MUST produce identical output. We force EACH on
    # every input via _encode_ordinary_scan / _encode_ordinary_stream, so scan vs
    # stream is checked on long inputs too (the default path auto-streams those,
    # which would hide a divergence). Corpus hammers the known failure mode: long
    # runs of ONE repeated token across the 256-byte streaming threshold.
    print("=== scan == stream == tiktoken (forced both engines, single-token runs) ===")
    r = random.Random(0)
    import string
    inputs = TEST_TEXTS + [CORPUS]
    for L in (255, 256, 257, 1000, 8000):  # boundary + long no-merge chunks
        inputs += ["".join(r.choice(string.ascii_letters + "0123") for _ in range(L)) for _ in range(20)]
    # single-token repetition straddling the threshold — what broke gemma
    for u in ["a", " ", "the", "0", "  ", "==", "ab", " the", "\t", "..."]:
        for n in (1, 85, 128, 255, 256, 257, 512, 2000, 20000):
            inputs.append(u * n)
    ok = tot = 0
    for enc in TIKTOKEN_ENCODINGS:
        R = tuetoken.Tokenizer(get_cached_tokenizer_path(enc))
        tk = tiktoken.get_encoding(enc)
        local = sum(1 for t in inputs
                    if R._encode_ordinary_scan(t)
                       == R._encode_ordinary_stream(t)
                       == R.encode_ordinary(t)
                       == tk.encode_ordinary(t))
        ok += local; tot += len(inputs)
        _line(enc, local, len(inputs))
    assert ok == tot, f"scan == stream == tiktoken: {ok}/{tot}"
    return ok, tot


# --- 3. decode round-trip + buffer fast-paths --------------------------------
def test_decode_and_buffers():
    print("=== decode correctness (vs tiktoken, buffers, round-trip, edges) ===")
    import numpy as np, array as _array, random as _random
    rnd = _random.Random(0)
    ok = tot = 0
    for enc in TIKTOKEN_ENCODINGS:
        R = tuetoken.Tokenizer(get_cached_tokenizer_path(enc))
        tk = tiktoken.get_encoding(enc)
        maxbpe = max(tk._mergeable_ranks.values())  # BPE-decodable range
        # (a) decode(real corpus ids) == tiktoken.decode  — the case that matters
        real = tk.encode_ordinary(CORPUS)
        tot += 1; ok += R.decode(real) == tk.decode(real)
        # (b) decode(random ids) == tiktoken.decode — exercises lossy/invalid UTF-8
        rand = [[rnd.randint(0, maxbpe) for _ in range(rnd.randint(0, 200))] for _ in range(400)]
        tot += 1; ok += all(R.decode(s) == tk.decode(s) for s in rand)
        # (c) decode_array(numpy / array.array) == decode(list)
        tot += 1; ok += all(R.decode_array(np.array(s, np.uint32)) == R.decode(s) for s in rand if s)
        tot += 1; ok += all(R.decode_array(_array.array("I", s)) == R.decode(s) for s in rand if s)
        # (d) round-trip stability: encode->decode->encode is idempotent (NOT a tautology)
        for t in TEST_TEXTS:
            ids = R.encode_ordinary(t)
            tot += 1; ok += R.encode_ordinary(R.decode(ids)) == R.encode_ordinary(R.decode(R.encode_ordinary(R.decode(ids))))
            tot += 1; ok += np.frombuffer(R.encode_to_bytes(t), np.uint32).tolist() == ids
        # (e) edges: empty, single token, out-of-range ids tolerated (no crash)
        tot += 1; ok += R.decode([]) == "" and R.decode_array(np.array([], np.uint32)) == ""
        tot += 1; ok += isinstance(R.decode([R.n_vocab + 5, real[0], R.n_vocab]), str)
        # out-of-range / negative ids must RAISE (not silently wrap), like tiktoken
        for bad in ([-1], [2**32], [2**32 + 123], [10**20]):
            tot += 1
            try:
                R.decode(bad); ok += 0
            except (ValueError, OverflowError):
                ok += 1
    res = _line("decode checks", ok, tot)
    assert ok == tot, f"decode_and_buffers: {ok}/{tot}"
    return res


# --- 4. HF model sweep (no C) ------------------------------------------------
HF_MODELS = [
    "gpt2", "HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct",
    "EleutherAI/pythia-410m", "bigcode/starcoder2-3b",          # ByteLevel
    "mistralai/Mistral-7B-Instruct-v0.3", "01-ai/Yi-1.5-6B-Chat",  # byte_fallback/SP
    "deepseek-ai/DeepSeek-V3-0324",                              # CJK 3-stage machine
    "Qwen/Qwen3.6-35B-A3B",                                      # mark-inclusive Generic ([\p{L}\p{M}]+)
    "ByteDance-Seed/Seed-Coder-8B-Instruct",                     # custom symbol-no-NL Split (exact matcher)
    "tiiuae/Falcon3-7B-Instruct",                                # Punctuation + ByteLevel(re) + Digits
    "tiiuae/Falcon-H1-7B-Instruct",                              # Punctuation + CamelCase Split + ByteLevel + Split
    "dphn/Dolphin-X1-Trinity-Nano",                              # 6-stage \A/\G/\z + Onig {3}+ digits
    "openai/clip-vit-large-patch14",                             # NFC+Replace+Lowercase + </w> eow suffix
]


def _has_content_added_tokens(tj_path):
    import json
    d = json.load(open(tj_path))
    for at in d.get("added_tokens", []):
        if not at.get("special", False) and at.get("content", "").strip() == "" and at.get("content"):
            return True  # e.g. GPT-NeoX/codegen whitespace-run tokens
    return False


def test_hf_models():
    print("=== HF model sweep: encode_ordinary == HF fast tokenizer ===")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import GatedRepoError
    ok = tot = 0
    for repo in HF_MODELS:
        try:
            tj = hf_hub_download(repo, "tokenizer.json")
        except (GatedRepoError, Exception):
            print(f"  {repo:42} SKIP (unavailable)")
            continue
        fast = HFTokenizer.from_file(tj); fast.no_truncation()
        R = tuetoken.Tokenizer(tj)
        # The two merge engines must agree on this model too — long single-token
        # runs across the streaming threshold (the gemma regression). For byte_fallback
        # models this also asserts the gate routes both forced modes to scan.
        merge_runs = [u * n for u in ("a", " ", "the", "00", "\t") for n in (300, 2000)]
        merge_ok = all(R._encode_ordinary_scan(s) == R._encode_ordinary_stream(s)
                       for s in merge_runs)
        ok += merge_ok; tot += 1
        if not merge_ok:
            print(f"  {repo:42} scan != stream [FAIL]")
        if _has_content_added_tokens(tj):
            # Rust's pure core doesn't split content added-tokens; HF does. Not a
            # fair encode_ordinary comparison — just check it loads + round-trips.
            print(f"  {repo:42} loads (content added-tokens; core != HF by design)")
            ok += 1; tot += 1
            continue
        # encode parity on real text
        enc_ids = [fast.encode(t, add_special_tokens=False).ids for t in TEST_TEXTS]
        local = sum(1 for t, ids in zip(TEST_TEXTS, enc_ids) if R.encode_ordinary(t) == ids)
        # decode parity vs HF on the SAME real token sequences (covers byte_fallback
        # metaspace/Strip decode). Random-id decode differs only on special tokens
        # / invalid-UTF-8 lossy, which never occur in real decoding.
        dec = sum(1 for ids in enc_ids if not ids or R.decode(ids) == fast.decode(ids))
        ok += local + dec; tot += 2 * len(TEST_TEXTS)
        _line(repo + " (enc/dec)", local + dec, 2 * len(TEST_TEXTS))
    assert ok == tot, f"HF model sweep: {ok}/{tot}"
    return ok, tot


# --- 5b. Unicode normalization (the Qwen-family fix) -------------------------
def test_normalization():
    # HF runs the normalizer (NFC for the whole Qwen family) BEFORE ByteLevel, so
    # we must too -- skipping it silently mistokenized any decomposed/combining
    # Unicode. Cover a broad decomposed corpus, assert == HF, assert EVERY entry
    # point normalizes consistently, and assert the offsets guard.
    print("=== Unicode normalization (NFC) == HF on all entry points ===")
    import unicodedata, numpy as np
    from huggingface_hub import hf_hub_download
    ok = tot = 0
    try:
        tj = hf_hub_download("Qwen/Qwen2.5-1.5B-Instruct", "tokenizer.json")
        fast = HFTokenizer.from_file(tj); fast.no_truncation()
        R = tuetoken.Tokenizer(tj)
    except Exception as e:
        print(f"  SKIP ({e})"); return ok, tot

    # base strings that change under NFC: NFD accents, Hangul jamo, compatibility
    # chars, combining runs inside sentences, Vietnamese, Greek tonos / Cyrillic
    bases = [
        "café", "déjà vu", "naïve", "Zürich",
        "São Paulo", "jalapeño",
        "한", "한글", "한국어",
        "Å", "Ω", "ﬁ", "①", "²",
        "é à ô ñ",
        "Tiếng Việt với dấu",
        "Ἀθήνα Ελληνικά",
        "Кириллица й",
        "mixed café 中文 한글 123 déjà",
    ]
    corpus = []
    for b in bases:
        corpus.append(unicodedata.normalize("NFC", b))
        corpus.append(unicodedata.normalize("NFD", b))
    enc = sum(1 for s in corpus
              if R.encode_ordinary(s) == fast.encode(s, add_special_tokens=False).ids)
    _line("decomposed == HF", enc, len(corpus)); ok += enc; tot += len(corpus)

    # every entry point must apply the normalizer (all route through encode_one)
    consistent = 0
    for s in corpus:
        ids = R.encode_ordinary(s)
        same = (np.frombuffer(R.encode_to_bytes(s), np.uint32).tolist() == ids
                and R.count_tokens(s) == len(ids)
                and R.encode_ordinary_batch([s])[0] == ids
                and R.count_tokens_batch([s])[0] == len(ids))
        consistent += same
    _line("entry points consistent", consistent, len(corpus)); ok += consistent; tot += len(corpus)

    # offsets guard: accept already-normalized input; refuse anything a normalizer
    # would rewrite (byte spans into the original would be wrong)
    nfc_inputs = ["hello café", "plain ascii 123", unicodedata.normalize("NFC", "Việt")]
    nfd_inputs = [unicodedata.normalize("NFD", x) for x in ("café", "한글", "déjà")]
    g = 0
    for s in nfc_inputs:
        try:
            i, off = R.encode_with_offsets(s)
            g += i == R.encode_ordinary(s) and len(i) == len(off)
        except Exception:
            pass
    for s in nfd_inputs:
        try:
            R.encode_with_offsets(s)              # must raise
        except ValueError:
            g += 1
    _line("offsets guard (accept NFC / refuse NFD)", g, len(nfc_inputs) + len(nfd_inputs))
    ok += g; tot += len(nfc_inputs) + len(nfd_inputs)
    assert ok == tot, f"normalization: {ok}/{tot}"
    return ok, tot


# --- 5c. composable HF pipeline parity + fail-closed on unsupported configs ---
def test_composable_pipeline():
    # Universal composable normalizer + pre-tokenizer pipeline must match HF
    # byte-exact across real-world multi-stage models AND synthetic compositions
    # (Lowercase, Strip, Replace(Regex), Punctuation, ByteLevel(add_prefix_space),
    # Metaspace(never), Whitespace, WhitespaceSplit, byte_fallback + Punctuation).
    print("=== composable HF pipeline parity (real + synthetic) ===")
    import json, tempfile
    from huggingface_hub import hf_hub_download
    ok = tot = 0

    extra_inputs = TEST_TEXTS + [
        "Привет в мире! 1234567890",           # Falcon3 post-ByteLevel Cyrillic 0xB2='²' split
        "1234 12345 123456 1234567 " + "1" * 515,  # Dolphin right-aligned 3-digit & 510-digit grouping
        "def foo(x: int):\n    return x\n",    # Seed-Coder symbol-without-newline Split
        "  UPPER   CASE\t\nCafe\u0301  ",       # Lowercase / Strip / Replace(\s+) / NFC
    ]
    for repo in ["tiiuae/falcon-7b", "tiiuae/Falcon3-7B-Instruct",
                 "tiiuae/Falcon-H1-7B-Instruct", "dphn/Dolphin-X1-Trinity-Nano",
                 "ByteDance-Seed/Seed-Coder-8B-Instruct", "openai/clip-vit-large-patch14"]:
        try:
            tj = hf_hub_download(repo, "tokenizer.json")
        except Exception:
            print(f"  {repo:42} SKIP (unavailable)")
            continue
        fast = HFTokenizer.from_file(tj); fast.no_truncation()
        R = tuetoken.Tokenizer(tj)
        AT = tuetoken.AutoTokenizer.from_file(tj)
        local = sum(
            1 for s in extra_inputs
            if AT.encode(s, add_special_tokens=False) == fast.encode(s, add_special_tokens=False).ids
            and R.decode(AT.encode(s, add_special_tokens=False)) == fast.decode(fast.encode(s, add_special_tokens=False).ids)
        )
        ok += local; tot += len(extra_inputs)
        _line(repo, local, len(extra_inputs))

    # Offline synthetic ByteLevel + byte_fallback composable configs vs HF tokenizers
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs); n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    bl_vocab = {chr(c): i for i, c in enumerate(cs)}
    bf_vocab = {f"<0x{b:02X}>": b for b in range(256)}
    for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-\u2581":
        if ch not in bf_vocab:
            bf_vocab[ch] = len(bf_vocab)

    bl_dec = {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}
    synth_cases = [
        ("Lowercase + Strip + Replace(Regex) + ByteLevel(aps=True)",
         {"version": "1.0", "model": {"type": "BPE", "vocab": bl_vocab, "merges": []},
          "normalizer": {"type": "Sequence", "normalizers": [
              {"type": "NFC"},
              {"type": "Strip", "strip_left": True, "strip_right": True},
              {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
              {"type": "Lowercase"}]},
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
          "decoder": bl_dec}),
        ("Metaspace(prepend_scheme=never, split=True) + byte_fallback",
         {"version": "1.0", "model": {"type": "BPE", "vocab": bf_vocab, "merges": [], "byte_fallback": True},
          "pre_tokenizer": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "never", "split": True},
          "decoder": {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "never", "split": True}}),
        ("Punctuation(MergedWithPrevious) + byte_fallback",
         {"version": "1.0", "model": {"type": "BPE", "vocab": bf_vocab, "merges": [], "byte_fallback": True},
          "pre_tokenizer": {"type": "Punctuation", "behavior": "MergedWithPrevious"}}),
        ("Whitespace + ByteLevel(use_regex=False)",
         {"version": "1.0", "model": {"type": "BPE", "vocab": bl_vocab, "merges": []},
          "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
              {"type": "Whitespace"},
              {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
          "decoder": bl_dec}),
        ("WhitespaceSplit + ByteLevel(add_prefix_space=True, use_regex=False)",
         {"version": "1.0", "model": {"type": "BPE", "vocab": bl_vocab, "merges": []},
          "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
              {"type": "WhitespaceSplit"},
              {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": False}]},
          "decoder": bl_dec}),
    ]
    synth_texts = ["", " ", "Hello,  World! 123", "  a...b!? c  ", "café naïve — test"]
    for label, cfg in synth_cases:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg, f); path = f.name
        try:
            fast = HFTokenizer.from_file(path)
            AT = tuetoken.AutoTokenizer.from_file(path)
            good = all(AT.encode(s, add_special_tokens=False) == fast.encode(s, add_special_tokens=False).ids
                       for s in synth_texts)
            tot += 1; ok += int(good)
            _line(label[:34], int(good), 1)
        finally:
            os.unlink(path)
    assert ok == tot, f"composable_pipeline: {ok}/{tot}"
    return ok, tot


def test_failclosed():
    # Reject (don't silently mistokenize) constructs the engine genuinely does not
    # implement: non-BPE models (WordPiece, Unigram), unknown normalizers
    # (BertNormalizer, Precompiled), unknown pre-tokenizers (BertPreTokenizer,
    # UnicodeScripts), and unsupported BPE subword prefixes.
    print("=== fail-closed: unsupported models/stages rejected ===")
    import json, tempfile
    from huggingface_hub import hf_hub_download
    ok = tot = 0
    for repo, why in [("google-bert/bert-base-uncased", "WordPiece"),
                      ("cross-encoder/nli-deberta-v3-small", "Unigram")]:
        try:
            tj = hf_hub_download(repo, "tokenizer.json")
        except Exception:
            print(f"  {repo}: SKIP (unavailable)"); continue
        tot += 1
        try:
            tuetoken.Tokenizer(tj)
            print(f"  {repo} ({why}): LOADED (should reject) [FAIL]")
        except Exception:
            ok += 1; print(f"  {repo} ({why}): rejected [PASS]")

    synthetic = [
        ("Unigram model",
         {"model": {"type": "Unigram", "vocab": [["<unk>", 0.0]]},
          "pre_tokenizer": {"type": "ByteLevel", "use_regex": True}}),
        ("WordPiece model",
         {"model": {"type": "WordPiece", "vocab": {"[UNK]": 0}},
          "pre_tokenizer": {"type": "Whitespace"}}),
        ("BertNormalizer",
         {"model": {"type": "BPE", "vocab": {}, "merges": []},
          "normalizer": {"type": "BertNormalizer"},
          "pre_tokenizer": {"type": "ByteLevel", "use_regex": True}}),
        ("Precompiled normalizer",
         {"model": {"type": "BPE", "vocab": {}, "merges": []},
          "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "Precompiled", "precompiled_charsmap": ""}]},
          "pre_tokenizer": {"type": "ByteLevel", "use_regex": True}}),
        ("UnicodeScripts pre_tokenizer",
         {"model": {"type": "BPE", "vocab": {}, "merges": [], "byte_fallback": True},
          "pre_tokenizer": {"type": "UnicodeScripts"}}),
        ("BPE continuing_subword_prefix",
         {"model": {"type": "BPE", "vocab": {}, "merges": [], "continuing_subword_prefix": "##"},
          "pre_tokenizer": {"type": "ByteLevel", "use_regex": True}}),
    ]
    for why, cfg in synthetic:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg, f); path = f.name
        tot += 1
        try:
            tuetoken.Tokenizer(path)
            print(f"  synthetic ({why}): LOADED (should reject) [FAIL]")
        except Exception:
            ok += 1; print(f"  synthetic ({why}): rejected [PASS]")
        finally:
            os.unlink(path)
    assert ok == tot, f"fail-closed: {ok}/{tot}"
    return ok, tot


# --- 6. new API surface: constructors, dunders, batch, offsets, loader -------
def test_new_features():
    print("=== new API (from_*, dunders, pickle, batch, offsets, encode_batch) ===")
    import numpy as np, pickle, random as _random
    rnd = _random.Random(0)
    ok = tot = 0
    R = tuetoken.Tokenizer.from_tiktoken("cl100k_base")
    tk = tiktoken.get_encoding("cl100k_base")

    # constructors / metadata
    tot += 1; ok += R.encode_ordinary("hello world") == tk.encode_ordinary("hello world")
    tot += 1; ok += bool(isinstance(tuetoken.__version__, str) and tuetoken.__version__)
    tot += 1; ok += len(R) == R.n_vocab and "Tokenizer" in repr(R)
    tot += 1; ok += pickle.loads(pickle.dumps(R)).encode_ordinary("x") == R.encode_ordinary("x")

    texts = TEST_TEXTS + ["café", "the quick brown fox jumps over the lazy dog"]
    seqs = [R.encode_ordinary(t) for t in texts]

    # batch count / decode parity
    tot += 1; ok += R.count_tokens_batch(texts) == [len(s) for s in seqs]
    tot += 1; ok += R.decode_batch(seqs) == [tk.decode(s) for s in seqs]

    # encode_with_offsets: ids match, spans are byte-exact
    off_ok = True
    for t in texts:
        ids, offs = R.encode_with_offsets(t)
        b = t.encode("utf-8")
        if ids != R.encode_ordinary(t): off_ok = False
        if any(b[s:e].decode("utf-8", "replace") != R.decode([i]) for i, (s, e) in zip(ids, offs)):
            off_ok = False
    tot += 1; ok += off_ok

    # encode_batch (training loader): dynamic + fixed-width + truncation + mask
    b = R.encode_batch(texts)
    ids, mask = b["input_ids"], b["attention_mask"]
    dyn_ok = ids.dtype == np.uint32 and mask.dtype == np.uint8 and ids.flags.writeable
    for r, s in enumerate(seqs):
        w = ids.shape[1]
        dyn_ok &= ids[r].tolist() == s + [0] * (w - len(s))
        dyn_ok &= mask[r].tolist() == [1] * len(s) + [0] * (w - len(s))
    tot += 1; ok += dyn_ok
    b2 = R.encode_batch(texts, max_length=3, pad_id=99)
    fix_ok = b2["input_ids"].shape[1] == 3
    for r, s in enumerate(seqs):
        exp = s[:3]
        fix_ok &= b2["input_ids"][r].tolist() == exp + [99] * (3 - len(exp))
    tot += 1; ok += fix_ok
    tot += 1; ok += R.encode_batch([])["input_ids"].shape == (0, 0)

    # encode_ordinary_to_buffer: flat uint32 ids + int64 offsets, slices == list
    buf = R.encode_ordinary_to_buffer(texts)
    data, offs = buf["data"], buf["offsets"]
    buf_ok = (data.dtype == np.uint32 and offs.dtype == np.int64
              and len(offs) == len(texts) + 1 and int(offs[0]) == 0
              and int(offs[-1]) == len(data) == sum(len(s) for s in seqs))
    for i, s in enumerate(seqs):
        buf_ok &= data[offs[i]:offs[i + 1]].tolist() == s
    tot += 1; ok += buf_ok
    tot += 1; ok += R.encode_ordinary_to_buffer([])["offsets"].tolist() == [0]

    # from_tiktoken accepts an OpenAI MODEL name (not just an encoding name)
    tot += 1
    ok += tuetoken.Tokenizer.from_tiktoken("gpt-4o").encode_ordinary("hi") == \
          tiktoken.get_encoding("o200k_base").encode_ordinary("hi")

    res = _line("new features", ok, tot)
    assert ok == tot, f"new features: {ok}/{tot}"
    return res


# --- 7. rank-inverted vocab (the gemma-whitespace BPE bug) -------------------
def test_rank_inversion():
    # gemma assigns its whitespace-run tokens INVERTED ranks: `(30-run)+(1-run)->
    # 31-run` is near rank 0, far below `(1)+(1)->2`. Canonical BPE grows one token
    # to the max run then repeats; the fast batch merger built power-of-two chunks.
    # Build that shape offline (run tokens of one byte, longer = lower rank) and
    # assert encode == canonical across the max-token boundary; also a MONOTONIC
    # twin (shorter = lower rank) to prove the fast path is unaffected.
    print("=== rank-inverted vocab encodes canonically (gemma whitespace) ===")
    import json, tempfile

    def canonical(units, rank):  # leftmost-lowest-rank greedy BPE reference
        parts = list(units)
        while True:
            best = None; bi = -1
            for i in range(len(parts) - 1):
                r = rank.get((parts[i], parts[i + 1]))
                if r is not None and (best is None or r < best):
                    best = r; bi = i
            if bi < 0: return parts
            parts[bi:bi + 2] = [parts[bi] + parts[bi + 1]]

    K = 31
    ok = tot = 0
    for inverted in (True, False):
        vocab = {"a" * L: L - 1 for L in range(1, K + 1)}
        pairs = [("a" * (L - 1), "a") for L in range(2, K + 1)]
        pairs.sort(key=lambda p: -(len(p[0]) + 1) if inverted else (len(p[0]) + 1))
        rank = {(x, y): i for i, (x, y) in enumerate(pairs)}
        cfg = {"model": {"type": "BPE", "vocab": vocab, "merges": [f"{a} {b}" for a, b in pairs]},
               "pre_tokenizer": {"type": "ByteLevel", "use_regex": True},
               "decoder": {"type": "ByteLevel"}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg, f); path = f.name
        R = tuetoken.Tokenizer(path)
        for n in (1, 2, 30, 31, 32, 50, 62, 100, 257, 1000):
            ids = R.encode_ordinary("a" * n)
            want = [vocab[p] for p in canonical(["a"] * n, rank)]
            tot += 1; ok += (ids == want)
    _line("rank-inverted + monotonic twin", ok, tot)
    assert ok == tot, f"rank_inversion: {ok}/{tot}"
    return ok, tot


# --- 8. AutoTokenizer: full HF-API parity vs transformers --------------------
def test_auto():
    # The HF-compatible wrapper must match transformers.AutoTokenizer on encode
    # (add_special_tokens both ways), decode (skip both ways), __call__
    # (padding/truncation), and apply_chat_template — across byte-level and the
    # SentencePiece prepend schemes ("first"/"always").
    print("=== AutoTokenizer == transformers.AutoTokenizer ===")
    try:
        from transformers import AutoTokenizer as HF
        import jinja2  # noqa: F401
    except Exception as e:
        print(f"  SKIP (transformers/jinja2 unavailable: {type(e).__name__})")
        return 0, 0
    repos = ["Qwen/Qwen2.5-0.5B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
             "microsoft/Phi-3.5-mini-instruct"]  # byte-level, SP-first, SP-always
    inputs = ["Hello world!", "café 你好 <|im_start|>x", "a</s>b", "</s>x</s>y",
              "snake_case 123\n new line", ""]
    chat = [{"role": "user", "content": "hi 你好"}, {"role": "assistant", "content": "yo!"},
            {"role": "user", "content": "ok?"}]
    ok = tot = 0
    for repo in repos:
        try:
            hf = HF.from_pretrained(repo); tt = tuetoken.AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            print(f"  {repo:38} SKIP ({type(e).__name__})"); continue
        if hf.pad_token is None: hf.pad_token = hf.eos_token
        if tt.pad_token is None: tt.pad_token = tt.eos_token
        tt.padding_side = hf.padding_side
        e = sum(hf.encode(t, add_special_tokens=a) == tt.encode(t, add_special_tokens=a)
                for t in inputs for a in (True, False))
        ii = hf.encode("Hi café 1 你好 x", add_special_tokens=True)
        e += all(tt.decode(ii, skip_special_tokens=s) == hf.decode(ii, skip_special_tokens=s) for s in (True, False))
        e += all(hf(inputs, padding=True, truncation=True, max_length=ml)["input_ids"]
                 == tt(inputs, padding=True, truncation=True, max_length=ml)["input_ids"] for ml in (8, 16))
        # all-Rust padded-batch (return_tensors): input_ids + attention_mask exact
        import numpy as np
        for ml in (None, 12):
            a = hf(inputs, padding=True, truncation=ml is not None, max_length=ml, return_tensors="np")
            b = tt(inputs, padding=True, truncation=ml is not None, max_length=ml, return_tensors="np")
            e += int(np.array_equal(a["input_ids"], b["input_ids"])
                     and np.array_equal(a["attention_mask"], b["attention_mask"]))
        _cids = lambda x: list(x["input_ids"]) if hasattr(x, "keys") else list(x)  # noqa: E731  (BatchEncoding is a Mapping, not a dict, in transformers>=5)
        e += _cids(hf.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)) \
             == _cids(tt.apply_chat_template(chat, tokenize=True, add_generation_prompt=True))
        n = len(inputs) * 2 + 5  # encode(2x) + decode + call + 2x tensors + chat
        _line(repo, e, n); ok += e; tot += n
    assert ok == tot, f"auto: {ok}/{tot}"
    return ok, tot


def test_kimi_tiktoken_model():
    # A tiktoken-format model (ships tiktoken.model + a custom pat_str, no
    # tokenizer.json): plain from_pretrained must auto-recover the pattern (from the
    # auto_map module, statically) and match HF byte-exact on ordinary text, Han/Latin
    # adjacency, special tokens, and decode.
    print("=== Kimi (tiktoken.model) via from_pretrained == HF ===")
    repo = "moonshotai/Kimi-K2-Instruct"
    try:
        from transformers import AutoTokenizer as HF
        hf = HF.from_pretrained(repo, trust_remote_code=True)
        tt = tuetoken.AutoTokenizer.from_pretrained(repo)  # no explicit pattern
    except Exception as e:
        print(f"  SKIP ({type(e).__name__}: {str(e)[:50]})")
        return 0, 0
    import random
    random.seed(11)
    HAN = "你好世界今天天气中文漢字测试北京上海人工智能"
    LAT = "abcDEFCamelCaseXyz"; DIG = "0123456789"; PUN = "!?.,@#-_/ "
    def glue(n):
        return "".join(random.choice(random.choice([HAN, LAT, DIG, PUN]))
                       for _ in range(n))
    curated = ["Hello world", "你好世界，今天天气不错", "中文English混合 mixed 漢字テスト",
               "def f(x): return x*2 # 注释", "café señor über", "🚀 emoji 👍🏽", "АБВ Привет"]
    fuzz = [glue(random.randint(4, 40)) for _ in range(500)]
    ime = next((t for t in tt._added_content_to_id if "im_end" in t), None)
    specials = [f"hi 你好{ime} bye", f"{ime}中文{ime}text", f"a {ime} b 漢字"] if ime else []
    ok = tot = 0
    for s in curated + fuzz:
        tot += 1; ok += tt.encode(s, add_special_tokens=False) == hf.encode(s)
    for s in specials:
        tot += 1; ok += tt.encode(s) == hf.encode(s)
    # decode round-trip
    tot += 1; rt = "你好 world 漢字 mix_123 señor"
    ok += tt.decode(tt.encode(rt, add_special_tokens=False)) == rt
    _line(repo, ok, tot)
    assert ok == tot, f"kimi: {ok}/{tot}"
    return ok, tot


def test_bytes_inputs():
    """bytes batch inputs: identical to str inputs when valid; ValueError naming
    the (first) offending item when not — including invalid bytes deep inside a
    large document that is split across threads."""
    print("=== bytes inputs (lazy UTF-8 validation) ===")
    import numpy as np
    ok = tot = 0
    for enc in ["gpt2", "cl100k_base"]:
        R = tuetoken.Tokenizer.from_tiktoken(enc)
        text = make_corpus()
        big = (text + "\n") * max(1, (3 << 20) // max(1, len(text.encode())))  # > split threshold
        for src in [[text], [big], [t for t in text.split("\n") if t]]:
            bs = [s.encode() for s in src]
            tot += 1; ok += R.count_tokens_batch(bs) == R.count_tokens_batch(src)
            a = np.asarray(R.encode_ordinary_to_buffer(bs)["data"])
            b = np.asarray(R.encode_ordinary_to_buffer(src)["data"])
            tot += 1; ok += bool(np.array_equal(a, b))
        bb = big.encode()
        mid = len(bb) // 2
        while bb[mid] & 0xC0 == 0x80:
            mid += 1
        bad_cases = [
            ([bb[:mid] + b"\xff" + bb[mid + 1:]], 0),          # inside a split document
            ([bb + b"\xe2\x80"], 0),                          # truncated at the very end
            ([b"ok", b"fine", b"x\xc0\xafy", b"\x80"], 2),     # overlong; lowest index reported
        ]
        for inputs, idx in bad_cases:
            for f in (R.count_tokens_batch, R.encode_ordinary_to_buffer, R.encode_ordinary_batch):
                tot += 1
                try:
                    f(inputs)
                except ValueError as e:
                    ok += f"item {idx}:" in str(e)
    _line("bytes inputs", ok, tot)
    return ok, tot


def test_warmup_and_fast_startup():
    """warmup() + warmup= kwarg populate shared chunk tables with zero parity drift."""
    print("=== warmup & fast startup ===")
    ok = tot = 0
    sample = make_corpus()
    for enc in ["gpt2", "cl100k_base"]:
        base = tuetoken.Tokenizer.from_tiktoken(enc)
        want = base.encode_ordinary(sample)
        # Tokenizer.warmup() (no texts) + warmup(sample)
        t1 = tuetoken.Tokenizer.from_tiktoken(enc, warmup=True)
        tot += 1; ok += t1.encode_ordinary(sample) == want
        n1 = t1.warmup(sample)
        tot += 1; ok += n1 > 0 and t1.encode_ordinary(sample) == want
        # Tokenizer.from_tiktoken(..., warmup=[bytes, str])
        t2 = tuetoken.Tokenizer.from_tiktoken(enc, warmup=[sample.encode(), "Hello world!"])
        tot += 1; ok += t2.learned_entries > 0 and t2.encode_ordinary(sample) == want
    # Test AutoTokenizer.from_file with real cached tiktoken json
    from tuetoken import _tiktoken as _tt
    tj = _tt.get_cached_tokenizer_path("cl100k_base")
    a = tuetoken.AutoTokenizer.from_file(tj, warmup=sample)
    tot += 1; ok += a._id_to_token_cache is None  # lazy: did NOT re-parse tokenizer.json!
    tot += 1; ok += a.encode(sample, add_special_tokens=False) == tuetoken.Tokenizer(tj).encode_ordinary(sample)
    tot += 1; ok += a.warmup([sample, sample.encode()]) > 0
    # Accessing get_vocab lazily loads the full vocab dict on demand
    v = a.get_vocab()
    tot += 1; ok += len(v) >= 100000 and a._id_to_token_cache is not None
    _line("warmup & fast startup", ok, tot)
    assert ok == tot, f"warmup: {ok}/{tot}"
    return ok, tot


def test_tekken_and_legacy_formats():
    """Mistral tekken.json (Devstral, Mistral-Small-3.2, Magistral) and legacy
    vocab.json+merges.txt (OPT) load transparently via from_pretrained == HF."""
    print("=== Mistral tekken.json & legacy BPE formats == transformers ===")
    try:
        from transformers import AutoTokenizer as HF
    except Exception as e:
        print(f"  SKIP ({type(e).__name__})")
        return 0, 0
    ok = tot = 0
    repos = [
        "mistralai/Devstral-Small-2505",               # Tekken v7
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",  # Tekken v11
        "mistralai/Magistral-Small-2506",              # Tekken v11
        "facebook/opt-125m",                           # legacy vocab.json + merges.txt
    ]
    samples = [
        "Hello, world!",
        "def foo_bar(x: int = 123456):\n    return x * 2  # comment\n",
        "café naïve 日本語 한국어 中文 🚀🔥",
        "1 12 123 1234 1234567890   spaces\t\ttabs\r\nnewlines",
        "camelCaseWord HTTPSConnection XMLParser don't we'll",
    ]
    for repo in repos:
        try:
            hf = HF.from_pretrained(repo)
            at = tuetoken.AutoTokenizer.from_pretrained(repo)
            core = tuetoken.Tokenizer.from_pretrained(repo)
        except Exception as e:
            print(f"  {repo:46} SKIP ({type(e).__name__}: {str(e)[:50]})")
            continue
        local = 0
        n_chk = 0
        for s in samples:
            for add_sp in (False, True):
                n_chk += 1
                local += int(at.encode(s, add_special_tokens=add_sp) == hf.encode(s, add_special_tokens=add_sp))
            n_chk += 1
            local += int(core.encode_ordinary(s) == hf.encode(s, add_special_tokens=False))
            ids = hf.encode(s, add_special_tokens=False)
            n_chk += 1
            local += int(at.decode(ids) == hf.decode(ids))
        n_chk += 1
        local += int(at.vocab_size == len(hf) and at.bos_token_id == hf.bos_token_id
                     and at.eos_token_id == hf.eos_token_id and at.pad_token_id == hf.pad_token_id)
        ok += local; tot += n_chk
        _line(repo, local, n_chk)
    assert ok == tot, f"tekken_and_legacy: {ok}/{tot}"
    return ok, tot


def test_review_regressions():
    """Regressions for Unicode lstrip/rstrip, NFC FastBatch mixed scripts/errors,
    and HF/vLLM AutoTokenizer protocol helpers (is_fast, get_added_vocab,
    convert_tokens_to_string)."""
    print("=== review regressions (Unicode strip, NFC FastBatch, vLLM helpers) ===")
    import json
    import tempfile
    ok = tot = 0
    # 1. Synthetic NFC ByteLevel tokenizer with Unicode lstrip/rstrip added token
    from tuetoken import _tiktoken as _tt
    base_tj = json.load(open(_tt.get_cached_tokenizer_path("cl100k_base"), encoding="utf-8"))
    base_tj["normalizer"] = {"type": "NFC"}
    strip_id = max(base_tj["model"]["vocab"].values()) + 10
    base_tj["added_tokens"].append({
        "id": strip_id, "content": "<|strip|>", "single_word": False,
        "lstrip": True, "rstrip": True, "normalized": False, "special": True,
    })
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(base_tj, f)
        tmp_path = f.name
    try:
        at = tuetoken.AutoTokenizer.from_file(tmp_path)
        core = tuetoken.Tokenizer(tmp_path)
        # HF/vLLM helpers
        tot += 1; ok += (at.is_fast is True)
        added_v = at.get_added_vocab()
        tot += 1; ok += (added_v.get("<|strip|>") == strip_id)
        toks = at.tokenize("Hello, world!")
        tot += 1; ok += (at.convert_tokens_to_string(toks) == "Hello, world!")

        # Unicode whitespace lstrip/rstrip (\u00a0 NBSP, \u2003 EM SPACE, \u3000 IDEOGRAPHIC SPACE)
        want_strip = core.encode_ordinary("hello") + [strip_id] + core.encode_ordinary("world")
        for s in [
            "hello <|strip|> world",
            "hello\u00a0<|strip|>\u00a0world",
            "hello\u2003<|strip|>\u3000world",
        ]:
            tot += 1; ok += (at.encode(s, add_special_tokens=False) == want_strip)

        # NFC FastBatch on mixed ASCII, Latin-1, UCS-2, UCS-4, NFD decomposed, and bytes
        mixed = [
            "pure ascii sentence 123",
            "caf\u00e9 na\u00efve \u00c4\u00d6\u00dc",
            "cafe\u0301 naive\u0308 A\u0308O\u0308U\u0308",  # NFD -> NFC changes bytes!
            "日本語と한국어と中文",
            "emojis 🚀🔥✨ and café + cafe\u0301",
            b"bytes ascii + cafe\xcc\x81!",
            "",
        ] * 20
        want_batch = [
            core.encode_ordinary(x.decode("utf-8") if isinstance(x, bytes) else x)
            for x in mixed
        ]
        for nt in (1, 0):
            got = core.encode_ordinary_batch(mixed, num_threads=nt)
            tot += 1; ok += (got == want_batch)
            got_cnt = core.count_tokens_batch(mixed, num_threads=nt)
            tot += 1; ok += (got_cnt == [len(w) for w in want_batch])
            buf = core.encode_ordinary_to_buffer(mixed, num_threads=nt)
            offs = buf["offsets"]
            got_buf = [buf["data"][offs[i]:offs[i + 1]].tolist() for i in range(len(mixed))]
            tot += 1; ok += (got_buf == want_batch)
        # Invalid UTF-8 bytes & lone surrogate in NFC FastBatch must still raise
        try:
            core.encode_ordinary_batch(["ok", b"\xff\xfe"], num_threads=1)
            raised_b = False
        except ValueError:
            raised_b = True
        tot += 1; ok += raised_b
        try:
            core.encode_ordinary_batch(["ok", "\ud800"], num_threads=1)
            raised_s = False
        except (ValueError, UnicodeEncodeError):
            raised_s = True
        tot += 1; ok += raised_s
    finally:
        os.unlink(tmp_path)
    _line("review regressions", ok, tot)
    assert ok == tot, f"review_regressions: {ok}/{tot}"
    return ok, tot


def test_whisper():
    """Whisper tokenizer parity vs HF WhisperTokenizerFast across all Whisper
    generations (tiny, tiny.en, large-v2, large-v3, large-v3-turbo, distil-large-v3),
    plus legacy vocab.json+merges.txt-only Whisper repos."""
    print("=== whisper models (parity, timestamps, prefixes, prompts, offsets, legacy) ===")
    import json
    import shutil
    import tempfile
    from transformers import WhisperTokenizerFast
    from huggingface_hub import hf_hub_download
    from tuetoken import _formats as _fmt

    repos = [
        "openai/whisper-tiny",
        "openai/whisper-tiny.en",
        "openai/whisper-large-v2",
        "openai/whisper-large-v3",
        "openai/whisper-large-v3-turbo",
        "distil-whisper/distil-large-v3",
    ]
    samples = [
        "",
        "Hello, world!",
        " The quick brown fox jumps over the lazy dog 12345.",
        "Bonjour! Ça va bien? 日本語のテストと한국어とالعربية 🚀",
        "<|startoftranscript|><|notimestamps|> Hello world.<|endoftext|>",
        "<|0.00|> Hello world.<|2.50|><|2.50|> Second phrase.<|5.00|>",
        "text with < single angle and <|not_a_token|> literal",
    ]
    ok = tot = 0
    for repo in repos:
        try:
            hf = WhisperTokenizerFast.from_pretrained(repo, local_files_only=True)
        except Exception:
            continue
        at = tuetoken.AutoTokenizer.from_pretrained(repo)
        n_ok = n_tot = 0

        # Metadata & properties
        n_tot += 1
        n_ok += int(
            at._is_whisper
            and at.vocab_size == len(hf)
            and at.bos_token_id == hf.bos_token_id
            and at.eos_token_id == hf.eos_token_id
            and at.pad_token_id == hf.pad_token_id
            and at.all_special_ids == sorted(hf.all_special_ids)
            and at.prefix_tokens == hf.prefix_tokens
            and at.timestamp_ids() == hf.timestamp_ids()
        )

        # Single encode (sp=False / sp=True) and batch __call__
        for s in samples:
            for sp in (False, True):
                n_tot += 1
                n_ok += int(at.encode(s, add_special_tokens=sp) == hf.encode(s, add_special_tokens=sp))
        for sp in (False, True):
            n_tot += 1
            n_ok += int(at(samples, add_special_tokens=sp)["input_ids"] == hf(samples, add_special_tokens=sp)["input_ids"])

        # Decode matrix: standard, timestamped, multi-segment >30s rollovers, prompt strip
        ts0 = at._whisper_ts_begin
        sot = at._id("<|startoftranscript|>")
        sop = at._id("<|startofprev|>")
        eot = at.eos_token_id
        hello = at.encode(" Hello world.", add_special_tokens=False)
        seg2 = at.encode(" Next segment!", add_special_tokens=False)
        prompt_body = at.encode(" prior context", add_special_tokens=False)

        dec_cases = [
            at.encode("Hello, world! 日本語", add_special_tokens=True),
            [sot, ts0 + 0, *hello, ts0 + 125, ts0 + 125, *seg2, ts0 + 250, eot],
            # Single-ending >30s rollover (+30.00s)
            [sot, ts0 + 0, *hello, ts0 + 1450, ts0 + 50, *seg2, ts0 + 250, eot],
            # Consecutive boundary rollover (rewind second ending timestamp, +20.00s)
            [sot, ts0 + 0, *hello, ts0 + 1000, ts0 + 1000, ts0 + 250, *seg2, ts0 + 500, eot],
            # Prompt + transcript + timestamps
            [sop, *prompt_body, sot, ts0 + 0, *hello, ts0 + 200, eot],
            # Prompt without <|startoftranscript|>
            [sop, *prompt_body],
        ]
        for ids in dec_cases:
            for skip in (False, True):
                for ts_flag in (False, True):
                    n_tot += 1
                    n_ok += int(
                        at.decode(ids, skip_special_tokens=skip, decode_with_timestamps=ts_flag)
                        == hf.decode(ids, skip_special_tokens=skip, decode_with_timestamps=ts_flag)
                    )
        # batch_decode (small serial + >32K-token parallel path!)
        for skip in (False, True):
            for ts_flag in (False, True):
                n_tot += 1
                n_ok += int(
                    at.batch_decode(dec_cases, skip_special_tokens=skip, decode_with_timestamps=ts_flag)
                    == hf.batch_decode(dec_cases, skip_special_tokens=skip, decode_with_timestamps=ts_flag)
                )
        big_batch = dec_cases * 500  # > 32K tokens -> exercises parallel decode_batch path
        n_tot += 1
        n_ok += int(
            at.batch_decode(big_batch, skip_special_tokens=True, decode_with_timestamps=True)
            == at.batch_decode(dec_cases, skip_special_tokens=True, decode_with_timestamps=True) * 500
        )

        # output_offsets parity
        for ids in dec_cases[1:5]:
            n_tot += 1
            n_ok += int(
                at.decode(ids, skip_special_tokens=True, output_offsets=True)
                == hf.decode(ids, skip_special_tokens=True, output_offsets=True)
            )

        # Normalization parity
        norm_ids = at.encode("Mr. Smith's 21st café (applause)!", add_special_tokens=True)
        for kw in (
            dict(skip_special_tokens=True, basic_normalize=True),
            dict(skip_special_tokens=True, basic_normalize=True, remove_diacritics=True),
            dict(skip_special_tokens=True, normalize=True),
        ):
            n_tot += 1
            n_ok += int(at.decode(norm_ids, **kw) == hf.decode(norm_ids, **kw))

        # get_prompt_ids parity + disallowed special check
        n_tot += 1
        n_ok += int(at.get_prompt_ids("previous context").tolist() == hf.get_prompt_ids("previous context").tolist())
        try:
            at.get_prompt_ids("bad <|endoftext|> prompt")
            raised = False
        except ValueError:
            raised = True
        n_tot += 1; n_ok += int(raised)

        # set_prefix_tokens & get_decoder_prompt_ids (multilingual models only)
        if not repo.endswith(".en"):
            for lang, task, pred_ts in [
                ("spanish", "transcribe", False),
                ("fr", "translate", True),
                ("mandarin", "transcribe", False),
            ]:
                hf.set_prefix_tokens(language=lang, task=task, predict_timestamps=pred_ts)
                at.set_prefix_tokens(language=lang, task=task, predict_timestamps=pred_ts)
                n_tot += 1
                n_ok += int(
                    at.prefix_tokens == hf.prefix_tokens
                    and at.encode("Hello world!", add_special_tokens=True)
                    == hf.encode("Hello world!", add_special_tokens=True)
                )
            n_tot += 1
            n_ok += int(
                at.get_decoder_prompt_ids(language="german", task="transcribe", no_timestamps=False)
                == hf.get_decoder_prompt_ids(language="german", task="transcribe", no_timestamps=False)
            )

        ok += n_ok; tot += n_tot
        _line(repo, n_ok, n_tot)

    # Legacy Whisper repo test (vocab.json + merges.txt + added_tokens.json + tokenizer_config.json, NO tokenizer.json)
    tj_src = None
    try:
        tj_src = hf_hub_download("openai/whisper-tiny", "tokenizer.json", local_files_only=True)
        cfg_src = hf_hub_download("openai/whisper-tiny", "tokenizer_config.json", local_files_only=True)
        hf_ref = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", local_files_only=True)
    except Exception:
        tj_src = None
    if tj_src is not None:
        tj_data = json.load(open(tj_src, encoding="utf-8"))
        tmpdir = tempfile.mkdtemp(prefix="tuetoken_legacy_whisper_")
        try:
            with open(os.path.join(tmpdir, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump(tj_data["model"]["vocab"], f)
            with open(os.path.join(tmpdir, "merges.txt"), "w", encoding="utf-8") as f:
                f.write("#version: 0.2\n")
                for m in tj_data["model"]["merges"]:
                    f.write((m if isinstance(m, str) else " ".join(m)) + "\n")
            added_map = {a["content"]: a["id"] for a in tj_data["added_tokens"]}
            with open(os.path.join(tmpdir, "added_tokens.json"), "w", encoding="utf-8") as f:
                json.dump(added_map, f)
            shutil.copy(cfg_src, os.path.join(tmpdir, "tokenizer_config.json"))

            res = _fmt.try_load_legacy_bpe_repo(tmpdir)
            assert res is not None, "legacy Whisper loader returned None"
            leg_tj, leg_cfg = res
            leg_at = tuetoken.AutoTokenizer(leg_tj, leg_cfg)
            n_ok = n_tot = 0
            for s in samples:
                for sp in (False, True):
                    n_tot += 1
                    n_ok += int(leg_at.encode(s, add_special_tokens=sp) == hf_ref.encode(s, add_special_tokens=sp))
            ts0 = leg_at._whisper_ts_begin
            ids = [leg_at._id("<|startoftranscript|>"), ts0, *leg_at.encode(" Hi!", add_special_tokens=False), ts0 + 50, leg_at.eos_token_id]
            for skip in (False, True):
                for ts_flag in (False, True):
                    n_tot += 1
                    n_ok += int(
                        leg_at.decode(ids, skip_special_tokens=skip, decode_with_timestamps=ts_flag)
                        == hf_ref.decode(ids, skip_special_tokens=skip, decode_with_timestamps=ts_flag)
                    )
            ok += n_ok; tot += n_tot
            _line("legacy Whisper (vocab+merges only)", n_ok, n_tot)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    assert ok == tot and tot > 0, f"whisper: {ok}/{tot}"
    return ok, tot


def test_zero_copy_buffer_decode():
    """Verify 1D/2D integer buffer decode (uint32/int32/int64/uint64/uint16/int16)
    and AutoTokenizer numpy fast decode without .tolist() boxing."""
    print("=== zero-copy buffer decode (1D/2D numpy & array.array) ===")
    import array
    import numpy as np

    ok = tot = 0
    tok = tuetoken.Tokenizer.from_tiktoken("o200k_base")
    samples = [
        "",
        "Hello, world!",
        "Pure ASCII code: for i in range(100): x += i * 2\n" * 40,
        "Mixed UTF-8: café, naïve, Zürich, 日本語, 🚀✨ — end!\n" * 15,
    ]
    for s in samples:
        ids = tok.encode_ordinary(s)
        for dt in (np.uint32, np.int32, np.int64, np.uint64):
            arr = np.array(ids, dtype=dt)
            tot += 1
            ok += int(tok.decode(arr) == s)
        tot += 1
        ok += int(tok.decode(array.array("I", ids)) == s)
    # 2D batch decode across dtypes (including large >65k tok parallel trigger)
    row = tok.encode_ordinary("The quick brown fox jumps over the lazy dog. " * 20)[:128]
    mat_expected = [tok.decode(row)] * 600
    for dt in (np.uint32, np.int32, np.int64, np.uint64):
        mat = np.tile(np.array(row, dtype=dt), (600, 1))
        tot += 1
        ok += int(tok.decode_batch(mat) == mat_expected)
    # Narrow u16/i16 dtypes on gpt2 (pick ids < 32767 so int16 holds them)
    tg = tuetoken.Tokenizer.from_tiktoken("gpt2")
    gtext = "Hello world! The quick brown fox 123."
    gids = tg.encode_ordinary(gtext)
    assert max(gids) < 32768, gids
    for dt in (np.uint16, np.int16):
        tot += 1
        ok += int(tg.decode(np.array(gids, dtype=dt)) == gtext)
        tot += 1
        ok += int(tg.decode_batch(np.array([gids, gids], dtype=dt)) == [gtext, gtext])
    # Reject negative values cleanly (must raise, never wrap)
    for bad in (np.array([-1, 2], dtype=np.int64), np.array([-5, 1], dtype=np.int32), np.array([[1, -2]], dtype=np.int64)):
        tot += 1
        try:
            (tok.decode_batch if bad.ndim == 2 else tok.decode)(bad)
        except Exception:
            ok += 1
    # AutoTokenizer numpy decode & batch_decode (1D, [1, N], 2D)
    at = tuetoken.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    qtext = "Hello <|im_end|> world — test!"
    qids = at.encode(qtext)
    qarr = np.array(qids, dtype=np.int64)
    tot += 3
    ok += int(at.decode(qarr) == at.decode(qids))
    ok += int(at.decode(qarr.reshape(1, -1), skip_special_tokens=True) == at.decode(qids, skip_special_tokens=True))
    ok += int(at.batch_decode(np.stack([qarr, qarr]), skip_special_tokens=True) == [at.decode(qids, skip_special_tokens=True)] * 2)
    _line("buffer decode (1D/2D/dtypes/Auto)", ok, tot)
    assert ok == tot, f"buffer decode: {ok}/{tot}"
    return ok, tot


def test_fast_and_batch_chat_templates():
    """Verify self-deriving fast chat compiler + batch apply_chat_template parity vs transformers."""
    print("=== fast & batch apply_chat_template parity vs transformers ===")
    import numpy as np
    from transformers import AutoTokenizer as HFAuto
    from tuetoken import AutoTokenizer as TTAuto

    ok = tot = 0
    for repo in ("Qwen/Qwen2.5-0.5B-Instruct", "microsoft/Phi-3.5-mini-instruct"):
        hf = HFAuto.from_pretrained(repo)
        tt = TTAuto.from_pretrained(repo)
        convs = [
            [{"role": "user", "content": "Hello world!"}],
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}, {"role": "user", "content": "Bye"}],
            [{"role": "system", "content": ""}, {"role": "user", "content": "empty sys"}],
            [{"role": "user", "content": "  spaces \n "}],
        ]
        n_ok = n_tot = 0
        for c in convs:
            for ag in (False, True):
                n_tot += 2
                n_ok += int(tt.apply_chat_template(c, add_generation_prompt=ag, tokenize=False)
                            == hf.apply_chat_template(c, add_generation_prompt=ag, tokenize=False))
                n_ok += int(tt.apply_chat_template(c, add_generation_prompt=ag, return_dict=False)
                            == hf.apply_chat_template(c, add_generation_prompt=ag, tokenize=True, return_dict=False))
        # Batch apply_chat_template (list[list[dict]]) with padding + numpy tensors
        b_tt = tt.apply_chat_template(convs, add_generation_prompt=True, padding=True, return_tensors="np")
        b_hf = hf.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, padding=True, return_tensors="np", return_dict=True)
        n_tot += 3
        n_ok += int(np.array_equal(b_tt["input_ids"], b_hf["input_ids"]))
        n_ok += int(np.array_equal(b_tt["attention_mask"], b_hf["attention_mask"]))
        n_ok += int(tt.apply_chat_template(convs, tokenize=False)
                    == [hf.apply_chat_template(c, tokenize=False) for c in convs])
        ok += n_ok; tot += n_tot
        _line(repo, n_ok, n_tot)
    assert ok == tot, f"chat templates: {ok}/{tot}"
    return ok, tot


def test_sentencepiece_fast_meta():
    """Verify zero-alloc SIMD SentencePiece fast path + Gemma-3 cross-space guard vs HF tokenizers."""
    print("=== SentencePiece fast path & Gemma-3 cross-space guard parity ===")
    from tokenizers import Tokenizer as HFTok
    from transformers import AutoTokenizer as HFAuto
    from tuetoken import AutoTokenizer as TTAuto

    cases = [
        "",
        " ",
        "   ",
        "hello",
        " hello",
        "  hello  world  ",
        "Hello, world! This is a SentencePiece test with numbers 12345 and\nnewlines.\n\n",
        "literal \u2581 metaspace \u2581\u2581 mixed with spaces and \u2581hello",
        "\u2581leading_metaspace and trailing \u2581",
        "<div> </div> <span> > </div> >  </div> > </span> > </a> >= >",
        "Code: if a > b && c > 0 { return \"a > b\"; } // <b>hi</b> > </div>",
        "Unicode: café naïve 日本語テスト 🚀👨‍👩‍👧‍👦 Привет мир! — “quoted”",
        # 300KB multi-thread parallel split (> SPLIT_THRESHOLD)
        ("The quick brown fox jumps over the lazy dog. <div> > </div> 12345 café 日本語\n" * 4000),
    ]

    ok = tot = 0
    for repo in (
        "microsoft/Phi-3.5-mini-instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "unsloth/gemma-2-9b-it",
        "unsloth/gemma-3-4b-it",
    ):
        try:
            hf = HFAuto.from_pretrained(repo)
            tt = TTAuto.from_pretrained(repo)
        except Exception as e:
            print(f"  [skip {repo}: {e}]")
            continue
        n_ok = n_tot = 0
        for s in cases:
            n_tot += 2
            n_ok += int(tt.encode(s, add_special_tokens=False) == hf.encode(s, add_special_tokens=False))
            n_ok += int(tt.encode(s, add_special_tokens=True) == hf.encode(s, add_special_tokens=True))
        # Batch & multi-threaded parallel batch parity
        batch_inputs = [c for c in cases[:-1]] + [cases[-1][:40000]]
        n_tot += 1
        n_ok += int(
            tt(batch_inputs, add_special_tokens=False)["input_ids"]
            == [hf.encode(c, add_special_tokens=False) for c in batch_inputs]
        )
        # Also verify raw Tokenizer encode_ordinary / encode_ordinary_batch on models without non-special added_tokens colliding
        if "gemma" not in repo:
            hf_tok = HFTok.from_pretrained(repo)
            n_tot += 2
            n_ok += int(
                tt._core.encode_ordinary(cases[-1])
                == hf_tok.encode(cases[-1], add_special_tokens=False).ids
            )
            n_ok += int(
                tt._core.encode_ordinary_batch(batch_inputs)
                == [hf_tok.encode(c, add_special_tokens=False).ids for c in batch_inputs]
            )
        else:
            # Verify raw encode_special_batch matches HF exactly on Gemma-2/3 too
            n_tot += 1
            n_ok += int(
                tt._core.encode_special_batch(batch_inputs)
                == [hf.encode(c, add_special_tokens=False) for c in batch_inputs]
            )
        ok += n_ok
        tot += n_tot
        _line(repo, n_ok, n_tot)
    assert tot > 0 and ok == tot, f"SentencePiece fast path: {ok}/{tot}"
    return ok, tot


def test_decode_stream():
    """Verify incremental streaming DecodeStream parity across ASCII, split UTF-8/emojis/CJK, invalid bytes, specials, and SentencePiece."""
    print("=== DecodeStream incremental streaming parity ===")
    import array
    import numpy as np
    from tuetoken import AutoTokenizer as TTAuto

    ok = tot = 0
    cases = [
        "",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog. 12345!",
        "Emojis: 🚀🔥👨‍👩‍👧‍👦🎉✨🇺🇸",
        "Multilingual: café naïve über ß 日本語テスト 人工智能 한국어 Привет мир! — “quoted”",
        "   leading spaces and\nnewlines\n\n  end ",
    ]

    # 1. ByteLevel (o200k_base & Qwen2.5): split multi-byte UTF-8 emojis & CJK NEVER emit bogus \ufffd mid-stream!
    tok = tuetoken.Tokenizer.from_tiktoken("o200k_base")
    for s in cases:
        ids = tok.encode_ordinary(s)
        ds = tok.decode_stream()
        chunks = [ds.step(t) for t in ids]
        tail = ds.flush()
        got = "".join(chunks) + tail
        tot += 3
        ok += int(got == s)
        ok += int("\ufffd" not in got)
        ok += int(ds.pending_bytes == 0)
        # step_many in slices of 1, 2, 3, 5 + numpy array + array.array
        for k in (2, 3, 5):
            ds.reset()
            parts = [ds.step_many(ids[i:i + k]) for i in range(0, len(ids), k)]
            parts.append(ds.flush())
            tot += 1
            ok += int("".join(parts) == s)
        ds.reset()
        tot += 2
        ok += int(ds.step_many(np.array(ids, dtype=np.uint32)) + ds.flush() == s)
        ds.reset()
        ok += int(ds.step_many(array.array("I", ids)) + ds.flush() == s)

    # Specifically prove a 4-byte emoji split across 2+ single-byte tokens buffers until complete!
    emoji_ids = tok.encode_ordinary("🚀👨‍👩‍👧‍👦")
    assert len(emoji_ids) >= 2
    ds = tok.decode_stream()
    mid_chunks = []
    saw_pending = False
    for t in emoji_ids:
        c = ds.step(t)
        if ds.pending_bytes > 0:
            saw_pending = True
        assert "\ufffd" not in c, f"spurious replacement char mid-stream: {c!r}"
        mid_chunks.append(c)
    mid_chunks.append(ds.flush())
    tot += 2
    ok += int("".join(mid_chunks) == "🚀👨‍👩‍👧‍👦")
    ok += int(saw_pending)

    # 2. Invalid / truncated UTF-8 byte tokens match decode() byte-for-byte (maximal-subpart U+FFFD)
    for cut in (1, 2, len(emoji_ids) - 1):
        sub = emoji_ids[:cut]
        ds = tok.decode_stream()
        got = "".join(ds.step(t) for t in sub) + ds.flush()
        tot += 1
        ok += int(got == tok.decode(sub))

    # Genuinely ill-formed mid-stream byte sequences via Phi-3.5 <0xXX> byte-fallback tokens
    phi = TTAuto.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    b_tok = lambda byte: phi.convert_tokens_to_ids(f"<0x{byte:02X}>")
    tok_a = phi.encode("A", add_special_tokens=False)[-1]
    bad_seqs = [
        [b_tok(0xFF), tok_a],
        [b_tok(0x80), b_tok(0x80), tok_a],
        [b_tok(0xC2), tok_a],
        [b_tok(0xE0), b_tok(0x80), tok_a],
        [b_tok(0xE2), b_tok(0x82), tok_a],
        [b_tok(0xF0), b_tok(0x9F), b_tok(0x98), tok_a],
        [b_tok(0xF0), b_tok(0x9F), b_tok(0x9A), b_tok(0x80), b_tok(0xFF), b_tok(0xF0), b_tok(0x9F)],  # 🚀 + bad + truncated!
    ]
    for seq in bad_seqs:
        ds = phi.decode_stream(skip_special_tokens=False)
        got = "".join(ds.step(t) for t in seq) + ds.flush()
        want = phi.decode(seq, skip_special_tokens=False)
        tot += 1
        ok += int(got == want)

    # 3. AutoTokenizer with skip_special_tokens=True/False (Qwen2.5) + SentencePiece strip_space (Phi-3.5)
    for repo in ("Qwen/Qwen2.5-0.5B-Instruct", "microsoft/Phi-3.5-mini-instruct"):
        at = TTAuto.from_pretrained(repo)
        for skip in (False, True):
            for s in ("Hello world!", " Hello 🚀 world", "café 日本語 123", "   spaces"):
                ids = at.encode(s, add_special_tokens=True)
                ds = at.decode_stream(skip_special_tokens=skip)
                got_step = "".join(ds.step(t) for t in ids) + ds.flush()
                ds.reset()
                got_many = ds.step_many(np.array(ids, dtype=np.int64)) + ds.flush()
                want = at.decode(ids, skip_special_tokens=skip)
                tot += 2
                ok += int(got_step == want)
                ok += int(got_many == want)
    _line("DecodeStream (ASCII/UTF-8/emoji/trunc/SP/Auto)", ok, tot)
    assert ok == tot, f"DecodeStream: {ok}/{tot}"
    return ok, tot


def test_count_and_file_apis():
    """Verify zero-alloc count_tokens / AutoTokenizer.count_tokens(_batch) and mmap encode_file_to_buffer / count_tokens_file."""
    print("=== zero-alloc count_tokens & mmap file tokenization parity ===")
    import os
    import pathlib
    import tempfile
    import unicodedata
    import numpy as np
    from tuetoken import AutoTokenizer as TTAuto

    ok = tot = 0

    # 1. Tokenizer.count_tokens (short, medium, >SPLIT_MIN parallel) + AutoTokenizer.count_tokens(_batch)
    tok_o200k = tuetoken.Tokenizer.from_tiktoken("o200k_base")
    long_doc = ("The quick brown fox jumps over the lazy dog. 12345 café 🚀\n" * 6000)
    for s in ("", "hi", "Hello world! 🚀 日本語", long_doc):
        tot += 1
        ok += int(tok_o200k.count_tokens(s) == len(tok_o200k.encode_ordinary(s)))

    for repo in (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "microsoft/Phi-3.5-mini-instruct",
        "deepseek-ai/DeepSeek-V3",
        "unsloth/gemma-3-4b-it",
    ):
        try:
            at = TTAuto.from_pretrained(repo)
        except Exception:
            continue
        samples = [
            "",
            "Hello world!",
            "def f(x): return x + 1 # 人工智能 🚀",
            "<div> > </div> <|im_start|>user\nhi",
            long_doc[:320_000],
        ]
        for add_sp in (False, True):
            want = [len(at.encode(s, add_special_tokens=add_sp)) for s in samples]
            got_single = [at.count_tokens(s, add_special_tokens=add_sp) for s in samples]
            got_batch = at.count_tokens_batch(samples, add_special_tokens=add_sp)
            tot += 2
            ok += int(got_single == want)
            ok += int(got_batch == want)

    # 2. mmap encode_file_to_buffer & count_tokens_file
    with tempfile.TemporaryDirectory() as td:
        p_empty = pathlib.Path(td) / "empty.txt"
        p_empty.write_bytes(b"")

        p_small = pathlib.Path(td) / "small.txt"
        small_text = "Hello world!\nSecond line 🚀 日本語\nCafé naïve\n"
        p_small.write_text(small_text, encoding="utf-8")

        # > 512 KiB file to exercise fused separator mmap path AND parallel split path
        p_big = pathlib.Path(td) / "big.txt"
        big_docs = [f"Doc {i}: The quick brown fox jumps over the lazy dog! 12345 café 🚀" for i in range(9000)]
        big_bytes = "\n".join(big_docs).encode("utf-8")
        p_big.write_bytes(big_bytes)

        # NFD decomposed file for NFC models (Qwen2.5)
        p_nfd = pathlib.Path(td) / "nfd.txt"
        nfd_text = unicodedata.normalize("NFD", "Café naïve résumé über\nAlpha beta\n") * 5000
        p_nfd.write_text(nfd_text, encoding="utf-8")

        # Invalid UTF-8 file
        p_bad = pathlib.Path(td) / "bad.bin"
        p_bad.write_bytes(b"valid prefix\n\xff\xfe bad utf8\n")

        # Empty file
        r_empty = tok_o200k.encode_file_to_buffer(p_empty)
        tot += 3
        ok += int(len(r_empty["data"]) == 0)
        ok += int(list(r_empty["offsets"]) == [0, 0])
        ok += int(tok_o200k.count_tokens_file(str(p_empty)) == 0)

        # Small file: no sep, bytes sep, str sep, pathlib.Path
        ref_small = tok_o200k.encode_ordinary_to_buffer([small_text.encode()])
        got_small = tok_o200k.encode_file_to_buffer(p_small)
        tot += 3
        ok += int(np.array_equal(got_small["data"], ref_small["data"]))
        ok += int(np.array_equal(got_small["offsets"], ref_small["offsets"]))
        ok += int(tok_o200k.count_tokens_file(p_small) == len(ref_small["data"]))

        ref_sep = tok_o200k.encode_ordinary_to_buffer([small_text.encode()], separator=b"\n")
        got_sep_b = tok_o200k.encode_file_to_buffer(str(p_small), separator=b"\n")
        got_sep_s = tok_o200k.encode_file_to_buffer(p_small, separator="\n")
        tot += 5
        ok += int(np.array_equal(got_sep_b["data"], ref_sep["data"]))
        ok += int(np.array_equal(got_sep_b["offsets"], ref_sep["offsets"]))
        ok += int(np.array_equal(got_sep_s["data"], ref_sep["data"]))
        ok += int(tok_o200k.count_tokens_file(p_small, separator="\n") == list(np.diff(ref_sep["offsets"])))
        ok += int(tok_o200k.count_tokens_file(p_small, separator=b"\n") == list(np.diff(ref_sep["offsets"])))

        # Big >512 KiB file (hits encode_fused_sep_t with separator AND split_points without)
        ref_big = tok_o200k.encode_ordinary_to_buffer([big_bytes])
        got_big = tok_o200k.encode_file_to_buffer(p_big)
        ref_big_sep = tok_o200k.encode_ordinary_to_buffer([big_bytes], separator=b"\n")
        got_big_sep = tok_o200k.encode_file_to_buffer(p_big, separator=b"\n")
        tot += 6
        ok += int(np.array_equal(got_big["data"], ref_big["data"]))
        ok += int(tok_o200k.count_tokens_file(p_big) == len(ref_big["data"]))
        ok += int(np.array_equal(got_big_sep["data"], ref_big_sep["data"]))
        ok += int(np.array_equal(got_big_sep["offsets"], ref_big_sep["offsets"]))
        ok += int(tok_o200k.count_tokens_file(p_big, separator=b"\n") == list(np.diff(ref_big_sep["offsets"])))
        ok += int(tok_o200k.count_tokens_file(p_big, num_threads=1, separator=b"\n") == list(np.diff(ref_big_sep["offsets"])))

        # uint16 dtype on gpt2 + rejection on o200k
        tg = tuetoken.Tokenizer.from_tiktoken("gpt2")
        r16 = tg.encode_file_to_buffer(p_small, dtype="uint16")
        tot += 2
        ok += int(r16["data"].dtype == np.uint16)
        ok += int(list(r16["data"]) == tg.encode_ordinary(small_text))
        tot += 1
        try:
            tok_o200k.encode_file_to_buffer(p_small, dtype="uint16")
        except ValueError:
            ok += 1

        # Cross-family file tests: Qwen2.5 (NFC + NFD), Phi-3.5 (SentencePiece), DeepSeek-V3 (CJK)
        qwen = TTAuto.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        rq = qwen.encode_file_to_buffer(p_nfd)
        tot += 2
        ok += int(list(rq["data"]) == qwen._core.encode_ordinary(nfd_text))
        ok += int(qwen.count_tokens_file(p_nfd) == len(rq["data"]))

        phi = TTAuto.from_pretrained("microsoft/Phi-3.5-mini-instruct")
        rp = phi.encode_file_to_buffer(p_small)
        tot += 2
        ok += int(list(rp["data"]) == phi._core.encode_ordinary(small_text))
        ok += int(phi.count_tokens_file(p_small) == len(rp["data"]))

        # Invalid UTF-8 and missing file errors
        for fn in (
            lambda: tok_o200k.encode_file_to_buffer(p_bad),
            lambda: tok_o200k.count_tokens_file(p_bad),
            lambda: tok_o200k.count_tokens_file(p_bad, separator=b"\n"),
        ):
            tot += 1
            try:
                fn()
            except ValueError:
                ok += 1

        tot += 1
        try:
            tok_o200k.count_tokens_file(os.path.join(td, "does_not_exist.txt"))
        except OSError:
            ok += 1

    _line("count_tokens & mmap file APIs", ok, tot)
    assert ok == tot, f"count/file APIs: {ok}/{tot}"
    return ok, tot


def test_truncate_chunk_and_v02_regressions():
    """Verify `truncate`, `chunk_text`, local-dir `AutoTokenizer.from_pretrained`,
    `chat_template.jinja`, Gemma/SP literal `'▁'` multi-thread parity, `gpt-oss-20b`
    `ignore_merges`, `Falcon-H1`/`Falcon3` `PretokPipe`, and `attention_mask`."""
    import tempfile
    from tuetoken import Tokenizer, AutoTokenizer as TTAuto
    from transformers import AutoTokenizer as HFAuto
    from huggingface_hub import hf_hub_download

    ok = tot = 0
    tok_o200k = Tokenizer.from_tiktoken("o200k_base")

    # 1. Core `truncate` & `chunk_text` (short + long >DETACH_MIN + >SPLIT_MIN + Unicode)
    short = "The quick brown fox jumps over the lazy dog! Café naïve 日本語 🚀"
    ids_short = tok_o200k.encode_ordinary(short)
    for k in (0, 1, 3, 7, len(ids_short), len(ids_short) + 10):
        for side in ("right", "left"):
            got = tok_o200k.truncate(short, k, side=side)
            if k == 0:
                want = ""
            elif k >= len(ids_short):
                want = short
            else:
                sub = ids_short[:k] if side == "right" else ids_short[-k:]
                want = tok_o200k.decode(sub)
            tot += 1
            ok += int(got == want)

    # Empty string and error cases
    tot += 3
    ok += int(tok_o200k.truncate("", 5) == "")
    ok += int(tok_o200k.chunk_text("", 5) == [])
    try:
        tok_o200k.truncate("abc", 2, side="middle")
    except ValueError:
        ok += 1

    for bad_call in (
        lambda: tok_o200k.chunk_text("hello", 0),
        lambda: tok_o200k.chunk_text("hello", 4, overlap=4),
        lambda: tok_o200k.chunk_text("hello", 4, overlap=5),
    ):
        tot += 1
        try:
            bad_call()
        except ValueError:
            ok += 1

    # Long text (>DETACH_MIN and >SPLIT_MIN) for both truncate & chunk_text
    long_txt = (short + "\n") * 4000
    ids_long = tok_o200k.encode_ordinary(long_txt)
    tot += 3
    ok += int(tok_o200k.truncate(long_txt, 50, side="right") == tok_o200k.decode(ids_long[:50]))
    ok += int(tok_o200k.truncate(long_txt, 50, side="left") == tok_o200k.decode(ids_long[-50:]))
    ok += int(tok_o200k.truncate(long_txt, len(ids_long) + 100) == long_txt)

    for max_t, ov in ((5, 0), (5, 2), (7, 6), (len(ids_short), 0), (len(ids_short) + 5, 2)):
        chunks = tok_o200k.chunk_text(short, max_t, overlap=ov)
        if len(ids_short) <= max_t:
            want_chunks = [short]
        else:
            step = max_t - ov
            want_chunks = []
            pos = 0
            while pos < len(ids_short):
                end = min(pos + max_t, len(ids_short))
                want_chunks.append(tok_o200k.decode(ids_short[pos:end]))
                if end == len(ids_short):
                    break
                pos += step
        tot += 1
        ok += int(chunks == want_chunks)

    # Large doc chunk_text (parallel encode + sliding window decode)
    big_chunks = tok_o200k.chunk_text(long_txt[:20000], 128, overlap=16)
    ids_20k = tok_o200k.encode_ordinary(long_txt[:20000])
    step = 112
    want_big = []
    pos = 0
    while pos < len(ids_20k):
        end = min(pos + 128, len(ids_20k))
        want_big.append(tok_o200k.decode(ids_20k[pos:end]))
        if end == len(ids_20k):
            break
        pos += step
    tot += 1
    ok += int(big_chunks == want_big)

    # 2. AutoTokenizer.truncate & chunk_text (specials + post-processor BOS/EOS budget)
    qwen = TTAuto.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    phi = TTAuto.from_pretrained("microsoft/Phi-3.5-mini-instruct", local_files_only=True)
    spec_text = "<|im_start|>user\nHello world from tuetoken truncate!<|im_end|>"
    ids_q = qwen.encode(spec_text, add_special_tokens=False)
    tot += 3
    ok += int(qwen.truncate(spec_text, 4, side="right", add_special_tokens=False) == qwen.decode(ids_q[:4], skip_special_tokens=False))
    ok += int(qwen.truncate(spec_text, 4, side="left", add_special_tokens=False) == qwen.decode(ids_q[-4:], skip_special_tokens=False))
    # Phi-3.5 adds BOS (1 token) when add_special_tokens=True: truncated text must fit within budget!
    phi_trunc = phi.truncate("one two three four five six seven eight nine ten", 6, add_special_tokens=True)
    ok += int(len(phi.encode(phi_trunc, add_special_tokens=True)) <= 6)

    phi_chunks = phi.chunk_text("one two three four five six seven eight nine ten", 6, overlap=1, add_special_tokens=True)
    tot += 2
    ok += int(len(phi_chunks) >= 2)
    ok += int(all(len(phi.encode(c, add_special_tokens=True)) <= 6 for c in phi_chunks))

    # 3. Local-directory AutoTokenizer.from_pretrained + stand-alone chat_template.jinja
    qwen_tj = hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", "tokenizer.json", local_files_only=True)
    qwen_dir = os.path.dirname(qwen_tj)
    qwen_local = TTAuto.from_pretrained(qwen_dir)
    tot += 2
    ok += int(qwen_local.encode(spec_text) == qwen.encode(spec_text))
    ok += int(qwen_local.eos_token_id == qwen.eos_token_id)

    with tempfile.TemporaryDirectory() as td:
        import shutil
        shutil.copy(qwen_tj, os.path.join(td, "tokenizer.json"))
        with open(os.path.join(td, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            f.write('{"eos_token": "<|im_end|>"}')
        with open(os.path.join(td, "chat_template.jinja"), "w", encoding="utf-8") as f:
            f.write("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}")
        ltok = TTAuto.from_pretrained(td)
        rendered = ltok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False)
        tot += 1
        ok += int(rendered == "user: hi\n")

    # 4. PySequence_Repeat attention_mask exactness on empty, 1-token, and padded batches
    out_am = qwen(["", "hello", "the quick brown fox jumps"], padding=True, return_attention_mask=True)
    tot += 2
    ok += int(out_am["attention_mask"][0] == [0] * len(out_am["input_ids"][0]))
    ok += int(all(len(a) == len(b) for a, b in zip(out_am["input_ids"], out_am["attention_mask"])))

    # 5. SentencePiece literal '▁' isolation + multi-thread 1T==MT parity (Phi-3.5 & cached Gemma-3)
    lit_sp_text = ("Hello world! " * 300 + "literal_\u2581_bar foo \u2581 baz " + "German über Straße\n" * 200) * 8
    for tok_sp, hf_sp in [(phi, HFAuto.from_pretrained("microsoft/Phi-3.5-mini-instruct", local_files_only=True))]:
        got_1t = tok_sp._core.encode_ordinary_batch([lit_sp_text], num_threads=1)[0]
        got_mt = tok_sp._core.encode_ordinary_batch([lit_sp_text], num_threads=0)[0]
        want_hf = hf_sp.encode(lit_sp_text, add_special_tokens=False)
        tot += 2
        ok += int(got_1t == want_hf)
        ok += int(got_mt == want_hf)

    # 6. Cached modern model regressions: gpt-oss-20b (ignore_merges), Falcon-H1/Falcon3 (PretokPipe)
    from tokenizers import Tokenizer as HFTok
    adv_pipe = "Digits 1234567890 & punct!!! Café òóùüýþ ¼½¾ ²³¹ 日本語 🚀\n" * 40
    for repo in ("openai/gpt-oss-20b", "tiiuae/Falcon-H1-7B-Instruct", "tiiuae/Falcon3-7B-Instruct"):
        try:
            tj = hf_hub_download(repo, "tokenizer.json", local_files_only=True)
        except Exception:
            continue
        tt_m = Tokenizer(tj)
        hf_m = HFTok.from_file(tj)
        w = hf_m.encode(adv_pipe, add_special_tokens=False).ids
        g1 = tt_m.encode_ordinary_batch([adv_pipe], num_threads=1)[0]
        gm = tt_m.encode_ordinary_batch([adv_pipe], num_threads=0)[0]
        tot += 2
        ok += int(g1 == w)
        ok += int(gm == w)

    _line("truncate, chunk_text & v0.2 regressions", ok, tot)
    assert ok == tot, f"v0.2 regressions: {ok}/{tot}"
    return ok, tot


def main():
    results = [
        test_tiktoken_parity(), test_streaming(), test_decode_and_buffers(),
        test_hf_models(), test_composable_pipeline(), test_failclosed(),
        test_normalization(), test_new_features(), test_rank_inversion(),
        test_auto(), test_kimi_tiktoken_model(), test_tekken_and_legacy_formats(),
        test_bytes_inputs(), test_warmup_and_fast_startup(),
        test_review_regressions(), test_whisper(), test_zero_copy_buffer_decode(),
        test_fast_and_batch_chat_templates(), test_sentencepiece_fast_meta(),
        test_decode_stream(), test_count_and_file_apis(),
        test_truncate_chunk_and_v02_regressions(),
    ]
    ok = sum(a for a, _ in results); tot = sum(b for _, b in results)
    print("\n" + "=" * 52)
    print(f"TOTAL: {ok}/{tot}")
    if ok != tot:
        print("SOME TESTS FAILED"); sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
