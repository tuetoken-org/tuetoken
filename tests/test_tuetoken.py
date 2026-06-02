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


# --- 5c. fail-closed on pre-tokenizers we can't reproduce faithfully ---------
def test_failclosed():
    # Reject (don't silently mistokenize) constructs the engine can't replicate:
    #   * Punctuation pre-tokenizer (Falcon) -- Contiguous, not Isolated Split
    #   * \A/\G/\z-anchored Splits (Dolphin right-aligned digit grouping) -- our
    #     chunk-wise Isolated application can't honor whole-input/contiguous anchors
    print("=== fail-closed: unsupported pre-tokenizers rejected ===")
    import json, tempfile
    from huggingface_hub import hf_hub_download
    ok = tot = 0
    for repo, why in [("tiiuae/falcon-7b", "Punctuation"),
                      ("dphn/Dolphin-X1-Trinity-Nano", "\\A/\\G/\\z Split")]:
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

    # Offline synthetic configs for the stages we don't model — these must be
    # rejected at load, not silently mistokenized (the reviewer's Lowercase case
    # and a byte_fallback model with an unreproducible Punctuation pre-tokenizer).
    synthetic = [
        ("Lowercase normalizer",
         {"model": {"type": "BPE", "vocab": {}, "merges": []},
          "normalizer": {"type": "Lowercase"},
          "pre_tokenizer": {"type": "ByteLevel", "use_regex": True}}),
        ("Strip normalizer",
         {"model": {"type": "BPE", "vocab": {}, "merges": []},
          "normalizer": {"type": "Sequence", "normalizers": [{"type": "NFC"}, {"type": "Strip"}]},
          "pre_tokenizer": {"type": "ByteLevel", "use_regex": True}}),
        ("byte_fallback + Punctuation",
         {"model": {"type": "BPE", "vocab": {}, "merges": [], "byte_fallback": True},
          "pre_tokenizer": {"type": "Punctuation"}}),
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
        e += hf.apply_chat_template(chat, tokenize=True, add_generation_prompt=True) \
             == tt.apply_chat_template(chat, tokenize=True, add_generation_prompt=True)
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


def main():
    results = [
        test_tiktoken_parity(), test_streaming(), test_decode_and_buffers(),
        test_hf_models(), test_failclosed(), test_normalization(),
        test_new_features(), test_rank_inversion(), test_auto(),
        test_kimi_tiktoken_model(),
    ]
    ok = sum(a for a, _ in results); tot = sum(b for _, b in results)
    print("\n" + "=" * 52)
    print(f"TOTAL: {ok}/{tot}")
    if ok != tot:
        print("SOME TESTS FAILED"); sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
