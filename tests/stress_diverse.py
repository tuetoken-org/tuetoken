#!/usr/bin/env python
"""Diverse-structure correctness audit of the `tuetoken` Rust package vs HF.

Goes wider than stress_rust_models.py on two axes the user asked about:

  * STRUCTURE — chat/instruct vs base vs code vs multilingual vs SentencePiece,
    different normalizers (NFC/NFKC/metaspace/none) and pre-tok grammars
    (Word/CamelCase/Generic/Split-fallback), PLUS real *agentic chat* transcripts
    rendered through each model's own chat template (system + multi-turn +
    tool-call JSON), which is the realest input an LLM serving stack tokenizes.

  * SIZE — empty, 1 char, ~10, ~100, ~1K, ~10K, ~100K, ~200K chars, plus the
    adversarial no-space long runs that stress the O(n) merger.

Ground truth = the HF fast tokenizer loaded from the SAME tokenizer.json. We
compare encode_ordinary, decode, round-trip, the numpy buffer paths, offsets,
batch helpers, AND scan-vs-stream merge parity. A mismatch counts as a real bug
("genuine") only if HF did NOT use a special/added-token id — encode_ordinary is
special-token-unaware by design, so those diffs are expected and labelled.
"""
import warnings, sys, os, json, random, unicodedata
warnings.simplefilter("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import numpy as np
import tuetoken
from tokenizers import Tokenizer as HFTokenizer
from huggingface_hub import hf_hub_download

# ------------------------------------------------------------------ model list
# Grouped by structure so a failure points at a family. Mostly ungated/cached.
MODELS = [
    # -- ByteLevel, plain Word grammar (gpt2 family), normalizer = none --
    "openai-community/gpt2", "distilgpt2", "EleutherAI/gpt-neo-125m",
    "bigscience/bloom-560m", "EleutherAI/gpt-j-6b",
    # -- ByteLevel CamelCase (o200k-style), NFC --
    "Qwen/Qwen2-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/QwQ-32B-Preview", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "MiniMaxAI/MiniMax-M2.7", "XiaomiMiMo/MiMo-V2.5-Pro",
    # -- ByteLevel via Split-regex fallback (DeepSeek empty-Seq norm, gpt-oss) --
    "openai/gpt-oss-20b", "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/deepseek-coder-1.3b-instruct", "zai-org/GLM-5.1",
    # -- code models (ByteLevel; some with content added-tokens = partial parity) --
    "bigcode/starcoder2-3b", "bigcode/santacoder", "Salesforce/codegen-350M-mono",
    "Qwen/Qwen3-Coder-Next",
    # -- Generic / cl100k-style, NFC / NFKC --
    "EleutherAI/pythia-410m", "EleutherAI/gpt-neox-20b",
    "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct", "stabilityai/stablelm-2-1_6b",
    # -- byte_fallback / SentencePiece metaspace (Replace-only & Prepend+Replace) --
    "mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-v0.1",
    "01-ai/Yi-1.5-6B-Chat", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-mini-instruct",
    "codellama/CodeLlama-7b-hf", "upstage/SOLAR-10.7B-Instruct-v1.0",
    "NousResearch/Hermes-2-Pro-Mistral-7B", "openbmb/BitCPM-CANN-8B",
    "openbmb/MiniCPM5-1B", "OBLITERATUS/gemma-4-E4B-it-OBLITERATED",
    # -- base (no chat template) to contrast with instruct --
    "microsoft/phi-2", "microsoft/phi-4", "allenai/OLMo-2-1124-7B-Instruct",
    "tiiuae/Falcon3-1B-Instruct", "ibm-granite/granite-3.1-8b-instruct",
    # -- must REJECT (unreproducible pre-tok) — fail-closed check --
    "tiiuae/falcon-7b", "dphn/Dolphin-X1-Trinity-Nano",
]

# ---------------------------------------------------------------- agentic chat
# A realistic multi-turn tool-using conversation. Rendered per-model through the
# model's OWN chat template (so the control tokens / role markers are whatever
# that model uses), then tokenized as ordinary text. This is the structure a
# real agent serving loop feeds the tokenizer.
AGENT_MESSAGES = [
    {"role": "system", "content":
        "You are an autonomous coding agent. You have tools: read_file(path), "
        "run(cmd), search(query). Think step by step. Always cite file:line."},
    {"role": "user", "content":
        "The build fails with `error[E0382]: borrow of moved value: tok`. "
        "Find the cause in src/lib.rs and propose a fix. Also: what's 2**10?"},
    {"role": "assistant", "content":
        "I'll inspect the file, then reason about the move.\n\n"
        '```json\n{"tool":"read_file","args":{"path":"src/lib.rs","range":[740,780]}}\n```'},
    {"role": "user", "content":
        "Tool output:\n```\n754  let toks = encode_one(text);\n"
        "755  emit(toks);            // moves toks\n756  log(toks.len());        // E0382 here\n```"},
    {"role": "assistant", "content":
        "Root cause: `emit(toks)` moves `toks` at src/lib.rs:755, then "
        "`toks.len()` at :756 borrows after move. Fix: pass `&toks` to `emit`, or "
        "compute `let n = toks.len();` before the move.\n\n"
        "And 2**10 = 1024. Patch:\n```rust\n-    emit(toks);\n+    let n = toks.len();\n"
        "+    emit(&toks);\n+    log(n);\n```\nShall I apply it?"},
    {"role": "user", "content": "yes, apply and run the tests. 你好, merci, спасибо!"},
]

def render_chat(tok):
    """Best-effort: render the agentic convo via the model's chat template.
    Returns the rendered string, or None if the model has no usable template."""
    try:
        if getattr(tok, "chat_template", None) is None:
            return None
        return tok.apply_chat_template(AGENT_MESSAGES, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        # some templates require tool schemas / reject roles — drop assistant
        # tool turns and retry with a plain system+user+assistant+user convo
        try:
            simple = [m for m in AGENT_MESSAGES if m["role"] in ("system", "user", "assistant")][:4]
            return tok.apply_chat_template(simple, tokenize=False, add_generation_prompt=True)
        except Exception:
            return None

# a hand-written agentic transcript WITHOUT special-token strings — pure text, so
# encode_ordinary MUST match HF exactly (genuine bug if not)
CHAT_PLAIN = (
    "System: You are a helpful coding agent.\n"
    "User: refactor get_user() in api/users.py:42 and add a test.\n"
    "Assistant: Plan:\n  1. extract validation into _validate(uid)\n"
    "  2. add tests/test_users.py::test_get_user_404\n"
    "I will call read_file then write_file.\n"
    'Action: {"tool": "read_file", "args": {"path": "api/users.py"}}\n'
    "Observation: def get_user(uid):\\n    return db.query(uid)  # no 404 handling\n"
    "Assistant: patching now — return 404 when db.query returns None. Tests green: 12 passed.\n"
)

# ---------------------------------------------------------------- size sweep
def size_sweep():
    r = random.Random(13)
    base = ("The agent reads file:line, runs cmd, and returns JSON. "
            "Café 你好 спасибо 1234567890 def f(x): return x*x; "
            "https://ex.com/p?q=1 — MixedCamelCASE_snake-kebab.\n")
    out = ["", "a", "你", " ", "\n"]
    for target in (10, 100, 1_000, 10_000, 100_000, 200_000):
        s = (base * (target // len(base) + 1))[:target]
        out.append(s)
    # adversarial: long runs that defeat naive O(n^2) merging
    out.append("".join(r.choice("abcdef0123") for _ in range(50_000)))
    out.append("x" * 100_000)
    out.append("1234567890" * 5_000)
    out.append(" " * 20_000)
    out.append(("世界" * 30_000))
    return out

# ---------------------------------------------------------------- base corpora
def realistic():
    return "\n".join([
        "The quick brown fox jumps over the lazy dog. 0123456789!",
        "In 2024, GPT-4 cost $0.03/1K tokens — a 10× drop from 2023.",
        "def fib(n):\n    if n < 2: return n\n    return fib(n-1)+fib(n-2)  # O(2^n)",
        "  leading and    multiple   internal   spaces\tand\ttabs  ",
        "Café naïve résumé — Zürich, Köln, São Paulo, façade, jalapeño.",
        "Россия, Москва. Ελληνικά. עברית מימין לשמאל. العربية كذلك.",
        "日本語のテキスト。中文文本。한국어 텍스트입니다。ไทยไม่มีช่องว่าง",
        "Emoji: 👨‍👩‍👧‍👦 🏳️‍🌈 🇺🇳 👋🏽 🤦‍♂️ 🧑🏿‍🚀 — ZWJ + skin tone + flags.",
        "Math: ∑∫∂∇ ℝ⊆ℂ, α+β=γ, ∀x∃y. ½ ¾ ⅞. x₁₂₃, x²³.",
        '{"key":"value","arr":[1,2.5e-3,true,null],"深":"嵌套"}',
        "MixedCASEwithCamelAndSNAKE_case_AND-kebab-case-IDENTIFIERS_v2",
    ])

def edges():
    e = ["", " ", "  ", "\t", "\n", "\r\n", "\n\n\n", " \t\n \r ",
         "a", "Z", "0", " a", "a ", ".", "...", "—", "…", "​", "﻿",
         "a\x00b", "\x00", "\x01\x02\x1f", "\x7f",
         "the", " the", "The", "THE", " 007 ", "0000",
         "é", "é", "👋", "👨‍👩‍👧‍👦", "🇺🇸", "中文", "한국어", "العربية",
         "<|endoftext|>", "<s>", "</s>", "[INST]", "<|im_start|>assistant"]
    e += [chr(c) for c in range(32, 127)]
    e += [" " + chr(c) for c in range(33, 127)]
    for cp in [0xA0, 0xFF, 0x100, 0x391, 0x4E00, 0x1F600, 0x1F1E6, 0x10437, 0x10FFFE]:
        try: e.append(chr(cp))
        except ValueError: pass
    return e

def fuzz(n, seed):
    r = random.Random(seed)
    pools = ["abcXYZ ", "0123 .,", "abc 123 \t\n", "中文日本語한국어ไทย ",
             "éàçñü ", "👋🏽👨‍👩‍👧‍👦🇺🇳 ", " ", "".join(chr(c) for c in range(0x20, 0x7f))]
    out = []
    for _ in range(n):
        pool = r.choice(pools)
        L = r.choice([0, 1, 2, 3, 5, 13, 50, 200, 1000])
        out.append("".join(r.choice(pool) for _ in range(L)))
    return out

REALISTIC = realistic()
EDGES = edges()
FUZZ = fuzz(300, 0)
SIZES = size_sweep()
GLOBAL = [REALISTIC, CHAT_PLAIN] + EDGES + FUZZ + SIZES

# ---------------------------------------------------------------- helpers
def short(s, n=70):
    r = repr(s)
    return r if len(r) <= n else r[:n] + "…"

def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y: return i
    return min(len(a), len(b))

def has_content_added_tokens(tj):
    d = json.load(open(tj))
    return any(not a.get("special", False) for a in d.get("added_tokens", []))

# ---------------------------------------------------------------- per-model audit
def audit(repo):
    rec = {"repo": repo, "status": "", "enc": (0, 0), "genuine": 0, "dec": (0, 0),
           "rt": (0, 0), "buf": (0, 0), "merge": (0, 0), "off": "?", "batch": "",
           "chat": "-", "samples": [], "note": ""}
    try:
        tj = hf_hub_download(repo, "tokenizer.json")
    except Exception as e:
        rec["status"] = "SKIP"; rec["note"] = type(e).__name__; return rec
    try:
        hf = HFTokenizer.from_file(tj); hf.no_truncation(); hf.no_padding()
    except Exception as e:
        rec["status"] = "SKIP"; rec["note"] = "HF load: " + str(e)[:40]; return rec
    added_ids = {a["id"] for a in json.load(open(tj)).get("added_tokens", [])}
    try:
        R = tuetoken.Tokenizer(tj)
    except Exception as e:
        rec["status"] = "REJECT"; rec["note"] = str(e).splitlines()[0][:70]; return rec
    rec["status"] = "OK"
    if has_content_added_tokens(tj):
        rec["note"] = "content-added-tokens (enc parity partial by design)"

    # per-model corpus = global + this model's own rendered agentic chat
    corpus = list(GLOBAL)
    try:
        from transformers import AutoTokenizer
        atok = AutoTokenizer.from_pretrained(repo)
        rendered = render_chat(atok)
        if rendered:
            corpus.append(rendered)
            rec["chat"] = f"render({len(rendered)}c)"
    except Exception as e:
        rec["chat"] = "no-tmpl"

    enc_ok = enc_tot = dec_ok = dec_tot = rt_ok = rt_tot = buf_ok = buf_tot = 0
    merge_ok = merge_tot = genuine = 0
    for t in corpus:
        try: hf_ids = hf.encode(t, add_special_tokens=False).ids
        except Exception: hf_ids = None
        r_ids = R.encode_ordinary(t)
        if hf_ids is not None:
            enc_tot += 1
            if r_ids == hf_ids:
                enc_ok += 1
            else:
                if not any(i in added_ids for i in hf_ids):
                    genuine += 1
                    if len(rec["samples"]) < 8:
                        i = first_diff(r_ids, hf_ids)
                        rec["samples"].append(
                            f"ENC {short(t)}\n      R ={r_ids[max(0,i-2):i+4]} (len {len(r_ids)})"
                            f"\n      HF={hf_ids[max(0,i-2):i+4]} (len {len(hf_ids)}) @tok{i}")
        # decode parity on the SAME ids
        ids = hf_ids if hf_ids is not None else r_ids
        try:
            dec_tot += 1
            if R.decode(ids) == hf.decode(ids, skip_special_tokens=False):
                dec_ok += 1
            elif len(rec["samples"]) < 10:
                rec["samples"].append(f"DEC {short(t)}\n      R ={short(R.decode(ids))}"
                                      f"\n      HF={short(hf.decode(ids, skip_special_tokens=False))}")
        except Exception: pass
        # our own round-trip stability
        rt_tot += 1
        once = R.encode_ordinary(R.decode(r_ids)); twice = R.encode_ordinary(R.decode(once))
        rt_ok += (once == twice)
        # numpy buffer paths == list paths
        buf_tot += 1
        ok_b = np.frombuffer(R.encode_to_bytes(t), np.uint32).tolist() == r_ids
        if r_ids:
            ok_b &= R.decode_array(np.array(r_ids, np.uint32)) == R.decode(r_ids)
        buf_ok += ok_b
        # scan-vs-stream merge engines must agree (long inputs auto-stream)
        merge_tot += 1
        merge_ok += (R._encode_ordinary_scan(t) == R._encode_ordinary_stream(t) == r_ids)

    rec.update(enc=(enc_ok, enc_tot), genuine=genuine, dec=(dec_ok, dec_tot),
               rt=(rt_ok, rt_tot), buf=(buf_ok, buf_tot), merge=(merge_ok, merge_tot))

    # offsets (ByteLevel only) on NFC text
    try:
        co = unicodedata.normalize("NFC", REALISTIC)
        ids, offs = R.encode_with_offsets(co); b = co.encode("utf-8")
        ok = (ids == R.encode_ordinary(co)) and all(
            b[s:e].decode("utf-8", "replace") == R.decode([i]) for i, (s, e) in zip(ids, offs))
        rec["off"] = "ok" if ok else "BAD"
    except ValueError:
        rec["off"] = "n/a(bf)"
    except Exception as e:
        rec["off"] = "ERR"

    # batch helpers
    try:
        texts = [REALISTIC] + EDGES[:20] + FUZZ[:20]
        seqs = [R.encode_ordinary(x) for x in texts]
        cb = R.count_tokens_batch(texts) == [len(s) for s in seqs]
        db = R.decode_batch(seqs) == [R.decode(s) for s in seqs]
        bb = R.encode_batch(texts, max_length=64, pad_id=0)["input_ids"]
        rb = all(bb[r].tolist()[:min(64, len(s))] == s[:64] for r, s in enumerate(seqs))
        rec["batch"] = "ok" if (cb and db and bb.shape == (len(texts), 64) and rb) else "FAIL"
    except Exception as e:
        rec["batch"] = "ERR"
    return rec

# ---------------------------------------------------------------- run + report
def pc(t):
    ok, tot = t
    return f"{ok}/{tot}" + ("" if ok == tot else " <<")

print(f"corpus/model: realistic + chat_plain + {len(EDGES)} edges + {len(FUZZ)} fuzz "
      f"+ {len(SIZES)} sizes (≤200K) + 1 rendered agentic chat = ~{len(GLOBAL)+1} inputs")
print("genuine = encode diffs NOT explained by special/added tokens (real bugs)\n")
hdr = (f"{'model':46} {'st':4} {'enc(vsHF)':11} {'gen':5} {'dec':9} {'rt':7} "
       f"{'buf':7} {'merge':9} {'off':8} {'batch':5} chat")
print(hdr); print("-" * len(hdr))
records = []
for repo in MODELS:
    r = audit(repo); records.append(r)
    if r["status"] in ("SKIP", "REJECT"):
        print(f"{repo:46} {r['status']:4} {r['note']}")
    else:
        g = r["genuine"]
        print(f"{repo:46} {r['status']:4} {pc(r['enc']):11} "
              f"{(str(g) if g==0 else str(g)+'!'):5} {pc(r['dec']):9} {pc(r['rt']):7} "
              f"{pc(r['buf']):7} {pc(r['merge']):9} {r['off']:8} {r['batch']:5} {r['chat']}")

print("\n" + "=" * 72 + "\nDIAGNOSIS (anything not by-design)\n" + "=" * 72)
clean = True
for r in records:
    if r["status"] != "OK": continue
    # SentencePiece decode strips a leading metaspace each pass, so encode->decode
    # is intentionally not idempotent (HF behaves identically) — don't flag rt on
    # byte_fallback models.
    rt_bad = r["rt"][0] != r["rt"][1] and r["off"] != "n/a(bf)"
    bad = (r["genuine"] or r["dec"][0] != r["dec"][1] or rt_bad
           or r["buf"][0] != r["buf"][1] or r["merge"][0] != r["merge"][1]
           or r["off"] in ("BAD", "ERR") or r["batch"] not in ("ok",))
    if bad:
        clean = False
        print(f"\n### {r['repo']}  gen={r['genuine']} dec={pc(r['dec'])} rt={pc(r['rt'])} "
              f"merge={pc(r['merge'])} off={r['off']} batch={r['batch']}")
        for s in r["samples"][:8]:
            print("  " + s.replace("\n", "\n  "))
if clean:
    print("\n  none — every loaded model is byte-exact vs HF on ordinary text across all"
          "\n  structures (chat/base/code/SP/multilingual) and sizes (≤200K), and matches"
          "\n  on decode, round-trip, buffers, scan==stream, offsets, batch."
          "\n  (special-token literals differ by design.)")

ok = [r for r in records if r["status"] == "OK"]
rej = [r for r in records if r["status"] == "REJECT"]
print(f"\nSUMMARY: {len(ok)} loaded · {sum(1 for r in records if r['status']=='SKIP')} skipped "
      f"· {len(rej)} rejected (fail-closed)")
if rej:
    for r in rej: print(f"  REJECT {r['repo']}: {r['note']}")
gen_total = sum(r["genuine"] for r in ok)
print(f"genuine encode bugs across all loaded models: {gen_total}")
sys.exit(1 if (gen_total or not clean) else 0)
