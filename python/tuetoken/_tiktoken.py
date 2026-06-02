"""Convert tiktoken encodings to tuetoken's tokenizer.json format.

Downloads raw .tiktoken BPE rank files directly from OpenAI's public CDN.
No tiktoken dependency required.

Supports all 6 OpenAI BPE encodings:
  gpt2, r50k_base, p50k_base, p50k_edit, cl100k_base, o200k_base

Also maps OpenAI model names to their encoding.
"""

import ast
import base64
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

# Download timeout (seconds) for fetching .tiktoken rank files.
_DOWNLOAD_TIMEOUT = float(os.environ.get("TUETOKEN_DOWNLOAD_TIMEOUT", "30"))
_DOWNLOAD_RETRIES = 3

# OpenAI model name -> encoding name mapping
# Based on tiktoken.model.MODEL_TO_ENCODING
MODEL_TO_ENCODING = {
    # GPT-2
    "gpt2": "gpt2",
    # GPT-3
    "davinci": "r50k_base",
    "curie": "r50k_base",
    "babbage": "r50k_base",
    "ada": "r50k_base",
    "text-davinci-001": "r50k_base",
    "text-curie-001": "r50k_base",
    "text-babbage-001": "r50k_base",
    "text-ada-001": "r50k_base",
    "code-davinci-002": "p50k_base",
    "code-cushman-001": "p50k_base",
    "text-davinci-002": "p50k_base",
    "text-davinci-003": "p50k_base",
    "text-davinci-edit-001": "p50k_edit",
    "code-davinci-edit-001": "p50k_edit",
    # GPT-3.5 / GPT-4
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # GPT-4o / o-series
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
}

# All valid encoding names
ENCODING_NAMES = {"gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base"}

# --- Encoding definitions (URLs, patterns, special tokens) ---
# Mirrors tiktoken_ext/openai_public.py without importing it.

_CDN = "https://openaipublic.blob.core.windows.net"

# Expected SHA-256 of each .tiktoken blob, keyed by URL. Mirrors the hashes
# pinned in tiktoken's tiktoken_ext/openai_public.py so a truncated or
# tampered download is rejected rather than silently cached.
_EXPECTED_SHA256 = {
    f"{_CDN}/encodings/r50k_base.tiktoken":
        "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    f"{_CDN}/encodings/p50k_base.tiktoken":
        "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
    f"{_CDN}/encodings/cl100k_base.tiktoken":
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    f"{_CDN}/encodings/o200k_base.tiktoken":
        "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
}

_GPT2_PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_GPT2_PAT_POSSESSIVE = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s+"

_CL100K_PAT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)

_O200K_PAT = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

_ENCODING_DEFS = {
    "gpt2": {
        "url": f"{_CDN}/encodings/r50k_base.tiktoken",
        "pat_str": _GPT2_PAT,
        "special_tokens": {"<|endoftext|>": 50256},
        "vocab_size": 50257,
    },
    "r50k_base": {
        "url": f"{_CDN}/encodings/r50k_base.tiktoken",
        "pat_str": _GPT2_PAT_POSSESSIVE,
        "special_tokens": {"<|endoftext|>": 50256},
        "vocab_size": 50257,
    },
    "p50k_base": {
        "url": f"{_CDN}/encodings/p50k_base.tiktoken",
        "pat_str": _GPT2_PAT_POSSESSIVE,
        "special_tokens": {"<|endoftext|>": 50256},
        "vocab_size": 50281,
    },
    "p50k_edit": {
        "url": f"{_CDN}/encodings/p50k_base.tiktoken",
        "pat_str": _GPT2_PAT_POSSESSIVE,
        "special_tokens": {
            "<|endoftext|>": 50256,
            "<|fim_prefix|>": 50281,
            "<|fim_middle|>": 50282,
            "<|fim_suffix|>": 50283,
        },
        "vocab_size": 50284,
    },
    "cl100k_base": {
        "url": f"{_CDN}/encodings/cl100k_base.tiktoken",
        "pat_str": _CL100K_PAT,
        "special_tokens": {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
        "vocab_size": 100277,
    },
    "o200k_base": {
        "url": f"{_CDN}/encodings/o200k_base.tiktoken",
        "pat_str": _O200K_PAT,
        "special_tokens": {
            "<|endoftext|>": 199999,
            "<|endofprompt|>": 200018,
        },
        "vocab_size": 200019,
    },
}

# --- Byte-level BPE helpers ---

def _byte_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_B2U = _byte_to_unicode()


def _bytes_to_str(token_bytes):
    return "".join(_B2U[b] for b in token_bytes)


def _bpe_merge_for_token(token_bytes, ranks):
    """Reconstruct which merge produced a token by simulating BPE."""
    parts = [bytes([b]) for b in token_bytes]
    while len(parts) > 2:
        best_rank = float('inf')
        best_idx = -1
        for i in range(len(parts) - 1):
            pair = parts[i] + parts[i + 1]
            if pair in ranks and ranks[pair] < best_rank:
                best_rank = ranks[pair]
                best_idx = i
        if best_idx == -1:
            break
        parts[best_idx] = parts[best_idx] + parts[best_idx + 1]
        parts.pop(best_idx + 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    return None


# --- Download and parse .tiktoken files ---

def _download(url):
    """Download ``url`` with a timeout, retries, and SHA-256 verification.

    Returns the raw bytes. Raises RuntimeError on repeated failure or if the
    content hash does not match the pinned expected value.
    """
    expected = _EXPECTED_SHA256.get(url)
    last_err = None
    for attempt in range(_DOWNLOAD_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp:
                data = resp.read()
        except (urllib.error.URLError, OSError) as e:
            last_err = e
            continue
        if expected is not None:
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected:
                last_err = RuntimeError(
                    f"SHA-256 mismatch for {url}: expected {expected}, got "
                    f"{actual} (truncated or tampered download)."
                )
                continue  # retry — likely a truncated body
        return data
    raise RuntimeError(
        f"Failed to download {url} after {_DOWNLOAD_RETRIES} attempts: {last_err}"
    )


def _load_tiktoken_bpe(url):
    """Download and parse a .tiktoken file. Returns {token_bytes: rank}."""
    data = _download(url)
    ranks = {}
    for line in data.splitlines():
        if not line:
            continue
        token_b64, rank_str = line.split()
        ranks[base64.b64decode(token_b64)] = int(rank_str)
    return ranks


def _convert_ranks_to_tokenizer_json(encoding_name, ranks):
    """Convert a {token_bytes: rank} dict to HuggingFace tokenizer.json dict."""
    defn = _ENCODING_DEFS[encoding_name]
    pat_str = defn["pat_str"]
    special_tokens = defn["special_tokens"]

    # Build vocab: byte-level unicode string -> rank
    vocab = {}
    for token_bytes, rank in ranks.items():
        vocab[_bytes_to_str(token_bytes)] = rank

    # Reconstruct merges via BPE simulation
    merges = []
    for token_bytes, rank in sorted(ranks.items(), key=lambda x: x[1]):
        if len(token_bytes) <= 1:
            continue
        result = _bpe_merge_for_token(token_bytes, ranks)
        if result:
            merges.append(f"{_bytes_to_str(result[0])} {_bytes_to_str(result[1])}")

    # Build pre_tokenizer config that our C detector recognizes
    # GPT-2 family: plain ByteLevel with use_regex=True
    # Others: Split(regex) + ByteLevel(use_regex=False)
    is_gpt2 = encoding_name in ("gpt2", "r50k_base", "p50k_base", "p50k_edit")
    if is_gpt2:
        pre_tokenizer = {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        }
    else:
        pre_tokenizer = {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": pat_str},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False,
                 "trim_offsets": True, "use_regex": False},
            ],
        }

    added_tokens = []
    for content, tid in special_tokens.items():
        added_tokens.append({
            "id": tid, "content": content,
            "single_word": False, "lstrip": False, "rstrip": False,
            "normalized": False, "special": True,
        })

    return {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "pre_tokenizer": pre_tokenizer,
        "decoder": {"type": "ByteLevel"},
        "added_tokens": added_tokens,
    }


def convert_tiktoken_encoding(encoding_name):
    """Convert a tiktoken encoding to HuggingFace tokenizer.json dict.

    Downloads the raw .tiktoken file from OpenAI's public CDN.
    No tiktoken installation required.
    """
    if encoding_name not in _ENCODING_DEFS:
        raise ValueError(f"Unknown encoding: {encoding_name!r}")
    url = _ENCODING_DEFS[encoding_name]["url"]
    ranks = _load_tiktoken_bpe(url)
    return _convert_ranks_to_tokenizer_json(encoding_name, ranks)


def resolve_encoding_name(name):
    """Resolve a model name or encoding name to an encoding name.

    Returns the encoding name, or None if not recognized.
    """
    if name in ENCODING_NAMES:
        return name
    return MODEL_TO_ENCODING.get(name)


def get_cached_tokenizer_path(name, cache_dir=None):
    """Get path to cached tokenizer.json, downloading and converting if needed.

    ``name`` may be an encoding name (e.g. "cl100k_base") OR an OpenAI model name
    (e.g. "gpt-4o", "gpt-3.5-turbo"), which is resolved to its encoding via
    ``MODEL_TO_ENCODING`` — otherwise ``Tokenizer.from_tiktoken("gpt-4o")`` would
    fail even though the mapping exists.

    Downloads from OpenAI's public CDN on first use and caches the result.
    No tiktoken installation required.

    Returns the path to the tokenizer.json file.
    """
    encoding_name = resolve_encoding_name(name)
    if encoding_name is None:
        raise ValueError(
            f"Unknown tiktoken encoding or model name: {name!r} "
            f"(valid encodings: {', '.join(sorted(ENCODING_NAMES))})"
        )
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "tuetoken"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / f"{encoding_name}.json"

    if cached_path.exists():
        return str(cached_path)

    # Download, convert, and cache
    tokenizer_json = convert_tiktoken_encoding(encoding_name)

    # Write atomically
    tmp_path = str(cached_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f)
    os.replace(tmp_path, str(cached_path))

    return str(cached_path)


# --- tiktoken.model models (Kimi etc.): ranks file + custom pattern, no JSON ---
# Some models (e.g. moonshotai/Kimi-K2) use the tiktoken ALGORITHM but ship only a
# `tiktoken.model` rank file plus a custom split pattern in their tokenization_*.py
# — no tokenizer.json. They load the same way OpenAI encodings do: parse the ranks,
# reconstruct merges, and build a tokenizer.json with the model's own split pattern.

def load_tiktoken_model_file(path):
    """Parse a tiktoken `.model` file (`base64(token_bytes) rank` per line)."""
    ranks = {}
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            b64, rk = line.split()
            ranks[base64.b64decode(b64)] = int(rk)
    return ranks


def build_tiktoken_model_json(ranks, pattern, special_tokens=None):
    """Build a HuggingFace tokenizer.json dict from tiktoken `ranks` + a split
    `pattern`. `special_tokens` is an optional {content: id} added as special
    tokens. Like `_convert_ranks_to_tokenizer_json` but for an arbitrary pattern
    and caller-supplied specials (tiktoken-format models ship neither)."""
    vocab = {_bytes_to_str(tb): rk for tb, rk in ranks.items()}
    merges = []
    for tb, rk in sorted(ranks.items(), key=lambda x: x[1]):
        if len(tb) <= 1:
            continue
        r = _bpe_merge_for_token(tb, ranks)
        if r:
            merges.append(f"{_bytes_to_str(r[0])} {_bytes_to_str(r[1])}")
    added = []
    for content, tid in (special_tokens or {}).items():
        added.append({"id": int(tid), "content": content, "single_word": False,
                      "lstrip": False, "rstrip": False, "normalized": False,
                      "special": True})
    return {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern},
             "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False,
             "trim_offsets": True, "use_regex": False}]},
        "decoder": {"type": "ByteLevel"},
        "added_tokens": added,
    }


def _read_special_tokens_from_config(repo, revision=None):
    """Read {content: id} special tokens from a repo's tokenizer_config.json
    (`added_tokens_decoder`). Returns {} if unavailable."""
    from huggingface_hub import hf_hub_download
    kw = {"revision": revision} if revision else {}
    try:
        with open(hf_hub_download(repo, "tokenizer_config.json", **kw),
                  encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {}
    out = {}
    for sid, info in (cfg.get("added_tokens_decoder") or {}).items():
        content = info.get("content") if isinstance(info, dict) else None
        if content:
            out[content] = int(sid)
    return out


def get_tiktoken_model_path(source, pattern, special_tokens=None, revision=None,
                            model_file="tiktoken.model", cache_dir=None):
    """Path to a cached tokenizer.json built from a tiktoken-format model's ranks +
    `pattern`. `source` is a local `tiktoken.model` path OR an HF repo id (downloads
    the rank file). Specials default to the repo's tokenizer_config.json. Cached by
    (rank-file, pattern), so the slow merge reconstruction runs once."""
    if os.path.exists(source):
        model_path, repo = source, None
    else:
        from huggingface_hub import hf_hub_download
        kw = {"revision": revision} if revision else {}
        model_path = hf_hub_download(source, model_file, **kw)
        repo = source
    if special_tokens is None and repo is not None:
        special_tokens = _read_special_tokens_from_config(repo, revision)

    key = hashlib.sha1(
        (os.path.abspath(model_path) + "\x00" + pattern).encode("utf-8")
    ).hexdigest()[:16]
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "tuetoken"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"tiktoken_model_{key}.json"
    if not cached.exists():
        ranks = load_tiktoken_model_file(model_path)
        tj = build_tiktoken_model_json(ranks, pattern, special_tokens)
        tmp = str(cached) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(tj, f)
        os.replace(tmp, str(cached))
    return str(cached)


# --- auto pattern discovery (config-driven, no model-name hardcode, no exec) ---
# tiktoken-format HF tokenizers keep their split pattern in a `tokenization_*.py`
# (referenced by tokenizer_config.json's `auto_map`), as `pat_str = "|".join([...])`
# or a literal. We recover it by STATIC AST evaluation — never importing/executing
# the module — so `from_pretrained` can load these models with no explicit pattern.

def _eval_static_str(node):
    """Evaluate a static string expression (literals, `+`/implicit concat, and
    `sep.join([literals])`). Returns a str, a list[str] (for the join arg), or None
    if the expression isn't a pure constant. No code execution."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _eval_static_str(node.left), _eval_static_str(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        return None
    if isinstance(node, (ast.List, ast.Tuple)):
        parts = [_eval_static_str(e) for e in node.elts]
        return parts if all(isinstance(p, str) for p in parts) else None
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "join" and len(node.args) == 1 and not node.keywords):
        sep = _eval_static_str(node.func.value)
        items = _eval_static_str(node.args[0])
        if isinstance(sep, str) and isinstance(items, list):
            return sep.join(items)
    return None


def _extract_pat_str_from_source(src, var="pat_str"):
    """Static-extract the `pat_str` split pattern from a tokenizer module's source."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id == var:
                v = _eval_static_str(node.value)
                if isinstance(v, str):
                    return v
    return None


def tiktoken_pattern_from_repo(repo, revision=None):
    """Best-effort split pattern for a tiktoken-format model: read the
    `tokenization_*.py` named by tokenizer_config.json's `auto_map` (config-driven,
    not by model name) and static-extract its `pat_str`. None if not applicable."""
    from huggingface_hub import hf_hub_download
    kw = {"revision": revision} if revision else {}
    try:
        with open(hf_hub_download(repo, "tokenizer_config.json", **kw),
                  encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None
    ref = (cfg.get("auto_map") or {}).get("AutoTokenizer")
    if isinstance(ref, (list, tuple)):
        ref = ref[0] if ref else None
    if not isinstance(ref, str) or "." not in ref:
        return None
    module = ref.rsplit(".", 1)[0]  # "tokenization_kimi.TikTokenTokenizer" -> module
    try:
        with open(hf_hub_download(repo, module + ".py", **kw), encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return None
    return _extract_pat_str_from_source(src)
