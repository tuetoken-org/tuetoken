"""Convert tiktoken encodings to tuetoken's tokenizer.json format.

Downloads raw .tiktoken BPE rank files directly from OpenAI's public CDN.
No tiktoken dependency required.

Supports all 6 OpenAI BPE encodings:
  gpt2, r50k_base, p50k_base, p50k_edit, cl100k_base, o200k_base

Also maps OpenAI model names to their encoding.
"""

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


def get_cached_tokenizer_path(encoding_name, cache_dir=None):
    """Get path to cached tokenizer.json, downloading and converting if needed.

    Downloads from OpenAI's public CDN on first use and caches the result.
    No tiktoken installation required.

    Returns the path to the tokenizer.json file.
    """
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
