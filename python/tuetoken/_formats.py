"""Converters for HuggingFace BPE tokenizer formats that do not ship `tokenizer.json`.

Currently supports:
  1. Mistral `tekken.json` (v3-v13+, used by Devstral-Small, Mistral-Small-3.2,
     Magistral-Small, Pixtral, Voxtral, etc.) — pure Python, zero `mistral_common`
     dependency; converts to a cached `tokenizer.json` that hits tuetoken's full
     SIMD CamelCase fast path.
  2. Legacy ByteLevel BPE (`vocab.json` + `merges.txt`, e.g. `facebook/opt-125m`
     and older GPT-2/RoBERTa/BART/CodeGen repos) — strictly fail-closed unless
     the repo metadata confirms a standard ByteLevel BPE configuration.
"""

import base64
import hashlib
import json
import os
from pathlib import Path

from ._tiktoken import _bytes_to_str

# Standard special tokens for Tekken <= v7 (when `special_tokens` is omitted
# from `tekken.json`). Mirrors `mistral_common.tokens.tokenizers.tekken`.
_TEKKEN_V7_SPECIALS = (
    (0, "<unk>"),
    (1, "<s>"),
    (2, "</s>"),
    (3, "[INST]"),
    (4, "[/INST]"),
    (5, "[AVAILABLE_TOOLS]"),
    (6, "[/AVAILABLE_TOOLS]"),
    (7, "[TOOL_RESULTS]"),
    (8, "[/TOOL_RESULTS]"),
    (9, "[TOOL_CALLS]"),
    (10, "[IMG]"),
    (11, "<pad>"),
    (12, "[IMG_BREAK]"),
    (13, "[IMG_END]"),
    (14, "[PREFIX]"),
    (15, "[MIDDLE]"),
    (16, "[SUFFIX]"),
    (17, "[SYSTEM_PROMPT]"),
    (18, "[/SYSTEM_PROMPT]"),
    (19, "[TOOL_CONTENT]"),
)


def _bpe_merge_fast(token_bytes, ranks):
    """Reconstruct the final BPE merge (left, right) for `token_bytes`."""
    if len(token_bytes) == 2:
        return (token_bytes[:1], token_bytes[1:])
    parts = [bytes([b]) for b in token_bytes]
    while len(parts) > 2:
        best_rank = None
        best_idx = -1
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_idx = i
        if best_idx < 0:
            break
        parts[best_idx] = parts[best_idx] + parts[best_idx + 1]
        parts.pop(best_idx + 1)
    if len(parts) == 2:
        return (parts[0], parts[1])
    return None


def _parse_tekken_specials(raw):
    """Return ordered list of `(rank, token_str)` of length `default_num_special_tokens`."""
    cfg = raw["config"]
    num_special = int(cfg["default_num_special_tokens"])
    spec_list = raw.get("special_tokens")
    if spec_list is None:
        specials = [(r, s) for r, s in _TEKKEN_V7_SPECIALS if r < num_special]
    else:
        specials = [(int(s["rank"]), s["token_str"]) for s in spec_list]
    seen = {r for r, _ in specials}
    for i in range(num_special):
        if i not in seen:
            specials.append((i, f"<SPECIAL_{i}>"))
    specials.sort(key=lambda x: x[0])
    return specials


def build_tekken_tokenizer_json(raw):
    """Convert a parsed `tekken.json` dict to `(tokenizer_json_dict, default_cfg)`."""
    cfg = raw["config"]
    pattern = cfg["pattern"]
    vocab_size = int(cfg["default_vocab_size"])
    num_special = int(cfg["default_num_special_tokens"])
    inner_vocab_size = vocab_size - num_special

    raw_vocab = raw["vocab"][:inner_vocab_size]
    ranks = {}
    ordered_bytes = []
    for entry in raw_vocab:
        tb = base64.b64decode(entry["token_bytes"])
        rk = int(entry["rank"])
        ranks[tb] = rk
        ordered_bytes.append((rk, tb))
    ordered_bytes.sort(key=lambda x: x[0])

    # Build ByteLevel vocab (BPE tokens live at id = rank + num_special)
    b2u_str = {}
    vocab = {}
    for rk, tb in ordered_bytes:
        s = _bytes_to_str(tb)
        b2u_str[tb] = s
        vocab[s] = rk + num_special

    # Reconstruct merges in ascending rank order
    merges = []
    for rk, tb in ordered_bytes:
        if len(tb) <= 1:
            continue
        pair = _bpe_merge_fast(tb, ranks)
        if pair is not None:
            merges.append(f"{b2u_str[pair[0]]} {b2u_str[pair[1]]}")

    specials = _parse_tekken_specials(raw)
    added_tokens = []
    spec_by_str = {}
    for sid, content in specials:
        spec_by_str[content] = sid
        if content not in vocab:
            vocab[content] = sid
        added_tokens.append({
            "id": sid,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        })

    # Post-processor matching transformers' MistralCommonTokenizer: <s> $A </s>
    bos_id = spec_by_str.get("<s>")
    eos_id = spec_by_str.get("</s>")
    post_processor = None
    if bos_id is not None and eos_id is not None:
        post_processor = {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
                {"SpecialToken": {"id": "<s>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "</s>", "type_id": 1}},
            ],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [bos_id], "tokens": ["<s>"]},
                "</s>": {"id": "</s>", "ids": [eos_id], "tokens": ["</s>"]},
            },
        }

    tj = {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": pattern},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": post_processor,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "added_tokens": added_tokens,
    }

    default_cfg = {}
    if "<s>" in spec_by_str:
        default_cfg["bos_token"] = "<s>"
    if "</s>" in spec_by_str:
        default_cfg["eos_token"] = "</s>"
    if "<pad>" in spec_by_str:
        default_cfg["pad_token"] = "<pad>"
    if "<unk>" in spec_by_str:
        default_cfg["unk_token"] = "<unk>"
    return tj, default_cfg


def get_tekken_tokenizer_path(tekken_path, cache_dir=None):
    """Convert a local `tekken.json` to a cached `tokenizer.json`.

    Returns `(cached_tokenizer_json_path, default_cfg_dict)`.
    """
    abs_path = os.path.abspath(tekken_path)
    st = os.stat(abs_path)
    key = hashlib.sha1(f"v1\x00{abs_path}\x00{st.st_size}\x00{st.st_mtime_ns}".encode()).hexdigest()[:16]
    cdir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "tuetoken")
    cdir.mkdir(parents=True, exist_ok=True)
    cached_tj = cdir / f"tekken_{key}.json"
    cached_cfg = cdir / f"tekken_{key}.cfg.json"
    if cached_tj.exists() and cached_cfg.exists():
        with open(cached_cfg, encoding="utf-8") as f:
            return str(cached_tj), json.load(f)

    with open(abs_path, encoding="utf-8") as f:
        raw = json.load(f)
    tj, default_cfg = build_tekken_tokenizer_json(raw)

    tmp_tj = str(cached_tj) + ".tmp"
    with open(tmp_tj, "w", encoding="utf-8") as f:
        json.dump(tj, f)
    os.replace(tmp_tj, str(cached_tj))

    tmp_cfg = str(cached_cfg) + ".tmp"
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        json.dump(default_cfg, f)
    os.replace(tmp_cfg, str(cached_cfg))
    return str(cached_tj), default_cfg


# --- Legacy vocab.json + merges.txt (OPT, GPT-2, RoBERTa, BART, CodeGen) ----

_LEGACY_BYTELEVEL_CLASSES = {
    "GPT2Tokenizer",
    "GPT2TokenizerFast",
    "RobertaTokenizer",
    "RobertaTokenizerFast",
    "BartTokenizer",
    "BartTokenizerFast",
    "CodeGenTokenizer",
    "CodeGenTokenizerFast",
    "OPTTokenizer",
}

_LEGACY_BYTELEVEL_MODEL_TYPES = {"gpt2", "roberta", "bart", "opt", "codegen"}


def _tok_content(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("content")
    return None


def try_load_legacy_bpe_repo(repo_id, revision=None, cache_dir=None):
    """Best-effort converter for HF repos shipping `vocab.json` + `merges.txt`
    without `tokenizer.json` (e.g. `facebook/opt-125m`).

    Strictly fail-closed: returns `None` unless the repo metadata unambiguously
    identifies a standard ByteLevel BPE model.
    """
    from huggingface_hub import hf_hub_download

    kw = {"revision": revision} if revision else {}
    try:
        vocab_path = hf_hub_download(repo_id, "vocab.json", **kw)
        merges_path = hf_hub_download(repo_id, "merges.txt", **kw)
    except Exception:
        return None

    cfg = {}
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        try:
            with open(hf_hub_download(repo_id, name, **kw), encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    model_cfg = {}
    try:
        with open(hf_hub_download(repo_id, "config.json", **kw), encoding="utf-8") as f:
            model_cfg = json.load(f)
    except Exception:
        pass

    tok_cls = cfg.get("tokenizer_class") or ""
    model_type = (model_cfg.get("model_type") or "").lower()
    if tok_cls not in _LEGACY_BYTELEVEL_CLASSES and model_type not in _LEGACY_BYTELEVEL_MODEL_TYPES:
        return None

    abs_v = os.path.abspath(vocab_path)
    abs_m = os.path.abspath(merges_path)
    key = hashlib.sha1(
        f"v1\x00{abs_v}\x00{abs_m}\x00{tok_cls}\x00{model_type}\x00{cfg.get('add_prefix_space', False)}".encode()
    ).hexdigest()[:16]
    cdir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "tuetoken")
    cdir.mkdir(parents=True, exist_ok=True)
    cached_tj = cdir / f"legacy_bpe_{key}.json"
    if cached_tj.exists():
        return str(cached_tj), cfg

    with open(abs_v, encoding="utf-8") as f:
        vocab = json.load(f)
    if not isinstance(vocab, dict) or "\u0120" not in vocab:
        return None
    merges = []
    with open(abs_m, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#version"):
                continue
            merges.append(line)

    # Collect added / special tokens from config + optional added_tokens.json
    added_map = {}
    try:
        at_path = hf_hub_download(repo_id, "added_tokens.json", **kw)
        with open(at_path, encoding="utf-8") as f:
            for content, tid in json.load(f).items():
                added_map[content] = (int(tid), False)
    except Exception:
        pass

    for field in ("bos_token", "eos_token", "unk_token", "pad_token", "cls_token", "sep_token", "mask_token"):
        c = _tok_content(cfg.get(field))
        if c and c in vocab:
            added_map[c] = (int(vocab[c]), True)
    for item in cfg.get("additional_special_tokens") or []:
        c = _tok_content(item)
        if c and c in vocab:
            added_map[c] = (int(vocab[c]), True)

    added_tokens = [
        {
            "id": tid,
            "content": content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": is_spec,
        }
        for content, (tid, is_spec) in sorted(added_map.items(), key=lambda x: x[1][0])
    ]

    add_prefix_space = bool(cfg.get("add_prefix_space", False))
    post_processor = None
    if model_type == "opt":
        bos = _tok_content(cfg.get("bos_token")) or "</s>"
        if cfg.get("add_bos_token", True) and bos in vocab:
            post_processor = {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": bos, "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                ],
                "pair": [
                    {"SpecialToken": {"id": bos, "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": bos, "type_id": 1}},
                    {"Sequence": {"id": "B", "type_id": 1}},
                ],
                "special_tokens": {
                    bos: {"id": bos, "ids": [int(vocab[bos])], "tokens": [bos]},
                },
            }
    elif model_type in ("roberta", "bart") or tok_cls.startswith(("Roberta", "Bart")):
        cls_t = _tok_content(cfg.get("cls_token")) or "<s>"
        sep_t = _tok_content(cfg.get("sep_token")) or "</s>"
        if cls_t in vocab and sep_t in vocab:
            post_processor = {
                "type": "RobertaProcessing",
                "cls": [cls_t, int(vocab[cls_t])],
                "sep": [sep_t, int(vocab[sep_t])],
                "trim_offsets": True,
                "add_prefix_space": add_prefix_space,
            }

    tj = {
        "version": "1.0",
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": add_prefix_space,
            "trim_offsets": True,
            "use_regex": True,
        },
        "post_processor": post_processor,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "added_tokens": added_tokens,
    }
    tmp = str(cached_tj) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tj, f)
    os.replace(tmp, str(cached_tj))
    return str(cached_tj), cfg


def try_load_tekken_repo(repo_id, revision=None, cache_dir=None):
    """Download `tekken.json` from an HF repo (or load a local path) and convert
    to a cached `tokenizer.json`. Returns `(tj_path, cfg_dict)` or `None`."""
    if os.path.exists(repo_id):
        tj_path, default_cfg = get_tekken_tokenizer_path(repo_id, cache_dir=cache_dir)
        return tj_path, default_cfg
    from huggingface_hub import hf_hub_download

    kw = {"revision": revision} if revision else {}
    try:
        tk_path = hf_hub_download(repo_id, "tekken.json", **kw)
    except Exception:
        return None
    tj_path, default_cfg = get_tekken_tokenizer_path(tk_path, cache_dir=cache_dir)
    cfg = dict(default_cfg)
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        try:
            with open(hf_hub_download(repo_id, name, **kw), encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return tj_path, cfg


def resolve_non_json_repo(repo_id, revision=None, cache_dir=None):
    """Resolve an HF repo without `tokenizer.json` to a cached `tokenizer.json`
    path for `Tokenizer.from_pretrained`: tries `tekken.json`, `tiktoken.model`,
    then legacy `vocab.json` + `merges.txt`. Returns `str` path or `None`."""
    res = try_load_tekken_repo(repo_id, revision=revision, cache_dir=cache_dir)
    if res is not None:
        return res[0]
    from . import _tiktoken as _tt

    pattern = _tt.tiktoken_pattern_from_repo(repo_id, revision)
    if pattern is not None:
        return _tt.get_tiktoken_model_path(repo_id, pattern=pattern, revision=revision, cache_dir=cache_dir)
    legacy = try_load_legacy_bpe_repo(repo_id, revision=revision, cache_dir=cache_dir)
    if legacy is not None:
        return legacy[0]
    return None


