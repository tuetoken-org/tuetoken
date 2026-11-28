"""HF-compatible tokenizer API on top of tuetoken's fast Rust core.

`AutoTokenizer` mirrors enough of `transformers.PreTrainedTokenizerFast` to be a
drop-in for LLM use: special-token-aware `encode`, `__call__` with
padding/truncation/`return_tensors`, `decode(skip_special_tokens=...)`,
`apply_chat_template` (Jinja2), and the usual conversion helpers/properties.

The heavy BPE stays in Rust (`tuetoken._core.Tokenizer.encode_ordinary`); only the
special-token splitting, post-processor (BOS/EOS/CLS/SEP), padding/truncation, and
chat-template rendering live here. Parsed from the model's own `tokenizer.json` +
`tokenizer_config.json`, never the model name.
"""
import json
import os
from ._core import Tokenizer as _Core


def _tok_str(v):
    """A special token in config may be a plain string or an AddedToken dict."""
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return v.get("content")
    return None


class _ChatDict(dict):
    """Tokenized chat output ({input_ids, attention_mask}) that also supports
    `.to(device)` for the common `model.generate(**tok.apply_chat_template(...))`
    / `.to(model.device)` pattern. No torch import unless tensors are present."""
    def to(self, device):
        return _ChatDict((k, v.to(device) if hasattr(v, "to") else v)
                         for k, v in self.items())


class AutoTokenizer:
    is_fast = True

    # ---- construction -------------------------------------------------------
    def __init__(self, tokenizer_json, config=None, warmup=None,
                 language=None, task=None, predict_timestamps=None,
                 normalizer_json=None):
        self._tokenizer_json = tokenizer_json
        self._core = _Core(tokenizer_json)
        self._cfg = config or {}
        self._normalizer_json = normalizer_json

        # Fast metadata extracted during the single Rust parse (avoids re-parsing
        # the full 12+ MB tokenizer.json in Python; full id<->token dicts are
        # built lazily only if get_vocab/convert_*_to_* are called).
        added_rows, post_json, prepend_scheme, known_ids = self._core._auto_meta()
        self._id_to_token_cache = None
        self._token_to_id_cache = None
        self._added = [
            {"content": c, "id": int(i), "special": sp, "lstrip": ls, "rstrip": rs}
            for (c, i, sp, ls, rs) in added_rows
        ]
        self._added_content_to_id = {}
        self._special_ids = set()
        self._fast_id_to_token = {}
        self._fast_token_to_id = {t: int(i) for (t, i) in known_ids}
        for c, i, sp, _, _ in added_rows:
            ii = int(i)
            self._fast_id_to_token[ii] = c
            self._fast_token_to_id[c] = ii
            self._added_content_to_id[c] = ii
            if sp:
                self._special_ids.add(ii)

        self._post = json.loads(post_json) if post_json else None
        self._prepend_scheme = prepend_scheme

        # special tokens / chat template / limits from tokenizer_config.json
        c = self._cfg
        self.bos_token = _tok_str(c.get("bos_token"))
        self.eos_token = _tok_str(c.get("eos_token"))
        self.unk_token = _tok_str(c.get("unk_token"))
        self.pad_token = _tok_str(c.get("pad_token")) or self.eos_token
        self.sep_token = _tok_str(c.get("sep_token"))
        self.cls_token = _tok_str(c.get("cls_token"))
        self.mask_token = _tok_str(c.get("mask_token"))
        self.chat_template = c.get("chat_template")
        self.add_bos_token = c.get("add_bos_token")
        self.add_eos_token = c.get("add_eos_token")
        mml = c.get("model_max_length")
        self.model_max_length = mml if isinstance(mml, int) and mml < int(1e15) else None
        self.padding_side = c.get("padding_side", "right")
        self.truncation_side = c.get("truncation_side", "right")
        # Some tokenizers (RoBERTa/BERT family) keep their pad token only in the
        # transformers class default, not any config file. Fall back to the common
        # pad-token surfaces by content (convention, not model name).
        if self.pad_token is None:
            for cand in ("<pad>", "[PAD]", "<|pad|>", "<|endoftext|>"):
                if cand in self._fast_token_to_id:
                    self.pad_token = cand
                    break

        # Hand the added tokens to the Rust core so `encode_special` splits + BPEs a
        # whole chat string in one pass (no Python regex / per-fragment crossing).
        _scheme_map = {"none": 0, "always": 1, "first": 2, "never": 3}
        if self._prepend_scheme not in _scheme_map:
            raise ValueError(
                f"unsupported SentencePiece prepend_scheme {self._prepend_scheme!r}: "
                "tuetoken cannot reproduce this tokenizer"
            )
        scheme = _scheme_map[self._prepend_scheme]
        self._core.set_special_tokens(
            [a["content"] for a in self._added],
            [int(a["id"]) for a in self._added],
            [bool(a.get("lstrip", False)) for a in self._added],
            [bool(a.get("rstrip", False)) for a in self._added],
            scheme,
        )

        # Config-driven Whisper support (timestamps, prefix tokens, prompt strip).
        from . import _whisper as _wh
        self._is_whisper = _wh.is_whisper_tokenizer(self._cfg, self._added_content_to_id)
        self.language = language
        self.task = task
        self.predict_timestamps = bool(predict_timestamps) if predict_timestamps is not None else False
        self._whisper_ts_begin = None
        if self._is_whisper:
            ts0 = self._added_content_to_id.get("<|0.00|>")
            if ts0 is None and self._special_ids:
                ts0 = max(self._special_ids) + 1
            if ts0 is not None:
                self._whisper_ts_begin = int(ts0)
                sop = int(self._added_content_to_id.get("<|startofprev|>", 0xFFFF_FFFF))
                sot = int(self._added_content_to_id.get("<|startoftranscript|>", 0xFFFF_FFFF))
                self._core.set_whisper_decode(self._whisper_ts_begin, sop, sot)
            if language is not None or task is not None or predict_timestamps is not None:
                self.set_prefix_tokens(language=language, task=task, predict_timestamps=predict_timestamps)

        if warmup is True:
            self.warmup()
        elif warmup:
            self.warmup(warmup)

    def _ensure_vocab_maps(self):
        if self._id_to_token_cache is not None:
            return
        with open(self._tokenizer_json, encoding="utf-8") as f:
            tj = json.load(f)
        id2t = {}
        vocab = tj.get("model", {}).get("vocab", {})
        if isinstance(vocab, dict):
            for tok, i in vocab.items():
                id2t[int(i)] = tok
        for a in self._added:
            id2t[int(a["id"])] = a["content"]
        self._id_to_token_cache = id2t
        self._token_to_id_cache = {t: i for i, t in id2t.items()}

    @property
    def _id_to_token(self):
        self._ensure_vocab_maps()
        return self._id_to_token_cache

    @property
    def _token_to_id(self):
        self._ensure_vocab_maps()
        return self._token_to_id_cache

    def warmup(self, texts=None, num_threads=0):
        """Pre-warm thread pools, Python int table, and (when `texts` is given)
        populate the shared chunk cache for representative domain text(s).

        Passing a small sample (~0.5-5 MB) of your target corpus warms the
        cache in a few milliseconds so production requests start at full warm
        throughput (3-5x faster than cold BPE). Returns cached entry count.
        """
        return self._core.warmup(texts, num_threads)

    @classmethod
    def from_file(cls, tokenizer_json, config=None, warmup=None,
                  language=None, task=None, predict_timestamps=None,
                  normalizer_json=None):
        cfg = None
        if isinstance(config, str):
            with open(config, encoding="utf-8") as f:
                cfg = json.load(f)
        elif isinstance(config, dict):
            cfg = config
        return cls(tokenizer_json, cfg, warmup=warmup, language=language,
                   task=task, predict_timestamps=predict_timestamps,
                   normalizer_json=normalizer_json)

    @classmethod
    def from_tiktoken_model(cls, source, pattern, special_tokens=None, revision=None, warmup=None):
        """Load a tiktoken-format model (ships a `tiktoken.model` rank file + a
        custom split regex, e.g. Kimi-K2) that has no `tokenizer.json`.

        `source` is a local `tiktoken.model` path or an HF repo id. `pattern` is the
        model's tiktoken split regex — its `pat_str`, found in the model's
        tokenization_*.py (it lives in no config file, so it must be supplied).
        Special tokens default to the repo's tokenizer_config.json. Returns a full
        AutoTokenizer with specials + chat template wired, like `from_pretrained`.
        """
        if not pattern:
            raise ValueError(
                "from_tiktoken_model requires the model's tiktoken split `pattern` "
                "(its pat_str, from the model's tokenization_*.py); it is not stored "
                "in any config file"
            )
        from . import _tiktoken as _tt
        tj_path = _tt.get_tiktoken_model_path(source, pattern, special_tokens, revision)
        cfg = {}
        if not os.path.exists(source):
            from huggingface_hub import hf_hub_download
            kw = {"revision": revision} if revision else {}
            for name in ("special_tokens_map.json", "tokenizer_config.json"):
                try:
                    with open(hf_hub_download(source, name, **kw), encoding="utf-8") as f:
                        cfg.update(json.load(f))
                except Exception:
                    pass
        return cls(tj_path, cfg, warmup=warmup)

    @classmethod
    def from_tekken(cls, source, revision=None, warmup=None):
        """Load a Mistral `tekken.json` model (e.g. Devstral-Small, Mistral-Small-3.2,
        Magistral-Small) from a local `tekken.json` path or HF repo id."""
        from . import _formats as _fmt
        res = _fmt.try_load_tekken_repo(source, revision=revision)
        if res is None:
            raise FileNotFoundError(f"no tekken.json found for {source!r}")
        tj_path, cfg = res
        return cls(tj_path, cfg, warmup=warmup)

    @classmethod
    def from_pretrained(cls, repo_id, revision=None, warmup=None,
                        language=None, task=None, predict_timestamps=None,
                        local_files_only=False, cache_dir=None, token=None, **_):
        extra = dict(language=language, task=task, predict_timestamps=predict_timestamps)

        # Fast local-directory loading (zero huggingface_hub / network calls).
        if isinstance(repo_id, (str, os.PathLike)) and os.path.isdir(repo_id):
            d = os.fspath(repo_id)
            tj = os.path.join(d, "tokenizer.json")
            if os.path.isfile(tj):
                cfg = {}
                for name in ("special_tokens_map.json", "tokenizer_config.json"):
                    p = os.path.join(d, name)
                    if os.path.isfile(p):
                        with open(p, encoding="utf-8") as f:
                            cfg.update(json.load(f))
                if not cfg.get("chat_template"):
                    ct = os.path.join(d, "chat_template.jinja")
                    if os.path.isfile(ct):
                        with open(ct, encoding="utf-8") as f:
                            cfg["chat_template"] = f.read()
                norm_json = None
                if (cfg.get("tokenizer_class") or "").startswith("Whisper"):
                    cand = os.path.join(d, "normalizer.json")
                    if os.path.isfile(cand):
                        norm_json = cand
                return cls(tj, cfg, warmup=warmup, normalizer_json=norm_json, **extra)
            tek = os.path.join(d, "tekken.json")
            if os.path.isfile(tek):
                return cls.from_tekken(tek, warmup=warmup)
            from . import _formats as _fmt
            legacy = _fmt.try_load_legacy_bpe_repo(d, revision=revision)
            if legacy is not None:
                tj_path, cfg = legacy
                norm_json = None
                if (cfg.get("tokenizer_class") or "").startswith("Whisper"):
                    cand = os.path.join(d, "normalizer.json")
                    if os.path.isfile(cand):
                        norm_json = cand
                return cls(tj_path, cfg, warmup=warmup, normalizer_json=norm_json, **extra)
            raise FileNotFoundError(
                f"no tokenizer.json, tekken.json, or vocab.json+merges.txt found in {d!r}"
            )

        from huggingface_hub import hf_hub_download
        kw = {}
        if revision:
            kw["revision"] = revision
        if local_files_only:
            kw["local_files_only"] = True
        if cache_dir is not None:
            kw["cache_dir"] = cache_dir
        if token is not None:
            kw["token"] = token
        try:
            tj = hf_hub_download(repo_id, "tokenizer.json", **kw)
        except Exception as e:
            # 1. Mistral Tekken format (tekken.json — Devstral, Mistral-Small-3.2,
            #    Magistral, Pixtral, ...)?
            from . import _formats as _fmt
            res = _fmt.try_load_tekken_repo(repo_id, revision=revision)
            if res is not None:
                tj_path, cfg = res
                return cls(tj_path, cfg, warmup=warmup, **extra)
            # 2. tiktoken-format model (ships tiktoken.model + a tokenization_*.py,
            #    e.g. Kimi-K2 / Kimi-VL)? Recover pat_str via pure static AST.
            from . import _tiktoken as _tt
            pattern = _tt.tiktoken_pattern_from_repo(repo_id, revision)
            if pattern is not None:
                return cls.from_tiktoken_model(repo_id, pattern=pattern, revision=revision, warmup=warmup)
            # 3. Legacy ByteLevel BPE (vocab.json + merges.txt, e.g. OPT / legacy Whisper)?
            legacy = _fmt.try_load_legacy_bpe_repo(repo_id, revision=revision)
            if legacy is not None:
                tj_path, cfg = legacy
                norm_json = None
                if (cfg.get("tokenizer_class") or "").startswith("Whisper"):
                    try:
                        norm_json = hf_hub_download(repo_id, "normalizer.json", **kw)
                    except Exception:
                        pass
                return cls(tj_path, cfg, warmup=warmup, normalizer_json=norm_json, **extra)
            # Not auto-recoverable: actionable error if it IS tiktoken-format, else
            # re-raise the original download failure.
            try:
                hf_hub_download(repo_id, "tiktoken.model", **kw)
            except Exception:
                raise e
            raise ValueError(
                f"{repo_id!r} ships a tiktoken.model but its split pattern could not "
                "be auto-extracted; load it with AutoTokenizer.from_tiktoken_model("
                "repo_id, pattern=<the model's pat_str>)."
            ) from None
        cfg = {}
        # special_tokens_map.json is legacy; tokenizer_config.json wins (matches
        # transformers), so load the map FIRST and let the config override it.
        for name in ("special_tokens_map.json", "tokenizer_config.json"):
            try:
                with open(hf_hub_download(repo_id, name, **kw), encoding="utf-8") as f:
                    cfg.update(json.load(f))
            except Exception:
                pass
        if not cfg.get("chat_template"):
            try:
                with open(hf_hub_download(repo_id, "chat_template.jinja", **kw), encoding="utf-8") as f:
                    cfg["chat_template"] = f.read()
            except Exception:
                pass
        norm_json = None
        if (cfg.get("tokenizer_class") or "").startswith("Whisper"):
            try:
                norm_json = hf_hub_download(repo_id, "normalizer.json", **kw)
            except Exception:
                pass
        return cls(tj, cfg, warmup=warmup, normalizer_json=norm_json, **extra)

    # ---- token <-> id helpers ----------------------------------------------
    def get_vocab(self):
        return dict(self._token_to_id)

    def get_added_vocab(self):
        return dict(self._added_content_to_id)

    def convert_tokens_to_string(self, tokens):
        ids = [i for i in (self._token_to_id.get(t) for t in tokens) if i is not None]
        return self._core.decode(ids)

    @property
    def vocab_size(self):
        return self._core.n_vocab

    def __len__(self):
        return self._core.n_vocab

    def _id(self, tok):
        if tok is None:
            return None
        tid = self._fast_token_to_id.get(tok)
        if tid is not None:
            return tid
        return self._token_to_id.get(tok)

    @property
    def bos_token_id(self):
        return self._id(self.bos_token)

    @property
    def eos_token_id(self):
        return self._id(self.eos_token)

    @property
    def pad_token_id(self):
        return self._id(self.pad_token)

    @property
    def unk_token_id(self):
        return self._id(self.unk_token)

    @property
    def all_special_ids(self):
        return sorted(self._special_ids)

    @property
    def all_special_tokens(self):
        return [self._fast_id_to_token[i] for i in sorted(self._special_ids)]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self._id_to_token.get(ids)
        out = []
        for i in ids:
            if skip_special_tokens and i in self._special_ids:
                continue
            out.append(self._id_to_token.get(int(i)))
        return out

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._token_to_id.get(tokens)
        return [self._token_to_id.get(t) for t in tokens]

    def tokenize(self, text, add_special_tokens=False):
        return self.convert_ids_to_tokens(self.encode(text, add_special_tokens=add_special_tokens))

    # ---- core encode (special-token aware) ----------------------------------
    def _encode_pieces(self, text):
        """Special-token-aware encode (no post-processor): one Rust pass that splits
        on the registered added tokens, BPEs each gap, and splices the special ids."""
        return self._core.encode_special(text)

    def _special_id_for(self, name):
        # TemplateProcessing.special_tokens[name].ids, else the configured token
        if self._post:
            pp = self._post
            tables = []
            if pp.get("type") == "Sequence":
                tables = [p for p in pp.get("processors", [])]
            else:
                tables = [pp]
            for p in tables:
                st = p.get("special_tokens") or {}
                if name in st and st[name].get("ids"):
                    return list(st[name]["ids"])
        tid = self._id(name)
        return [tid] if tid is not None else []

    def _apply_post(self, ids_a, ids_b=None):
        """Add special tokens per the model's post_processor. Returns (ids, type_ids)."""
        pp = self._post
        if pp is None:
            ids = ids_a + (ids_b or [])
            return ids, [0] * len(ids)
        procs = pp.get("processors", [pp]) if pp.get("type") == "Sequence" else [pp]
        # find the processor that actually adds specials
        for p in procs:
            t = p.get("type")
            if t == "TemplateProcessing":
                tmpl = p.get("pair" if ids_b is not None else "single", [])
                ids, types = [], []
                for piece in tmpl:
                    if "SpecialToken" in piece:
                        sid = piece["SpecialToken"]["id"]
                        ti = piece["SpecialToken"].get("type_id", 0)
                        st = (p.get("special_tokens") or {}).get(sid, {})
                        for x in st.get("ids", []):
                            ids.append(x)
                            types.append(ti)
                    elif "Sequence" in piece:
                        which = piece["Sequence"]["id"]
                        ti = piece["Sequence"].get("type_id", 0)
                        seq = ids_a if which == "A" else (ids_b or [])
                        ids.extend(seq)
                        types.extend([ti] * len(seq))
                return ids, types
            if t == "RobertaProcessing":
                cls = self._special_id_for(p.get("cls", ["<s>"])[0] if isinstance(p.get("cls"), list) else "<s>")
                sep = self._special_id_for(p.get("sep", ["</s>"])[0] if isinstance(p.get("sep"), list) else "</s>")
                cls = cls[:1] or [self._id("<s>")]
                sep = sep[:1] or [self._id("</s>")]
                if ids_b is None:
                    ids = cls + ids_a + sep
                else:
                    ids = cls + ids_a + sep + sep + ids_b + sep
                return ids, [0] * len(ids)
            if t == "BertProcessing":
                clsid = self._id("[CLS]")
                sepid = self._id("[SEP]")
                if ids_b is None:
                    ids = [clsid] + ids_a + [sepid]
                    return ids, [0] * len(ids)
                ids = [clsid] + ids_a + [sepid] + ids_b + [sepid]
                types = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)
                return ids, types
        # ByteLevel / none post_processor: no template specials, but many models
        # (Llama/DeepSeek style) add BOS/EOS via the `add_bos_token`/`add_eos_token`
        # config flags instead. Apply those here (single sequence only).
        ids = list(ids_a)
        if ids_b is None:
            if self.add_bos_token and self.bos_token_id is not None:
                ids = [self.bos_token_id] + ids
            if self.add_eos_token and self.eos_token_id is not None:
                ids = ids + [self.eos_token_id]
        else:
            ids = ids + ids_b
        return ids, [0] * len(ids)

    def encode(self, text, text_pair=None, add_special_tokens=True):
        ids_a = self._encode_pieces(text)
        if text_pair is None:
            if not add_special_tokens:
                return ids_a
            prefix, suffix = self._post_prefix_suffix()
            if self.__dict__.get("_ps_exact", False):
                if not prefix and not suffix:
                    return ids_a
                return prefix + ids_a + suffix
            ids, _ = self._apply_post(ids_a, None)
            return ids
        ids_b = self._encode_pieces(text_pair)
        if add_special_tokens:
            ids, _ = self._apply_post(ids_a, ids_b)
            return ids
        return ids_a + ids_b

    # ---- Whisper helpers (prefix tokens, timestamps, prompts, normalizers) --
    @property
    def prefix_tokens(self):
        if not self._is_whisper:
            raise AttributeError("prefix_tokens is only defined for Whisper tokenizers")
        from . import _whisper as _wh
        return _wh.whisper_prefix_tokens(
            self._added_content_to_id, self.language, self.task, self.predict_timestamps
        )

    def set_prefix_tokens(self, language=None, task=None, predict_timestamps=None):
        if not self._is_whisper:
            raise AttributeError("set_prefix_tokens is only defined for Whisper tokenizers")
        from . import _whisper as _wh
        if language is not None:
            self.language = language
        if task is not None:
            self.task = task
        if predict_timestamps is not None:
            self.predict_timestamps = bool(predict_timestamps)
        prefix_ids = self.prefix_tokens
        eos = self.eos_token or "<|endoftext|>"
        eos_id = self.eos_token_id
        self._post = _wh.whisper_post_processor(self._fast_id_to_token, eos, eos_id, prefix_ids)
        self.__dict__.pop("_ps_cache", None)
        self.__dict__.pop("_ps_exact", None)

    def get_decoder_prompt_ids(self, task=None, language=None, no_timestamps=True):
        self.set_prefix_tokens(task=task, language=language, predict_timestamps=not no_timestamps)
        forced = self.prefix_tokens[1:]
        return [(rank + 1, tok) for rank, tok in enumerate(forced)]

    def get_prompt_ids(self, text, return_tensors="np"):
        if not self._is_whisper:
            raise AttributeError("get_prompt_ids is only defined for Whisper tokenizers")
        sop_id = self._added_content_to_id["<|startofprev|>"]
        prompt_ids = self._encode_pieces(" " + text.strip())
        min_special = min(self._special_ids) if self._special_ids else 0
        for x in prompt_ids:
            if x >= min_special:
                tok = self.convert_ids_to_tokens(x)
                raise ValueError(
                    f"Encountered text in the prompt corresponding to disallowed special token: {tok}."
                )
        ids = [sop_id] + prompt_ids
        if return_tensors == "np":
            import numpy as np
            return np.array(ids)
        if return_tensors == "pt":
            import torch
            return torch.tensor(ids)
        return ids

    def timestamp_ids(self, time_precision=0.02):
        if not self._is_whisper:
            raise AttributeError("timestamp_ids is only defined for Whisper tokenizers")
        return [self._id("<|%.2f|>" % (i * time_precision)) for i in range(1501)]

    def normalize(self, text):
        from . import _whisper as _wh
        return _wh.english_normalize(text, self._normalizer_json)

    @staticmethod
    def basic_normalize(text, remove_diacritics=False):
        from . import _whisper as _wh
        return _wh.basic_normalize(text, remove_diacritics=remove_diacritics)

    # ---- decode -------------------------------------------------------------
    def decode(self, ids, skip_special_tokens=False, decode_with_timestamps=False,
               output_offsets=False, time_precision=0.02, segment_size=1500,
               normalize=False, basic_normalize=False, remove_diacritics=False, **_):
        if isinstance(ids, int):
            ids = [ids]
        elif hasattr(ids, "ndim"):
            # Zero-copy buffer fast path for 1D (or [1, N]) numpy arrays / torch CPU tensors
            # (skips Python `.tolist()` int-boxing completely!).
            if hasattr(ids, "detach"):
                ids = ids.detach().cpu().numpy()
            if ids.ndim == 2 and ids.shape[0] == 1:
                ids = ids[0]
            elif ids.ndim != 1:
                ids = ids.tolist()
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()
            if ids and isinstance(ids[0], list) and len(ids) == 1:
                ids = ids[0]
        elif not isinstance(ids, list):
            ids = [int(i) for i in ids]
        text = self._core.decode(
            ids, bool(skip_special_tokens), bool(decode_with_timestamps),
            float(time_precision), int(segment_size),
        )
        if not decode_with_timestamps:
            if normalize:
                text = self.normalize(text)
            elif basic_normalize:
                text = self.basic_normalize(text, remove_diacritics=remove_diacritics)
        if output_offsets and self._is_whisper and self._whisper_ts_begin is not None:
            from . import _whisper as _wh
            id_list = ids.tolist() if hasattr(ids, "tolist") else ids
            offsets = _wh.whisper_compute_offsets(
                id_list, self._core, self._whisper_ts_begin, float(time_precision), int(segment_size)
            )
            return {"text": text, "offsets": offsets}
        return text

    def batch_decode(self, sequences, skip_special_tokens=False, decode_with_timestamps=False,
                     output_offsets=False, time_precision=0.02, segment_size=1500,
                     normalize=False, basic_normalize=False, remove_diacritics=False, **_):
        if hasattr(sequences, "ndim"):
            if hasattr(sequences, "detach"):
                sequences = sequences.detach().cpu().numpy()
            if sequences.ndim != 2:
                sequences = sequences.tolist()
        elif hasattr(sequences, "tolist"):
            sequences = sequences.tolist()
        elif sequences and not isinstance(sequences[0], list):
            sequences = [s.tolist() if hasattr(s, "tolist") else [int(i) for i in s] for s in sequences]
        texts = self._core.decode_batch(
            sequences, 0, bool(skip_special_tokens), bool(decode_with_timestamps),
            float(time_precision), int(segment_size),
        )
        if not decode_with_timestamps:
            if normalize:
                texts = [self.normalize(t) for t in texts]
            elif basic_normalize:
                texts = [self.basic_normalize(t, remove_diacritics=remove_diacritics) for t in texts]
        if output_offsets and self._is_whisper and self._whisper_ts_begin is not None:
            from . import _whisper as _wh
            seq_lists = sequences.tolist() if hasattr(sequences, "tolist") else sequences
            return [
                {
                    "text": t,
                    "offsets": _wh.whisper_compute_offsets(
                        s, self._core, self._whisper_ts_begin, float(time_precision), int(segment_size)
                    ),
                }
                for t, s in zip(texts, seq_lists)
            ]
        return texts

    def decode_stream(self, skip_special_tokens=False):
        """Return an incremental UTF-8-aware `DecodeStream` (`.step(id)`,
        `.step_many(ids)`, `.flush()`, `.reset()`)."""
        return self._core.decode_stream(bool(skip_special_tokens))

    def count_tokens(self, text, add_special_tokens=True):
        """Fast token count matching `len(self.encode(text, add_special_tokens=...))`
        without constructing a Python `list[int]`."""
        if not add_special_tokens:
            return self._core.count_special_tokens(text)
        prefix, suffix = self._post_prefix_suffix()
        if self.__dict__.get("_ps_exact", False):
            return len(prefix) + self._core.count_special_tokens(text) + len(suffix)
        return len(self.encode(text, add_special_tokens=True))

    def count_tokens_batch(self, texts, add_special_tokens=True, num_threads=0):
        """Parallel batch token counts matching `[len(self.encode(t, ...)) for t in texts]`
        with zero Python `list[int]` boxing."""
        if not add_special_tokens:
            return self._core.count_special_tokens_batch(texts, int(num_threads))
        prefix, suffix = self._post_prefix_suffix()
        if self.__dict__.get("_ps_exact", False):
            extra = len(prefix) + len(suffix)
            counts = self._core.count_special_tokens_batch(texts, int(num_threads))
            return [c + extra for c in counts] if extra else counts
        return [len(self.encode(t, add_special_tokens=True)) for t in texts]

    def encode_file_to_buffer(self, path, num_threads=0, separator=None, dtype=None):
        """Zero-copy mmap ordinary-encoding of a file on disk into
        `{"data": np.ndarray, "offsets": np.ndarray}`."""
        return self._core.encode_file_to_buffer(path, int(num_threads), separator, dtype)

    def count_tokens_file(self, path, num_threads=0, separator=None):
        """Zero-copy mmap ordinary token count for a file on disk (returns `int`
        when `separator is None`, or `list[int]` per separated document)."""
        return self._core.count_tokens_file(path, int(num_threads), separator)

    def truncate(self, text, max_tokens, side="right", add_special_tokens=False):
        """Truncate `text` to fit within `max_tokens` tokens without allocating
        a Python `list[int]`. When `add_special_tokens=True`, reserves budget for
        the tokenizer's post-processor prefix/suffix tokens."""
        max_tokens = int(max_tokens)
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', not {side!r}")
        if not add_special_tokens:
            return self._core.truncate(text, max_tokens, side, True)
        prefix, suffix = self._post_prefix_suffix()
        if self.__dict__.get("_ps_exact", False):
            budget = max(0, max_tokens - len(prefix) - len(suffix))
            return self._core.truncate(text, budget, side, True)
        ids = self.encode(text, add_special_tokens=True)
        if len(ids) <= max_tokens:
            return text
        if max_tokens == 0:
            return ""
        sub = ids[:max_tokens] if side == "right" else ids[-max_tokens:]
        return self.decode(sub, skip_special_tokens=True)

    def chunk_text(self, text, max_tokens, overlap=0, add_special_tokens=False):
        """Split `text` into consecutive chunks of at most `max_tokens` tokens
        each (with `overlap` shared tokens between adjacent chunks) directly in
        Rust without Python `list[int]` boxing."""
        max_tokens = int(max_tokens)
        overlap = int(overlap)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if overlap < 0 or overlap >= max_tokens:
            raise ValueError("overlap must satisfy 0 <= overlap < max_tokens")
        if not add_special_tokens:
            return self._core.chunk_text(text, max_tokens, overlap, True)
        prefix, suffix = self._post_prefix_suffix()
        if self.__dict__.get("_ps_exact", False):
            budget = max_tokens - len(prefix) - len(suffix)
            if budget <= 0:
                raise ValueError(
                    f"max_tokens ({max_tokens}) must exceed post-processor special tokens ({len(prefix) + len(suffix)})"
                )
            if overlap >= budget:
                raise ValueError(
                    f"overlap ({overlap}) must be < effective content budget ({budget})"
                )
            return self._core.chunk_text(text, budget, overlap, True)
        return self._core.chunk_text(text, max_tokens, overlap, True)

    def _post_prefix_suffix(self):
        """Special ids the post-processor adds before/after a single sequence
        (computed once via a sentinel; covers TemplateProcessing/Bert/Roberta/
        add_bos/eos — all of which put the content in one contiguous slot)."""
        cached = self.__dict__.get("_ps_cache")
        if cached is not None:
            return cached
        sentinel = -987654321
        out, _ = self._apply_post([sentinel], None)
        if sentinel in out:
            i = out.index(sentinel)
            ps = (out[:i], out[i + 1:])
            # exact iff the content appears exactly once (prefix + content + suffix)
            self._ps_exact = out.count(sentinel) == 1
        else:
            ps = ([], [])
            self._ps_exact = self._post is None
        self._ps_cache = ps
        return ps

    # ---- __call__ : padding / truncation / tensors --------------------------
    def __call__(self, text, text_pair=None, add_special_tokens=True, padding=False,
                 truncation=False, max_length=None, return_tensors=None,
                 return_attention_mask=True, return_token_type_ids=False, **_):
        single = isinstance(text, str)
        texts = [text] if single else (text if type(text) is list else list(text))
        pairs = ([text_pair] if single else list(text_pair)) if text_pair is not None else [None] * len(texts)

        # All-Rust fast path for the tensor output (single-sequence post-processor):
        # encode + specials + truncation + padding + mask happen in one Rust pass,
        # no Python per-row work or int-boxing.
        if return_tensors in ("np", "pt") and text_pair is None:
            prefix, suffix = self._post_prefix_suffix() if add_special_tokens else ([], [])
            if not add_special_tokens or self.__dict__.get("_ps_exact", False):
                d = self._core.encode_special_padded(
                    texts, prefix, suffix, max_length, self.pad_token_id or 0,
                    bool(truncation), padding == "max_length",
                    self.padding_side == "left", self.truncation_side == "left", 0)
                out = {"input_ids": d["input_ids"]}
                if return_attention_mask:
                    out["attention_mask"] = d["attention_mask"]
                if return_token_type_ids:
                    import numpy as np
                    out["token_type_ids"] = np.zeros_like(d["input_ids"])
                if return_tensors == "pt":
                    import torch
                    out = {k: torch.as_tensor(v) for k, v in out.items()}
                return out

        # Rust fast path for lists (single sequences): encode + specials +
        # truncation build the rows in one Rust pass; Python only adds masks and
        # padding. Used only when the post-processor is exactly prefix+content+
        # suffix (verified with a sentinel); anything else takes the path below.
        if text_pair is None:
            prefix, suffix = self._post_prefix_suffix() if add_special_tokens else ([], [])
            if not add_special_tokens or self.__dict__.get("_ps_exact", False):
                budget = (max(0, max_length - len(prefix) - len(suffix))
                          if (truncation and max_length) else None)
                want_mask = bool(return_attention_mask or padding)
                seqs, masks = self._core.encode_special_post_masked(
                    texts, list(prefix), list(suffix), budget, self.truncation_side == "left", want_mask, 0)
                types = [[0] * len(q) for q in seqs] if return_token_type_ids else None
                if padding:
                    width = max_length if (padding == "max_length" and max_length) else max((len(q) for q in seqs), default=0)
                    pid = self.pad_token_id or 0
                    left = self.padding_side != "right"
                    for k in range(len(seqs)):
                        pad = width - len(seqs[k])
                        if pad > 0:
                            if left:
                                seqs[k] = [pid] * pad + seqs[k]
                                masks[k] = [0] * pad + masks[k]
                                if types is not None:
                                    types[k] = [0] * pad + types[k]
                            else:
                                seqs[k] = seqs[k] + [pid] * pad
                                masks[k] = masks[k] + [0] * pad
                                if types is not None:
                                    types[k] = types[k] + [0] * pad
                out = {"input_ids": seqs}
                if return_attention_mask:
                    out["attention_mask"] = masks
                if return_token_type_ids:
                    out["token_type_ids"] = types
                if single and return_tensors is None:
                    out = {k: v[0] for k, v in out.items()}
                return out

        def _trunc(v, k):
            if k >= len(v):
                return v
            return v[:k] if self.truncation_side == "right" else v[len(v) - k:]

        # BPE all inputs in ONE parallel Rust pass (instead of a Python loop) — the
        # cheap post-processing (specials, padding) then runs per row in Python.
        a_all = self._core.encode_special_batch(texts, 0)
        has_pairs = any(p is not None for p in pairs)
        b_all = [self._core.encode_special(p) if p is not None else None for p in pairs] if has_pairs \
            else [None] * len(texts)

        seqs, types = [], []
        for a, b in zip(a_all, b_all):
            # Truncate the CONTENT, reserving room for the special tokens (HF
            # truncates so the post-processor specials always survive).
            if truncation and max_length:
                ns = len(self._apply_post([], [] if b is not None else None)[0]) if add_special_tokens else 0
                budget = max(0, max_length - ns)
                if b is None:
                    a = _trunc(a, budget)
                else:
                    while len(a) + len(b) > budget and (a or b):  # longest_first
                        if len(a) >= len(b) and a:
                            a = a[:-1] if self.truncation_side == "right" else a[1:]
                        elif b:
                            b = b[:-1] if self.truncation_side == "right" else b[1:]
            if add_special_tokens:
                ids, ty = self._apply_post(a, b)
            else:
                ids = a + (b or [])
                ty = [0] * len(ids)
            seqs.append(ids)
            types.append(ty)

        masks = [[1] * len(s) for s in seqs]
        if padding:
            width = max_length if (padding == "max_length" and max_length) else max((len(s) for s in seqs), default=0)
            pid = self.pad_token_id or 0
            for k in range(len(seqs)):
                pad = width - len(seqs[k])
                if pad > 0:
                    if self.padding_side == "right":
                        seqs[k] = seqs[k] + [pid] * pad
                        masks[k] = masks[k] + [0] * pad
                        types[k] = types[k] + [0] * pad
                    else:
                        seqs[k] = [pid] * pad + seqs[k]
                        masks[k] = [0] * pad + masks[k]
                        types[k] = [0] * pad + types[k]

        out = {"input_ids": seqs}
        if return_attention_mask:
            out["attention_mask"] = masks
        if return_token_type_ids:
            out["token_type_ids"] = types
        if single and return_tensors is None:
            out = {k: v[0] for k, v in out.items()}
        if return_tensors in ("np", "pt"):
            import numpy as np
            out = {k: np.array(v) for k, v in out.items()}
            if return_tensors == "pt":
                import torch
                out = {k: torch.as_tensor(v) for k, v in out.items()}
        return out

    # ---- chat templates (Jinja2) -------------------------------------------
    def _chat_template_str(self, tools=None):
        """chat_template may be a string or a list of {name, template} dicts
        (e.g. `default` + `tool_use`); pick the right one."""
        ct = self.chat_template
        if isinstance(ct, list):
            by = {d.get("name"): d.get("template") for d in ct}
            if tools and "tool_use" in by:
                return by["tool_use"]
            return by.get("default") or next(iter(by.values()), None)
        return ct

    def _compiled_template(self, tmpl_str):
        """Compile (and cache) a Jinja2 chat template once — recompiling per call
        was costing ~7 ms and dominated apply_chat_template."""
        cache = self.__dict__.setdefault("_jinja_cache", {})
        tmpl = cache.get(tmpl_str)
        if tmpl is not None:
            return tmpl
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        env = self.__dict__.get("_jinja_env")
        if env is None:
            def raise_exception(msg):
                raise jinja2.exceptions.TemplateError(msg)
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False)
            env.globals["raise_exception"] = raise_exception
            self._jinja_env = env
        tmpl = env.from_string(tmpl_str)
        cache[tmpl_str] = tmpl
        return tmpl

    def _fast_chat_compiler(self, tmpl_str):
        """Self-deriving, Jinja-verified transition-graph renderer for role-based
        chat templates (ChatML/Qwen, Llama-3, Phi-3.5, SmolLM2, Mistral, Gemma, ...).

        Learns `(start[role], trans[(prev, cur)], end[(last, add_gen)])` directly
        from Jinja2 probe renders and exhaustively cross-checks against Jinja2 on
        multi-turn, whitespace-padded, and empty-content conversations. Returns a
        fast renderer `(messages, add_gen) -> str | None` (returning None to fall
        back to Jinja2 on any unusual input) or None if the template is dynamic.
        """
        cache = self.__dict__.setdefault("_fast_chat_cache", {})
        if tmpl_str in cache:
            return cache[tmpl_str]

        # Bail immediately if template uses time/random/string-mutating filters or
        # reasoning/thinking switches that could depend on message content.
        bad_markers = (
            "strftime", "now(", "date_string", ".strip(", ".lstrip(", ".rstrip(",
            ".replace(", ".split(", ".lower(", ".upper(", "think", "reasoning",
        )
        low = tmpl_str.lower()
        if any(m in low for m in bad_markers):
            cache[tmpl_str] = None
            return None

        tmpl = self._compiled_template(tmpl_str)
        bos = self.bos_token or ""
        eos = self.eos_token or ""
        unk = self.unk_token or ""
        pad = self.pad_token or ""

        def jrender(msgs, add_gen=False):
            return tmpl.render(
                messages=msgs, tools=None, add_generation_prompt=add_gen,
                bos_token=bos, eos_token=eos, unk_token=unk, pad_token=pad,
            )

        def split_sentinels(s, sents):
            parts = []
            pos = 0
            for tok in sents:
                idx = s.find(tok, pos)
                if idx < 0 or s.find(tok, idx + len(tok)) >= 0:
                    return None
                parts.append(s[pos:idx])
                pos = idx + len(tok)
            parts.append(s[pos:])
            return parts

        S0, U1, A1, U2, A2 = "__TTS0__", "__TTU1__", "__TTA1__", "__TTU2__", "__TTA2__"
        try:
            r_suau = jrender([
                {"role": "system", "content": S0},
                {"role": "user", "content": U1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": U2},
            ], False)
            r_uau = jrender([
                {"role": "user", "content": U1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": U2},
            ], False)
            r_sua = jrender([
                {"role": "system", "content": S0},
                {"role": "user", "content": U1},
                {"role": "assistant", "content": A1},
            ], False)
            r_suau_g = jrender([
                {"role": "system", "content": S0},
                {"role": "user", "content": U1},
                {"role": "assistant", "content": A1},
                {"role": "user", "content": U2},
            ], True)
            r_sua_g = jrender([
                {"role": "system", "content": S0},
                {"role": "user", "content": U1},
                {"role": "assistant", "content": A1},
            ], True)
        except Exception:
            cache[tmpl_str] = None
            return None

        p_suau = split_sentinels(r_suau, (S0, U1, A1, U2))
        p_uau = split_sentinels(r_uau, (U1, A1, U2))
        p_sua = split_sentinels(r_sua, (S0, U1, A1))
        p_suau_g = split_sentinels(r_suau_g, (S0, U1, A1, U2))
        p_sua_g = split_sentinels(r_sua_g, (S0, U1, A1))
        if not (p_suau and p_uau and p_sua and p_suau_g and p_sua_g):
            cache[tmpl_str] = None
            return None

        start = {"system": p_suau[0], "user": p_uau[0]}
        trans = {
            ("system", "user"): p_suau[1],
            ("user", "assistant"): p_suau[2],
            ("assistant", "user"): p_suau[3],
        }
        end = {
            ("user", False): p_suau[4],
            ("user", True): p_suau_g[4],
            ("assistant", False): p_sua[3],
            ("assistant", True): p_sua_g[3],
        }

        # Optional consecutive same-role transitions (user->user, assistant->assistant)
        for rname, s1, s2 in (("user", U1, U2), ("assistant", A1, A2)):
            try:
                base_msgs = [{"role": "user", "content": U1}] if rname == "assistant" else []
                probe = base_msgs + [{"role": rname, "content": s1}, {"role": rname, "content": s2}]
                sents = ([U1] if rname == "assistant" else []) + [s1, s2]
                rp = split_sentinels(jrender(probe, False), sents)
                if rp is not None:
                    trans[(rname, rname)] = rp[-2]
            except Exception:
                pass

        def render_graph(msgs, add_gen):
            m0 = msgs[0]
            r0 = m0["role"]
            st = start.get(r0)
            if st is None:
                return None
            out = [st, m0["content"]]
            prev = r0
            for m in msgs[1:]:
                role = m["role"]
                tr = trans.get((prev, role))
                if tr is None:
                    return None
                out.append(tr)
                out.append(m["content"])
                prev = role
            en = end.get((prev, add_gen))
            if en is None:
                return None
            out.append(en)
            return "".join(out)

        # Exhaustive differential verification against Jinja2 across shapes & edge content.
        probes = [
            [{"role": "user", "content": "hi"}],
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}],
            [
                {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
                {"role": "user", "content": "q3"},
            ],
            [
                {"role": "system", "content": "  keep spaces \n "},
                {"role": "user", "content": "\n  keep spaces  \t"},
                {"role": "assistant", "content": "  reply \n"},
            ],
        ]
        for pm in probes:
            for ag in (False, True):
                try:
                    expected = jrender(pm, ag)
                except Exception:
                    cache[tmpl_str] = None
                    return None
                got = render_graph(pm, ag)
                if got != expected:
                    cache[tmpl_str] = None
                    return None

        # Check whether empty string contents also match (e.g. Phi-3.5 omits empty system).
        allow_empty = True
        try:
            empty_probe = [
                {"role": "system", "content": ""},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""},
            ]
            if render_graph(empty_probe, False) != jrender(empty_probe, False):
                allow_empty = False
        except Exception:
            allow_empty = False

        # Build the ultra-fast inline closure for verified plain conversations.
        _start, _trans, _end = start, trans, end
        _allow_empty = allow_empty

        def fast_render(msgs, add_gen):
            n = len(msgs)
            if n == 0:
                return None
            m0 = msgs[0]
            if type(m0) is not dict or len(m0) != 2:
                return None
            r0 = m0.get("role")
            c0 = m0.get("content")
            if type(c0) is not str or (not _allow_empty and not c0):
                return None
            st = _start.get(r0)
            if st is None:
                return None
            if n == 1:
                en = _end.get((r0, add_gen))
                return (st + c0 + en) if en is not None else None
            out = [st, c0]
            prev = r0
            for i in range(1, n):
                m = msgs[i]
                if type(m) is not dict or len(m) != 2:
                    return None
                role = m.get("role")
                c = m.get("content")
                if type(c) is not str or (not _allow_empty and not c):
                    return None
                tr = _trans.get((prev, role))
                if tr is None:
                    return None
                out.append(tr)
                out.append(c)
                prev = role
            en = _end.get((prev, add_gen))
            if en is None:
                return None
            out.append(en)
            return "".join(out)

        cache[tmpl_str] = fast_render
        return fast_render

    def apply_chat_template(self, conversation, tools=None, add_generation_prompt=False,
                            tokenize=True, padding=False, truncation=False, max_length=None,
                            return_tensors=None, return_dict=True, **kwargs):
        tmpl_str = self._chat_template_str(tools)
        if not tmpl_str:
            raise ValueError("this tokenizer has no chat_template")

        is_batch = isinstance(conversation, (list, tuple)) and len(conversation) > 0 and isinstance(conversation[0], (list, tuple))
        add_gen = bool(add_generation_prompt)
        fast = self._fast_chat_compiler(tmpl_str) if (tools is None and not kwargs) else None

        def render_one(conv):
            if fast is not None:
                res = fast(conv, add_gen)
                if res is not None:
                    return res
            template = self._compiled_template(tmpl_str)
            return template.render(
                messages=conv, tools=tools,
                add_generation_prompt=add_gen,
                bos_token=self.bos_token or "", eos_token=self.eos_token or "",
                unk_token=self.unk_token or "", pad_token=self.pad_token or "",
                **kwargs,
            )

        if is_batch:
            rendered = [render_one(c) for c in conversation]
            if not tokenize:
                return rendered
            # Chat templates already embed their special tokens -> add_special_tokens=False.
            out = self(
                rendered,
                add_special_tokens=False,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_dict,
            )
            if return_dict:
                return _ChatDict(out)
            return out["input_ids"]

        rendered = render_one(conversation)
        if not tokenize:
            return rendered

        # Fast path for the common single-conversation call (no padding/truncation).
        if not padding and not truncation:
            ids = self._core.encode_special(rendered)
            if return_tensors not in ("np", "pt"):
                if return_dict:
                    return _ChatDict(input_ids=ids, attention_mask=[1] * len(ids))
                return ids
            import numpy as np
            arr = np.array([ids])               # batch-of-1, shape [1, len]
            if return_tensors == "pt":
                import torch
                t_ids = torch.as_tensor(arr)
                if return_dict:
                    return _ChatDict(input_ids=t_ids, attention_mask=torch.ones_like(t_ids))
                return t_ids
            if return_dict:
                return _ChatDict(input_ids=arr, attention_mask=np.ones_like(arr))
            return arr

        out = self(
            [rendered],
            add_special_tokens=False,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_attention_mask=return_dict,
        )
        if return_tensors not in ("np", "pt"):
            out = {k: v[0] for k, v in out.items()}
        if return_dict:
            return _ChatDict(out)
        return out["input_ids"]

    def __repr__(self):
        return f"AutoTokenizer(vocab_size={self.vocab_size}, core={self._core!r})"
