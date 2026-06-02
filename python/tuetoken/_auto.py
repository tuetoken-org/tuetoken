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


class AutoTokenizer:
    # ---- construction -------------------------------------------------------
    def __init__(self, tokenizer_json, config=None):
        with open(tokenizer_json, encoding="utf-8") as f:
            tj = json.load(f)
        self._core = _Core(tokenizer_json)
        self._cfg = config or {}

        # id <-> token-string maps (vocab + added tokens), for convert_*_to_*.
        model = tj.get("model", {})
        self._id_to_token = {}
        vocab = model.get("vocab", {})
        if isinstance(vocab, dict):
            for tok, i in vocab.items():
                self._id_to_token[int(i)] = tok
        self._added = tj.get("added_tokens", [])
        self._added_content_to_id = {}
        self._special_ids = set()
        for a in self._added:
            i = int(a["id"])
            self._id_to_token[i] = a["content"]
            self._added_content_to_id[a["content"]] = i
            if a.get("special", False):
                self._special_ids.add(i)
        self._token_to_id = {t: i for i, t in self._id_to_token.items()}

        self._post = tj.get("post_processor")

        # SentencePiece metaspace prepend scheme, for splicing fragments around
        # special tokens correctly: "always" prepends ▁ to every fragment, "first"
        # only to the first fragment of the whole sequence, "none" never.
        pre = tj.get("pre_tokenizer") or {}
        def _find_ms(p):
            if p.get("type") == "Metaspace":
                return p
            if p.get("type") == "Sequence":
                for x in p.get("pretokenizers", []):
                    if x.get("type") == "Metaspace":
                        return x
            return None
        ms = _find_ms(pre)
        norm = tj.get("normalizer") or {}
        nstages = norm.get("normalizers", []) if norm.get("type") == "Sequence" else ([norm] if norm else [])
        has_prepend_norm = any(s.get("type") == "Prepend" and s.get("prepend") == "▁" for s in nstages)
        if ms is not None:
            self._prepend_scheme = ms.get("prepend_scheme", "always")
        elif has_prepend_norm:
            self._prepend_scheme = "always"
        else:
            self._prepend_scheme = "none"
        # NOTE: models with `legacy: true` SentencePiece (e.g. Yi-1.5, a Replace-only
        # normalizer with no Prepend) apply a dummy-prefix that isn't in tokenizer.json;
        # we match the tokenizer.json (== the raw `tokenizers` fast tokenizer) but not
        # transformers' legacy path for those.

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
                if cand in self._token_to_id:
                    self.pad_token = cand
                    break

        # Hand the added tokens to the Rust core so `encode_special` splits + BPEs a
        # whole chat string in one pass (no Python regex / per-fragment crossing).
        # The core reproduces "first" (prepend ▁ to the first fragment) and "always"
        # (every fragment); HF's "never" (suppress the prepend) has no core mode, so
        # fail closed with a clear message instead of a cryptic KeyError.
        _scheme_map = {"none": 0, "always": 1, "first": 2}
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

    @classmethod
    def from_file(cls, tokenizer_json, config=None):
        cfg = None
        if isinstance(config, str):
            with open(config, encoding="utf-8") as f:
                cfg = json.load(f)
        elif isinstance(config, dict):
            cfg = config
        return cls(tokenizer_json, cfg)

    @classmethod
    def from_tiktoken_model(cls, source, pattern, special_tokens=None, revision=None):
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
        return cls(tj_path, cfg)

    @classmethod
    def from_pretrained(cls, repo_id, revision=None, **_):
        from huggingface_hub import hf_hub_download
        kw = {"revision": revision} if revision else {}
        try:
            tj = hf_hub_download(repo_id, "tokenizer.json", **kw)
        except Exception as e:
            # tiktoken-format model (ships tiktoken.model + a tokenization_*.py, no
            # tokenizer.json — e.g. Kimi)? Recover its split pattern statically from
            # the module named in tokenizer_config.json's auto_map (config-driven,
            # never the model name; pure AST, no code execution) and load it
            # transparently — so from_pretrained "just works".
            from . import _tiktoken as _tt
            pattern = _tt.tiktoken_pattern_from_repo(repo_id, revision)
            if pattern is not None:
                return cls.from_tiktoken_model(repo_id, pattern=pattern, revision=revision)
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
        return cls(tj, cfg)

    # ---- token <-> id helpers ----------------------------------------------
    def get_vocab(self):
        return dict(self._token_to_id)

    @property
    def vocab_size(self):
        return self._core.n_vocab

    def __len__(self):
        return self._core.n_vocab

    def _id(self, tok):
        return self._token_to_id.get(tok) if tok is not None else None

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
        return [self._id_to_token[i] for i in sorted(self._special_ids)]

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
        tid = self._token_to_id.get(name)
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
                cls = cls[:1] or [self._token_to_id.get("<s>")]
                sep = sep[:1] or [self._token_to_id.get("</s>")]
                if ids_b is None:
                    ids = cls + ids_a + sep
                else:
                    ids = cls + ids_a + sep + sep + ids_b + sep
                return ids, [0] * len(ids)
            if t == "BertProcessing":
                clsid = self._token_to_id.get("[CLS]")
                sepid = self._token_to_id.get("[SEP]")
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
        ids_b = self._encode_pieces(text_pair) if text_pair is not None else None
        if add_special_tokens:
            ids, _ = self._apply_post(ids_a, ids_b)
            return ids
        return ids_a + (ids_b or [])

    # ---- decode -------------------------------------------------------------
    def decode(self, ids, skip_special_tokens=False, **_):
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            ids = [i for i in ids if i not in self._special_ids]
        return self._core.decode(ids)

    def batch_decode(self, sequences, skip_special_tokens=False, **_):
        if skip_special_tokens:
            sequences = [[int(i) for i in s if int(i) not in self._special_ids] for s in sequences]
        else:
            sequences = [[int(i) for i in s] for s in sequences]
        return self._core.decode_batch(sequences, 0)  # parallel Rust gather

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
        else:
            ps = ([], [])
        self._ps_cache = ps
        return ps

    # ---- __call__ : padding / truncation / tensors --------------------------
    def __call__(self, text, text_pair=None, add_special_tokens=True, padding=False,
                 truncation=False, max_length=None, return_tensors=None,
                 return_attention_mask=True, return_token_type_ids=False, **_):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        pairs = ([text_pair] if single else list(text_pair)) if text_pair is not None else [None] * len(texts)

        # All-Rust fast path for the tensor output (single-sequence post-processor):
        # encode + specials + truncation + padding + mask happen in one Rust pass,
        # no Python per-row work or int-boxing.
        if return_tensors in ("np", "pt") and text_pair is None:
            prefix, suffix = self._post_prefix_suffix() if add_special_tokens else ([], [])
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

    def apply_chat_template(self, conversation, tools=None, add_generation_prompt=False,
                            tokenize=True, return_tensors=None, **kwargs):
        tmpl_str = self._chat_template_str(tools)
        if not tmpl_str:
            raise ValueError("this tokenizer has no chat_template")
        template = self._compiled_template(tmpl_str)
        rendered = template.render(
            messages=conversation, tools=tools,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token or "", eos_token=self.eos_token or "",
            unk_token=self.unk_token or "", pad_token=self.pad_token or "",
            **kwargs,
        )
        if not tokenize:
            return rendered
        # chat templates already embed their special tokens -> don't add more
        ids = self.encode(rendered, add_special_tokens=False)
        if return_tensors in ("np", "pt"):
            import numpy as np
            arr = np.array([ids])
            if return_tensors == "pt":
                import torch
                return torch.as_tensor(arr)
            return arr
        return ids

    def __repr__(self):
        return f"AutoTokenizer(vocab_size={self.vocab_size}, core={self._core!r})"
