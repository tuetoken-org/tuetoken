"""Whisper-specific tokenizer helpers (prefix tokens, timestamps, offsets, normalization).

Config-driven: activated automatically by `AutoTokenizer` whenever a model's
`tokenizer_config.json` declares `WhisperTokenizer*` or its `tokenizer.json`
contains Whisper's signature `<|startoftranscript|>`, `<|notimestamps|>`, and
`<|0.00|>` timestamp tokens. Zero `transformers` runtime dependency.
"""

import json
import re
import unicodedata

# Full 100-language table matching OpenAI Whisper / HF WhisperTokenizer (v1-v3-turbo).
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

_LANG_KEYS = tuple(LANGUAGES.keys())

TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
    "mandarin": "zh",
}

TASK_IDS = ("translate", "transcribe")


def is_whisper_tokenizer(cfg, added_content_to_id):
    """Detect Whisper tokenizers from config/vocab metadata (never model name)."""
    tok_cls = (cfg or {}).get("tokenizer_class") or ""
    if tok_cls in ("WhisperTokenizer", "WhisperTokenizerFast"):
        return True
    return (
        "<|startoftranscript|>" in added_content_to_id
        and "<|notimestamps|>" in added_content_to_id
        and "<|0.00|>" in added_content_to_id
    )


def whisper_prefix_tokens(added_content_to_id, language=None, task=None, predict_timestamps=False):
    """Compute Whisper prefix token IDs matching `WhisperTokenizerFast.prefix_tokens`."""
    bos_id = added_content_to_id.get("<|startoftranscript|>")
    translate_id = added_content_to_id.get("<|translate|>")
    transcribe_id = added_content_to_id.get("<|transcribe|>")
    notimestamps_id = added_content_to_id.get("<|notimestamps|>")

    lang_code = None
    if language is not None:
        lang = language.lower()
        if lang in TO_LANGUAGE_CODE:
            lang_code = TO_LANGUAGE_CODE[lang]
        elif lang in LANGUAGES:
            lang_code = lang
        else:
            is_code = len(lang) == 2
            opts = list(LANGUAGES.keys()) if is_code else list(TO_LANGUAGE_CODE.keys())
            raise ValueError(f"Unsupported language: {language}. Language should be one of: {opts}.")

    if task is not None and task not in TASK_IDS:
        raise ValueError(f"Unsupported task: {task}. Task should be in: {list(TASK_IDS)}")

    seq = [bos_id]
    if lang_code is not None:
        # Match HF WhisperTokenizerFast: `bos_id + 1 + langs.index(lang_code)`
        lang_tok = f"<|{lang_code}|>"
        if lang_tok in added_content_to_id:
            seq.append(added_content_to_id[lang_tok])
        else:
            seq.append(bos_id + 1 + _LANG_KEYS.index(lang_code))
    if task is not None:
        seq.append(transcribe_id if task == "transcribe" else translate_id)
    if not predict_timestamps and notimestamps_id is not None:
        seq.append(notimestamps_id)
    return seq


def whisper_post_processor(id_to_token, eos_token, eos_id, prefix_ids):
    """Build a HF `TemplateProcessing` post-processor dict for Whisper prefix tokens."""
    prefixes = [id_to_token[i] for i in prefix_ids]
    special_tokens = {eos_token: {"id": eos_token, "ids": [eos_id], "tokens": [eos_token]}}
    for tok, tid in zip(prefixes, prefix_ids):
        special_tokens[tok] = {"id": tok, "ids": [tid], "tokens": [tok]}
    prefix_pieces = [{"SpecialToken": {"id": tok, "type_id": 0}} for tok in prefixes]
    return {
        "type": "TemplateProcessing",
        "single": [
            *prefix_pieces,
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": eos_token, "type_id": 0}},
        ],
        "pair": [
            *prefix_pieces,
            {"Sequence": {"id": "A", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
            {"SpecialToken": {"id": eos_token, "type_id": 1}},
        ],
        "special_tokens": special_tokens,
    }


def whisper_compute_offsets(token_ids, core, timestamp_begin, time_precision=0.02, segment_size=1500):
    """Compute Whisper timestamp offset dicts matching `WhisperTokenizerFast._compute_offsets`."""
    if hasattr(token_ids, "tolist"):
        ids = token_ids.tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) > 1:
                raise ValueError("Can only process a single input at a time")
            ids = ids[0]
    else:
        ids = [int(x) for x in token_ids]

    n = len(ids)
    ts_pos = [i for i, t in enumerate(ids) if t >= timestamp_begin]
    if not ts_pos:
        return []
    consecutive = [
        i + 1 for i in range(n - 1) if ids[i] >= timestamp_begin and ids[i + 1] >= timestamp_begin
    ]
    if not consecutive and len(ts_pos) <= 1:
        return []
    last_ts_end = ts_pos[-1] + 1
    if last_ts_end not in consecutive:
        consecutive.append(last_ts_end)

    offsets = []
    last_slice = ts_pos[0]
    cur_max_ts = 0
    prev_segments_len = 0
    for cur_slice in consecutive:
        sliced = ids[last_slice:cur_slice]
        if len(sliced) > 1:
            start_pos = sliced[0] - timestamp_begin
            end_pos = sliced[-1] - timestamp_begin
            if start_pos < cur_max_ts:
                is_single_ending = last_slice >= 2 and not (
                    ids[last_slice - 2] >= timestamp_begin and ids[last_slice - 1] >= timestamp_begin
                )
                if is_single_ending:
                    prev_segments_len += segment_size
                else:
                    prev_segments_len += cur_max_ts
            cur_max_ts = end_pos
            text = core.decode(sliced, False, False, time_precision, segment_size)
            offsets.append({
                "text": text,
                "timestamp": (
                    start_pos * time_precision + prev_segments_len * time_precision,
                    end_pos * time_precision + prev_segments_len * time_precision,
                ),
            })
        last_slice = cur_slice
    return offsets


def basic_normalize(text, remove_diacritics=False):
    """Whisper BasicTextNormalizer (uses HF's if installed, else pure-Python fallback)."""
    try:
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        return BasicTextNormalizer(remove_diacritics=remove_diacritics)(text)
    except Exception:
        s = text.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
        s = re.sub(r"\(([^)]+?)\)", "", s)
        s = unicodedata.normalize("NFKD" if remove_diacritics else "NFKC", s)
        out = []
        for ch in s:
            cat = unicodedata.category(ch)
            if remove_diacritics and cat == "Mn":
                continue
            if cat[0] in ("M", "S", "P"):
                out.append(" ")
            else:
                out.append(ch)
        return re.sub(r"\s+", " ", "".join(out)).strip()


def english_normalize(text, normalizer_json_path=None):
    """Whisper EnglishTextNormalizer (uses HF's if installed, else basic_normalize)."""
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        mapping = None
        if normalizer_json_path:
            with open(normalizer_json_path, encoding="utf-8") as f:
                mapping = json.load(f)
        return EnglishTextNormalizer(mapping)(text)
    except Exception:
        return basic_normalize(text)
