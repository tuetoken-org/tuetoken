from os import PathLike
from typing import Any, Literal, Mapping, Sequence, overload

__version__: str

class DecodeStream:
    """Incremental UTF-8-aware streaming token decoder for LLM token-by-token
    generation (buffers incomplete multi-byte UTF-8 sequences and strips the
    initial SentencePiece leading space only once)."""

    def step(self, token_id: int) -> str:
        """Feed one token id and return any newly completed UTF-8 text."""

    def step_many(self, tokens: Any) -> str:
        """Feed a batch of token ids (list, numpy array, or int sequence) and
        return any newly completed UTF-8 text."""

    def flush(self) -> str:
        """Flush any trailing incomplete UTF-8 bytes at end-of-stream."""

    def reset(self) -> None:
        """Reset buffered bytes and SentencePiece leading-space state."""

class Tokenizer:
    """A fast BPE tokenizer loaded from a HuggingFace ``tokenizer.json``."""

    n_vocab: int

    def __init__(self, path: str) -> None:
        """Load from a ``tokenizer.json`` file path."""

    @staticmethod
    def from_pretrained(
        repo_id: str,
        revision: str | None = ...,
        warmup: Any = ...,
    ) -> "Tokenizer":
        """Download ``tokenizer.json`` from a HuggingFace repo (needs ``huggingface_hub``)."""

    @staticmethod
    def from_tiktoken(name: str, warmup: Any = ...) -> "Tokenizer":
        """Load an OpenAI tiktoken encoding by name, e.g. ``"o200k_base"`` or ``"cl100k_base"``."""

    @staticmethod
    def from_tekken(
        source: str,
        revision: str | None = ...,
        warmup: Any = ...,
    ) -> "Tokenizer":
        """Load a Mistral ``tekken.json`` tokenizer from a local path or HF repo."""

    def warmup(self, texts: Any = ..., num_threads: int = ...) -> int:
        """Pre-populate the shared chunk cache from sample text(s); returns learned entries."""

    def cache_limit_mb(self) -> int: ...
    def set_cache_limit_mb(self, mb: int) -> None: ...
    def learned_entries(self) -> int: ...

    # --- encode ---
    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text to token ids (special tokens in text are literal bytes)."""

    def encode_special(self, text: str) -> list[int]:
        """Encode text recognizing registered special tokens."""

    def encode_special_batch(
        self, texts: Sequence[str], num_threads: int = ...
    ) -> list[list[int]]:
        """Encode many texts in parallel recognizing registered special tokens."""

    def encode_to_bytes(self, text: str) -> bytes:
        """Encode to a raw native-endian uint32 buffer (``numpy.frombuffer(..., uint32)``)."""

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode and return ``(ids, offsets)`` where each offset is the token's
        ``(start, end)`` byte span in ``text``. ByteLevel models only."""

    def encode_with_offsets_buffer(self, text: str) -> dict[str, Any]:
        """Zero-copy numpy offset encoding returning
        ``{"input_ids": np.ndarray[uint32, (N,)], "offsets": np.ndarray[int64, (N, 2)]}``
        with byte spans ``[start, end]`` in ``text``. ByteLevel models only."""

    def count_tokens(self, text: str) -> int:
        """Number of ordinary tokens, without building the id list."""

    def count_special_tokens(self, text: str) -> int:
        """Number of tokens (recognizing registered specials), without building the id list."""

    def count_special_tokens_batch(
        self, texts: Sequence[str], num_threads: int = ...
    ) -> list[int]:
        """Parallel token counts recognizing registered specials, zero output buffer."""

    def encode_ordinary_batch(
        self, texts: Sequence[str | bytes], num_threads: int = ...
    ) -> list[list[int]]:
        """Encode many texts in parallel (``num_threads=0`` = all cores)."""

    def count_tokens_batch(
        self,
        texts: Sequence[str | bytes],
        num_threads: int = ...,
        separator: bytes | None = ...,
    ) -> list[int]:
        """Token counts for many texts in parallel. With ``separator``, each text is
        first split into documents on it (like ``bytes.split``), inside Rust."""

    def encode_ordinary_to_buffer(
        self,
        texts: Sequence[str | bytes],
        num_threads: int = ...,
        separator: bytes | None = ...,
        dtype: Any = ...,
    ) -> dict[str, Any]:
        """Bulk-corpus fast path: encode many texts in parallel into a flat
        ``uint32``/``uint16`` numpy array plus ``int64`` offsets."""

    def encode_file_to_buffer(
        self,
        path: str | PathLike[str],
        num_threads: int = ...,
        separator: str | bytes | None = ...,
        dtype: Any = ...,
    ) -> dict[str, Any]:
        """Zero-copy mmap file tokenization into ``{"data", "offsets"}`` numpy arrays."""

    @overload
    def count_tokens_file(
        self,
        path: str | PathLike[str],
        num_threads: int = ...,
        separator: None = ...,
    ) -> int: ...
    @overload
    def count_tokens_file(
        self,
        path: str | PathLike[str],
        num_threads: int = ...,
        separator: str | bytes = ...,
    ) -> list[int]: ...
    def count_tokens_file(
        self,
        path: str | PathLike[str],
        num_threads: int = ...,
        separator: str | bytes | None = ...,
    ) -> int | list[int]:
        """Zero-copy mmap file token counting with zero output token buffer."""

    def encode_batch(
        self,
        texts: Sequence[str],
        max_length: int | None = ...,
        pad_id: int = ...,
        num_threads: int = ...,
    ) -> dict[str, Any]:
        """Encode into a fixed-width padded batch (``input_ids``, ``attention_mask``)."""

    def truncate(
        self,
        text: str,
        max_tokens: int,
        side: Literal["right", "left"] = ...,
        special: bool = ...,
    ) -> str:
        """Truncate ``text`` to at most ``max_tokens`` tokens without Python list boxing."""

    def chunk_text(
        self,
        text: str,
        max_tokens: int,
        overlap: int = ...,
        special: bool = ...,
    ) -> list[str]:
        """Split ``text`` into consecutive chunks of at most ``max_tokens`` tokens
        with optional ``overlap`` shared tokens between adjacent chunks."""

    # --- decode ---
    def decode(
        self,
        tokens: Any,
        skip_special_tokens: bool = ...,
        decode_with_timestamps: bool = ...,
        time_precision: float = ...,
        segment_size: int = ...,
    ) -> str:
        """Decode token ids (list, numpy array, or CPU tensor) back to text."""

    def decode_array(self, tokens: Any) -> str:
        """Decode from a C-contiguous uint32 buffer (numpy/array.array/tensor)."""

    def decode_batch(
        self,
        sequences: Any,
        num_threads: int = ...,
        skip_special_tokens: bool = ...,
        decode_with_timestamps: bool = ...,
        time_precision: float = ...,
        segment_size: int = ...,
    ) -> list[str]:
        """Decode many token sequences (list of lists or 2D array/tensor) in parallel."""

    def decode_stream(self, skip_special_tokens: bool = ...) -> DecodeStream:
        """Create an incremental UTF-8-aware streaming decoder."""

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class AutoTokenizer:
    """HuggingFace-compatible tokenizer wrapper over tuetoken's Rust core."""

    is_fast: bool
    bos_token: str | None
    eos_token: str | None
    unk_token: str | None
    pad_token: str | None
    sep_token: str | None
    cls_token: str | None
    mask_token: str | None
    chat_template: str | None
    add_bos_token: bool | None
    add_eos_token: bool | None
    model_max_length: int | None
    padding_side: str
    truncation_side: str

    def __init__(
        self,
        tokenizer_json: str,
        config: Mapping[str, Any] | None = ...,
        warmup: Any = ...,
        language: str | None = ...,
        task: str | None = ...,
        predict_timestamps: bool | None = ...,
        normalizer_json: str | None = ...,
    ) -> None: ...

    @classmethod
    def from_file(
        cls,
        tokenizer_json: str,
        config: str | Mapping[str, Any] | None = ...,
        warmup: Any = ...,
        language: str | None = ...,
        task: str | None = ...,
        predict_timestamps: bool | None = ...,
        normalizer_json: str | None = ...,
    ) -> "AutoTokenizer": ...

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str | PathLike[str],
        revision: str | None = ...,
        warmup: Any = ...,
        language: str | None = ...,
        task: str | None = ...,
        predict_timestamps: bool | None = ...,
        local_files_only: bool = ...,
        cache_dir: str | None = ...,
        token: str | bool | None = ...,
        **kwargs: Any,
    ) -> "AutoTokenizer": ...

    @classmethod
    def from_tiktoken_model(
        cls,
        source: str,
        pattern: str,
        special_tokens: Mapping[str, int] | None = ...,
        revision: str | None = ...,
        warmup: Any = ...,
    ) -> "AutoTokenizer": ...

    @classmethod
    def from_tekken(
        cls,
        source: str,
        revision: str | None = ...,
        warmup: Any = ...,
    ) -> "AutoTokenizer": ...

    def warmup(self, texts: Any = ..., num_threads: int = ...) -> int: ...

    @property
    def vocab_size(self) -> int: ...
    @property
    def bos_token_id(self) -> int | None: ...
    @property
    def eos_token_id(self) -> int | None: ...
    @property
    def pad_token_id(self) -> int | None: ...
    @property
    def unk_token_id(self) -> int | None: ...
    @property
    def all_special_ids(self) -> list[int]: ...
    @property
    def all_special_tokens(self) -> list[str]: ...
    @property
    def prefix_tokens(self) -> list[int]: ...

    def get_vocab(self) -> dict[str, int]: ...
    def get_added_vocab(self) -> dict[str, int]: ...
    def convert_ids_to_tokens(
        self, ids: int | Sequence[int], skip_special_tokens: bool = ...
    ) -> str | None | list[str | None]: ...
    def convert_tokens_to_ids(
        self, tokens: str | Sequence[str]
    ) -> int | None | list[int | None]: ...
    def convert_tokens_to_string(self, tokens: Sequence[str]) -> str: ...
    def tokenize(self, text: str, add_special_tokens: bool = ...) -> list[str]: ...

    def encode(
        self,
        text: str,
        text_pair: str | None = ...,
        add_special_tokens: bool = ...,
    ) -> list[int]: ...

    def __call__(
        self,
        text: str | Sequence[str],
        text_pair: str | Sequence[str] | None = ...,
        add_special_tokens: bool = ...,
        padding: bool | str = ...,
        truncation: bool | str = ...,
        max_length: int | None = ...,
        return_tensors: Literal["np", "pt"] | None = ...,
        return_attention_mask: bool = ...,
        return_token_type_ids: bool = ...,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    def decode(
        self,
        ids: Any,
        skip_special_tokens: bool = ...,
        decode_with_timestamps: bool = ...,
        output_offsets: bool = ...,
        time_precision: float = ...,
        segment_size: int = ...,
        normalize: bool = ...,
        basic_normalize: bool = ...,
        remove_diacritics: bool = ...,
        **kwargs: Any,
    ) -> str | dict[str, Any]: ...

    def batch_decode(
        self,
        sequences: Any,
        skip_special_tokens: bool = ...,
        decode_with_timestamps: bool = ...,
        output_offsets: bool = ...,
        time_precision: float = ...,
        segment_size: int = ...,
        normalize: bool = ...,
        basic_normalize: bool = ...,
        remove_diacritics: bool = ...,
        **kwargs: Any,
    ) -> list[str] | list[dict[str, Any]]: ...

    def decode_stream(self, skip_special_tokens: bool = ...) -> DecodeStream: ...

    def count_tokens(self, text: str, add_special_tokens: bool = ...) -> int: ...
    def count_tokens_batch(
        self,
        texts: Sequence[str],
        add_special_tokens: bool = ...,
        num_threads: int = ...,
    ) -> list[int]: ...

    def encode_file_to_buffer(
        self,
        path: str | PathLike[str],
        num_threads: int = ...,
        separator: str | bytes | None = ...,
        dtype: Any = ...,
    ) -> dict[str, Any]: ...

    @overload
    def count_tokens_file(
        self,
        path: str | PathLike[str],
        num_threads: int = ...,
        separator: None = ...,
    ) -> int: ...
    @overload
    def count_tokens_file(
        self,
        path: str | PathLike[str],
        num_threads: int = ...,
        separator: str | bytes = ...,
    ) -> list[int]: ...
    def count_tokens_file(
        self,
        path: str | PathLike[str],
        num_threads: int = ...,
        separator: str | bytes | None = ...,
    ) -> int | list[int]: ...

    def truncate(
        self,
        text: str,
        max_tokens: int,
        side: Literal["right", "left"] = ...,
        add_special_tokens: bool = ...,
    ) -> str: ...

    def chunk_text(
        self,
        text: str,
        max_tokens: int,
        overlap: int = ...,
        add_special_tokens: bool = ...,
    ) -> list[str]: ...

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]] | Sequence[Sequence[Mapping[str, Any]]],
        tools: Sequence[Mapping[str, Any]] | None = ...,
        add_generation_prompt: bool = ...,
        continue_final_message: bool = ...,
        tokenize: bool = ...,
        padding: bool | str = ...,
        truncation: bool | str = ...,
        max_length: int | None = ...,
        return_tensors: Literal["np", "pt"] | None = ...,
        return_dict: bool = ...,
        chat_template: str | None = ...,
        **template_kwargs: Any,
    ) -> Any: ...

    # --- Whisper helpers ---
    def set_prefix_tokens(
        self,
        language: str | None = ...,
        task: str | None = ...,
        predict_timestamps: bool | None = ...,
    ) -> None: ...
    def get_decoder_prompt_ids(
        self,
        task: str | None = ...,
        language: str | None = ...,
        no_timestamps: bool = ...,
    ) -> list[tuple[int, int]]: ...
    def get_prompt_ids(self, text: str, return_tensors: str = ...) -> Any: ...
    def timestamp_ids(self, time_precision: float = ...) -> dict[float, int]: ...
    def normalize(self, text: str) -> str: ...
    def basic_normalize(self, text: str, remove_diacritics: bool = ...) -> str: ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

__all__: list[str]
