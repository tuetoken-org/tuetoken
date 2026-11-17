from typing import Any, Sequence

__version__: str

class Tokenizer:
    """A fast BPE tokenizer loaded from a HuggingFace ``tokenizer.json``."""

    n_vocab: int

    def __init__(self, path: str) -> None:
        """Load from a ``tokenizer.json`` file path."""

    @staticmethod
    def from_pretrained(repo_id: str, revision: str | None = ...) -> "Tokenizer":
        """Download ``tokenizer.json`` from a HuggingFace repo (needs ``huggingface_hub``)."""

    @staticmethod
    def from_tiktoken(name: str) -> "Tokenizer":
        """Load an OpenAI tiktoken encoding by name, e.g. ``"cl100k_base"``."""

    # --- encode ---
    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text to token ids (special tokens in text are literal bytes)."""

    def encode_to_bytes(self, text: str) -> bytes:
        """Encode to a raw native-endian uint32 buffer (``numpy.frombuffer(..., uint32)``)."""

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode and return ``(ids, offsets)`` where each offset is the token's
        ``(start, end)`` byte span in ``text``. ByteLevel models only."""

    def count_tokens(self, text: str) -> int:
        """Number of tokens, without building the id list."""

    def encode_ordinary_batch(self, texts: Sequence[str | bytes], num_threads: int = ...) -> list[list[int]]:
        """Encode many texts in parallel (``num_threads=0`` = all cores)."""

    def count_tokens_batch(
        self, texts: Sequence[str | bytes], num_threads: int = ..., separator: bytes | None = ...
    ) -> list[int]:
        """Token counts for many texts in parallel. With ``separator``, each text is
        first split into documents on it (like ``bytes.split``), inside Rust."""

    def encode_ordinary_to_buffer(
        self, texts: Sequence[str | bytes], num_threads: int = ..., separator: bytes | None = ...
    ) -> dict[str, Any]:
        """Bulk-corpus fast path: encode many texts in parallel and return all ids
        as one flat ``uint32`` numpy array plus an ``int64`` offsets array (length
        rows+1). Document ``i``'s tokens are ``data[offsets[i]:offsets[i+1]]``.
        Avoids the per-token Python boxing of ``encode_ordinary_batch`` — use it to
        tokenize a whole corpus offline. Returns ``{"data", "offsets"}``. Needs numpy.
        With ``separator`` (e.g. ``b"<|endoftext|>"``), each text is first split into
        documents on it exactly like ``bytes.split`` (separators dropped, one row per
        piece), in parallel inside Rust — no Python list of documents needed."""

    def encode_batch(
        self,
        texts: Sequence[str],
        max_length: int | None = ...,
        pad_id: int = ...,
        num_threads: int = ...,
    ) -> dict[str, Any]:
        """Encode into a fixed-width padded batch for training. Returns a dict with
        ``input_ids`` (uint32 numpy array, shape ``[rows, width]``) and
        ``attention_mask`` (uint8). ``max_length`` sets the width (truncating longer
        sequences); when ``None`` the width is the longest sequence in the batch.
        Requires numpy."""

    # --- decode ---
    def decode(self, tokens: Sequence[int]) -> str:
        """Decode token ids back to text (lossy on invalid UTF-8)."""

    def decode_array(self, tokens: Any) -> str:
        """Decode from a C-contiguous uint32 buffer (numpy/array.array/tensor) — fastest."""

    def decode_batch(self, sequences: Sequence[Sequence[int]], num_threads: int = ...) -> list[str]:
        """Decode many token sequences in parallel."""

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
