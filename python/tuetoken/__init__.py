"""tuetoken — a fast Rust/PyO3 BPE tokenizer.

Loads any HuggingFace ``tokenizer.json`` (ByteLevel + byte_fallback/SentencePiece)
and beats tiktoken/HF on encode and decode, with an O(n) streaming merger that
keeps adversarial (no-merge) inputs from blowing up.

Two entry points:

  * `Tokenizer` — the fast core: ordinary (special-token-unaware) encode/decode.

        from tuetoken import Tokenizer
        tok = Tokenizer.from_tiktoken("cl100k_base")        # or from_pretrained(repo)
        ids = tok.encode_ordinary("hello world")            # <|...|> -> literal bytes

  * `AutoTokenizer` — a HF-compatible wrapper over that core: special-token-aware
    `encode`, `__call__` (padding/truncation/return_tensors), `decode`,
    `apply_chat_template`. Drop-in for `transformers.AutoTokenizer` on LLMs.

        from tuetoken import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        ids = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                      add_generation_prompt=True)
"""
from ._core import Tokenizer, __version__
from ._auto import AutoTokenizer

__all__ = ["Tokenizer", "AutoTokenizer", "__version__"]
