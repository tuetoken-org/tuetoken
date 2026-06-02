# tuetoken

**The fastest tokenizer for modern LLMs, up to 30x faster.**

tuetoken is a BPE tokenizer with a fast, safe Rust core. It is a drop-in replacement
for 🤗 `transformers.AutoTokenizer`: it loads any model's own `tokenizer.json` and
reproduces tokenization exactly (special tokens, chat templates, padding/truncation),
up to 30x faster. It also loads OpenAI/tiktoken encodings natively, and
its O(n) merger stays fast even on adversarial inputs (hashes, base64, minified code)
where other tokenizers degrade to O(n²).

```python
from tuetoken import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
out = tok.apply_chat_template(messages, add_generation_prompt=True)   # {"input_ids", "attention_mask"}
```

Detection is 100% config-driven (the model's `tokenizer.json`, never its name), so
the same code works across families: Llama, Qwen, Mistral/Mixtral, DeepSeek, Gemma,
Phi, GPT-OSS, GLM, Kimi, and more.

## Install

```bash
pip install tuetoken
```
Build from source (development, or a platform with no prebuilt wheel):

```bash
git clone https://github.com/tuetoken-org/tuetoken && cd tuetoken
pip install maturin
maturin develop --release
```

## Performances

![Performances](https://raw.githubusercontent.com/tuetoken-org/tuetoken/main/bench_tokenizers.png)

## Drop-in `AutoTokenizer`

The full 🤗 API, byte-exact with `transformers.AutoTokenizer`:

```python
from tuetoken import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

tok.encode("Hello <|eot_id|> world")                 # special-token aware -> list[int]
tok.decode(ids, skip_special_tokens=True)            # -> str
tok(texts, padding=True, truncation=True,            # batch dict: input_ids + attention_mask
    max_length=512, return_tensors="np")             #   (also "pt" for torch)
tok.apply_chat_template(messages, add_generation_prompt=True)  # -> {input_ids, attention_mask}
tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)  # -> list[int]
tok.batch_decode(...) ; tok.convert_ids_to_tokens(...) ; tok.tokenize(...)
tok.bos_token_id ; tok.eos_token ; tok.pad_token_id ; tok.vocab_size
```

This matches `transformers.AutoTokenizer` token-for-token across byte-level models
(Llama, Qwen, DeepSeek, …) and SentencePiece models (Mistral, Phi-3, CodeLlama, …).

## OpenAI / tiktoken encodings

tuetoken loads OpenAI's encodings natively, with no `tiktoken` dependency, and is
faster than tiktoken itself, by up to an order of magnitude on long inputs:

```python
from tuetoken import Tokenizer
enc = Tokenizer.from_tiktoken("cl100k_base")   # also "o200k_base", "gpt2", ...
enc.encode_ordinary("Hello world")             # list[int]
```

## Lower-level core

`Tokenizer` is the raw BPE engine (no special tokens or chat templates; that is what
`AutoTokenizer` is for). Reach for it when you only need fast token ids or counts:

```python
from tuetoken import Tokenizer
enc = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")   # or Tokenizer("tokenizer.json")

enc.encode_ordinary("Hello world")                # list[int]
enc.encode_ordinary_batch(texts, num_threads=0)   # parallel (0 = all cores), GIL released
enc.decode(ids) ; enc.count_tokens("Hi") ; len(enc)
```

For ML pipelines there are zero-copy numpy paths, a padded training-batch helper, and
byte-span offsets:

```python
import numpy as np
arr   = np.frombuffer(enc.encode_to_bytes(text), dtype=np.uint32)   # skip per-token boxing
text  = enc.decode_array(arr)
batch = enc.encode_batch(texts, max_length=512, pad_id=0)           # input_ids + attention_mask
ids, offsets = enc.encode_with_offsets("Hello café")                # byte spans, ByteLevel only
```

## Coverage

tuetoken works with essentially every modern LLM tokenizer: ByteLevel BPE (Llama,
Qwen, DeepSeek, Mistral/Mixtral, GPT-OSS, GLM, Phi, OLMo, Yi, …), SentencePiece
(Llama-2, Mistral, Phi-3, CodeLlama, Gemma, …), and OpenAI/tiktoken encodings. We
extend coverage constantly; if a tokenizer you need isn't supported yet, please open
an issue.

Anything tuetoken can't reproduce exactly **fails closed** (raises at load) rather
than mistokenizing, so you never get silently wrong tokens.

## Linear-time on any input

Classic BPE is O(n²) per chunk and collapses on long, poorly-merging content (random
identifiers, hashes, base64, minified code), tiktoken included. tuetoken's merger is
O(n), so adversarial inputs that hang other tokenizers for minutes stay in the
millisecond range, byte-identical.

## Correctness

Every claim above is validated byte-exact against the reference tokenizers
(`transformers`, `tokenizers`, `tiktoken`) on a large, adversarial corpus. That is the
only reason those libraries appear in the **test** dependencies; they are **not**
runtime dependencies of tuetoken.
