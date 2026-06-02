# tuetoken — tests & benchmarks

Self-contained Python tests and the benchmark/plot generator. **No dependency on
any other tuetoken package** — references are OpenAI `tiktoken` and HuggingFace
`tokenizers`/`transformers`. Run from anywhere.

## Files

| file | what it does | needs |
|---|---|---|
| `test_tuetoken.py` | Standalone suite: tiktoken parity (gpt2/cl100k/o200k), streaming==scan==tiktoken, decode round-trip, `encode_to_bytes`/`decode_array` buffer paths, an HF model sweep, unsupported-pretok rejection | `tiktoken`, `tokenizers`, `numpy`, `huggingface_hub`, network |
| `bench_three_way_rust.py` | Benchmark + 6-panel plot: tuetoken vs tiktoken vs HuggingFace → `bench_three_way_rust.png` | `matplotlib`, `tiktoken`, `tokenizers`; `--plot-only` re-renders from JSON |
| `_tiktoken_json.py` | Vendored helper: downloads OpenAI `.tiktoken` rank files and builds a `tokenizer.json` (so tests can load tiktoken encodings into the Rust tokenizer). No extra deps. | network |

The benchmark corpus is the optional `data/corpus.txt`; if absent it is generated
in-memory, so the bench runs with no external data.

## Running

```bash
pip install -e ".[test]"          # tiktoken, tokenizers, numpy
python tests/test_tuetoken.py
python tests/bench_three_way_rust.py --encoding o200k_base
```

The crate's own `cargo test` (run from the repo root) is fully offline — no
Python, data, or network — and gates CI.