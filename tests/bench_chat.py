#!/usr/bin/env python
"""tuetoken.AutoTokenizer vs transformers.AutoTokenizer on chat formats.

Checks CORRECTNESS (token-exact) and SPEED for:
  * plain      — short system/user/assistant multi-turn
  * long       — a big multi-turn transcript (code + multilingual)
  * agentic    — tools + assistant tool_calls + tool responses (function calling)
Speed is measured two ways: apply_chat_template(tokenize=True) end-to-end
(Jinja render + encode), and __call__ on a batch of rendered chat strings
(encode only — where the Rust core does its work). best-of-N (min wall time).
"""
import warnings, time, json
warnings.simplefilter("ignore")
import tuetoken
from transformers import AutoTokenizer as HF

MODELS = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3",
          "NousResearch/Hermes-3-Llama-3.1-8B", "microsoft/Phi-3.5-mini-instruct",
          "deepseek-ai/DeepSeek-V3-0324"]

PLAIN = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the Doppler effect simply."},
    {"role": "assistant", "content": "As a source moves toward you the waves bunch up, so the pitch rises; moving away, it drops."},
    {"role": "user", "content": "Does it work for light too? 中文也行吗?"},
]

# a long transcript: many turns with code, prose, multilingual, numbers
_turn = [
    {"role": "user", "content": "Refactor get_user() in api/users.py:42 and add a 404 test. Café ☕ 你好."},
    {"role": "assistant", "content": "Plan:\n1. extract `_validate(uid)`\n2. add tests/test_users.py::test_404\n```python\ndef get_user(uid):\n    u = db.query(uid)\n    if u is None:\n        raise HTTPException(404)\n    return u\n```\nTests: 12 passed in 0.43s. Стоимость $1,234.56."},
]
LONG = [{"role": "system", "content": "You are an autonomous coding agent."}] + _turn * 30

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name."},
            "unit": {"type": "string", "description": "celsius or fahrenheit."}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Search the web.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query."}}, "required": ["query"]}}},
]
AGENTIC = [
    {"role": "system", "content": "You are an agent with tools. Think step by step."},
    {"role": "user", "content": "What's the weather in Paris, and who won the 2022 World Cup?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}]},
    {"role": "tool", "name": "get_weather", "content": "{\"temp\": 15, \"sky\": \"rainy\"}"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "web_search", "arguments": {"query": "2022 World Cup winner"}}}]},
    {"role": "tool", "name": "web_search", "content": "Argentina won the 2022 FIFA World Cup."},
    {"role": "assistant", "content": "Paris is 15°C and rainy ☔. And Argentina 🇦🇷 won the 2022 World Cup."},
    {"role": "user", "content": "thanks!"},
]

CONVS = [("plain", PLAIN, None), ("long", LONG, None), ("agentic", AGENTIC, TOOLS)]


def best(fn, r):
    ts = []
    for _ in range(r):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    print(f"{'model':34} {'conv':8} {'correct':8} {'render+enc (tue/HF)':>22} {'spd':>5} {'encode-only':>16} {'spd':>5}")
    print("-" * 104)
    for repo in MODELS:
        try:
            hf = HF.from_pretrained(repo); tt = tuetoken.AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            print(f"{repo:34} SKIP {type(e).__name__}"); continue
        for name, conv, tools in CONVS:
            kw = {"tokenize": True, "add_generation_prompt": True}
            if tools:
                kw["tools"] = tools
            try:
                h_ids = hf.apply_chat_template(conv, **kw)
            except Exception as e:
                print(f"{repo:34} {name:8} HF-template-error ({type(e).__name__}); skipping"); continue
            try:
                t_ids = tt.apply_chat_template(conv, **kw)
            except Exception as e:
                print(f"{repo:34} {name:8} tuetoken ERROR {type(e).__name__}: {str(e)[:40]}"); continue
            correct = (list(h_ids) == list(t_ids))
            # also confirm the rendered STRING matches (isolates template vs encode)
            rk = dict(kw); rk["tokenize"] = False
            str_ok = (hf.apply_chat_template(conv, **rk) == tt.apply_chat_template(conv, **rk))
            ntok = len(h_ids)
            reps = max(20, 200000 // max(ntok, 1))
            th = best(lambda: hf.apply_chat_template(conv, **kw), reps)
            ttt = best(lambda: tt.apply_chat_template(conv, **kw), reps)
            # encode-only: render once, then time __call__ on a batch of copies
            rendered = tt.apply_chat_template(conv, **rk)
            batch = [rendered] * 64
            te_h = best(lambda: hf(batch, add_special_tokens=False)["input_ids"], 20)
            te_t = best(lambda: tt(batch, add_special_tokens=False)["input_ids"], 20)
            mark = "OK" if (correct and str_ok) else ("STR!!" if not str_ok else "IDS!!")
            print(f"{repo:34} {name:8} {mark:8} {ttt*1e3:7.2f}/{th*1e3:6.2f}ms{'':3}{th/ttt:4.1f}x "
                  f"{te_t*1e3:6.2f}/{te_h*1e3:6.2f}ms {te_h/te_t:4.1f}x")
        print()
    print("render+enc = full apply_chat_template(tokenize=True); encode-only = __call__ on 64 rendered strings.")
    print("correct: OK = ids AND rendered string match transformers; IDS!!/STR!! = mismatch.")


if __name__ == "__main__":
    main()
