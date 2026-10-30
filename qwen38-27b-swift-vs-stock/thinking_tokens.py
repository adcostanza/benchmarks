#!/usr/bin/env python3
"""Thinking vs answer tokens, stock vs Swift, from the saved responses in results/.

The step summaries only have completion_tokens (thinking + answer). This splits them: each
response's reasoning text (`reasoning_content` in GSM8K, `reasoning` in SPEED-Bench and the card
sets) is tokenized with the model's tokenizer.json (both models share Qwen3.8's tokenizer). Only
prompts where both models stopped on their own (finish_reason "stop") are counted, so answers cut
off at the token limit don't skew the means; the n column says how many that leaves.

Usage: thinking_tokens.py <tokenizer.json>   (pip install tokenizers)
"""

import json
import statistics
import sys
from pathlib import Path

from tokenizers import Tokenizer

RESULTS = Path(__file__).resolve().parent / "results"
SETS = ["gsm8k", "math500", "humaneval", "mbpp", "mt-bench"]
ROWS = [("GSM8K, all 1,319", "medium", "nvfp4-quality", "gsm8k/daily.jsonl"),
        ("SPEED-Bench qualitative", "medium", "dflash8", "speed-bench/requests_qualitative_c1.jsonl")]
ROWS += [(s, effort, "dflash8", f"accept-sets/requests_{proto}_{s}_c1.jsonl")
         for proto, effort in (("ref", "xhigh"), ("daily", "medium")) for s in SETS]
DIRS = {"nvfp4-quality": ("stock-nvfp4-quality", "swift-nvfp4-quality"),
        "dflash8": ("dflash8", "swift-dflash8")}


def load(path, tok):
    out = {}
    for line in path.open():
        d = json.loads(line)
        key = (d.get("idx", d.get("id")), d.get("turn", 0))
        reasoning = d.get("reasoning_content") or d.get("reasoning") or ""
        out[key] = (len(tok.encode(reasoning).ids), d["completion_tokens"], d["finish_reason"])
    return out


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    print("| Set | Effort | n (both stopped) | Stock thinking | Swift thinking | Δ | Stock total | Swift total | Δ |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, effort, pair, rel in ROWS:
        stock_dir, swift_dir = DIRS[pair]
        a, b = load(RESULTS / stock_dir / rel, tok), load(RESULTS / swift_dir / rel, tok)
        keys = [k for k in a if k in b and a[k][2] == "stop" and b[k][2] == "stop"]
        mean = lambda d, i: statistics.mean(d[k][i] for k in keys)
        ta, tb, ca, cb = mean(a, 0), mean(b, 0), mean(a, 1), mean(b, 1)
        print(f"| {name} | {effort} | {len(keys)} of {len(a)} | {ta:,.0f} | {tb:,.0f} | {(tb / ta - 1) * 100:+.0f}% "
              f"| {ca:,.0f} | {cb:,.0f} | {(cb / ca - 1) * 100:+.0f}% |")


if __name__ == "__main__":
    main()
