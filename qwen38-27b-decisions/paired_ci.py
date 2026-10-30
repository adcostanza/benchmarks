#!/usr/bin/env python3
"""Paired agreement gaps between configs, with 95% bootstrap intervals over cases.

Per case, the share of its 5 decisions whose argmax matches the gold argmax (the dataset
scorer's score_one, in ../scripts/decisions/sets.py), for pass 1 of each config and, for the
NVFP4 configs, after the T+B calibration in results/<config>/calibration.json. A gap is the
mean per-decision difference over the same 400 cases; the interval resamples cases (10,000
resamples, seed 0), since the 5 decisions of one case are not independent.

A second table counts, for the quantization pairs, the top answers that differ between A and B
(pass 1, raw), whatever the gold says: similar agreement rates can hide different answers.

Usage (host, from this folder):
  uv run --no-project --with pyarrow --with numpy --with scipy paired_ci.py "$DATASETS_DIR" \
      > results/paired-agreement.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "scripts" / "decisions"))
import calibrate  # noqa: E402
import sets  # noqa: E402

TD = sets.TypedDecisions

PAIRS = [
    ("stock-bf16", "stock-fp8"), ("stock-bf16", "stock-nvfp4"),
    ("swift-bf16", "swift-fp8"), ("swift-bf16", "swift-nvfp4"),
    ("stock-bf16", "swift-bf16"), ("stock-fp8", "swift-fp8"), ("stock-nvfp4", "swift-nvfp4"),
    ("stock-nvfp4", "stock-nvfp4 T+B"), ("swift-nvfp4", "swift-nvfp4 T+B"),
    ("stock-nvfp4 T+B", "swift-nvfp4 T+B"),
]


def hits(cfg, cases, calibrated):
    rows = calibrate.read_rows(here / "results" / cfg / "decisions" / "pass1.jsonl")
    if calibrated:
        p = json.load(open(here / "results" / cfg / "calibration.json"))["temperature_and_bias"]
        rows = calibrate.calibrated_rows(TD, rows, cases, lambda cid: (p["temperatures"], p["bias"]))
    return TD.case_scores(rows, cases)


def top_answers(cfg):
    out = {}
    for line in open(here / "results" / cfg / "decisions" / "pass1.jsonl"):
        rec = json.loads(line)
        for name, a in (rec.get("answers") or {}).items():
            if probs := a.get("probabilities"):
                out[(rec["id"], name)] = max(probs, key=probs.get)
    return out


def main():
    cases = TD.load(sys.argv[1], "test")
    scores = {}
    for a, b in PAIRS:
        for c in (a, b):
            if c not in scores:
                scores[c] = hits(c.removesuffix(" T+B"), cases, c.endswith("T+B"))
    ids = sorted(c["id"] for c in cases)
    resample = np.random.default_rng(0).integers(0, len(ids), (10_000, len(ids)))
    print("| A | B | agreement A | agreement B | B − A | 95% CI |")
    print("|---|---|---|---|---|---|")
    for a, b in PAIRS:
        d = np.array([scores[b][i] - scores[a][i] for i in ids])
        lo, hi = np.percentile(d[resample].mean(1), [2.5, 97.5])
        acc = lambda c: np.mean([scores[c][i] for i in ids])  # noqa: E731
        print(f"| `{a}` | `{b}` | {acc(a):.4f} | {acc(b):.4f} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    print("\n| A | B | decisions | top answers that differ |")
    print("|---|---|---|---|")
    for a, b in PAIRS[:4]:
        ta, tb = top_answers(a), top_answers(b)
        keys = ta.keys() & tb.keys()
        diff = sum(ta[k] != tb[k] for k in keys)
        print(f"| `{a}` | `{b}` | {len(keys)} | {diff} ({diff / len(keys):.1%}) |")


if __name__ == "__main__":
    main()
