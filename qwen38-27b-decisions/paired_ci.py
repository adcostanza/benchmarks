#!/usr/bin/env python3
"""Paired agreement gaps between configs, with 95% bootstrap intervals over cases.

Per case, the number of its 5 decisions whose argmax matches the gold argmax (the dataset
scorer's score_one, via ../scripts/decisions/run.py), for pass 1 of each config and, for the
NVFP4 configs, after the T+B calibration in results/<config>/calibration.json. A gap is the
mean per-decision difference over the same 400 cases; the interval resamples cases (10,000
resamples, seed 0), since the 5 decisions of one case are not independent.

Usage (host, from this folder):
  uv run --no-project --with pyarrow --with numpy --with scipy paired_ci.py \
      "$DATASETS_DIR/typed-decisions/all/test-00000-of-00001.parquet" > results/paired-agreement.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "scripts" / "decisions"))
import calibrate  # noqa: E402
from run import score_one  # noqa: E402

PAIRS = [
    ("stock-bf16", "stock-fp8"), ("stock-bf16", "stock-nvfp4"),
    ("swift-bf16", "swift-fp8"), ("swift-bf16", "swift-nvfp4"),
    ("stock-bf16", "swift-bf16"), ("stock-fp8", "swift-fp8"), ("stock-nvfp4", "swift-nvfp4"),
    ("stock-nvfp4", "stock-nvfp4 T+B"), ("swift-nvfp4", "swift-nvfp4 T+B"),
    ("stock-nvfp4 T+B", "swift-nvfp4 T+B"),
]


def raw_hits(answers, gold):
    out = {}
    for line in open(answers):
        r = json.loads(line)
        out[r["id"]] = sum(score_one(r["answers"][q], gq)[0] for q, gq in gold[r["id"]].items())
    return out


def tb_hits(cfg, answers, data, gold):
    p = json.load(open(here / "results" / cfg / "calibration.json"))["temperature_and_bias"]
    # calibrate.load flattens decisions in file order, case by case, gold question order
    ids = [json.loads(line)["id"] for line in open(answers)]
    case_of = [i for i in ids for _ in gold[i]]
    out = dict.fromkeys(ids, 0)
    for i, it in zip(case_of, calibrate.load(answers, data)):
        q = calibrate.apply(it, p["temperatures"], p["bias"])
        out[i] += int(np.argmax(q) == np.argmax(it["g"]))
    return out


def main():
    data = sys.argv[1]
    gold = {r["id"]: json.loads(r["gold"]) for r in pq.read_table(data).to_pylist()}
    hits = {}
    for a, b in PAIRS:
        for c in (a, b):
            if c in hits:
                continue
            cfg = c.removesuffix(" T+B")
            answers = here / "results" / cfg / "decisions" / "pass1.jsonl"
            hits[c] = tb_hits(cfg, answers, data, gold) if c.endswith("T+B") else raw_hits(answers, gold)
    ids = sorted(gold)
    resample = np.random.default_rng(0).integers(0, len(ids), (10_000, len(ids)))
    print("| A | B | agreement A | agreement B | B − A | 95% CI |")
    print("|---|---|---|---|---|---|")
    for a, b in PAIRS:
        d = np.array([hits[b][i] - hits[a][i] for i in ids]) / 5
        lo, hi = np.percentile(d[resample].mean(1), [2.5, 97.5])
        acc = lambda c: sum(hits[c].values()) / (5 * len(ids))  # noqa: E731
        print(f"| `{a}` | `{b}` | {acc(a):.4f} | {acc(b):.4f} | {d.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] |")


if __name__ == "__main__":
    main()
