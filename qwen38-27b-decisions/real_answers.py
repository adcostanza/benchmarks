#!/usr/bin/env python3
"""Tables for the real-answer sets (results-real/): every config raw and calibrated, paired gaps
with 95% bootstrap intervals, and for Typed Decision Bench the published systems' answers scored
on the same items.

Per set, the per-case score is the set's own (sets.py case_scores): Typed Decision Bench the
item's proper score x 100, procedural the case's proper score x 100, calibrated typed decisions
the hit (accuracy). A gap is the mean difference over the same cases; the interval resamples
cases within groups (task or source; 10,000 resamples, seed 0), as Typed Decision Bench's own
paired bootstrap stratifies by task.

Inputs: results-real/<config>/<set>-<split>/pass1.jsonl, the calibrated answers in
results-real/<config>/calibrated/<set>-<split>/{T,T+B}.jsonl (calibrate.py --rows-out).

Usage (host, from this folder):
  uv run --no-project --with pyarrow --with numpy --with scipy real_answers.py "$DATASETS_DIR" \
      > results-real/tables.md
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "scripts" / "decisions"))
import sets  # noqa: E402

CONFIGS = ["stock-nvfp4", "swift-nvfp4"]
FITS = ["raw", "T", "T+B"]
RUNS = [("typed-decision-bench", "test"), ("procedural-typed-decisions", "test"),
        ("calibrated-typed-decisions", "test"), ("calibrated-typed-decisions", "heldout")]
# Typed Decision Bench's published runs (results/<system>/ in the dataset repo)
TDB_SYSTEMS = ["jev-1.13.0", "jevfish-v4-r4b", "decider-2b-v8", "s1-scorer-4b-v2b", "openjev-0.1", "nanojev-0.6b"]
COLUMNS = {
    "typed-decision-bench": [("DecisionScore", "decision_score", ".2f"), ("accuracy", "accuracy", ".3f")],
    "procedural-typed-decisions": [("proper score", "proper_score", ".2f"), ("accuracy", "accuracy", ".3f")],
    "calibrated-typed-decisions": [("accuracy", "accuracy", ".3f"), ("log loss", "log_loss", ".3f"),
                                   ("Brier", "brier", ".3f"), ("ECE", "ece", ".3f")],
}


def read_rows(path):
    return [json.loads(line) for line in open(path)]


def rows_for(cfg, set_name, split, fit):
    if fit == "raw":
        return read_rows(here / "results-real" / cfg / f"{set_name}-{split}" / "pass1.jsonl")
    return read_rows(here / "results-real" / cfg / "calibrated" / f"{set_name}-{split}" / f"{fit}.jsonl")


def tdb_published(root, system):
    rows = []
    for f in sorted((Path(root) / "typed-decision-bench" / "results" / system).glob("*.jsonl")):
        rows.extend(json.loads(line) for line in open(f))
    return rows


def paired(scores, cases, a, b):
    by_group = defaultdict(list)
    for i, c in enumerate(cases):
        by_group[c["group"]].append(i)
    d = np.array([scores[b][c["id"]] - scores[a][c["id"]] for c in cases])
    rng = np.random.default_rng(0)
    means = np.zeros(10_000)
    for idx in by_group.values():
        idx = np.array(idx)
        means += d[idx][rng.integers(0, len(idx), (10_000, len(idx)))].sum(1)
    means /= len(cases)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean(), lo, hi


def main():
    root = sys.argv[1]
    for set_name, split in RUNS:
        ds = sets.get(set_name)
        cases = ds.load(root, split)
        cols = COLUMNS[set_name]
        print(f"## {set_name} `{split}` ({len(cases)} cases)\n")
        print("| Config | Fit | " + " | ".join(c[0] for c in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        scores = {}
        for cfg in CONFIGS:
            for fit in FITS:
                rows = rows_for(cfg, set_name, split, fit)
                m = ds.metrics(rows, cases)
                scores[f"{cfg} {fit}"] = ds.case_scores(rows, cases)
                print(f"| `{cfg}` | {fit} | " + " | ".join(format(m[k], f) for _, k, f in cols) + " |")
        if set_name == "typed-decision-bench":
            for system in TDB_SYSTEMS:
                rows = tdb_published(root, system)
                m = ds.metrics(rows, cases)
                scores[system] = ds.case_scores(rows, cases)
                print(f"| {system} (published) | — | " + " | ".join(format(m[k], f) for _, k, f in cols) + " |")
        pairs = [("stock-nvfp4 raw", "swift-nvfp4 raw"), ("stock-nvfp4 T+B", "swift-nvfp4 T+B")]
        pairs += [(f"{c} raw", f"{c} {fit}") for c in CONFIGS for fit in ("T", "T+B")]
        if set_name == "typed-decision-bench":
            pairs += [(s, f"{c} {fit}") for s in ("openjev-0.1", "decider-2b-v8", "jev-1.13.0")
                      for c in CONFIGS for fit in ("raw", "T+B")]
        unit = "accuracy" if set_name == "calibrated-typed-decisions" else "score"
        print(f"\nPaired gaps ({unit}, B − A, 95% CI resampling cases within groups):\n")
        print("| A | B | B − A | 95% CI |")
        print("|---|---|---|---|")
        for a, b in pairs:
            mean, lo, hi = paired(scores, cases, a, b)
            scale = 100 if set_name != "calibrated-typed-decisions" else 1
            fmt = ".2f" if scale == 100 else ".3f"
            print(f"| `{a}` | `{b}` | {mean * scale:+{fmt}} | [{lo * scale:+{fmt}}, {hi * scale:+{fmt}}] |")
        print()


if __name__ == "__main__":
    main()
