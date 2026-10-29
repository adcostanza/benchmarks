#!/usr/bin/env python3
"""Side-by-side table of bench-serving and BFCL results for two or more result dirs.

Each dir holds bench-serving.jsonl and bfcl/{score,result} as written by
scripts/bench-serving.sh and scripts/bfcl.sh. The delta column is second vs first.

Usage: scripts/compare.py <dir-a> <dir-b> [<dir-c> ...]
"""

import glob
import json
import statistics
import sys
from pathlib import Path

SERVING_METRICS = [
    ("output_throughput", "out tok/s", 1),
    ("median_tpot_ms", "TPOT ms (p50)", 2),
    ("median_ttft_ms", "TTFT ms (p50)", 0),
    ("input_throughput", "in tok/s", 0),
    ("accept_length", "accept len", 2),
]


def serving(path):
    rows = {}
    for line in open(path / "bench-serving.jsonl"):
        d = json.loads(line)
        rows[d["tag"]] = d
    return rows


def flat(x):
    return [y for i in x for y in flat(i)] if isinstance(x, list) else [x]


def bfcl(path):
    acc, tokens, latency = {}, [], []
    for f in glob.glob(str(path / "bfcl/score/*/**/*_score.json"), recursive=True):
        head = json.loads(open(f).readline())
        cat = Path(f).name.removeprefix("BFCL_v4_").removesuffix("_score.json")
        acc[cat] = (head["correct_count"], head["total_count"])
    for f in glob.glob(str(path / "bfcl/result/**/*.json"), recursive=True):
        for line in open(f):
            d = json.loads(line)
            tokens += [float(t) for t in flat(d.get("output_token_count", [])) if t not in ("", None)]
            latency += [float(t) for t in flat(d.get("latency", [])) if t not in ("", None)]
    return acc, tokens, latency


def table(header, rows):
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    print(fmt(header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(fmt(r))
    print()


def main(dirs):
    paths = [Path(d) for d in dirs]
    labels = [p.name for p in paths]
    serv = [serving(p) for p in paths]
    rows = []
    for tag in serv[0]:
        for key, name, nd in SERVING_METRICS:
            vals = [s.get(tag, {}).get(key) for s in serv]
            if all(v is None for v in vals):
                continue
            cells = [f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-" for v in vals]
            delta = ""
            if all(isinstance(v, (int, float)) and v for v in vals[:2]):
                delta = f"{(vals[1] / vals[0] - 1) * 100:+.1f}%"
            rows.append([tag, name, *cells, delta])
    print("## Serving (reasoning_effort=medium)\n")
    table(["scenario", "metric", *labels, f"{labels[1]} vs {labels[0]}"], rows)

    res = [bfcl(p) for p in paths]
    if not any(r[0] for r in res):
        return
    cats = sorted(set().union(*(r[0] for r in res)))
    rows, tot = [], [[0, 0] for _ in paths]
    for c in cats:
        cells = []
        for i, r in enumerate(res):
            ok, n = r[0].get(c, (0, 0))
            tot[i][0] += ok
            tot[i][1] += n
            cells.append(f"{ok}/{n} ({ok / n * 100:.0f}%)" if n else "-")
        rows.append([c, *cells])
    rows.append(["**total**", *[f"{ok}/{n} ({ok / n * 100:.1f}%)" if n else "-" for ok, n in tot]])
    for name, idx in (("mean out tokens / call", 1), ("mean latency s / call", 2)):
        rows.append([name, *[f"{statistics.mean(r[idx]):.2f}" if r[idx] else "-" for r in res]])
    rows.append(["calls hitting 4096 cap", *[str(sum(t >= 4096 for t in r[1])) for r in res]])
    print("## BFCL v4 subset\n")
    table(["category", *labels], rows)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
