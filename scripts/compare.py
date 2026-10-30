#!/usr/bin/env python3
"""Side-by-side tables of suite results for two or more result dirs.

Each dir holds what the suite scripts write:
  bench-serving.jsonl                  serving table; delta column is second vs first
  bfcl/rep<i>/{score,result}           BFCL v4 subset, one dir per repeat (scripts/bfcl.sh); results
  bfcl/{score,result}                  from before repeats existed (a single run) are read too
  <benchmark>/summary.json             generic table, one row per <benchmark>/<protocol>/<metric>
                                       with "value ± stderr" (suite v1 contract); "-" where missing

BFCL: accuracy ± one binomial standard error sqrt(p(1-p)/n), and for each later dir the gap to
the first dir with "within 2 SE" when |gap| < 2 * sqrt(se_a^2 + se_b^2) (gaps under ~2
standard errors count as "no difference"). With several repeats the cells show the mean
over repeats and a "repeat SE" row shows std/sqrt(repeats); the gap test stays on the binomial SE.

Usage: scripts/compare.py <dir-a> <dir-b> [<dir-c> ...]
       scripts/compare.py --bfcl-summary <out>/bfcl --repeats N --limit L --model M --temperature T
           (called by scripts/bfcl.sh: writes <out>/bfcl/summary.json)
"""

import glob
import json
import math
import re
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
    f = path / "bench-serving.jsonl"
    if not f.exists():
        return rows
    for line in open(f):
        d = json.loads(line)
        rows[d["tag"]] = d
    return rows


def flat(x):
    return [y for i in x for y in flat(i)] if isinstance(x, list) else [x]


def bfcl_roots(path):
    """Project roots of the BFCL runs under <path>/bfcl: rep<i> dirs, else the old single run."""
    base = path / "bfcl"
    reps = sorted(
        (d for d in base.glob("rep*") if re.fullmatch(r"rep\d+", d.name) and (d / "score").is_dir()),
        key=lambda d: int(d.name[3:]),
    )
    if reps:
        return reps
    return [base] if (base / "score").is_dir() else []


def bfcl_one(root):
    acc, tokens, latency = {}, [], []
    for f in glob.glob(str(root / "score/*/**/*_score.json"), recursive=True):
        head = json.loads(open(f).readline())
        cat = Path(f).name.removeprefix("BFCL_v4_").removesuffix("_score.json")
        acc[cat] = (head["correct_count"], head["total_count"])
    for f in glob.glob(str(root / "result/**/*.json"), recursive=True):
        for line in open(f):
            d = json.loads(line)
            tokens += [float(t) for t in flat(d.get("output_token_count", [])) if t not in ("", None)]
            latency += [float(t) for t in flat(d.get("latency", [])) if t not in ("", None)]
    return acc, tokens, latency


def bfcl(path):
    """(per-repeat {category: (correct, total)}, pooled output tokens, pooled latencies)."""
    runs = [bfcl_one(r) for r in bfcl_roots(path)]
    return [r[0] for r in runs], [t for r in runs for t in r[1]], [t for r in runs for t in r[2]]


def binom_se(p, n):
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def bfcl_stats(reps):
    """Per-category and overall (mean accuracy, n, mean correct, binomial SE, repeat SE or None)."""
    cats = sorted(set().union(*reps)) if reps else []
    out = {}
    for c in [*cats, None]:
        accs, oks, n = [], [], 0
        for r in reps:
            if c is None:
                ok, tot = sum(v[0] for v in r.values()), sum(v[1] for v in r.values())
            else:
                ok, tot = r.get(c, (0, 0))
            if tot:
                accs.append(ok / tot)
                oks.append(ok)
                n = tot
        if not accs:
            continue
        p = statistics.mean(accs)
        rep_se = statistics.stdev(accs) / math.sqrt(len(accs)) if len(accs) > 1 else None
        out[c] = (p, n, statistics.mean(oks), binom_se(p, n), rep_se)
    return out


def bfcl_summary(base, repeats, limit, model, temperature):
    """Write <base>/summary.json for the repeats under base (suite v1 contract)."""
    base = Path(base)
    reps, tokens, latency = bfcl(base.parent)
    if len(reps) != repeats:
        sys.exit(f"bfcl summary: expected {repeats} scored repeats under {base}, found {len(reps)}")
    stats = bfcl_stats(reps)
    overall = stats[None]
    metrics, stderr, stderr_repeat, counts = {}, {}, {}, {}
    for c, (p, n, ok, se, rse) in stats.items():
        name = "accuracy" if c is None else f"accuracy.{c}"
        metrics[name], stderr[name] = p, se
        counts[name] = {"mean_correct": ok, "n": n}
        if rse is not None:
            stderr_repeat[name] = rse
    if tokens:
        metrics["mean_output_tokens"] = statistics.mean(tokens)
        metrics["calls_at_4096_cap_per_repeat"] = sum(t >= 4096 for t in tokens) / repeats
    if latency:
        metrics["mean_latency_s"] = statistics.mean(latency)
    doc = {
        "benchmark": "bfcl",
        "suite_version": 1,
        "dataset": {"repo": "bfcl-eval (pip package, bundled data)", "revision": "2026.3.23"},
        "runs": [{
            "protocol": "daily",
            "n": overall[1],
            "metrics": metrics,
            "stderr": stderr,
            "stderr_repeat": stderr_repeat,
            "counts": counts,
            "repeat_accuracy": [
                sum(v[0] for v in r.values()) / sum(v[1] for v in r.values()) for r in reps
            ],
            "settings": {
                "handler": model, "temperature": temperature, "thinking": True,
                "reasoning_effort": "default", "repeats": repeats,
                "case_limit": limit or None, "stderr": "binomial sqrt(p(1-p)/n) on one run's n",
                "stderr_repeat": "std of repeat accuracies / sqrt(repeats), only with 2+ repeats",
            },
        }],
    }
    json.dump(doc, open(base / "summary.json", "w"), indent=2)
    print(f"bfcl: accuracy {overall[0]:.4f} ± {overall[3]:.4f} (binomial), n={overall[1]}, "
          f"repeats={repeats} -> {base / 'summary.json'}")


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
    for tag in dict.fromkeys(t for s in serv for t in s):
        for key, name, nd in SERVING_METRICS:
            vals = [s.get(tag, {}).get(key) for s in serv]
            if all(v is None for v in vals):
                continue
            cells = [f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-" for v in vals]
            delta = ""
            if all(isinstance(v, (int, float)) and v for v in vals[:2]):
                delta = f"{(vals[1] / vals[0] - 1) * 100:+.1f}%"
            rows.append([tag, name, *cells, delta])
    if rows:
        print("## Serving (reasoning_effort=medium)\n")
        table(["scenario", "metric", *labels, f"{labels[1]} vs {labels[0]}"], rows)

    res = [bfcl(p) for p in paths]
    if any(r[0] for r in res):
        bfcl_section(labels, res)
    generic_section(paths, labels)


def bfcl_section(labels, res):
    stats = [bfcl_stats(r[0]) for r in res]
    cats = sorted(set().union(*(set(s) - {None} for s in stats)))
    multi = any(len(r[0]) > 1 for r in res)
    rows = []
    for c in cats:
        cells = []
        for s in stats:
            if c not in s:
                cells.append("-")
                continue
            p, n, ok, _, _ = s[c]
            cells.append(f"{ok:.1f}/{n} ({p * 100:.0f}%)" if multi else f"{ok:.0f}/{n} ({p * 100:.0f}%)")
        rows.append([c, *cells])
    cells = []
    for s in stats:
        if None not in s:
            cells.append("-")
            continue
        p, n, ok, _, _ = s[None]
        tot = sum(s[c][1] for c in s if c is not None)
        cells.append(f"{ok:.1f}/{tot} ({p * 100:.1f}%)" if multi else f"{ok:.0f}/{tot} ({p * 100:.1f}%)")
    rows.append(["**total**", *cells])
    for name, idx in (("mean out tokens / call", 1), ("mean latency s / call", 2)):
        rows.append([name, *[f"{statistics.mean(r[idx]):.2f}" if r[idx] else "-" for r in res]])
    nreps = [max(len(r[0]), 1) for r in res]
    rows.append(["calls hitting 4096 cap", *[f"{sum(t >= 4096 for t in r[1]) / k:g}" for r, k in zip(res, nreps)]])
    rows.append(["repeats", *[str(len(r[0])) for r in res]])
    rows.append(["accuracy ± SE (binomial)", *[
        f"{s[None][0] * 100:.1f}% ± {s[None][3] * 100:.1f}" if None in s else "-" for s in stats]])
    if multi:
        rows.append(["accuracy ± SE (repeats)", *[
            f"{s[None][0] * 100:.1f}% ± {s[None][4] * 100:.1f}" if None in s and s[None][4] is not None else "-"
            for s in stats]])
    first = stats[0].get(None)
    gap = []
    for i, s in enumerate(stats):
        if i == 0 or first is None or None not in s:
            gap.append("-")
            continue
        d = s[None][0] - first[0]
        se = math.hypot(s[None][3], first[3])
        verdict = "within 2 SE" if abs(d) < 2 * se else "beyond 2 SE"
        gap.append(f"{d * 100:+.1f} pts ± {se * 100:.1f} ({verdict})")
    rows.append([f"gap vs {labels[0]} (SE of gap)", *gap])
    print("## BFCL v4 subset\n")
    table(["category", *labels], rows)


def fmt_value(v, se):
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "-"
    cell = f"{v:.4g}" if isinstance(v, float) else str(v)
    if isinstance(se, (int, float)) and not isinstance(se, bool):
        cell += f" ± {se:.2g}"
    return cell


def summaries(path):
    """{(benchmark, protocol, metric): (value, stderr)} from every <path>/*/summary.json."""
    out = {}
    for f in sorted(glob.glob(str(path / "*" / "summary.json"))):
        doc = json.load(open(f))
        bench = doc.get("benchmark", Path(f).parent.name)
        seen = {}
        for run in doc.get("runs", []):
            proto = run.get("protocol", "?")
            seen[proto] = seen.get(proto, 0) + 1
            if seen[proto] > 1:
                proto = f"{proto}#{seen[proto]}"
            se = run.get("stderr", {})
            for m, v in run.get("metrics", {}).items():
                out[(bench, proto, m)] = (v, se.get(m))
    return out


def generic_section(paths, labels):
    data = [summaries(p) for p in paths]
    keys = []
    for d in data:
        keys += [k for k in d if k not in keys]
    if not keys:
        return
    rows = []
    for k in keys:
        rows.append(["/".join(k), *[fmt_value(*d[k]) if k in d else "-" for d in data]])
    print("## Suite summaries (value ± standard error)\n")
    table(["benchmark/protocol/metric", *labels], rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--bfcl-summary":
        import argparse

        ap = argparse.ArgumentParser(prog="compare.py --bfcl-summary")
        ap.add_argument("bfcl_dir")
        ap.add_argument("--repeats", type=int, required=True)
        ap.add_argument("--limit", type=int, default=0)
        ap.add_argument("--model", required=True)
        ap.add_argument("--temperature", type=float, required=True)
        a = ap.parse_args(sys.argv[2:])
        bfcl_summary(a.bfcl_dir, a.repeats, a.limit, a.model, a.temperature)
    elif len(sys.argv) < 3:
        sys.exit(__doc__)
    else:
        main(sys.argv[1:])
