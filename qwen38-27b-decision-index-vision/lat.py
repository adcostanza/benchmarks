#!/usr/bin/env python3
"""Latency (median/mean/p80 ms, kit's total_wall_ms) overall, per benchmark and per image count, and status counts.
Usage: lat.py <results.jsonl> [more.jsonl ...]"""
import json, sys, statistics, collections
rows = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
def st(v):
    v = sorted(v); n = len(v)
    return f"n={n:5d} median={statistics.median(v):7.0f} mean={statistics.fmean(v):7.0f} p80={v[min(n-1, int(0.8*n))]:7.0f} max={v[-1]:7.0f}"
print("status:", dict(collections.Counter(r["status"] for r in rows)))
ok = [r for r in rows if r["status"] == "ok"]
print("ALL      ", st([r["total_wall_ms"] for r in ok]))
by = collections.defaultdict(list); byi = collections.defaultdict(list)
for r in ok:
    by[r["benchmark"]].append(r["total_wall_ms"])
    byi[len(r["payload"]["images"]) if "payload" in r else -1].append(r["total_wall_ms"])
for k, v in sorted(by.items()): print(f"{k:28s}", st(v))
for k, v in sorted(byi.items()): print(f"{k} image(s)".ljust(28), st(v))
