#!/usr/bin/env python3
"""Prefix cache check table rows: cached prompt-token share per phase and top-answer changes.

Usage: cache_check.py <result dir> ...   (dirs from `DECISIONS_LIMIT=50 ./run.sh`, default passes)

Phases come from server.log: prefill lines up to the 50th completed /v1/systemone request count as
pass 1, the next 50 as pass 2, /v1/decisions requests as native. Requests overlap at concurrency > 1,
so shares near the boundaries are approximate.
"""
import json
import re
import sys
from pathlib import Path

PREFILL = re.compile(r"Prefill batch, .*#new-token: (\d+), #cached-token: (\d+)")
DONE = re.compile(r'"POST /v1/(systemone|decisions) HTTP')


def cached_shares(log: Path) -> list[float]:
    tokens = {p: [0, 0] for p in ("pass1", "pass2", "native")}
    systemone = 0
    native = False
    for line in log.read_text(errors="replace").splitlines():
        if m := DONE.search(line):
            if m[1] == "systemone":
                systemone += 1
            else:
                native = True
            continue
        if m := PREFILL.search(line):
            phase = "native" if native or systemone >= 100 else "pass2" if systemone >= 50 else "pass1"
            tokens[phase][0] += int(m[1])
            tokens[phase][1] += int(m[2])
    return [c / (n + c) if n + c else 0.0 for n, c in tokens.values()]


def answers(path: Path) -> dict[tuple[str, str], tuple[str, dict]]:
    out = {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        for name, a in (rec.get("answers") or {}).items():
            probs = a.get("probabilities") or {}
            if probs:
                out[(rec["id"], name)] = (max(probs, key=probs.get), probs)
    return out


def compare(a: dict, b: dict) -> tuple[int, float, int]:
    keys = a.keys() & b.keys()
    changed = sum(a[k][0] != b[k][0] for k in keys)
    max_dp = max(
        (abs(a[k][1].get(l, 0.0) - b[k][1].get(l, 0.0)) for k in keys for l in a[k][1] | b[k][1]),
        default=0.0,
    )
    return changed, max_dp, len(keys)


def main() -> None:
    print("| Config | cached prompt tokens: pass 1 / pass 2 / native "
          "| top answers changed: pass 2 vs 1 / native vs 1 / native vs 2 | max \\|Δp\\| pass 2 vs 1 | decisions |")
    print("|---|---|---|---|---|")
    for d in map(Path, sys.argv[1:]):
        dec = d / "decisions"
        p1, p2, nat = (answers(dec / f) for f in ("pass1.jsonl", "pass2.jsonl", "native1.jsonl"))
        c21, dp, n = compare(p1, p2)
        cn1, _, _ = compare(p1, nat)
        cn2, _, _ = compare(p2, nat)
        shares = " / ".join(f"{s:.1%}" for s in cached_shares(d / "server.log"))
        print(f"| `{d.parent.name}/{d.name}` | {shares} | {c21} / {cn1} / {cn2} | {dp:.3f} | {n} |")


if __name__ == "__main__":
    main()
