#!/usr/bin/env python3
"""Prefix-cache hit rate from an SGLang server.log: sum(#cached-token) / sum(#cached-token + #new-token) over prefill batches."""
import re, sys
c = n = b = 0
for l in open(sys.argv[1]):
    m = re.search(r"Prefill batch.*#new-token: (\d+), #cached-token: (\d+)", l)
    if m: n += int(m[1]); c += int(m[2]); b += 1
print(f"prefill batches={b} new={n} cached={c} hit_rate={c/(c+n):.4f}")
