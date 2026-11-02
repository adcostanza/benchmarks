#!/usr/bin/env python3
"""Post chart and table: per-benchmark skill of stock/Swift vs the 27B-class fine-tunes on the board.
27B-class = board entrants with 26-28.5 B parameters, no active-parameter tag, autoregressive, not an inference technique (20 entries).
Usage: board_range.py svg|table board-vision.json results/stock-nvfp4/run/vision-scores.json results/swift-nvfp4/run/vision-scores.json"""
import json, sys
mode = sys.argv[1]
board = json.load(open(sys.argv[2])); stock = json.load(open(sys.argv[3])); swift = json.load(open(sys.argv[4]))
E = {e["name"]: e for e in board["entrants"] + board["refs"]}
tuned = [e for e in board["entrants"] if e["params"] and 26e9 < e["params"] < 28.5e9 and not e["active"] and e["technique"] == "autoregressive" and "inference" not in (e["kind"] or "")]
assert len(tuned) == 20
names = ["CV-Bench", "BLINK", "RealWorldQA", "CharXiv", "InfographicVQA", "Mind2Web", "Winoground", "KIE (CORD+FUNSD)", "Moderation (Hateful Memes)", "R-Bench-M", "MMMU-Pro vision"]
def med(v): v = sorted(v); n = len(v); return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2
rows = []
for n in names + [None]:
    v = [e["pub"] if n is None else e["bench"][n]["pub"] for e in tuned]
    ref = lambda r: E[r]["pub"] if n is None else E[r]["bench"][n]["pub"]
    label = "Public index" if n is None else n.replace("Moderation (Hateful Memes)", "Hateful Memes").replace(" (CORD+FUNSD)", "")
    st = stock["public_vision_index"] if n is None else 100 * stock["benchmarks"][n]["skill"]
    sw = swift["public_vision_index"] if n is None else 100 * swift["benchmarks"][n]["skill"]
    rows.append((label, st, sw, min(v), max(v), ref("Qwen3.5-397B (VLM)"), med(v), ref("Qwen3-VL-8B (VLM)"), sum(1 for x in v if x < st)))
if mode == "table":
    print("| Benchmark | Stock | Swift | Qwen3.5-397B | Qwen3-VL-8B | 20 tuned 27B: lowest | median | highest | below stock |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for n, st, sw, lo, hi, q, md, v8, nb in rows:
        b = "**" if n == "Public index" else ""
        print(f"| {b}{n}{b} | {b}{st:.2f}{b} | {b}{sw:.2f}{b} | {q:.2f} | {v8:.2f} | {lo:.2f} | {md:.2f} | {hi:.2f} | {nb} of 20 |")
    sys.exit()
X0, X1, Y0, DY = 170, 580, 100, 30
x = lambda s: X0 + (X1 - X0) * s / 100
H = Y0 + DY * len(rows) + 30
o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 {H}" font-family="system-ui,sans-serif" font-size="16" fill="#888">',
     '<text x="0" y="22" font-size="18" font-weight="600">Skill per benchmark and public index</text>',
     '<circle cx="6" cy="42" r="5" fill="#4e79a7"/><text x="17" y="47" font-size="14">Stock NVFP4</text>',
     '<circle cx="136" cy="42" r="5" fill="#f28e2b"/><text x="147" y="47" font-size="14">Swift 1.5 NVFP4</text>',
     '<path d="M280 37 l5 5 l-5 5 l-5 -5 z" fill="none" stroke="#888" stroke-width="2"/><text x="291" y="47" font-size="14">Qwen3.5-397B</text>',
     '<rect x="6" y="59" width="40" height="8" fill="#888" fill-opacity="0.35"/><text x="52" y="67" font-size="14">20 tuned 27B-class entries: range (bars clipped at 0)</text>',
     ]
for t in range(0, 101, 20):
    o.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{Y0 - 14}" y2="{Y0 + DY * len(rows) - 12}" stroke="#888" stroke-opacity="0.25"/><text x="{x(t):.1f}" y="{Y0 + DY * len(rows) + 6}" text-anchor="middle" font-size="12">{t}</text>')
for i, (n, st, sw, lo, hi, q, _md, _v8, _nb) in enumerate(rows):
    y = Y0 + DY * i; lo = max(lo, 0)
    if n == "Public index": o.append(f'<line x1="0" x2="600" y1="{y - 16}" y2="{y - 16}" stroke="#888" stroke-opacity="0.5"/>')
    o.append(f'<text x="162" y="{y + 5}" text-anchor="end" font-size="14"{" font-weight=\"600\"" if n == "Public index" else ""}>{n}</text>')
    o.append(f'<rect x="{x(lo):.1f}" y="{y - 4}" width="{x(hi) - x(lo):.1f}" height="8" fill="#888" fill-opacity="0.35"/>')
    o.append(f'<path d="M{x(q):.1f} {y - 6} l6 6 l-6 6 l-6 -6 z" fill="none" stroke="#888" stroke-width="2"/>')
    o.append(f'<circle cx="{x(sw):.1f}" cy="{y + 5}" r="4.5" fill="#f28e2b" stroke="#fff" stroke-width="1"/>')
    o.append(f'<circle cx="{x(st):.1f}" cy="{y - 5}" r="4.5" fill="#4e79a7" stroke="#fff" stroke-width="1"/>')
o.append("</svg>")
print("\n".join(o))
