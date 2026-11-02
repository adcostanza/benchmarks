#!/usr/bin/env python3
"""Per-benchmark public scores (this run vs board rows). Usage: board.py board-vision.json label=results/<label>/run/vision-scores.json ..."""
import json, sys
board = json.load(open(sys.argv[1]))
runs = {}
for a in sys.argv[2:]:
    k, p = a.split("=", 1); runs[k] = json.load(open(p))
names = ["CV-Bench", "BLINK", "RealWorldQA", "CharXiv", "InfographicVQA", "Mind2Web", "Winoground", "KIE (CORD+FUNSD)", "Moderation (Hateful Memes)", "R-Bench-M", "MMMU-Pro vision"]
E = {e["name"]: e for e in board["entrants"] + board["refs"]}
ref = ["Qwen3.5-397B (VLM)", "Qwen3-VL-8B (VLM)", "SigLIP 2 (contrastive)", "JEV-27B-VL", "jev-vision-27b", "VisionJEV 1.0.0 (27B)", "Sieve-Vision-27B", "RSI-Jev v6.1 VL 27B (effort auto)", "WeMM-JEV-27B"]
cols = list(runs) + ref
print("| Benchmark | " + " | ".join(cols) + " |"); print("|---|" + "---:|" * len(cols))
for n in names + ["Public index"]:
    cells = []
    for k, r in runs.items():
        if n == "Public index": cells.append(f"**{r['public_vision_index']:.2f}**")
        else: cells.append(f"{100 * r['benchmarks'][n]['skill']:.2f}")
    for e in ref:
        cells.append(f"{E[e]['pub']:.2f}" if n == "Public index" else f"{E[e]['bench'][n]['pub']:.2f}")
    print(f"| {n} | " + " | ".join(cells) + " |")
