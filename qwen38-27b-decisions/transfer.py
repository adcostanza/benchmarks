#!/usr/bin/env python3
"""Do per-type temperatures carry across datasets? Temperatures only (priors are keyed to one
dataset's questions, so they can't carry). Host side, CPU only, from the stored answers.

For each NVFP4 config, fits "T per type" (calibrate.fit, same objective as calibrate.py) on each
source dataset's fit split, then applies it to every target's scored split:
  sources  typed-decisions (train), procedural (validation), calibrated (validation),
           typed-decision-bench (all of test: it has no other split)
  targets  typed-decisions test, procedural test, calibrated test, calibrated heldout (PubMedQA,
           never a source), typed-decision-bench test
A type the source has no questions of keeps T = 1 (calibrated-typed-decisions has no `score`).
The diagonal is the in-domain fit (typed-decision-bench: 5-fold cross-fit stratified by group, as
calibrate.py --folds 5), so no target is scored with temperatures fitted on its own answers.
"pooled text" fits one set on all sources except the target, so it is out of sample too; it is
the best case for a single server-side default.

Passes: with several `--passes`, each pass's fits use the same pass of the fit split, and every
cell shows the mean over passes with the [min, max] across them: the spread from serving noise
(pass 1 is cold, later passes run on the prefix cache of the same server session).

Usage (host, from this folder):
  uv run --no-project --with pyarrow --with numpy --with scipy transfer.py "$DATASETS_DIR" > results-real/transfer.md
  ... transfer.py "$DATASETS_DIR" --td-train results-repeat/train --td-test results-repeat/test \\
      --real results-repeat/test --passes pass1 pass2 pass3 > results-repeat/transfer.md
"""
import argparse
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "scripts" / "decisions"))
import calibrate  # noqa: E402
import sets  # noqa: E402

CONFIGS = ["stock-nvfp4", "swift-nvfp4"]
FOLDS = 5
# name: (set, answers dir under its root, root key, fit split)
SOURCES = {
    "Typed Decisions": ("typed-decisions", "decisions", "td_train", "train"),
    "procedural": ("procedural-typed-decisions", "procedural-typed-decisions-validation", "real", "validation"),
    "Calibrated TD": ("calibrated-typed-decisions", "calibrated-typed-decisions-validation", "real", "validation"),
    "TD Bench": ("typed-decision-bench", "typed-decision-bench-test", "real", "test"),
    "ChaosNLI": ("chaosnli", "chaosnli-validation", "real", "validation"),
    "LeWiDi": ("lewidi", "lewidi-validation", "real", "validation"),
}
# name: (set, answers dir, root key, split, home source, metric key, higher is better)
TARGETS = {
    "Typed Decisions": ("typed-decisions", "decisions", "td_test", "test", "Typed Decisions", "kl_from_gold", False),
    "procedural": ("procedural-typed-decisions", "procedural-typed-decisions-test", "real", "test", "procedural", "proper_score", True),
    "Calibrated TD": ("calibrated-typed-decisions", "calibrated-typed-decisions-test", "real", "test", "Calibrated TD", "log_loss", False),
    "PubMedQA": ("calibrated-typed-decisions", "calibrated-typed-decisions-heldout", "real", "heldout", None, "log_loss", False),
    "TD Bench": ("typed-decision-bench", "typed-decision-bench-test", "real", "test", "TD Bench", "decision_score", True),
    "ChaosNLI": ("chaosnli", "chaosnli-test", "real", "test", "ChaosNLI", "kl_from_gold", False),
    "LeWiDi": ("lewidi", "lewidi-test", "real", "test", "LeWiDi", "kl_from_gold", False),
}
METRIC = {"kl_from_gold": "KL ↓", "proper_score": "proper score ↑", "log_loss": "log loss ↓", "decision_score": "DecisionScore ↑"}


def one_pass(roots, c, pas, load, sources, targets):
    """Fitted temperatures per source, {target: {row: metric}} and {target: its question types}
    for one config and pass."""
    def answers(key, sub):
        return calibrate.read_rows(roots[key] / c / sub / f"{pas}.jsonl")

    items, temps = {}, {}
    for name, (set_name, sub, key, split) in sources.items():
        items[name] = calibrate.decisions(sets.get(set_name), answers(key, sub), load(set_name, split))
        temps[name] = calibrate.fit(items[name], False)[0]

    def score(ds, rows, cases, params_for):
        return ds.metrics(calibrate.calibrated_rows(ds, rows, cases, params_for), cases)

    table, ttypes = {}, {}
    for tname, (set_name, sub, key, split, home, metric, _) in targets.items():
        ds, tc, trows = sets.get(set_name), load(set_name, split), answers(key, sub)
        ttypes[tname] = {i["type"] for i in calibrate.decisions(ds, trows, tc)}
        col = {"raw": ds.metrics(trows, tc)[metric]}
        for sname in sources:
            if sname == home and sources[sname][3] == split:  # fitted on the scored answers: cross-fit
                fold_of = {cid: k for k, ids in enumerate(sets.folds(tc, FOLDS)) for cid in ids}
                per_fold = [calibrate.fit([i for i in items[sname] if fold_of[i["id"]] != k], False)[0]
                            for k in range(FOLDS)]
                col[sname] = score(ds, trows, tc, lambda cid: (per_fold[fold_of[cid]], {}))[metric]
            else:
                col[sname] = score(ds, trows, tc, lambda cid, t=temps[sname]: (t, {}))[metric]
        pooled = calibrate.fit([i for s, its in items.items() if s != home for i in its], False)[0]
        col["pooled text"] = score(ds, trows, tc, lambda cid, t=pooled: (t, {}))[metric]
        table[tname] = col
    return temps, table, ttypes


def fmt(vs, digits=None):
    d = digits if digits is not None else (2 if abs(vs[0]) >= 10 else 3)
    s = lambda v: f"{v:.{d}f}"  # noqa: E731
    return s(np.mean(vs)) if len(vs) == 1 else f"{s(np.mean(vs))} [{s(min(vs))}, {s(max(vs))}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets_dir")
    ap.add_argument("--td-train", default="results-train", help="root with <config>/decisions/ (Typed Decisions train)")
    ap.add_argument("--td-test", default="results", help="root with <config>/decisions/ (Typed Decisions test)")
    ap.add_argument("--real", default="results-real", help="root with <config>/<set>-<split>/")
    ap.add_argument("--passes", nargs="+", default=["pass1"])
    args = ap.parse_args()
    roots = {"td_train": here / args.td_train, "td_test": here / args.td_test, "real": here / args.real}
    cases = {}

    def load(set_name, split):
        if (set_name, split) not in cases:
            cases[set_name, split] = sets.get(set_name).load(args.datasets_dir, split)
        return cases[set_name, split]

    print("# Temperature transfer across datasets (T per type only)\n")
    print(f"Answers: Typed Decisions `{args.td_train}` (train) and `{args.td_test}` (test), the rest "
          f"`{args.real}`; passes: {', '.join(args.passes)}. Rows: where the temperatures were fitted. "
          "Columns: where they are scored. `raw` is T = 1; the diagonal (bold) is the in-domain fit "
          "(Typed Decision Bench: 5-fold cross-fit); `pooled text` is one fit on every source except "
          "the column's dataset (`pooled text`). Sets without answers "
          "in these roots are left out."
          + (" Cells: mean over passes [min, max]." if len(args.passes) > 1 else "")
          + " Generated by `transfer.py`.\n")
    def have(c, key, sub):
        return all((roots[key] / c / sub / f"{p}.jsonl").exists() for p in args.passes)

    for c in CONFIGS:
        sources = {n: v for n, v in SOURCES.items() if have(c, v[2], v[1])}
        targets = {n: v for n, v in TARGETS.items() if have(c, v[2], v[1])}
        rows_order = ["raw", *sources, "pooled text"]
        runs = [one_pass(roots, c, p, load, sources, targets) for p in args.passes]
        print(f"## `{c}`\n")
        print("Fitted temperatures (choice / noul / score; `-` = no questions of that type, T = 1). "
              "A cell is `-` where the source has none of the target's question types (it would "
              "equal raw):\n")
        print("| Source | choice | noul | score |\n|---|---|---|---|")
        for name in sources:
            cells = []
            for k in ("choice", "noul", "score"):
                vs = [temps[name][k] for temps, _, _ in runs if k in temps[name]]
                cells.append(fmt(vs, 2) if vs else "-")
            print(f"| {name} | " + " | ".join(cells) + " |")
        print()
        print("| Fitted on \\ scored on | " + " | ".join(f"{t} ({METRIC[targets[t][5]]})" for t in targets) + " |")
        print("|---|" + "---|" * len(targets))
        for r in rows_order:
            cells = []
            for tname, t in targets.items():
                home, higher = t[4], t[6]
                if r in sources and not set(runs[0][0][r]) & runs[0][2][tname]:
                    cells.append("-")
                    continue
                vs = [table[tname][r] for _, table, _ in runs]
                raws = [table[tname]["raw"] for _, table, _ in runs]
                s = fmt(vs)
                if r == home:
                    s = f"**{s}**"
                elif r != "raw" and any((v < w) if higher else (v > w) for v, w in zip(vs, raws)):
                    s += " (worse than raw in a pass)"
                cells.append(s)
            print(f"| {r} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
