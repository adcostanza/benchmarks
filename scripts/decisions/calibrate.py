#!/usr/bin/env python3
"""Client-side calibration of decision answers: fit on one split's answers, score once on another.

Fits, by cross-entropy against each set's targets (sets.py: the gold distribution for Typed
Decisions, one-hot for hard labels, the exact probability for graded `noul`):
  T    one temperature per question type (choice, score, noul): p_i ∝ p_i^(1/T)
  T+B  the same plus one log-prior bias per (group, question id, answer key): p_i ∝ p_i^(1/T) e^(b_i)
       (L2-penalised; group is the set's workflow, task or source, so a bias belongs to one
       question of one group, like the Typed Decisions card's Amos 1 large "per-question priors" row).
Nothing touches the model or the server: this rescales the probabilities a client gets back.

Scoring is the set's own (sets.py) on the calibrated answers. A set without a split to fit on
(Typed Decision Bench) is cross-fitted instead: `--folds K` splits the test cases into K folds
stratified by group, fits on K-1 and calibrates the held-out fold, so no case is calibrated with
parameters fitted on itself. Writes <out> with the fitted parameters and raw/T/T+B metrics, and
prints the headlines. For soft gold (Typed Decisions) it also reports the gold's own ECE: gold
scored as a prediction against its own argmax, the floor for that ECE definition.

Usage (host, not the client container):
  uv run --no-project --with pyarrow --with numpy --with scipy scripts/decisions/calibrate.py \
      --set typed-decisions --data-dir "$DATASETS_DIR" \
      --fit-answers <dir>/decisions/pass1.jsonl --fit-split train \
      --test-answers <dir>/decisions/pass1.jsonl --test-split test --out <dir>/calibration.json
  ... --set typed-decision-bench --folds 5 --test-answers <dir>/typed-decision-bench-test/pass1.jsonl ...
"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import sets
from sets import ece

L2 = 0.1  # bias penalty, scaled by 1/n decisions


def read_rows(path):
    return [json.loads(line) for line in open(path)]


def decisions(dataset, rows, cases):
    """One entry per gold question, in row order: answer log-probs over the set's answer keys and the target."""
    by = {c["id"]: c for c in cases}
    out = []
    for row in rows:
        case = by[row["id"]]
        for qid, qtype, keys, target in dataset.targets(case):
            p = sets.answer_probs(row, qid)["probabilities"]
            out.append({"id": row["id"], "qid": qid, "key": f"{case['group']}/{qid}", "type": qtype, "opts": keys,
                        "lp": np.log(np.clip([float(p.get(k, 0.0)) for k in keys], 1e-9, None)),
                        "g": np.array(target, dtype=float)})
    return out


def apply(it, temps, bias):
    z = it["lp"] / temps.get(it["type"], 1.0)
    z = z + np.array([bias.get(f"{it['key']}/{o}", 0.0) for o in it["opts"]])
    z = np.exp(z - z.max())
    return z / z.sum()


def fit(items, use_bias):
    """Minimises the mean cross-entropy (plus the bias penalty) with its analytic gradient: all
    decisions' options in one flat array, softmax per decision segment."""
    types = sorted({i["type"] for i in items})
    bkeys = sorted({f"{i['key']}/{o}" for i in items for o in i["opts"]}) if use_bias else []
    bpos = {b: k for k, b in enumerate(bkeys)}
    nt, n = len(types), len(items)
    lp = np.concatenate([i["lp"] for i in items])
    g = np.concatenate([i["g"] for i in items])
    t_of = np.concatenate([np.full(len(i["opts"]), types.index(i["type"])) for i in items])
    b_of = np.array([bpos[f"{i['key']}/{o}"] for i in items for o in i["opts"]]) if use_bias else None
    starts = np.cumsum([0] + [len(i["opts"]) for i in items[:-1]])
    seg = np.repeat(np.arange(n), [len(i["opts"]) for i in items])

    def unpack(x):
        return ({t: math.exp(x[k]) for k, t in enumerate(types)},
                {b: float(x[nt + k]) for k, b in enumerate(bkeys)})

    def loss_grad(x):
        temps = np.exp(x[:nt])[t_of]
        z = lp / temps + (x[nt:][b_of] if use_bias else 0.0)
        z = z - np.maximum.reduceat(z, starts)[seg]
        logp = z - np.log(np.add.reduceat(np.exp(z), starts))[seg]
        r = np.exp(logp) * np.add.reduceat(g, starts)[seg] - g  # d CE / d z
        b = x[nt:]
        loss = -float(g @ logp) / n + L2 * float(b @ b) / n
        grad = np.zeros_like(x)
        grad[:nt] = np.bincount(t_of, r * (-lp / temps), minlength=nt) / n
        if use_bias:
            grad[nt:] = np.bincount(b_of, r, minlength=len(bkeys)) / n + 2 * L2 * b / n
        return loss, grad

    res = minimize(loss_grad, np.zeros(nt + len(bkeys)), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"fit did not converge: {res.message}")
    return unpack(res.x)


def calibrated_rows(dataset, rows, cases, params_for):
    """Rows with each gold question's probabilities replaced by the calibrated ones.
    params_for(case id) -> (temps, bias)."""
    out = {r["id"]: copy.deepcopy(r) for r in rows}
    for it in decisions(dataset, rows, cases):
        p = apply(it, *params_for(it["id"]))
        ans = out[it["id"]]["answers"][it["qid"]]
        ans["probabilities"] = dict(zip(it["opts"], p.tolist()))
        if it["type"] == "noul":
            ans["noul"] = float(p[it["opts"].index("yes")])
    return [out[r["id"]] for r in rows]


def gold_ece(items):
    return float(ece([True] * len(items), [float(i["g"].max()) for i in items]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="typed-decisions", choices=sorted(sets.SETS))
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fit-answers")
    ap.add_argument("--fit-split")
    ap.add_argument("--folds", type=int, default=0, help="cross-fit on the test answers instead of a fit split")
    ap.add_argument("--test-answers", required=True)
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-out", help="dir for the calibrated answers (T.jsonl, T+B.jsonl), for paired intervals")
    args = ap.parse_args()
    if bool(args.folds) == bool(args.fit_answers):
        ap.error("give either --fit-answers/--fit-split or --folds")

    dataset = sets.get(args.set)
    test_cases = dataset.load(args.data_dir, args.test_split)
    test_rows = read_rows(args.test_answers)
    result = {"set": args.set, "dataset": dataset.revision, "l2": L2,
              "scored_on": {"answers": args.test_answers, "split": args.test_split}}
    if args.folds:
        fold_of = {cid: k for k, ids in enumerate(sets.folds(test_cases, args.folds)) for cid in ids}
        items = decisions(dataset, test_rows, test_cases)
        params = {}
        for name, use_bias in (("T", False), ("T+B", True)):
            params[name] = [fit([i for i in items if fold_of[i["id"]] != k], use_bias) for k in range(args.folds)]
        result["fit_on"] = {"cross_fit_folds": args.folds, "stratified_by": "group", "n_decisions": len(items)}
        result["temperature_only"] = {"temperatures_per_fold": [p[0] for p in params["T"]]}
        result["temperature_and_bias"] = {"temperatures_per_fold": [p[0] for p in params["T+B"]],
                                          "bias_per_fold": [p[1] for p in params["T+B"]]}
        variants = {name: (lambda cid, ps=ps: ps[fold_of[cid]]) for name, ps in params.items()}
    else:
        fit_items = decisions(dataset, read_rows(args.fit_answers), dataset.load(args.data_dir, args.fit_split))
        t_only, t_bias = fit(fit_items, False), fit(fit_items, True)
        result["fit_on"] = {"answers": args.fit_answers, "split": args.fit_split, "n_decisions": len(fit_items)}
        result["temperature_only"] = {"temperatures": t_only[0]}
        result["temperature_and_bias"] = {"temperatures": t_bias[0], "bias": t_bias[1]}
        variants = {"T": lambda cid: t_only, "T+B": lambda cid: t_bias}

    if args.set == "typed-decisions":  # soft gold: report the floor of this ECE definition
        result["gold_ece_test"] = gold_ece(decisions(dataset, test_rows, test_cases))
    result["test"] = {"raw": dataset.metrics(test_rows, test_cases)}
    for name, params_for in variants.items():
        rows = calibrated_rows(dataset, test_rows, test_cases, params_for)
        result["test"][name] = dataset.metrics(rows, test_cases)
        if args.rows_out:
            Path(args.rows_out).mkdir(parents=True, exist_ok=True)
            Path(args.rows_out, f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    for name, m in result["test"].items():
        print(f"{name:4s} {m['headline']}")
    if "temperatures" in result["temperature_only"]:
        print("T only: " + ", ".join(f"{k} {v:.2f}" for k, v in result["temperature_only"]["temperatures"].items()))
    if "gold_ece_test" in result:
        print(f"gold's own ECE on test: {result['gold_ece_test']:.4f}")


if __name__ == "__main__":
    main()
