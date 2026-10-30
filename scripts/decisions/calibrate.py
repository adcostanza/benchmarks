#!/usr/bin/env python3
"""Client-side calibration of decision answers: fit on `train` answers, score once on `test`.

Fits, by soft cross-entropy against the gold distributions of the train split:
  T    one temperature per question type (choice, score, noul): p_i ∝ p_i^(1/T)
  T+B  the same plus one log-prior bias per (workflow, question id, option): p_i ∝ p_i^(1/T) e^(b_i)
       (L2-penalised; the dataset's questions are fixed per workflow, so a bias belongs to one
       question of one workflow, like the Amos 1 large "per-question priors" row on the card).
Nothing touches the model or the server: this rescales the probabilities a client gets back.

Inputs are run.py outputs (pass1.jsonl) on each split and the two parquet files. Scoring is
run.py's (the dataset scorer's score_one and ECE). Writes <out> with the fitted parameters and
raw/T/T+B metrics on test, and prints a table. Also reports the gold's own ECE: gold scored as
a prediction against its own argmax, the floor for this ECE definition with soft labels.

Usage (host, not the client container):
  uv run --no-project --with pyarrow --with numpy --with scipy scripts/decisions/calibrate.py \
      --train-answers <dir>/decisions/pass1.jsonl --train-data .../all/train-00000-of-00001.parquet \
      --test-answers  <dir>/decisions/pass1.jsonl --test-data  .../all/test-00000-of-00001.parquet \
      --out <dir>/calibration.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize

from run import ece, norm_probs, score_one

L2 = 0.1  # bias penalty, scaled by 1/n decisions


def load(answers_path, data_path):
    gold = {r["id"]: (r["workflow"], json.loads(r["gold"])) for r in pq.read_table(data_path).to_pylist()}
    items = []
    for line in open(answers_path):
        row = json.loads(line)
        wf, g = gold[row["id"]]
        for qid, gq in g.items():
            opts = list(gq["probabilities"])
            p = norm_probs(row["answers"][qid]["probabilities"])
            items.append({"key": f"{wf}/{qid}", "type": gq["type"], "opts": opts, "gq": gq,
                          "lp": np.log(np.clip([p.get(o, 0.0) for o in opts], 1e-9, None)),
                          "g": np.array([float(gq["probabilities"][o]) for o in opts])})
    return items


def apply(it, temps, bias):
    z = it["lp"] / temps.get(it["type"], 1.0)
    z = z + np.array([bias.get(f"{it['key']}/{o}", 0.0) for o in it["opts"]])
    z = np.exp(z - z.max())
    return z / z.sum()


def fit(items, use_bias):
    types = sorted({i["type"] for i in items})
    bkeys = sorted({f"{i['key']}/{o}" for i in items for o in i["opts"]}) if use_bias else []

    def unpack(x):
        return ({t: math.exp(x[k]) for k, t in enumerate(types)},
                {b: float(x[len(types) + k]) for k, b in enumerate(bkeys)})

    def loss(x):
        temps, bias = unpack(x)
        ce = -sum(float(i["g"] @ np.log(apply(i, temps, bias) + 1e-12)) for i in items) / len(items)
        return ce + L2 * float(np.sum(x[len(types):] ** 2)) / len(items)

    res = minimize(loss, np.zeros(len(types) + len(bkeys)), method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"fit did not converge: {res.message}")
    return unpack(res.x)


def evaluate(items, temps, bias):
    hits, kls, briers, confs = [], [], [], []
    for i in items:
        p = apply(i, temps, bias)
        # score_one renames yes/no for noul; our keys are already the gold's, so score as-is.
        ok, kl, brier, conf = score_one({"probabilities": dict(zip(i["opts"], p.tolist()))}, {**i["gq"], "type": "-"})
        hits.append(ok); kls.append(kl); briers.append(brier); confs.append(conf)
    return {"accuracy": float(np.mean(hits)), "kl_from_gold": float(np.mean(kls)),
            "brier": float(np.mean(briers)), "ece": float(ece(hits, confs)), "n_decisions": len(items)}


def gold_ece(items):
    return float(ece([True] * len(items), [float(i["g"].max()) for i in items]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-answers", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--test-answers", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train, test = load(args.train_answers, args.train_data), load(args.test_answers, args.test_data)
    t_only = fit(train, False)
    t_bias = fit(train, True)
    result = {
        "fit_on": {"answers": args.train_answers, "n_decisions": len(train)},
        "scored_on": {"answers": args.test_answers},
        "l2": L2,
        "temperature_only": {"temperatures": t_only[0]},
        "temperature_and_bias": {"temperatures": t_bias[0], "bias": t_bias[1]},
        "test": {"raw": evaluate(test, {}, {}), "T": evaluate(test, *t_only), "T+B": evaluate(test, *t_bias)},
        "gold_ece_test": gold_ece(test),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    for name, m in result["test"].items():
        print(f"{name:4s} acc {m['accuracy']:.4f}  KL {m['kl_from_gold']:.4f}  Brier {m['brier']:.4f}  ECE {m['ece']:.4f}")
    print("T only: " + ", ".join(f"{k} {v:.2f}" for k, v in t_only[0].items()))
    print(f"gold's own ECE on test: {result['gold_ece_test']:.4f}")


if __name__ == "__main__":
    main()
