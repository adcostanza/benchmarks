#!/usr/bin/env python3
"""Typed Decisions test split against SGLang's two decision APIs (client side of scripts/decisions.sh):
/v1/systemone (TypeSafe/Jev request shape, the one the dataset's scorer uses) and native
/v1/decisions (input + a list of choice/score/yes_no questions; see to_native for the mapping).
SGLang documents both as the same rendering and scoring for chat models; native_vs_systemone in
the summary checks that on our server.

One request per case (the case's state and all of its questions), as the dataset's own scorer
does (bench/score_with_jobs.py at the pinned revision). Scoring is that scorer's, unchanged:
per decision, argmax agreement with the gold distribution, KL(gold || pred), Brier over the
options and the top probability as confidence; ECE over 10 confidence bins. Gold is the mean of
three samples from a ~4B teacher (dataset card), so accuracy is agreement with that teacher.

Added on top of the dataset's scorer: per question type breakdown, mean `x_label_mass` (the
full-vocabulary probability of the answer labels; low = the model wanted to say something else),
per-request latency, and every raw answer, so configs can be compared with each other
(e.g. KL of a quantized build's answers against its BF16) without rerunning.

Passes: `--passes` runs over /v1/systemone, then `--native-passes` over /v1/decisions, all in the
same order. Pass 2 hits the server's prefix cache; SGLang documents probability shifts of up to
~0.07 between cold and cached requests on this architecture, so the pass-to-pass difference is
reported as the noise floor. Headline metrics are pass 1 (/v1/systemone, cold).

Writes <out>/decisions/pass<k>.jsonl and native<k>.jsonl (one row per case) and
<out>/decisions/summary.json.
Exits non-zero on any failed request.
"""
import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

DATASET = {"repo": "LocalLLaMA/typed-decisions", "config": "all", "split": "test",
           "revision": "1415d375cb86b418c53abec12795adf5b194e7fc"}
RETRIES = 3
N_BINS = 10


def score_one(pred, gold):
    """Dataset scorer's score_one (bench/score_with_jobs.py), unchanged."""
    opts = list(gold["probabilities"].keys())
    g = [float(gold["probabilities"][o]) for o in opts]
    if gold["type"] == "noul":  # engines report yes/no; gold uses true/false
        pred = dict(pred)
        pred["probabilities"] = {
            {"yes": "true", "no": "false"}.get(k, k): v for k, v in pred["probabilities"].items()
        }
    p = [float(pred["probabilities"].get(o, 0.0)) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)
    am_p = p.index(max(p))
    am_g = g.index(max(g))
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g))
    return am_p == am_g, kl, brier, max(p)


def ece(hits, confs):
    """Dataset scorer's ECE: 10 equal-width bins over the top probability."""
    n = len(hits)
    total, hit, conf = [0] * N_BINS, [0] * N_BINS, [0.0] * N_BINS
    for ok, c in zip(hits, confs):
        b = min(int(c * N_BINS), N_BINS - 1)
        total[b] += 1
        hit[b] += int(ok)
        conf[b] += c
    return sum(total[b] / n * abs(hit[b] / total[b] - conf[b] / total[b]) for b in range(N_BINS) if total[b])


def post(base_url, path, body):
    data = json.dumps(body).encode()
    last = None
    for attempt in range(RETRIES):
        req = urllib.request.Request(f"{base_url}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read()), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            # A 4xx names a refused prompt or label (SGLang decisions errors); retrying won't help.
            msg = e.read().decode(errors="replace")[:500]
            if 400 <= e.code < 500:
                raise RuntimeError(f"HTTP {e.code}: {msg}") from e
            last = f"HTTP {e.code}: {msg}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = repr(e)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request failed after {RETRIES} attempts: {last}")


def to_native(case):
    """The case as a native /v1/decisions request: noul -> yes_no (criteria true/false become the
    yes/no descriptions), choice criteria -> options, score criteria (lowest first) -> levels.
    Answers come back keyed like the gold: option names, level indexes, yes/no (the scorer maps
    yes/no to true/false)."""
    questions = []
    for qid, q in case["questions"].items():
        nq = {"id": qid, "question": q["instructions"]}
        crit = q.get("criteria")
        if q["type"] == "noul":
            nq["type"] = "yes_no"
            if crit:
                nq["yes"], nq["no"] = crit["true"], crit["false"]
        elif q["type"] == "choice":
            nq["type"] = "choice"
            nq["options"] = [{"name": k, "description": v} if v else {"name": k} for k, v in crit.items()]
        elif q["type"] == "score":
            nq["type"] = "score"
            nq["levels"] = list(crit)
        else:
            raise RuntimeError(f"case {case['id']}: unknown question type {q['type']!r}")
        questions.append(nq)
    return {"model": "decisions-bench", "input": case["state"], "questions": questions}


def run_pass(base_url, cases, concurrency, api):
    def one(case):
        if api == "systemone":
            path, body = "/v1/systemone", {"model": "decisions-bench", "state": case["state"],
                                           "questions": case["questions"]}
        else:
            path, body = "/v1/decisions", to_native(case)
        resp, latency = post(base_url, path, body)
        answers = resp["answers"]
        for ans in answers.values():
            # A System One noul answer is only P(yes) (SGLang decision_models docs, TypeSafe API
            # reference); spell it out as the yes/no distribution /v1/decisions returns.
            if ans.get("type") == "noul" and "probabilities" not in ans and "noul" in ans:
                ans["probabilities"] = {"yes": float(ans["noul"]), "no": 1.0 - float(ans["noul"])}
        return {"id": case["id"], "workflow": case["workflow"], "api": api, "latency_s": latency,
                "model": resp.get("model"), "usage": resp.get("usage"), "answers": answers}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, cases))


def norm_probs(probs):
    """Probabilities with yes/no renamed to true/false, as the scorer does for noul."""
    return {{"yes": "true", "no": "false"}.get(k, k): float(v) for k, v in probs.items()}


def metrics(rows, cases):
    gold_by_id = {c["id"]: c["gold"] for c in cases}
    per = {}
    allv = {"hit": [], "kl": [], "brier": [], "conf": [], "mass": []}
    for row in rows:
        for qid, gq in gold_by_id[row["id"]].items():
            ans = row["answers"].get(qid)
            if not ans or not ans.get("probabilities"):
                # The scorer would score a missing distribution as uniform; that hides a broken
                # response shape, so stop instead.
                raise RuntimeError(f"case {row['id']} question {qid}: no probabilities in answer {ans!r}")
            if not set(norm_probs(ans["probabilities"])) & set(gq["probabilities"]):
                # Keys that match no gold option would also score as uniform.
                raise RuntimeError(f"case {row['id']} question {qid}: answer keys "
                                   f"{sorted(ans['probabilities'])} match none of {sorted(gq['probabilities'])}")
            ok, kl, brier, conf = score_one(ans, gq)
            mass = ans.get("x_label_mass", ans.get("label_mass"))
            for d in (allv, per.setdefault(gq["type"], {"hit": [], "kl": [], "brier": [], "conf": [], "mass": []})):
                d["hit"].append(ok)
                d["kl"].append(kl)
                d["brier"].append(brier)
                d["conf"].append(conf)
                if mass is not None:
                    d["mass"].append(float(mass))

    def summarize(d):
        n = len(d["hit"])
        return {"n_decisions": n, "accuracy": sum(d["hit"]) / n, "kl_from_gold": sum(d["kl"]) / n,
                "brier": sum(d["brier"]) / n, "ece": ece(d["hit"], d["conf"]),
                "label_mass_mean": statistics.fmean(d["mass"]) if d["mass"] else None,
                "label_mass_p10": sorted(d["mass"])[len(d["mass"]) // 10] if d["mass"] else None}

    lat = sorted(r["latency_s"] for r in rows)
    out = summarize(allv)
    out["by_type"] = {t: summarize(d) for t, d in sorted(per.items())}
    out["latency_p50_s"] = lat[len(lat) // 2]
    out["latency_p95_s"] = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    return out


def drift(a_rows, b_rows):
    """Mean and max absolute probability difference, and top-answer changes, between two passes."""
    b_by_id = {r["id"]: r for r in b_rows}
    diffs, changed, n = [], 0, 0
    for ra in a_rows:
        rb = b_by_id[ra["id"]]
        for qid, qa in ra["answers"].items():
            pa = norm_probs(qa.get("probabilities", {}))
            pb = norm_probs(rb["answers"].get(qid, {}).get("probabilities", {}))
            if not pa or not pb:
                continue
            n += 1
            diffs.extend(abs(pa[k] - pb.get(k, 0.0)) for k in pa)
            if max(pa, key=pa.get) != max(pb, key=pb.get):
                changed += 1
    return {"mean_abs_prob_diff": statistics.fmean(diffs) if diffs else None,
            "max_abs_prob_diff": max(diffs) if diffs else None,
            "top_answer_changed": changed, "n_decisions": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--data", required=True, help="typed-decisions all/<split> parquet")
    ap.add_argument("--split", default="test", choices=["test", "train"], help="split the --data file holds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--passes", type=int, default=2, help="passes over /v1/systemone")
    ap.add_argument("--native-passes", type=int, default=1, help="passes over native /v1/decisions")
    args = ap.parse_args()

    cases = []
    for raw in pq.read_table(args.data).to_pylist():
        cases.append({"id": raw["id"], "workflow": raw["workflow"], "state": json.loads(raw["state"]),
                      "questions": json.loads(raw["questions"]), "gold": json.loads(raw["gold"])})
    if args.limit:
        cases = cases[: args.limit]

    out = Path(args.out) / "decisions"
    out.mkdir(parents=True, exist_ok=True)
    passes = []
    runs = [("systemone", k, f"pass{k}") for k in range(1, args.passes + 1)]
    runs += [("decisions", k, f"native{k}") for k in range(1, args.native_passes + 1)]
    for api, k, name in runs:
        t0 = time.time()
        rows = run_pass(args.base_url, cases, args.concurrency, api)
        wall = time.time() - t0
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        m = metrics(rows, cases)
        m["wall_s"] = wall
        m["api"] = api
        passes.append((name, rows, m))
        print(f"decisions {name} ({api}): {len(cases)} cases, acc {m['accuracy']:.4f}, KL {m['kl_from_gold']:.4f}, "
              f"Brier {m['brier']:.4f}, ECE {m['ece']:.4f}, label_mass {m['label_mass_mean']}, "
              f"p50 {m['latency_p50_s']*1000:.0f} ms, {wall:.0f}s", flush=True)

    by = {name: (rows, m) for name, rows, m in passes}
    first = passes[0]
    summary = {"suite_version": 1, "dataset": {**DATASET, "split": args.split}, "n_cases": len(cases), "limit": args.limit,
               "concurrency": args.concurrency, "endpoint": f"/v1/{first[2]['api']}",
               "served_model": first[1][0]["model"] if first[1] else None,
               **first[2], "passes": {name: m for name, _, m in passes}}
    if "pass1" in by and "pass2" in by:
        summary["pass_drift"] = drift(by["pass1"][0], by["pass2"][0])
    if "pass1" in by and "native1" in by:
        # Same rendering and scoring per the SGLang docs; this checks it on our server.
        summary["native_vs_systemone"] = drift(by["pass1"][0], by["native1"][0])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"decisions: {e}", file=sys.stderr)
        sys.exit(1)
