"""Decision datasets as one case format, each with its own scorer.

A case is {"id", "group", "state", "questions", "gold", "gold_source"}: `state` and `questions`
go to /v1/systemone as they are (TypeSafe System One shape: `noul`, `choice`, `score`), `group`
is the workflow, task or source (per-group breakdown and calibration biases), `gold` is the set's
own answer format, and `gold_source` says where the answers come from:
`human`, `rule` or `model:<name>` (`model:unnamed` when the dataset doesn't say).

Each set provides:
  load(root, split)        cases, from DATASETS_DIR (mounted at /datasets in the client)
  metrics(rows, cases)     the set's own scoring of run.py's rows (one per case)
  targets(case)            per gold question: (qid, type, answer keys, target distribution), the
                           target calibrate.py fits to (one-hot for hard labels)
  revision, gold_source    recorded in summary.json

Answers are scored as SGLang returns them: `probabilities` keyed by option name (`choice`), level
index as a string (`score`) or yes/no (`noul`, which also carries `noul` = P(yes)).
"""
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

N_BINS = 10


def ece(hits, confs):
    """Typed Decisions scorer's ECE: 10 equal-width bins over the top probability."""
    n = len(hits)
    total, hit, conf = [0] * N_BINS, [0] * N_BINS, [0.0] * N_BINS
    for ok, c in zip(hits, confs):
        b = min(int(c * N_BINS), N_BINS - 1)
        total[b] += 1
        hit[b] += int(ok)
        conf[b] += c
    return sum(total[b] / n * abs(hit[b] / total[b] - conf[b] / total[b]) for b in range(N_BINS) if total[b])


def norm_probs(probs):
    """Probabilities with yes/no renamed to true/false, as the Typed Decisions scorer does for noul."""
    return {{"yes": "true", "no": "false"}.get(k, k): float(v) for k, v in probs.items()}


def answer_probs(row, qid):
    ans = row["answers"].get(qid)
    if not ans or not ans.get("probabilities"):
        # A scorer would count a missing distribution as uniform or unanswered; that hides a
        # broken response shape, so stop instead.
        raise RuntimeError(f"case {row['id']} question {qid}: no probabilities in answer {ans!r}")
    return ans


def answer_keys(q):
    """The keys SGLang's answer probabilities use for one System One question."""
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    raise RuntimeError(f"unknown question type {q['type']!r}")


def mean_or_none(xs):
    return statistics.fmean(xs) if xs else None


# ------------------------------------------------------------------ Typed Decisions
class TypedDecisions:
    """LocalLLaMA/typed-decisions `all`: gold is the mean of three samples of an unnamed ~4B-class
    teacher (dataset card), so accuracy is agreement with that teacher. Scoring is the dataset's
    scorer (bench/score_with_jobs.py at the pinned revision), unchanged: per decision, argmax
    agreement with the gold distribution, KL(gold || pred), Brier over the options and the top
    probability as confidence; ECE over 10 confidence bins."""
    name = "typed-decisions"
    revision = {"repo": "LocalLLaMA/typed-decisions", "config": "all",
                "revision": "1415d375cb86b418c53abec12795adf5b194e7fc"}
    gold_source = "model:unnamed"
    splits = ("test", "train")

    @staticmethod
    def load(root, split):
        cases = []
        for raw in pq.read_table(Path(root) / "typed-decisions" / "all" / f"{split}-00000-of-00001.parquet").to_pylist():
            cases.append({"id": raw["id"], "group": raw["workflow"], "state": json.loads(raw["state"]),
                          "questions": json.loads(raw["questions"]), "gold": json.loads(raw["gold"]),
                          "gold_source": TypedDecisions.gold_source})
        return cases

    @staticmethod
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

    @staticmethod
    def targets(case):
        out = []
        for qid, gq in case["gold"].items():
            q = case["questions"][qid]
            keys = answer_keys(q)
            gkey = {"yes": "true", "no": "false"} if gq["type"] == "noul" else {}
            out.append((qid, gq["type"], keys, [float(gq["probabilities"].get(gkey.get(k, k), 0.0)) for k in keys]))
        return out

    @staticmethod
    def metrics(rows, cases):
        gold_by_id = {c["id"]: c["gold"] for c in cases}
        per = {}
        allv = {"hit": [], "kl": [], "brier": [], "conf": [], "mass": []}
        for row in rows:
            for qid, gq in gold_by_id[row["id"]].items():
                ans = answer_probs(row, qid)
                if not set(norm_probs(ans["probabilities"])) & set(gq["probabilities"]):
                    # Keys that match no gold option would also score as uniform.
                    raise RuntimeError(f"case {row['id']} question {qid}: answer keys "
                                       f"{sorted(ans['probabilities'])} match none of {sorted(gq['probabilities'])}")
                ok, kl, brier, conf = TypedDecisions.score_one(ans, gq)
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
                    "label_mass_mean": mean_or_none(d["mass"]),
                    "label_mass_p10": sorted(d["mass"])[len(d["mass"]) // 10] if d["mass"] else None}

        out = summarize(allv)
        out["by_type"] = {t: summarize(d) for t, d in sorted(per.items())}
        out["headline"] = f"acc {out['accuracy']:.4f}, KL {out['kl_from_gold']:.4f}, Brier {out['brier']:.4f}, ECE {out['ece']:.4f}"
        return out

    @staticmethod
    def case_scores(rows, cases):
        """Per case: hits over its decisions (paired intervals resample cases)."""
        gold_by_id = {c["id"]: c["gold"] for c in cases}
        return {r["id"]: statistics.fmean(TypedDecisions.score_one(answer_probs(r, q), gq)[0]
                                          for q, gq in gold_by_id[r["id"]].items()) for r in rows}


# ------------------------------------------------------------------ Typed Decision Bench
def tdb_item_proper_score(item, answers):
    """Typed Decision Bench's item_proper_score (blobfishai/jevfish
    benchmark/typed_decision_bench/site_data.py @ 80262b5), unchanged: mean over the gold
    questions of 1 - normalised Brier; None when no gold question was answered."""
    parts = []
    for qid, gold in item["gold"].items():
        q = item["questions"][qid]
        ans = (answers or {}).get(qid)
        if not ans:
            continue
        if q["type"] == "noul":
            p = ans.get("noul")
            if p is None:
                continue
            parts.append(1.0 - (float(p) - int(gold)) ** 2)
        else:
            probs = ans.get("probabilities") or {}
            if not probs:
                continue
            g = str(gold)
            keys = set(probs) | {g}
            parts.append(1.0 - sum((float(probs.get(k, 0.0)) - (1.0 if k == g else 0.0)) ** 2 for k in keys) / 2.0)
    return statistics.mean(parts) if parts else None


def tdb_top_answer(q, ans):
    """Typed Decision Bench's top_answer (quality.py @ 80262b5), unchanged."""
    if not ans:
        return None
    if q["type"] == "noul":
        p = ans.get("noul")
        return None if p is None else int(p >= 0.5)
    probs = ans.get("probabilities") or {}
    if not probs:
        return None
    best = min(probs, key=lambda k: (-probs[k], k))
    return int(best) if q["type"] == "score" else best


class TypedDecisionBench:
    """SamuelChien821/typed-decision-bench v0.3: 5,387 items in 25 tasks, labels from 21 public
    datasets and one generator (no LLM labels). The 359 withheld items (Abt-Buy, TrialGPT) are
    rebuilt with the jevfish builder (`python -m typed_decision_bench.freeze --only
    abt-buy-product-match,trialgpt-criterion-support --seed 20260919 --version 0.3.0`) and merged
    into bench-complete/; load() checks every suite file against the dataset's SHA256SUMS.

    Score: the benchmark's DecisionScore (mean per-item proper score x 100 over all items, an
    unanswered item scores 0) and its accuracy (top answer of each item's primary question)."""
    name = "typed-decision-bench"
    revision = {"repo": "SamuelChien821/typed-decision-bench", "version": "0.3.0",
                "revision": "79ec2b9bf193d69ceaaaf86ad9d0b79623adb101",
                "scorer": "blobfishai/jevfish@80262b5c7049179a5def5e15fc85c51e3f9d7314"}
    gold_source = "human"  # source datasets' own labels (and one rule generator); none from an LLM
    splits = ("test",)

    @staticmethod
    def load(root, split):
        base = Path(root) / "typed-decision-bench"
        sums = dict(reversed(line.split()) for line in (base / "bench" / "SHA256SUMS").read_text().splitlines() if line.strip())
        cases = []
        for rel, digest in sorted(sums.items()):
            data = (base / "bench-complete" / rel).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"{rel}: checksum differs from SHA256SUMS (rebuild the withheld items)")
            for line in data.decode().splitlines():
                it = json.loads(line)
                src = "rule" if it["meta"]["task"] == "trial-outcomes" else "human"
                cases.append({"id": it["id"], "group": it["meta"]["task"], "suite": it["suite"],
                              "primary": it["meta"].get("primary") or next(iter(it["gold"])),
                              "state": it["state"], "questions": it["questions"], "gold": it["gold"],
                              "gold_source": src})
        return cases

    @staticmethod
    def targets(case):
        out = []
        for qid, g in case["gold"].items():
            q = case["questions"][qid]
            keys = answer_keys(q)
            want = ("yes" if int(g) else "no") if q["type"] == "noul" else str(g)
            out.append((qid, q["type"], keys, [1.0 if k == want else 0.0 for k in keys]))
        return out

    @staticmethod
    def case_scores(rows, cases):
        by = {r["id"]: r for r in rows}
        return {c["id"]: tdb_item_proper_score(c, (by.get(c["id"]) or {}).get("answers")) or 0.0 for c in cases}

    @staticmethod
    def metrics(rows, cases):
        by = {r["id"]: r for r in rows}
        scores = TypedDecisionBench.case_scores(rows, cases)
        per_task, per_suite = defaultdict(list), defaultdict(list)
        hits = []
        answered = 0
        for c in cases:
            r = by.get(c["id"])
            if r and tdb_item_proper_score(c, r["answers"]) is not None:
                answered += 1
            per_task[c["group"]].append(scores[c["id"]])
            per_suite[c["suite"]].append(scores[c["id"]])
            q = c["questions"][c["primary"]]
            pred = tdb_top_answer(q, (r or {}).get("answers", {}).get(c["primary"]))
            if pred is not None:
                want = str(c["gold"][c["primary"]]) if q["type"] == "choice" else int(c["gold"][c["primary"]])
                hits.append(pred == want)
        out = {"n_items": len(cases), "answered": answered,
               "decision_score": 100 * statistics.fmean(scores.values()),
               "accuracy": sum(hits) / len(hits) if hits else None,
               "by_suite": {s: 100 * statistics.fmean(v) for s, v in sorted(per_suite.items())},
               "by_task": {t: 100 * statistics.fmean(v) for t, v in sorted(per_task.items())}}
        out["headline"] = f"DecisionScore {out['decision_score']:.2f}, accuracy {out['accuracy']:.4f}, answered {answered}/{len(cases)}"
        return out


# ------------------------------------------------------------------ procedural typed decisions
class ProceduralTypedDecisions:
    """tasksource/procedural-typed-decisions `all`: answers computed exactly from rules the state
    spells out (generators in sileod/reasoning-core). The split is large (18,000 test states), so
    load() takes a fixed stratified sample: PER_STRATUM states per (task, level), the ones with the
    smallest sha256(id), i.e. 18 tasks x 5 levels x 10 = 900 states.

    The dataset ships no scorer. Score: Typed Decision Bench's per-item proper score (mean over
    the questions of 1 - normalised Brier) with the gold as given, so graded `noul` answers (exact
    probabilities such as `incident_real`) are scored against the probability, not a rounded
    label; accuracy is over the questions with an exact label (graded `noul` excluded)."""
    name = "procedural-typed-decisions"
    revision = {"repo": "tasksource/procedural-typed-decisions", "config": "all",
                "revision": "05ca1d0388f57461a634a7caed67145d28e2eca7"}
    gold_source = "rule"
    splits = ("test", "validation")
    PER_STRATUM = 10

    @staticmethod
    def load(root, split):
        rows = pq.read_table(Path(root) / "procedural-typed-decisions" / "all" / f"{split}-00000-of-00001.parquet").to_pylist()
        strata = defaultdict(list)
        for r in rows:
            strata[(r["task"], r["level"])].append(r)
        cases = []
        for key in sorted(strata):
            pick = sorted(strata[key], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:ProceduralTypedDecisions.PER_STRATUM]
            for r in pick:
                try:
                    state = json.loads(r["state"])
                except json.JSONDecodeError:
                    state = r["state"]  # rendered text (tables, prose); sent as a string
                answers = json.loads(r["answers"])
                gold = {}
                for qid, a in answers.items():
                    gold[qid] = {"noul": a.get("noul"), "choice": a.get("choice"), "score": a.get("score")}[a["type"]]
                cases.append({"id": r["id"], "group": r["task"], "level": int(r["level"]), "state": state,
                              "questions": json.loads(r["questions"]), "gold": gold, "gold_source": "rule"})
        return cases

    @staticmethod
    def _target(q, g):
        keys = answer_keys(q)
        if q["type"] == "noul":
            return keys, [float(g), 1.0 - float(g)]
        want = str(g) if q["type"] == "choice" else str(int(g))
        return keys, [1.0 if k == want else 0.0 for k in keys]

    @staticmethod
    def targets(case):
        return [(qid, case["questions"][qid]["type"], *ProceduralTypedDecisions._target(case["questions"][qid], g))
                for qid, g in case["gold"].items()]

    @staticmethod
    def case_scores(rows, cases):
        by = {r["id"]: r for r in rows}
        out = {}
        for c in cases:
            parts = []
            for qid, keys, tgt in ((t[0], t[2], t[3]) for t in ProceduralTypedDecisions.targets(c)):
                p = answer_probs(by[c["id"]], qid)["probabilities"]
                pv = [float(p.get(k, 0.0)) for k in keys]
                if c["questions"][qid]["type"] == "noul":
                    parts.append(1.0 - (pv[0] - tgt[0]) ** 2)
                else:
                    extra = sum(float(v) ** 2 for k, v in p.items() if k not in keys)
                    parts.append(1.0 - (sum((a - b) ** 2 for a, b in zip(pv, tgt)) + extra) / 2.0)
            out[c["id"]] = statistics.fmean(parts)
        return out

    @staticmethod
    def metrics(rows, cases):
        by = {r["id"]: r for r in rows}
        scores = ProceduralTypedDecisions.case_scores(rows, cases)
        per_task, per_level = defaultdict(list), defaultdict(list)
        hits, hits_by_type = [], defaultdict(list)
        for c in cases:
            per_task[c["group"]].append(scores[c["id"]])
            per_level[c["level"]].append(scores[c["id"]])
            for qid, qtype, keys, tgt in ProceduralTypedDecisions.targets(c):
                if qtype == "noul" and tgt[0] not in (0.0, 1.0):
                    continue
                p = answer_probs(by[c["id"]], qid)["probabilities"]
                ok = max(keys, key=lambda k: float(p.get(k, 0.0))) == keys[tgt.index(1.0)]
                hits.append(ok)
                hits_by_type[qtype].append(ok)
        out = {"n_cases": len(cases), "proper_score": 100 * statistics.fmean(scores.values()),
               "accuracy": sum(hits) / len(hits), "n_exact_questions": len(hits),
               "accuracy_by_type": {t: sum(v) / len(v) for t, v in sorted(hits_by_type.items())},
               "by_task": {t: 100 * statistics.fmean(v) for t, v in sorted(per_task.items())},
               "by_level": {str(k): 100 * statistics.fmean(v) for k, v in sorted(per_level.items())}}
        out["headline"] = f"proper score {out['proper_score']:.2f}, accuracy {out['accuracy']:.4f} ({len(hits)} exact questions)"
        return out


# ------------------------------------------------------------------ calibrated typed decisions
class CalibratedTypedDecisions:
    """m-lagnajit/calibrated-typed-decisions v2.1: one question per row over a short text, with the
    source dataset's human label as `gold` (the soft teacher labels are not used here). Splits:
    `test` (six sources: CLINC150, GoEmotions, BANKING77, MASSIVE, Amazon polarity, SciCite),
    `heldout` (1,000 expert-labelled PubMedQA yes/no/maybe questions), `validation` (fits the
    calibration). Options in the dataset's canonical order, each option's description as its
    criterion.

    The dataset ships no scorer; its card reports accuracy, ECE and log loss. Score: accuracy of
    the top answer, log loss of the gold answer (probabilities clipped at 1e-6), multi-class Brier,
    and ECE over 10 equal-width bins of the top probability."""
    name = "calibrated-typed-decisions"
    revision = {"repo": "m-lagnajit/calibrated-typed-decisions", "version": "2.1",
                "revision": "34189779d116d0523db6af58c0d62538f7e7d07e"}
    gold_source = "human"
    splits = ("test", "heldout", "validation")
    CLIP = 1e-6

    @staticmethod
    def load(root, split):
        cases = []
        for r in pq.read_table(Path(root) / "calibrated-typed-decisions" / "data" / f"{split}-00000-of-00001.parquet").to_pylist():
            opts, descs = r["options"], r["option_descriptions"]
            if r["type"] == "noul":
                q = {"type": "noul", "instructions": r["instructions"]}
                gold = "yes" if r["gold"] == "true" else "no"
            else:
                q = {"type": "choice", "instructions": r["instructions"],
                     "criteria": {o: (d or None) for o, d in zip(opts, descs)}}
                gold = r["gold"]
            cases.append({"id": r["id"], "group": f"{r['source'].split('/')[-1]}/{r['type']}", "state": r["state"],
                          "questions": {"answer": q}, "gold": {"answer": gold}, "gold_source": "human"})
        return cases

    @staticmethod
    def targets(case):
        q = case["questions"]["answer"]
        keys = answer_keys(q)
        return [("answer", q["type"], keys, [1.0 if k == case["gold"]["answer"] else 0.0 for k in keys])]

    @staticmethod
    def _one(row, case):
        _, _, keys, tgt = CalibratedTypedDecisions.targets(case)[0]
        p = answer_probs(row, "answer")["probabilities"]
        pv = [float(p.get(k, 0.0)) for k in keys]
        s = sum(pv)
        pv = [v / s for v in pv] if s > 0 else [1.0 / len(keys)] * len(keys)
        g = tgt.index(1.0)
        top = max(range(len(keys)), key=lambda i: (pv[i], -i))
        logloss = -math.log(min(max(pv[g], CalibratedTypedDecisions.CLIP), 1 - CalibratedTypedDecisions.CLIP))
        return top == g, logloss, sum((a - b) ** 2 for a, b in zip(pv, tgt)), pv[top]

    @staticmethod
    def case_scores(rows, cases):
        """Per case: hit (paired intervals on accuracy)."""
        by = {c["id"]: c for c in cases}
        return {r["id"]: float(CalibratedTypedDecisions._one(r, by[r["id"]])[0]) for r in rows}

    @staticmethod
    def metrics(rows, cases):
        by = {c["id"]: c for c in cases}
        groups = defaultdict(list)
        allv = []
        for r in rows:
            m = CalibratedTypedDecisions._one(r, by[r["id"]])
            allv.append(m)
            groups[by[r["id"]]["group"]].append(m)

        def summarize(ms):
            return {"n": len(ms), "accuracy": statistics.fmean(m[0] for m in ms),
                    "log_loss": statistics.fmean(m[1] for m in ms), "brier": statistics.fmean(m[2] for m in ms),
                    "ece": ece([m[0] for m in ms], [m[3] for m in ms])}

        out = summarize(allv)
        out["by_group"] = {g: summarize(v) for g, v in sorted(groups.items())}
        out["headline"] = f"acc {out['accuracy']:.4f}, log loss {out['log_loss']:.4f}, Brier {out['brier']:.4f}, ECE {out['ece']:.4f}"
        return out


SETS = {s.name: s for s in (TypedDecisions, TypedDecisionBench, ProceduralTypedDecisions, CalibratedTypedDecisions)}


def get(name):
    if name not in SETS:
        raise RuntimeError(f"unknown set {name!r}; known: {', '.join(SETS)}")
    return SETS[name]


def folds(cases, k, seed=0):
    """Case ids split into k folds, stratified by group (cross-fitted calibration)."""
    by = defaultdict(list)
    for c in cases:
        by[c["group"]].append(c["id"])
    rng = random.Random(seed)
    out = [[] for _ in range(k)]
    i = 0
    for g in sorted(by):
        ids = sorted(by[g])
        rng.shuffle(ids)
        for cid in ids:
            out[i % k].append(cid)
            i += 1
    return out
