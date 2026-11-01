"""Decision datasets as one case format, each with its own scorer.

A case is {"id", "group", "state", "questions", "gold", "gold_source"}: `state` and `questions`
go to /v1/systemone as they are (TypeSafe System One shape: `noul`, `choice`, `score`), `group`
is the workflow, task or source (per-group breakdown and calibration biases), `gold` is the set's
own answer format, and `gold_source` says where the answers come from:
`human`, `rule` or `model:<name>` (`model:unnamed` when the dataset doesn't say). An image set's
cases also carry `images`: data URLs sent as the decision routes' top-level `images` field.

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


# ------------------------------------------------------------------ human label distributions
def sha_half(key):
    """'validation' or 'test' by the parity of sha256(key): a fixed split for sets with one split."""
    return "validation" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def sha_sample(cases, n, key=lambda c: c["id"]):
    """The n cases with the smallest sha256(key), in that order (a fixed sample)."""
    return sorted(cases, key=lambda c: hashlib.sha256(key(c).encode()).hexdigest())[:n] if n else cases


def image_data_url(image):
    """A parquet image cell ({"bytes", ...}) as a data URL; the type from the file's magic bytes."""
    import base64
    b = image["bytes"]
    mime = "image/png" if b[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg" if b[:3] == b"\xff\xd8\xff" else None
    if mime is None:
        raise RuntimeError(f"unknown image format (first bytes {b[:8]!r})")
    return f"data:{mime};base64," + base64.b64encode(b).decode()


class LabelDistributionSet:
    """Base for sets whose gold is, per question, a distribution over its answer keys (aligned with
    answer_keys: human vote shares, or one-hot for a single label). Per decision: accuracy of the
    top answer against the gold's top answer(s) (a tie in the gold counts any tied answer),
    KL(gold || pred) (probabilities clipped at 1e-9, as the Typed Decisions scorer; for one-hot gold
    it is the log loss), multi-class Brier, and ECE over 10 bins of the top probability. Per case
    (case_scores, paired intervals): mean KL over its decisions."""
    splits = ("validation", "test")

    @classmethod
    def targets(cls, case):
        return [(qid, q["type"], answer_keys(q), case["gold"][qid]) for qid, q in case["questions"].items()]

    @classmethod
    def _decisions(cls, row, case):
        out = []
        for qid, qtype, keys, g in cls.targets(case):
            p = answer_probs(row, qid)["probabilities"]
            pv = [float(p.get(k, 0.0)) for k in keys]
            s = sum(pv)
            if s <= 0:
                raise RuntimeError(f"case {row['id']} question {qid}: answer keys {sorted(p)} match none of {keys}")
            pv = [v / s for v in pv]
            top = max(range(len(keys)), key=lambda i: (pv[i], -i))
            hit = g[top] == max(g)
            kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, pv) if gi > 0)
            out.append((qtype, hit, kl, sum((a - b) ** 2 for a, b in zip(pv, g)), pv[top]))
        return out

    @classmethod
    def case_scores(cls, rows, cases):
        by = {c["id"]: c for c in cases}
        return {r["id"]: statistics.fmean(d[2] for d in cls._decisions(r, by[r["id"]])) for r in rows}

    @classmethod
    def metrics(cls, rows, cases):
        by = {c["id"]: c for c in cases}
        allv, per_type, per_group = [], defaultdict(list), defaultdict(list)
        for r in rows:
            for d in cls._decisions(r, by[r["id"]]):
                allv.append(d)
                per_type[d[0]].append(d)
                per_group[by[r["id"]]["group"]].append(d)

        def summarize(ds):
            return {"n_decisions": len(ds), "accuracy": statistics.fmean(d[1] for d in ds),
                    "kl_from_gold": statistics.fmean(d[2] for d in ds), "brier": statistics.fmean(d[3] for d in ds),
                    "ece": ece([d[1] for d in ds], [d[4] for d in ds])}

        out = summarize(allv)
        out["by_type"] = {t: summarize(v) for t, v in sorted(per_type.items())}
        out["by_group"] = {g: summarize(v) for g, v in sorted(per_group.items())}
        out["headline"] = f"acc {out['accuracy']:.4f}, KL {out['kl_from_gold']:.4f}, Brier {out['brier']:.4f}, ECE {out['ece']:.4f}"
        return out


class ChaosNLI(LabelDistributionSet):
    """earino/chaosnli (mirror of ChaosNLI v1.0, Nie et al. 2020): SNLI and MNLI-matched dev items
    with 100 new crowd labels each (`label_dist`, order e/n/c), as a 3-option choice; αNLI dev items
    with 100 labels over its two hypotheses, as a 2-option choice. Gold: the 100-vote distribution.
    One split upstream, so `validation` (calibration fit) and `test` are the two halves by
    sha256(uid) parity, per source. Licence cc-by-nc-4.0."""
    name = "chaosnli"
    revision = {"repo": "earino/chaosnli", "revision": "bf8d66a1d3434ac8b90ffc392b1b5e44bdf7e439"}
    gold_source = "human"
    NLI = {"entailment": "The hypothesis must be true if the premise is true.",
           "neutral": "The hypothesis may or may not be true given the premise.",
           "contradiction": "The hypothesis cannot be true if the premise is true."}

    @staticmethod
    def load(root, split):
        cases = []
        for src in ("snli", "mnli_m", "alphanli"):
            for line in open(Path(root) / "chaosnli" / "raw" / f"chaosNLI_{src}.jsonl"):
                r = json.loads(line)
                if sha_half(r["uid"]) != split:
                    continue
                ex = r["example"]
                if src == "alphanli":
                    state = {"beginning": ex["obs1"], "ending": ex["obs2"]}
                    q = {"type": "choice", "instructions": "Which explanation best connects the beginning to the ending?",
                         "criteria": {"hyp1": ex["hyp1"], "hyp2": ex["hyp2"]}}
                else:
                    state = {"premise": ex["premise"], "hypothesis": ex["hypothesis"]}
                    q = {"type": "choice", "instructions": "How does the hypothesis relate to the premise?",
                         "criteria": dict(ChaosNLI.NLI)}
                cases.append({"id": f"{src}-{r['uid']}", "group": src, "state": state, "questions": {"answer": q},
                              "gold": {"answer": [float(x) for x in r["label_dist"]]}, "gold_source": "human"})
        return cases


class LeWiDi(LabelDistributionSet):
    """tasksource/lewidi (LeWiDi soft labels): three English configs with real splits.
    `csc` sarcasm rated 1-6 by 4+ annotators -> `score` over 6 levels; `md_agreement` offensiveness
    (5 annotators) and `hs_brexit` hate speech (6) -> `noul`. Gold: the annotators' label shares
    (`soft_label`, order no/yes for the binary configs, checked on clear rows). `validation` fits
    the calibration, `test` is scored. Licence: each source dataset's terms."""
    name = "lewidi"
    revision = {"repo": "tasksource/lewidi", "revision": "9ba2e83b536ca127fe8b25f2a2725f1e9a489c82"}
    gold_source = "human"
    CSC_LEVELS = ["Not sarcastic at all", "Barely sarcastic", "Slightly sarcastic", "Somewhat sarcastic",
                  "Clearly sarcastic", "Extremely sarcastic"]
    BINARY = {"md_agreement": "Is this post offensive?", "hs_brexit": "Is this post hate speech?"}

    @staticmethod
    def load(root, split):
        cases = []
        base = Path(root) / "lewidi"
        for r in pq.read_table(base / "csc" / f"{split}-00000-of-00001.parquet").to_pylist():
            cases.append({"id": f"csc-{r['id']}", "group": "csc", "state": {"context": r["context"], "response": r["response"]},
                          "questions": {"answer": {"type": "score", "instructions": "How sarcastic is the response, given the context?",
                                                   "criteria": list(LeWiDi.CSC_LEVELS)}},
                          "gold": {"answer": [float(x) for x in r["soft_label"]]}, "gold_source": "human"})
        for cfg, question in LeWiDi.BINARY.items():
            for r in pq.read_table(base / cfg / f"{split}-00000-of-00001.parquet").to_pylist():
                no, yes = r["soft_label"]
                cases.append({"id": f"{cfg}-{r['id']}", "group": cfg, "state": r["text"],
                              "questions": {"answer": {"type": "noul", "instructions": question}},
                              "gold": {"answer": [float(yes), float(no)]}, "gold_source": "human"})
        return cases


class CIFAR10H(LabelDistributionSet):
    """MKZuziak/cifar10h (CIFAR-10H, Peterson et al. 2019): the 10,000 CIFAR-10 test images
    (32x32, sent as they are) with about 50 human labels each (`expert_probs`), as a 10-option
    choice. Gold: the human label distribution. One split upstream: halves by sha256 of the row
    index, then a fixed sample (smallest sha256) of 1,000 for `validation` and 2,000 for `test`."""
    name = "cifar10h"
    revision = {"repo": "MKZuziak/cifar10h", "revision": "0ec71d3ea7d6b3b9cc4bb03362f1268e9cdf4542"}
    gold_source = "human"
    CLASSES = {"airplane": None, "automobile": "car, not truck", "bird": None, "cat": None, "deer": None,
               "dog": None, "frog": None, "horse": None, "ship": None, "truck": "large truck, not car"}
    SIZES = {"validation": 1000, "test": 2000}

    @staticmethod
    def load(root, split):
        t = pq.read_table(Path(root) / "cifar10h" / "data" / "train-00000-of-00001.parquet")
        keep = sha_sample([{"id": f"cifar10h-{i}", "i": i} for i in range(t.num_rows) if sha_half(f"cifar10h-{i}") == split],
                          CIFAR10H.SIZES[split])
        rows = t.take([c["i"] for c in keep]).to_pylist()
        return [{"id": c["id"], "group": "cifar10h", "state": "The attached image.", "images": [image_data_url(r["image"])],
                 "questions": {"answer": {"type": "choice", "instructions": "What is the main object in the image?",
                                          "criteria": dict(CIFAR10H.CLASSES)}},
                 "gold": {"answer": [float(x) for x in r["expert_probs"]]}, "gold_source": "human"}
                for c, r in zip(keep, rows)]


class POPE(LabelDistributionSet):
    """lmms-lab-encoder/POPE (Li et al. 2023): yes/no "Is there a <object> in the image?" over 500
    COCO val2014 images in three settings (random, popular, adversarial negatives), 6 questions per
    image and setting. Gold: one-hot from COCO's object annotations (rule-derived from human
    boxes). One case per image and setting (its 6 questions in one request). One split upstream:
    halves by sha256(image), so no image is in both; then a fixed sample (smallest sha256 of the
    case id) of 300 cases for `validation` and 600 for `test` (6 questions each)."""
    name = "pope"
    revision = {"repo": "lmms-lab-encoder/POPE", "revision": "4db1276663dfa5eb8ad16a52d24c31a09e470896"}
    gold_source = "rule"
    SIZES = {"validation": 300, "test": 600}

    @staticmethod
    def load(root, split):
        cases = {}
        for f in sorted((Path(root) / "pope" / "data").glob("test-*.parquet")):
            for r in pq.read_table(f).to_pylist():
                if sha_half(r["image_source"]) != split:
                    continue
                cid = f"{r['image_source']}-{r['category']}"
                c = cases.setdefault(cid, {"id": cid, "group": r["category"], "state": "The attached image.",
                                           "images": [image_data_url(r["image"])], "questions": {}, "gold": {},
                                           "gold_source": "rule"})
                qid = f"q{r['question_id']}"
                c["questions"][qid] = {"type": "noul", "instructions": r["question"]}
                c["gold"][qid] = [1.0, 0.0] if r["answer"] == "yes" else [0.0, 1.0]
        return sha_sample(list(cases.values()), POPE.SIZES[split])


class SoftDigits(LabelDistributionSet):
    """majapavlo/image-level-soft-label-digits (soft-digits, arXiv 2605.18648): 5,530 28x28
    grayscale handwritten digits (MNIST and Ambiguous-MNIST) re-annotated by 3-10 Prolific raters
    each (6 on average), who marked every digit Yes/No/Unsure. Gold: `soft_label_yes_unc_equal`
    (the paper's equally weighted soft label: Yes and Unsure both count 1, per rater, averaged) over
    0-9 plus a "not a digit" class (a rater who rejected all ten), as an 11-option choice. Real
    splits: `val` (830) fits, `test` (830) is scored. Pixels are floats in [0, 1], sent as a
    grayscale PNG (run.py converts it to RGB). Licence cc-by-sa-4.0."""
    name = "soft-digits"
    revision = {"repo": "majapavlo/image-level-soft-label-digits", "revision": "3f231e9a1481a8b37f7a47e313221f66d9e22639"}
    gold_source = "human"
    OPTIONS = {**{str(d): None for d in range(10)}, "not a digit": "none of 0-9"}
    SPLITS = {"validation": "val", "test": "test"}

    @staticmethod
    def load(root, split):
        import base64
        import io
        from PIL import Image
        t = pq.read_table(Path(root) / "soft-digits" / "image_level.parquet",
                          columns=["images", "indices", "source", "soft_label_yes_unc_equal", "split"])
        cases = []
        for r in t.to_pylist():
            if r["split"] != SoftDigits.SPLITS[split]:
                continue
            im = Image.frombytes("L", (28, 28), bytes(round(255 * v) for v in r["images"]))
            buf = io.BytesIO()
            im.save(buf, "PNG")
            cases.append({"id": f"{r['source']}-{r['indices']}", "group": r["source"], "state": "The attached image.",
                          "images": ["data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()],
                          "questions": {"answer": {"type": "choice", "instructions": "Which digit is written in the image?",
                                                   "criteria": dict(SoftDigits.OPTIONS)}},
                          "gold": {"answer": [float(x) for x in json.loads(r["soft_label_yes_unc_equal"])]},
                          "gold_source": "human"})
        return cases


class RapidataStyle(LabelDistributionSet):
    """Rapidata/human-style-preferences-images: text-to-image outputs in pairs, with about 26
    crowd votes per pair on which image is preferred (`votes_image1/2`, unweighted counts). The
    first three of 53 shards (3,609 pairs, all DALL-E 3 vs Flux 1 pro, 238 prompts). The card
    doesn't give the raters' question, so the wording is ours. A 2-option choice over the two
    images, sent in an order set by sha256(id) parity (the shards always list DALL-E 3 first).
    Gold: the vote shares. One split upstream: halves by sha256(prompt), so no prompt is in
    both; then a fixed sample of 1,000 pairs for `validation` and 1,500 for `test`.
    Licence cdla-permissive-2.0."""
    name = "rapidata-style"
    revision = {"repo": "Rapidata/human-style-preferences-images", "revision": "0b9a8d940f4c8e5307019fb207b5707e1e31716c"}
    gold_source = "human"
    SHARDS = 3
    SIZES = {"validation": 1000, "test": 1500}

    @staticmethod
    def load(root, split):
        cases = []
        for k in range(RapidataStyle.SHARDS):
            t = pq.read_table(Path(root) / "rapidata-style" / "data" / f"train-{k:05d}-of-00053.parquet",
                              columns=["prompt", "image1", "image2", "votes_image1", "votes_image2", "image1_path", "image2_path"])
            for r in t.to_pylist():
                if sha_half(r["prompt"]) != split:
                    continue
                cid = f"{r['image1_path']}|{r['image2_path']}"
                n = r["votes_image1"] + r["votes_image2"]
                imgs, gold = [r["image1"], r["image2"]], [r["votes_image1"] / n, r["votes_image2"] / n]
                if int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 2:
                    imgs, gold = imgs[::-1], gold[::-1]
                cases.append({"id": cid, "group": "style", "state": {"prompt": r["prompt"]},
                              "images": [image_data_url(i) for i in imgs],
                              "questions": {"answer": {"type": "choice",
                                                       "instructions": "Both attached images were generated from the prompt. Which image do you prefer?",
                                                       "criteria": {"Image 1": "the first attached image",
                                                                    "Image 2": "the second attached image"}}},
                              "gold": {"answer": gold}, "gold_source": "human"})
        return sha_sample(cases, RapidataStyle.SIZES[split])


class VQAv2YesNo(LabelDistributionSet):
    """lmms-lab-encoder/VQAv2 (Goyal et al. 2017) validation, first six of 68 shards: the
    `yes/no` questions, each with 10 crowd answers. Gold: the share of `yes` among the `yes` and
    `no` answers, as a noul; questions with fewer than 8 such answers are dropped. COCO val2014
    photos, the 30 that are also POPE images left out. One case per image (its questions in one
    request). One split upstream: halves by sha256(image id); then a fixed sample of 500 images
    for `validation` and 1,000 for `test`. Licence cc-by-4.0."""
    name = "vqav2"
    revision = {"repo": "lmms-lab-encoder/VQAv2", "revision": "32665d35052eb4a6d4414851c3c829a72754915a"}
    gold_source = "human"
    SHARDS = 6
    MIN_YES_NO = 8
    SIZES = {"validation": 500, "test": 1000}

    @staticmethod
    def load(root, split):
        pope = set()
        for f in (Path(root) / "pope" / "data").glob("test-*.parquet"):
            pope.update(pq.read_table(f, columns=["image_source"]).column("image_source").to_pylist())
        cases = {}
        for k in range(VQAv2YesNo.SHARDS):
            t = pq.read_table(Path(root) / "vqav2" / "data" / f"validation-{k:05d}-of-00068.parquet",
                              columns=["answer_type", "answers", "image_id", "question_id", "question", "image"])
            for r in t.to_pylist():
                img = f"COCO_val2014_{r['image_id']:012d}"
                if r["answer_type"] != "yes/no" or img in pope or sha_half(img) != split:
                    continue
                a = [x["answer"] for x in r["answers"]]
                yes, no = a.count("yes"), a.count("no")
                if yes + no < VQAv2YesNo.MIN_YES_NO:
                    continue
                c = cases.setdefault(img, {"id": img, "group": "yes/no", "state": "The attached image.",
                                           "images": [image_data_url(r["image"])], "questions": {}, "gold": {},
                                           "gold_source": "human"})
                qid = f"q{r['question_id']}"
                c["questions"][qid] = {"type": "noul", "instructions": r["question"]}
                c["gold"][qid] = [yes / (yes + no), no / (yes + no)]
        return sha_sample(list(cases.values()), VQAv2YesNo.SIZES[split])


class AVA(LabelDistributionSet):
    """trojblue/AVA-aesthetics-10pct-min50-10bins (a subset of AVA, Murray et al. 2012): photos from
    DPChallenge.com photo contests, each rated 1-10 by its voters (at least 50, ~200 typical;
    `rating_counts`), as a 10-level score. Gold: the vote shares. Real splits: a fixed sample of
    1,000 of `validation` (5,110) fits, 2,000 of `test` (5,110, in `data/test-*`, outside the
    card's config) are scored. Licence: not stated on the card (AVA / DPChallenge terms)."""
    name = "ava"
    revision = {"repo": "trojblue/AVA-aesthetics-10pct-min50-10bins", "revision": "ee09ff2b4151b87c61cfb5165bb46c0ebe39b679"}
    gold_source = "human"
    LEVELS = ["1 (lowest)", "2", "3", "4", "5", "6", "7", "8", "9", "10 (highest)"]
    SIZES = {"validation": 1000, "test": 2000}

    @staticmethod
    def load(root, split):
        files = sorted((Path(root) / "ava" / "data").glob(f"{split}-*.parquet"))
        ids = [i for f in files for i in pq.read_table(f, columns=["image_id"]).column("image_id").to_pylist()]
        keep = {c["id"] for c in sha_sample([{"id": f"ava-{i}"} for i in ids], AVA.SIZES[split])}
        cases = []
        for f in files:
            for r in pq.read_table(f, columns=["image_id", "image", "total_votes", "rating_counts"]).to_pylist():
                if f"ava-{r['image_id']}" not in keep:
                    continue
                cases.append({"id": f"ava-{r['image_id']}", "group": "ava", "state": "The attached photo, entered in a photography contest.",
                              "images": [image_data_url(r["image"])],
                              "questions": {"answer": {"type": "score", "instructions": "How would the contest's voters rate this photo's aesthetic quality?",
                                                       "criteria": list(AVA.LEVELS)}},
                              "gold": {"answer": [n / r["total_votes"] for n in r["rating_counts"]]}, "gold_source": "human"})
        return sha_sample(cases, AVA.SIZES[split])


class KonIQ10k(LabelDistributionSet):
    """KonIQ-10k (Hosu et al. 2020): 10,073 Flickr photos (YFCC100M) at 1024x768, each rated for
    technical quality on a 5-point scale (bad ... excellent) by ~106 crowd workers. Images from
    xyliu6/koniq10k-1024x768, labels from chaofengc/IQA-PyTorch-Datasets-metainfo
    (`meta_info_KonIQ10kDataset.csv`: shares `c1..c5`), as a 5-level score. Gold: the rating
    shares. Official splits: `val` (1,000) fits, a fixed sample of 2,000 of `test` (2,015) is
    scored. Licence: images cc-by-4.0 (each its own CC licence), labels cc-by-sa-4.0."""
    name = "koniq10k"
    revision = {"repo": "xyliu6/koniq10k-1024x768", "revision": "e06f08f327a532537cdb059a91a07a2cd1f7580f",
                "labels": {"repo": "chaofengc/IQA-PyTorch-Datasets-metainfo", "revision": "4ee2b4f85138bac8f875d61dc9b4c50e006f8466"}}
    gold_source = "human"
    LEVELS = ["bad", "poor", "fair", "good", "excellent"]
    SPLITS = {"validation": "val", "test": "test"}
    SIZES = {"validation": 1000, "test": 2000}

    @staticmethod
    def load(root, split):
        import csv
        import zipfile
        rows = [r for r in csv.DictReader(open(Path(root) / "koniq10k-meta" / "meta_info_KonIQ10kDataset.csv"))
                if r["official_split"] == KonIQ10k.SPLITS[split]]
        rows = sha_sample(rows, KonIQ10k.SIZES[split], key=lambda r: r["img_name"])
        z = zipfile.ZipFile(Path(root) / "koniq10k" / "koniq10k_1024x768.zip")
        return [{"id": f"koniq-{r['img_name']}", "group": "koniq10k", "state": "The attached photo.",
                 "images": [image_data_url({"bytes": z.read(f"1024x768/{r['img_name']}")})],
                 "questions": {"answer": {"type": "score", "instructions": "How would you rate the technical quality of this photo?",
                                          "criteria": list(KonIQ10k.LEVELS)}},
                 "gold": {"answer": [float(r[f"c{i}"]) for i in range(1, 6)]}, "gold_source": "human"}
                for r in rows]


class VizWizAnswerable(LabelDistributionSet):
    """lmms-lab-encoder/VizWiz-VQA (Gurari et al. 2018) `val`: photos taken by blind people, each
    with the spoken question they asked about it and 10 crowd answers. Asked as a noul: can the
    question be answered from the photo? Gold: the share of the 10 answers that are not exactly
    `unanswerable` (the dataset's answerability label). Photos as uploaded (121x162 up to 2592x1936).
    One split upstream (`test` answers are hidden): halves by sha256(question id), then a fixed
    sample of 1,000 for `validation` and 2,000 for `test`. Licence: not stated on the mirror; the
    original is CC BY 4.0 (vizwiz.org)."""
    name = "vizwiz"
    revision = {"repo": "lmms-lab-encoder/VizWiz-VQA", "revision": "d428a2dae984f79cf1b9d99467dfa883e0c30686"}
    gold_source = "human"
    SIZES = {"validation": 1000, "test": 2000}

    @staticmethod
    def load(root, split):
        cases = []
        for f in sorted((Path(root) / "vizwiz" / "data").glob("val-*.parquet")):
            for r in pq.read_table(f, columns=["question_id", "image", "question", "answers"]).to_pylist():
                if sha_half(r["question_id"]) != split:
                    continue
                no = r["answers"].count("unanswerable") / len(r["answers"])
                cases.append({"id": r["question_id"], "group": "vizwiz", "state": {"question": r["question"].strip()},
                              "images": [image_data_url(r["image"])],
                              "questions": {"answer": {"type": "noul",
                                                       "instructions": "A blind person took the attached photo and asked the question above about it. Can the question be answered from the photo?"}},
                              "gold": {"answer": [1.0 - no, no]}, "gold_source": "human"})
        return sha_sample(cases, VizWizAnswerable.SIZES[split])


class RapidataLikert(LabelDistributionSet):
    """Rapidata/flux1.1-likert-scale-preference: 1,124 Flux1.1 [pro] images for 280 benchmark
    prompts (DrawBench, T2I-CompBench and others), each rated by 31-45 crowd annotators on how
    well the image matches the prompt (1: Not at all ... 5: Perfectly), as a 5-level score. Gold:
    the rating shares. One split upstream: halves by sha256(prompt), so no prompt is in both; all
    images of each half (no sample). Licence: none stated on the card."""
    name = "rapidata-likert"
    revision = {"repo": "Rapidata/flux1.1-likert-scale-preference", "revision": "a490293a5614b444b317855c7e9dcdd7da91e124"}
    gold_source = "human"
    LEVELS = ["1: Not at all", "2: A little", "3: Moderately", "4: Very well", "5: Perfectly"]

    @staticmethod
    def load(root, split):
        t = pq.read_table(Path(root) / "rapidata-likert" / "data" / "train-00000-of-00001.parquet",
                          columns=["image", "prompt", *RapidataLikert.LEVELS])
        cases = []
        for i, r in enumerate(t.to_pylist()):
            if sha_half(r["prompt"]) != split:
                continue
            n = sum(r[k] for k in RapidataLikert.LEVELS)
            cases.append({"id": f"likert-{i}", "group": "likert", "state": {"prompt": r["prompt"]},
                          "images": [image_data_url(r["image"])],
                          "questions": {"answer": {"type": "score", "instructions": "The attached image was generated from the prompt. How well does it match the prompt?",
                                                   "criteria": [k.split(": ", 1)[1] for k in RapidataLikert.LEVELS]}},
                          "gold": {"answer": [r[k] / n for k in RapidataLikert.LEVELS]}, "gold_source": "human"})
        return cases


SETS = {s.name: s for s in (TypedDecisions, TypedDecisionBench, ProceduralTypedDecisions, CalibratedTypedDecisions,
                            ChaosNLI, LeWiDi, CIFAR10H, POPE, SoftDigits, RapidataStyle, VQAv2YesNo, AVA, KonIQ10k,
                            VizWizAnswerable, RapidataLikert)}


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
