# qwen38-27b-decisions

**Question:** can Qwen3.8-27B, as served for chat, double as a decision model through SGLang's
`/v1/systemone` and `/v1/decisions`, and do Swift 1.5's fine-tune or the quantization
(BF16 / FP8 / NVFP4) hurt it? Answers come from the answer-label log-probs with thinking off
([SGLang decision models docs](https://docs.sglang.io/docs/supported-models/decision_models)).
Follow-ups: how much does client-side calibration (fitted on `train`, no weights or server
changes) fix, and how do the two NVFP4 configs do on datasets whose answers are human labels or
exact rules instead of a teacher model (`results-real/`)?

Run 2026-10-30, one RTX PRO 6000, one config at a time.

[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) @ `1415d375`: its
gold is the mean of three samples from an unnamed "roughly 4B-class" teacher, so these scores
measure agreement with that teacher, not correctness.

## Results (`test`, 400 cases, 2,000 decisions, `/v1/systemone` pass 1)

| Config | Agreement | KL from gold | Brier | ECE | Label mass | p50 / p95 latency per case |
|---|---|---|---|---|---|---|
| `stock-bf16` | 0.661 | 0.590 | 0.214 | 0.143 | 0.976 | 381 / 493 ms |
| `stock-fp8` | 0.663 | 0.558 | 0.208 | 0.133 | 0.988 | 241 / 312 ms |
| `stock-nvfp4` | 0.666 | 0.542 | 0.205 | 0.129 | 0.974 | 167 / 215 ms |
| `swift-bf16` | 0.674 | 0.709 | 0.229 | 0.157 | 0.986 | 386 / 501 ms |
| `swift-fp8` | 0.669 | 0.664 | 0.221 | 0.157 | 0.996 | 240 / 312 ms |
| `swift-nvfp4` | 0.683 | 0.676 | 0.216 | 0.140 | 0.993 | 166 / 214 ms |
| `swift-nvfp4-dflash8` | 0.683 | 0.676 | 0.216 | 0.140 | 0.993 | 170 / 218 ms |

Agreement is the dataset scorer's `accuracy` (argmax vs the gold argmax); see "Gold" below for
why it is agreement with a teacher, not correctness. Label mass is the full-vocabulary
probability of the answer labels (`x_label_mass`, mean). One request per case: the state (about
800 input tokens) and all five questions, concurrency 1.

By question type (agreement / KL):

| Config | `choice` (600) | `noul` (600) | `score` (800) |
|---|---|---|---|
| `stock-bf16` | 0.647 / 0.576 | 0.697 / 0.509 | 0.645 / 0.662 |
| `stock-fp8` | 0.652 / 0.588 | 0.697 / 0.407 | 0.646 / 0.649 |
| `stock-nvfp4` | 0.657 / 0.660 | 0.713 / 0.332 | 0.636 / 0.610 |
| `swift-bf16` | 0.662 / 0.691 | 0.717 / 0.641 | 0.651 / 0.775 |
| `swift-fp8` | 0.653 / 0.706 | 0.710 / 0.485 | 0.649 / 0.766 |
| `swift-nvfp4` | 0.667 / 0.723 | 0.747 / 0.612 | 0.647 / 0.688 |

Noise and API checks (`pass_drift` and `native_vs_systemone` in each `summary.json`), 2,000
decisions per config. These passes ran with an almost empty prefix cache (see "Prefix cache
check" below), so they compare cold with cold:

| Config | pass 2 vs pass 1: mean / max \|Δp\|, top answers changed | native vs systemone: mean / max \|Δp\|, changed |
|---|---|---|
| `stock-bf16` | 0.00003 / 0.029, 0 | 0.0000 / 0.029, 0 |
| `stock-fp8` | 0.0010 / 0.173, 3 | 0.0010 / 0.173, 3 |
| `stock-nvfp4` | 0.0002 / 0.277, 2 | 0.0002 / 0.277, 2 |
| `swift-bf16` | 0.00006 / 0.060, 1 | 0.0001 / 0.060, 1 |
| `swift-fp8` | 0.0010 / 0.220, 2 | 0.0010 / 0.220, 2 |
| `swift-nvfp4` | 0.0002 / 0.245, 0 | 0.0002 / 0.245, 0 |
| `swift-nvfp4-dflash8` | 0.0003 / 0.245, 1 | 0.0002 / 0.245, 0 |

Paired gaps over the same 400 cases, 95% bootstrap intervals that resample cases
(`paired_ci.py`, output `results/paired-agreement.txt`; the T+B rows are from the next section):

| A | B | B − A | 95% CI |
|---|---|---|---|
| `stock-bf16` | `stock-fp8` | +0.002 | [−0.006, +0.010] |
| `stock-bf16` | `stock-nvfp4` | +0.005 | [−0.008, +0.018] |
| `swift-bf16` | `swift-fp8` | −0.006 | [−0.013, +0.002] |
| `swift-bf16` | `swift-nvfp4` | +0.009 | [−0.003, +0.021] |
| `stock-bf16` | `swift-bf16` | +0.013 | [+0.006, +0.021] |
| `stock-fp8` | `swift-fp8` | +0.006 | [−0.003, +0.014] |
| `stock-nvfp4` | `swift-nvfp4` | +0.018 | [+0.003, +0.033] |
| `stock-nvfp4` | `stock-nvfp4` T+B | +0.077 | [+0.058, +0.097] |
| `swift-nvfp4` | `swift-nvfp4` T+B | +0.065 | [+0.046, +0.084] |
| `stock-nvfp4` T+B | `swift-nvfp4` T+B | +0.005 | [−0.009, +0.019] |

Same agreement rate doesn't mean the same answers. Top answers that differ from BF16 (pass 1,
raw, second table in `results/paired-agreement.txt`):

| A | B | top answers that differ (of 2,000) |
|---|---|---|
| `stock-bf16` | `stock-fp8` | 64 (3.2%) |
| `stock-bf16` | `stock-nvfp4` | 198 (9.9%) |
| `swift-bf16` | `swift-fp8` | 62 (3.1%) |
| `swift-bf16` | `swift-nvfp4` | 177 (8.8%) |

Read: for these separately produced exports (not a controlled isolation of quantization), every
BF16 → FP8/NVFP4 agreement interval contains 0, every point gap is under 0.01 and the widest
bound is +0.021; that doesn't establish equivalence, and NVFP4 changes about 9–10% of
top answers, gaining about as many matches with the teacher as it loses; adding the drafter changed none of 2,000 top answers (decisions are prefill only);
with a cold cache the two APIs differ on at most 3. Swift agrees with the teacher 0.006–0.018 more than stock
(interval above 0 at BF16 and NVFP4) and sits further from its distributions (KL 0.66–0.71 vs
0.54–0.59): sharper, with only a small agreement lead.

### Prefix cache check (`results-cache/`, `stock-nvfp4`, `swift-nvfp4`)

In the 400-case run, pass 2 and the native pass got about 5% of prompt tokens from the prefix
cache (server log `#cached-token`). The server keeps 117 Mamba states (`max_mamba_cache_size` in
the log) and a cached prefix is only usable where a state is stored, so by the time pass 2 came
back to a case its states were likely evicted `[INFERENCE]`. Rerun on the first 50 `test`
cases (250 decisions), same flags, two `/v1/systemone` passes then one native pass, so the
cache is warm after pass 1:

| Config | cached prompt tokens: pass 1 / pass 2 / native | top answers changed: pass 2 vs 1 / native vs 1 / native vs 2 | max \|Δp\| pass 2 vs 1 |
|---|---|---|---|
| `stock-nvfp4` | 0.9% / 67.7% / 75.5% | 25 / 27 / 8 | 0.388 |
| `swift-nvfp4` | 0.9% / 67.7% / 75.5% | 34 / 29 / 10 | 0.584 |

Phase shares are split at the completed-request count, so they are approximate at the boundaries. A cache
hit changes 10–14% of top answers against a cold prefill of the same prompt; two warm passes
through different APIs still differ on 3–4%. Aggregate agreement on these 50 cases barely moves
(0.636 / 0.640 / 0.640 Swift, 0.628 / 0.620 / 0.624 stock), so the headline numbers stand as
cold-cache numbers, but a single decision on a served model depends on what's in the cache. The
real-answer runs below were not cold: about 30% of Typed Decision Bench prompt tokens and 16–17%
of procedural ones came from the cache (none for Calibrated Typed Decisions).
The ablation below tests which flag is involved.

### Prefix cache ablation (`results-cache-ablation/`, `swift-nvfp4`)

Same 50-case check, one change to the shared flags per row (some rows change two or three flags) (`cache-ablation.sh`, table from
`cache_check.py`). Run 2026-10-30.

| Change | cached prompt tokens: pass 1 / pass 2 / native | top answers changed: pass 2 vs 1 / native vs 1 / native vs 2 | max \|Δp\| pass 2 vs 1 |
|---|---|---|---|
| none (`results-cache/`) | 0.9% / 67.7% / 75.5% | 34 / 29 / 10 | 0.584 |
| `--kv-cache-dtype auto` (BF16 KV) | 0.9% / 67.7% / 75.5% | 30 / 31 / 7 | 0.607 |
| `--attention-backend triton --enable-deterministic-inference` | 0.9% / 67.7% / 75.5% | 17 / 24 / 7 | 0.502 |
| both of the above | 0.9% / 67.7% / 75.5% | 18 / 22 / 8 | 0.547 |
| `--mamba-ssm-dtype float32` | 0.9% / 0.9% / 0.9% | 0 / 0 / 0 | 0 |
| no hierarchical cache | 0.9% / 0.9% / 0.9% | 0 / 0 / 0 | 0 |
| no hierarchical cache, `--mamba-radix-cache-strategy extra_buffer` | 0.9% / 0.9% / 0.9% | 0 / 0 / 0 | 0 |
| same, deterministic + triton | 0.9% / 0.9% / 0.9% | 0 / 0 / 0 | 0 |

Didn't start: `--disable-radix-cache` (exclusive with the hierarchical cache),
`--mamba-radix-cache-strategy no_buffer` (needs page size 1), `--disable-overlap-schedule` (the
lazy extra buffer requires the overlap scheduler).

Read: in the variants tested, answers change only with the hierarchical cache on; without it (or
with FP32 SSM state) 0.9% of prompt tokens come from the cache on every pass and repeats are
identical. With it,
hits change 7–14% of top answers. BF16 KV doesn't remove it (30). Deterministic inference
halves the changes but doesn't remove them, although SGLang's deterministic mode is meant to make
cached and uncached prefill identical. The log reports `NPU HiCache Mamba state transfer mode:
sync torch fallback`; that the restored Mamba state is where the difference comes from is my
reading, not tested.

SGLang's own check, `python3 -m sglang.test.test_deterministic --test-mode radix_cache` in the
container (`radix-test-{base,det}/test_deterministic.txt`), fails on both: the request with a
64-token cache hit and the same prompt uncached give the same top token with logprobs 0.486
apart (shared flags) and 0.197 apart (deterministic inference, Triton attention, BF16 KV).

Without the hierarchical cache, pass 2 is as fast as pass 1 (p50 104 ms); with it, warm passes
take 78–81 ms against 105–106 ms cold (deterministic runs).

Rerun: `DATASETS_DIR=... MODELS_DIR=... BENCH_CDI=1 ./cache-ablation.sh [variant ...]`, then
`./cache_check.py results-cache-ablation/*/swift-nvfp4`.

## Calibration (`stock-nvfp4`, `swift-nvfp4`)

`../scripts/decisions/calibrate.py`, fitted on the `train` split's answers (1,200 cases, 6,000
decisions, `results-train/`) by soft cross-entropy against the gold, then scored once on `test`
with the same scorer. Both act on the probabilities the client gets back; no weights, no server
change, chat on the same server untouched.

- **T:** one temperature per question type, `p_i ∝ p_i^(1/T)`. 3 parameters. Can't change any
  argmax, so agreement stays the same.
- **T+B:** the same plus one log-prior bias per workflow, question and option,
  `p_i ∝ p_i^(1/T) e^(b_i)`, L2-penalised, with the temperatures refit alongside the biases. 3 + 71 parameters. Same idea as the [Typed Decisions card's](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/1415d375cb86b418c53abec12795adf5b194e7fc/README.md)
  "Amos 1 large, per-question priors fitted on `train`" row.

| Config | Fit | Agreement | KL from gold | Brier | ECE |
|---|---|---|---|---|---|
| `stock-nvfp4` | raw | 0.666 | 0.542 | 0.205 | 0.129 |
| `stock-nvfp4` | T | 0.666 | 0.203 | 0.105 | 0.084 |
| `stock-nvfp4` | T+B | 0.743 | 0.133 | 0.069 | 0.135 |
| `swift-nvfp4` | raw | 0.683 | 0.676 | 0.216 | 0.140 |
| `swift-nvfp4` | T | 0.683 | 0.198 | 0.100 | 0.098 |
| `swift-nvfp4` | T+B | 0.748 | 0.132 | 0.068 | 0.135 |

Fitted temperatures (T only), `choice` / `noul` / `score`: stock 2.58 / 3.28 / 2.62, Swift
2.79 / 4.95 / 2.85. Both models are sharper than the teacher's soft gold; Swift most on yes/no.
Parameters and metrics: `results/<config>/calibration.json`.

ECE is not a usable calibration metric here: scored as a prediction against its own argmax, the
gold has ECE 0.341 on `test` (`gold_ece_test`). This ECE compares confidence with agreement on
the teacher's top label, not with its full distribution, so matching the teacher exactly doesn't
minimise it. KL and Brier are the calibration metrics.

## Datasets with real answers (`results-real/`, `stock-nvfp4`, `swift-nvfp4`)

Run 2026-10-30 14:13–14:55, same server flags, `/v1/systemone`, one pass, concurrency 1. Typed
Decision Bench uses its DecisionScore; Procedural Typed Decisions ships no scorer, so it uses
Typed Decision Bench's per-item proper score; Calibrated Typed Decisions uses accuracy, log loss,
Brier and ECE (`../scripts/decisions/sets.py`). Calibration is the same temperature (T) and
temperature + bias (T+B) method as above, fitted separately per set on data the scores don't use.
Full tables with paired 95% intervals: [`results-real/tables.md`](results-real/tables.md).

| Set (split, cases) | Gold | Metric | `stock-nvfp4` raw / T / T+B | `swift-nvfp4` raw / T / T+B |
|---|---|---|---|---|
| [Typed Decision Bench](https://huggingface.co/datasets/SamuelChien821/typed-decision-bench) (`test`, 5,387) | human labels from 21 datasets + 1 generator | DecisionScore | 77.18 / 80.28 / 83.97 | 76.63 / 80.58 / 84.17 |
| [Procedural Typed Decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) (`test` sample, 874) | exact rules | proper score | 74.28 / 76.46 / 79.64 | 74.38 / 77.41 / 80.30 |
| [Calibrated Typed Decisions](https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions) (`test`, 1,797) | human labels | accuracy | 0.823 / 0.823 / 0.878 | 0.838 / 0.838 / 0.874 |
| Calibrated Typed Decisions (`heldout` PubMedQA, 1,000) | expert labels | accuracy | 0.716 / 0.716 / 0.716 | 0.731 / 0.731 / 0.731 |

- **Swift vs stock, raw:** −0.55 DecisionScore [−0.86, −0.25] on Typed Decision Bench, +0.10
  [−0.63, +0.84] on procedural, +0.014 accuracy [+0.006, +0.023] on calibrated `test`, +0.015
  [+0.001, +0.029] on PubMedQA. After T+B the gaps are +0.19 [−0.05, +0.44], +0.66 [+0.04, +1.28]
  and −0.004 [−0.014, +0.005]. Small, mixed in sign, no consistent winner.
- **Calibration** helps on every set: T alone +2.2 to +4.0 score points, T+B +5.4 to +7.5. On
  calibrated `test`, log loss 0.60–0.67 → 0.32 and ECE 0.095–0.099 → 0.011 with T+B.
- **Out of distribution (PubMedQA):** the biases are per task and option, and none were fitted
  for PubMedQA, so T+B there is the temperatures plus nothing; accuracy can't change. The fitted
  temperatures still cut log loss 0.90 → 0.69 (stock) and 0.95 → 0.68 (Swift) and ECE
  0.15 → 0.06. T+B's temperatures (fitted jointly with the biases) do worse there than T's
  (log loss 0.72 / 0.70): on this one unseen task, the T-only fit transfers better.
- **Typed Decision Bench vs published systems** ([leaderboard](https://blobfish.ai/leaderboard);
  its published answer files in the dataset's `results/`, scored here with the same scorer,
  reproduce the leaderboard to two decimals): raw, both configs sit between OpenJev 0.1
  (75.56) and decider-2b-v8 (77.56) and 4.5–5.1 points under Jev 1.13.0 (81.70). With T+B, both
  are 2.3–2.5 points over Jev, but that fit is cross-fitted on the test items themselves (5 folds
  stratified by task; the set has no other split), so it's a fitted row against zero-shot rows,
  not like for like.

Sets:

- [SamuelChien821/typed-decision-bench](https://huggingface.co/datasets/SamuelChien821/typed-decision-bench)
  v0.3.0, revision `79ec2b9bf193d69ceaaaf86ad9d0b79623adb101`: 5,387 items in 25 tasks. The 359
  items it withholds (Abt-Buy, TrialGPT) are rebuilt with its builder
  (`blobfishai/jevfish` @ `80262b5c7049179a5def5e15fc85c51e3f9d7314`, `python -m
  typed_decision_bench.freeze --only abt-buy-product-match,trialgpt-criterion-support --seed
  20260919 --version 0.3.0`); the result matches the dataset's `SHA256SUMS`. Score:
  DecisionScore (mean per-item proper score × 100) and accuracy on each item's primary question.
- [tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions)
  `all`, revision `05ca1d0388f57461a634a7caed67145d28e2eca7`: answers computed from rules stated
  in the state. Fixed stratified sample, up to 10 states per (task, level) by smallest sha256(id):
  874 `test`, 880 `validation` (calibration fit). No scorer ships with it: Typed Decision Bench's
  per-item proper score against the gold as given; accuracy over the 2,787 questions with an
  exact label.
- [m-lagnajit/calibrated-typed-decisions](https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions)
  v2.1, revision `34189779d116d0523db6af58c0d62538f7e7d07e`: one question per short text, gold
  is the source dataset's human label (its soft teacher labels are not used). `test` six sources
  (CLINC150, GoEmotions, BANKING77, MASSIVE, Amazon polarity, SciCite), `heldout` PubMedQA,
  `validation` fits the calibration. Score: accuracy, log loss, Brier, ECE (10 bins).

### Three passes, two more sets (`results-repeat/`)

Run overnight 2026-10-30 to 2026-10-31 (`tonight.sh`), `stock-nvfp4` and `swift-nvfp4`, same server
flags, `/v1/systemone`, concurrency 1, 3 passes per run. Each dataset's passes share one server
session; `results-repeat/train` and `results-repeat/test` were separate `run.sh` invocations (separate
server launches), and the test datasets ran one after another in one session per config, so pass 1
of a later dataset is not an empty-cache request. Passes 2 and 3 can reuse cached prefixes. Answers
are in `results-repeat/{train,test}/`. The earlier one-pass run (`results-real/`, above) is kept.
Added sets, with where each one's answers come from:

- [ChaosNLI](https://huggingface.co/datasets/earino/chaosnli) (`earino/chaosnli` @
  `bf8d66a1d3434ac8b90ffc392b1b5e44bdf7e439`, mirror of ChaosNLI v1.0): SNLI, MNLI-matched and αNLI
  dev items, gold is the share of 100 human votes per item. One upstream split, so `validation` and
  `test` are halves by sha256 of the item id (2,334 and 2,311 cases).
- [LeWiDi](https://huggingface.co/datasets/tasksource/lewidi) (`tasksource/lewidi` @
  `9ba2e83b536ca127fe8b25f2a2725f1e9a489c82`): sarcasm (`score`, 6 levels), offensiveness and
  Brexit hate speech (`noul`); gold is the share of the annotators' votes (human). `validation` 1,976,
  `test` 3,929 cases.

Raw scores, pass 1 (`summary.json` of each run, `test` split; accuracy is the top answer against
the top gold option, KL is from the gold distribution; `results-repeat/test/<config>/<set>-test/`):

| Set (answers) | `stock-nvfp4` acc / KL / Brier / ECE | `swift-nvfp4` acc / KL / Brier / ECE |
|---|---|---|
| ChaosNLI `test` (human vote shares) | 0.7637 / 0.5788 / 0.1764 / 0.1055 | 0.7707 / 0.7500 / 0.1946 / 0.1322 |
| LeWiDi `test` (human vote shares) | 0.6908 / 0.3935 / 0.1922 / 0.0389 | 0.7182 / 0.4267 / 0.1922 / 0.0501 |

Accuracy differs between the configs by 0.007 on ChaosNLI and 0.027 on LeWiDi (Swift higher on both);
Swift's KL from the gold distribution is higher on both, and its fitted `noul` temperatures are
higher at every source (e.g. Typed Decisions 4.96 vs 3.28, below). No paired intervals for these sets.

### Serving noise across passes

`pass_drift` in `summary.json` is pass 2 vs pass 1 (first pass vs a pass that can reuse cached prefixes). Top answers changed
(`results-repeat/test/<config>/<set>/summary.json`, stock / Swift):

| Set | decisions | stock | Swift |
|---|---|---|---|
| Typed Decisions `test` | 2,000 | 2 | 0 |
| Typed Decision Bench `test` | 10,656 | 104 | 76 |
| Procedural Typed Decisions `test` | 2,882 | 10 | 13 |
| Procedural Typed Decisions `validation` | 2,918 | 26 | 18 |
| LeWiDi `test` | 3,929 | 0 | 2 |
| LeWiDi `validation` | 1,976 | 21 | 14 |
| ChaosNLI, Calibrated TD (all splits) | 1,000–2,334 | 0 | 0 |

ChaosNLI and Calibrated TD have no changed top answers between pass 1 and 2. In the transfer
tables the spread across passes is at most 0.07 points of DecisionScore or proper score (Swift TD
Bench raw 76.59–76.66) and 0.05 on a fitted `noul` temperature (stock procedural, 3.35–3.40).
Differences between sources smaller than these are not resolved.

### Temperature transfer across datasets (`results-repeat/transfer.md`)

**Question:** does a per-type temperature fitted on one dataset improve another? Run 2026-10-30 to
31, CPU only, from the stored answers of the three-pass runs above (`transfer.py`), `stock-nvfp4`
and `swift-nvfp4`.

Method. Temperatures only (T per question type, as in "Calibration"): the biases are keyed to one
dataset's questions and can't carry. Names in the tables, with where each dataset's answers come
from:

- Typed Decisions: the [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
  set, gold from the unnamed ~4B teacher, so its scores are agreement with that teacher. Metric
  KL from gold.
- procedural: Procedural Typed Decisions, answers computed from rules. Metric Typed Decision
  Bench's per-item proper score.
- Calibrated TD: Calibrated Typed Decisions `test`, human labels. Metric log loss.
- PubMedQA: Calibrated Typed Decisions `heldout`, expert labels. Metric log loss.
- TD Bench: Typed Decision Bench, human labels. Metric DecisionScore.
- ChaosNLI, LeWiDi: human vote shares. Metric KL from gold.

Sources (rows) are fitted on their fit split: Typed Decisions `train`, procedural, Calibrated TD,
ChaosNLI and LeWiDi `validation`, TD Bench on all of its `test` (it has no other split). PubMedQA
is a target only. Targets (columns) are scored on each set's scored split. A type the source has no
questions of keeps T = 1, and a cell is `-` where the source has none of the target's types (it
would equal raw). The diagonal (bold) is the in-domain fit; for TD Bench it is the 5-fold cross-fit
used above (stratified by task), so no target is scored with temperatures fitted on its own
answers. "pooled text" is one fit on every source except the column's dataset, so it is out of
sample. Each pass's fit uses the same pass of the fit split; cells are the mean over the three
passes with [min, max].

Full tables, both configs, with the fitted temperatures: [`results-repeat/transfer.md`](results-repeat/transfer.md).
Raw, in-domain and pooled text rows, mean over passes (metrics as in the file: KL ↓ for Typed
Decisions, ChaosNLI, LeWiDi; log loss ↓ for Calibrated TD and PubMedQA; proper score / DecisionScore
↑ for procedural and TD Bench):

| `stock-nvfp4` | Typed Decisions | procedural | Calibrated TD | PubMedQA | TD Bench | ChaosNLI | LeWiDi |
|---|---|---|---|---|---|---|---|
| raw | 0.541 | 74.26 | 0.602 | 0.902 | 77.21 | 0.579 | 0.394 |
| in-domain | 0.203 | 76.44 | 0.457 | - | 80.28 | 0.194 | 0.327 |
| pooled text | 0.219 | 75.95 | 0.463 | 0.678 | 79.94 | 0.240 | 0.361 |

| `swift-nvfp4` | Typed Decisions | procedural | Calibrated TD | PubMedQA | TD Bench | ChaosNLI | LeWiDi |
|---|---|---|---|---|---|---|---|
| raw | 0.677 | 74.47 | 0.668 | 0.948 | 76.63 | 0.750 | 0.427 |
| in-domain | 0.198 | 77.46 | 0.438 | - | 80.58 | 0.184 | 0.314 |
| pooled text | 0.210 | 76.82 | 0.446 | 0.657 | 80.25 | 0.251 | 0.354 |

- **Text to text mostly transfers:** 5 cells in the file are worse than raw in at least one pass
  (flagged "(worse than raw in a pass)"; stock 3, Swift 2). 4 of them come from the ChaosNLI fit;
  the fifth is Calibrated TD on LeWiDi, stock.
- **Worst cells:** the ChaosNLI fit (`choice` T 3.11 stock, 3.71 Swift, the highest of any source)
  does worse than raw on procedural (73.24 vs 74.26 stock, 72.99 vs 74.47 Swift) and TD Bench
  (76.90 vs 77.21, 76.04 vs 76.63). One text-to-text case outside ChaosNLI: Calibrated TD on
  LeWiDi, stock only (0.397 vs raw 0.394).
- **The pooled text fit** gets close to the in-domain fit on the sets with their own temperature
  types (Typed Decisions 0.219 vs 0.203 in-domain, raw 0.541, stock) but not all (ChaosNLI 0.240 vs
  0.194, LeWiDi 0.361 vs 0.327 stock). On PubMedQA, which has no fit of its own, it gives 0.678
  (stock) and 0.657 (Swift) log loss against raw 0.902 and 0.948.
- **Fitted `noul` temperatures:** Typed Decisions, procedural, Calibrated TD and TD Bench give
  2.92–3.83 (stock) and 4.08–5.08 (Swift); LeWiDi (1.58, 2.05) is lower.

Rerun (CPU, from this folder):

```sh
uv run --no-project --with pyarrow --with numpy --with scipy transfer.py "$DATASETS_DIR" \
    --td-train results-repeat/train --td-test results-repeat/test --real results-repeat/test \
    --passes pass1 pass2 pass3 > results-repeat/transfer.md
```

Caveats: SNLI and MNLI are likely in the model's pretraining data [INFERENCE]. Pooling fits one temperature per question type, weighted by that type's decisions. The pass-to-pass range is variation over three passes of a dataset, not sampling error of the scored split.

### How many labeled cases (`results-repeat/learning_curve.md`)

**Question:** how many labeled cases does "T per type" need? CPU only, from the stored pass-1
answers of the three-pass run (`learning_curve.py`). For each NVFP4 config and each dataset with a
fit split separate from its scored split (Typed Decisions `train`, Procedural Typed Decisions
`validation`, Calibrated Typed Decisions `validation`, ChaosNLI and LeWiDi `validation`), T per
type is fitted on n cases drawn at random from the fit split and scored on the same scored split as
above. Each n is drawn 20 times; cells show the median metric over draws with [10th, 90th
percentile]. "Share" is the median fraction of the full-split fit's gain over raw,
`(m_n − raw) / (m_full − raw)`: 1 = as good as fitting on the whole split, below 0 = worse than
raw; the second number is the worst draw. A case can hold several questions, so n counts cases.
Answers: Typed Decisions is agreement with the unnamed ~4B teacher (KL from gold); Procedural Typed
Decisions answers are rule-computed (proper score); Calibrated Typed Decisions, ChaosNLI and LeWiDi
are human labels or vote shares.

Median share at n = 25 / 50 / 100 / 200 (full tables in the file):

| Set | `stock-nvfp4` | `swift-nvfp4` |
|---|---|---|
| Typed Decisions | 1.00 / 1.00 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 / 1.00 |
| ChaosNLI | 0.99 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 |
| LeWiDi | 0.91 / 0.92 / 0.98 / 0.98 | 0.94 / 0.98 / 0.98 / 0.99 |
| procedural | 0.85 / 0.92 / 0.97 / 1.00 | 0.87 / 0.94 / 0.99 / 1.00 |
| Calibrated TD | 0.73 / 0.82 / 0.94 / 0.98 | 0.84 / 0.91 / 0.98 / 0.99 |

- **25 cases are enough for 0.9 or more on ChaosNLI, Typed Decisions and LeWiDi**, both configs.
  Procedural needs 50 (0.92 stock, 0.94 Swift), Calibrated TD 100 on stock (0.94; 0.91 at 50 on
  Swift).
- **Worst draws are the risk at small n.** Calibrated TD at n = 25 has a worst-draw share of −5.00
  (stock) and −6.82 (Swift), still −4.09 and −2.00 at n = 50, and 0.79 at n = 100 on both. LeWiDi
  at n = 25: −31.83 (stock), −13.17 (Swift). From n = 100 every worst draw is at least 0.79; from
  n = 200 at least 0.88.

Rerun (CPU, from this folder):

```sh
uv run --no-project --with pyarrow --with numpy --with scipy learning_curve.py "$DATASETS_DIR" \
    --td-train results-repeat/train --td-test results-repeat/test --real results-repeat/test \
    --pass pass1 > results-repeat/learning_curve.md
```

Caveats: draws come from fit splits of 880–2,334 cases (a 500-case draw covers 21–57% of the fit
split), NVFP4 configs only, intervals are percentiles over draws and say nothing about the scored
split's own sampling error.

## Benchmark

[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), `all`
config, revision `1415d375cb86b418c53abec12795adf5b194e7fc` (Apache-2.0): `test` 400 cases in four
workflows (agent trace observability, customer service, invoice processing, security incidents),
2,000 typed decisions (`noul`, `choice`, `score`); `train` 1,200 cases from a separate generation
run. Scored with the dataset's own scorer (`bench/score_with_jobs.py` at that revision, copied
unchanged into `../scripts/decisions/run.py`): agreement with the gold argmax, KL(gold ‖ pred),
Brier, ECE over 10 bins. Also recorded: label mass, latency per request, and every raw answer.

Both SGLang decision APIs run on every config: `/v1/systemone` (TypeSafe System One request
shape, the one the dataset's scorer uses; two passes, headline numbers from pass 1) and native
`/v1/decisions` (one pass; `noul` sent as `yes_no`, `choice` criteria as `options`, `score`
criteria as `levels`). SGLang documents both as the same prompt rendering and scoring for chat
models.

**Gold** is the mean of three samples from a ~4B-class teacher at temperature 0.7, per the
[Typed Decisions card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/1415d375cb86b418c53abec12795adf5b194e7fc/README.md),
which doesn't name the model. Agreement is agreement with that teacher. Card references: prior (label frequencies)
0.470, perfect factor recovery 0.704, teacher self-agreement 0.735. Differences between our
configs (same prompts, same scorer) are the result; absolute numbers compare with the Typed Decisions card's
zero-shot table only loosely (its rows are mostly vLLM or vendor APIs). Calibrated rows are fitted
on `train` and belong with the Typed Decisions card's second table ("fitted or fine-tuned on `train`").

## Configs

| Label | Checkpoint | Revision | `--mem-fraction-static` |
|---|---|---|---|
| `stock-bf16` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 0.76 |
| `stock-fp8` | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | 0.50 |
| `stock-nvfp4` | [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) | `319f741cce68d7914884900c138a1fbb70a42f30` | 0.41 |
| `swift-bf16` | [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) | `b4c84d42903a8646b25857eb2827288d92ed85a4` | 0.76 |
| `swift-fp8` | block FP8 of `swift-bf16`, built locally | 128×128 blocks, stock FP8's layout; public equivalent [realderpz/Swift-1.5-Qwen3.8-27B-FP8](https://huggingface.co/realderpz/Swift-1.5-Qwen3.8-27B-FP8) (llmcompressor `FP8_BLOCK`) | 0.50 |
| `swift-nvfp4` | [ukisai/Swift-1.5-Qwen3.8-27b-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4) | `25482027debd5485e8108897ba9fed8d3ba16595` (ModelOpt mixed, no KV scales) | 0.41 |
| `swift-nvfp4-dflash8` | as above + [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, 8 draft tokens | Swift NVFP4 with the DFlash2 drafter | 0.41 |

- Image: `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` for all.
- Same server flags everywhere ([../scripts/args/qwen38-27b.sh](../scripts/args/qwen38-27b.sh),
  FP8 KV cache and `--reasoning-parser qwen3` included), no speculative decoding except the last
  row. Chat requests are not part of this run.
- Two `/v1/systemone` passes per config, same order: pass 2 is prefix-cached, and its drift from
  pass 1 is the noise floor (SGLang's docs report up to ~0.07 between cold and cached requests).

## Rerun

```sh
# test split, all seven configs (about an hour with server starts)
DATASETS_DIR=/path/to/datasets MODELS_DIR=/path/to/models BENCH_CDI=1 ./run.sh

# train split answers for calibration (NVFP4 configs, one pass, ~15 min)
OUT=$PWD/results-train DECISIONS_RUNS=typed-decisions:train DECISIONS_PASSES=1 DECISIONS_NATIVE_PASSES=0 \
  DATASETS_DIR=... MODELS_DIR=... BENCH_CDI=1 ./run.sh stock-nvfp4 swift-nvfp4

# calibration (host, CPU only)
uv run --no-project --with pyarrow --with numpy --with scipy ../scripts/decisions/calibrate.py \
  --data-dir "$DATASETS_DIR" \
  --fit-answers results-train/swift-nvfp4/decisions/pass1.jsonl --fit-split train \
  --test-answers results/swift-nvfp4/decisions/pass1.jsonl --test-split test \
  --out results/swift-nvfp4/calibration.json

# real-answer sets (both NVFP4 configs, ~45 min), then calibration and tables (host, CPU)
OUT=$PWD/results-real DECISIONS_PASSES=1 DECISIONS_NATIVE_PASSES=0 \
  DECISIONS_RUNS="typed-decision-bench:test procedural-typed-decisions:validation procedural-typed-decisions:test calibrated-typed-decisions:validation calibrated-typed-decisions:test calibrated-typed-decisions:heldout" \
  DATASETS_DIR=... MODELS_DIR=... BENCH_CDI=1 ./run.sh stock-nvfp4 swift-nvfp4
DATASETS_DIR=... ./calibrate-real.sh
uv run --no-project --with pyarrow --with numpy --with scipy real_answers.py "$DATASETS_DIR" > results-real/tables.md
```

Overnight run (three passes of every text run, both NVFP4 configs; needs `chaosnli/` and `lewidi/`
under `DATASETS_DIR` at the revisions in "Three passes, two more sets"), then the tables (host, CPU):

```sh
DATASETS_DIR=... MODELS_DIR=... BENCH_CDI=1 ./tonight.sh
uv run --no-project --with pyarrow --with numpy --with scipy transfer.py "$DATASETS_DIR" \
    --td-train results-repeat/train --td-test results-repeat/test --real results-repeat/test \
    --passes pass1 pass2 pass3 > results-repeat/transfer.md
uv run --no-project --with pyarrow --with numpy --with scipy learning_curve.py "$DATASETS_DIR" \
    --td-train results-repeat/train --td-test results-repeat/test --real results-repeat/test \
    --pass pass1 > results-repeat/learning_curve.md
```

`DATASETS_DIR` needs `typed-decisions/all/{train,test}-00000-of-00001.parquet` at the revision
above; for `results-real/`, `typed-decision-bench/` (with the rebuilt items merged into
`bench-complete/`), `procedural-typed-decisions/all/` and `calibrated-typed-decisions/` at the
revisions in that section.

Prefix cache check: `OUT=$PWD/results-cache DECISIONS_LIMIT=50 DATASETS_DIR=... MODELS_DIR=...
BENCH_CDI=1 ./run.sh stock-nvfp4 swift-nvfp4` (default two passes plus native).

## Caveats

- One run per config. With a cold cache, repeat passes and the two APIs move at most 3 of 2,000
  top answers; with a warm prefix cache, 10–14% of top answers change (`results-cache/`). The
  paired intervals above are the uncertainty from the 400 cases at cold cache.
- Gold is an unnamed ~4B-class teacher's soft distribution, not human labels. "Calibrated" means calibrated to
  that teacher.
- The biases are per workflow, question and option: they only exist for questions with labeled
  `train` data. The temperatures are per question type; whether they carry to other datasets is
  in "Temperature transfer across datasets".
- Calibration was fitted for the two NVFP4 configs only.
- `results-real/`: one pass, NVFP4 configs only. Procedural is a 874-state sample of 18,000.
  Typed Decision Bench's calibrated rows are cross-fitted on its test items.
- `results-repeat/`: three passes per run; each dataset's passes share one server session (train and
  test were separate launches; pass 1 precedes the cache-reusing passes 2-3); NVFP4 configs only.
  SNLI and MNLI are likely in the model's pretraining data [INFERENCE].
