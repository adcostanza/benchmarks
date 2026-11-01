# qwen38-27b-decisions

**Question:** can Qwen3.8-27B, as served for chat, double as a decision model through SGLang's
`/v1/systemone` and `/v1/decisions`, and do Swift 1.5's fine-tune or the quantization
(BF16 / FP8 / NVFP4) hurt it? Answers come from the answer-label log-probs with thinking off
([SGLang decision models docs](https://docs.sglang.io/docs/supported-models/decision_models)).
Follow-ups: how much does client-side calibration (fitted on `train`, no weights or server
changes) fix, and how do the two NVFP4 configs do on datasets whose answers are human labels or
exact rules instead of a teacher model (`results-real/`)?

Run 2026-10-30, one RTX PRO 6000, one config at a time.

[Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/1415d375cb86b418c53abec12795adf5b194e7fc) @ `1415d375`: its
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
| [Typed Decision Bench](https://huggingface.co/datasets/SamuelChien821/typed-decision-bench/tree/79ec2b9bf193d69ceaaaf86ad9d0b79623adb101) (`test`, 5,387) | labels from 21 public datasets (human annotations and automated verifier outcomes) + 1 rule-derived generator | DecisionScore | 77.18 / 80.28 / 83.97 | 76.63 / 80.58 / 84.17 |
| [Procedural Typed Decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions/tree/05ca1d0388f57461a634a7caed67145d28e2eca7) (`test` sample, 874) | exact rules | proper score | 74.28 / 76.46 / 79.64 | 74.38 / 77.41 / 80.30 |
| [Calibrated Typed Decisions](https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions/tree/34189779d116d0523db6af58c0d62538f7e7d07e) (`test`, 1,797) | human labels | accuracy | 0.823 / 0.823 / 0.878 | 0.838 / 0.838 / 0.874 |
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

- [SamuelChien821/typed-decision-bench](https://huggingface.co/datasets/SamuelChien821/typed-decision-bench/tree/79ec2b9bf193d69ceaaaf86ad9d0b79623adb101)
  v0.3.0, revision `79ec2b9bf193d69ceaaaf86ad9d0b79623adb101`: 5,387 items in 25 tasks. The 359
  items it withholds (Abt-Buy, TrialGPT) are rebuilt with its builder
  (`blobfishai/jevfish` @ `80262b5c7049179a5def5e15fc85c51e3f9d7314`, `python -m
  typed_decision_bench.freeze --only abt-buy-product-match,trialgpt-criterion-support --seed
  20260919 --version 0.3.0`); the result matches the dataset's `SHA256SUMS`. Score:
  DecisionScore (mean per-item proper score × 100) and accuracy on each item's primary question.
- [tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions/tree/05ca1d0388f57461a634a7caed67145d28e2eca7)
  `all`, revision `05ca1d0388f57461a634a7caed67145d28e2eca7`: answers computed from rules stated
  in the state. Fixed stratified sample, up to 10 states per (task, level) by smallest sha256(id):
  874 `test`, 880 `validation` (calibration fit). No scorer ships with it: Typed Decision Bench's
  per-item proper score against the gold as given; accuracy over the 2,787 questions with an
  exact label.
- [m-lagnajit/calibrated-typed-decisions](https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions/tree/34189779d116d0523db6af58c0d62538f7e7d07e)
  v2.1, revision `34189779d116d0523db6af58c0d62538f7e7d07e`: one question per short text, gold
  is the source dataset's human label (its soft teacher labels are not used). `test` six sources
  (CLINC150, GoEmotions, BANKING77, MASSIVE, Amazon polarity, SciCite), `heldout` PubMedQA,
  `validation` fits the calibration. Score: accuracy, log loss, Brier, ECE (10 bins).

### Three passes, four more sets (`results-repeat/`, `results-vision/`)

Run overnight 2026-10-30 to 2026-10-31 (`tonight.sh`), `stock-nvfp4` and `swift-nvfp4`, same server
flags, `/v1/systemone`, concurrency 1, 3 passes per run. Each dataset's passes share one server
session; `results-repeat/train` and `results-repeat/test` were separate `run.sh` invocations (separate
server launches), and the test datasets ran one after another in one session per config, so pass 1
of a later dataset is not an empty-cache request. Passes 2 and 3 can reuse cached prefixes. Text answers are in `results-repeat/{train,test}/`, image
answers in `results-vision/`. The earlier one-pass run (`results-real/`, above) is kept. Added sets,
with where each one's answers come from:

- [ChaosNLI](https://huggingface.co/datasets/earino/chaosnli/tree/bf8d66a1d3434ac8b90ffc392b1b5e44bdf7e439) (`earino/chaosnli` @
  `bf8d66a1d3434ac8b90ffc392b1b5e44bdf7e439`, mirror of ChaosNLI v1.0): SNLI, MNLI-matched and αNLI
  dev items, gold is the share of 100 human votes per item. One upstream split, so `validation` and
  `test` are halves by sha256 of the item id (2,334 and 2,311 cases).
- [LeWiDi](https://huggingface.co/datasets/tasksource/lewidi/tree/9ba2e83b536ca127fe8b25f2a2725f1e9a489c82) (`tasksource/lewidi` @
  `9ba2e83b536ca127fe8b25f2a2725f1e9a489c82`): sarcasm (`score`, 6 levels), offensiveness and
  Brexit hate speech (`noul`); gold is the share of the annotators' votes (human). `validation` 1,976,
  `test` 3,929 cases.
- [CIFAR-10H](https://huggingface.co/datasets/MKZuziak/cifar10h/tree/0ec71d3ea7d6b3b9cc4bb03362f1268e9cdf4542) (`MKZuziak/cifar10h` @
  `0ec71d3ea7d6b3b9cc4bb03362f1268e9cdf4542`): the 10,000 CIFAR-10 test images with about 50 human
  votes each; gold is the share of votes. Images are sent as they are (32x32). Fixed samples of
  1,000 (`validation`) and 2,000 (`test`) images.
- [POPE](https://huggingface.co/datasets/lmms-lab-encoder/POPE/tree/4db1276663dfa5eb8ad16a52d24c31a09e470896) (`lmms-lab-encoder/POPE` @
  `4db1276663dfa5eb8ad16a52d24c31a09e470896`): "Is there a <object> in the image?" over COCO
  val2014 images; gold is rule-derived from COCO's object boxes (one-hot yes/no), there is no human
  vote distribution. One case = one image and setting (random / popular / adversarial) with 6
  questions: 300 cases (`validation`, 1,800 decisions), 600 (`test`, 3,600).

Raw scores, pass 1 (`summary.json` of each run, `test` split; accuracy is the top answer against
the top gold option, KL is from the gold distribution; ChaosNLI and LeWiDi are `results-repeat/test/<config>/<set>-test/`):

| Set (answers) | `stock-nvfp4` acc / KL / Brier / ECE | `swift-nvfp4` acc / KL / Brier / ECE |
|---|---|---|
| ChaosNLI `test` (human vote shares) | 0.7637 / 0.5788 / 0.1764 / 0.1055 | 0.7707 / 0.7500 / 0.1946 / 0.1322 |
| LeWiDi `test` (human vote shares) | 0.6908 / 0.3935 / 0.1922 / 0.0389 | 0.7182 / 0.4267 / 0.1922 / 0.0501 |
| CIFAR-10H `test` (human vote shares) | 0.9125 / 0.2640 / 0.0915 / 0.0217 | 0.9150 / 0.3205 / 0.0939 / 0.0060 |
| CIFAR-10H `validation` | 0.9090 / 0.2609 / 0.0873 / 0.0324 | 0.9180 / 0.3005 / 0.0856 / 0.0190 |
| POPE `test` (rule-derived from COCO boxes) | 0.8883 / 0.3482 / 0.1715 / 0.0576 | 0.8894 / 0.4007 / 0.1783 / 0.0655 |
| POPE `validation` | 0.8839 / 0.3405 / 0.1746 / 0.0622 | 0.8839 / 0.4375 / 0.1906 / 0.0729 |

Accuracy differs between the configs by at most 0.009 on ChaosNLI, CIFAR-10H and POPE (0.027 on
LeWiDi, Swift higher); Swift's KL from the gold distribution is higher on every set above, and its
fitted `noul` temperatures are higher at every source (e.g. Typed Decisions 4.96 vs 3.28, below).
No paired intervals for these sets.

### Do decisions work on images?

Decision requests carry the image as a data URL in the top-level `images` list (next to the state or input, not inside it), with the same question
and options as text. On these two sets the answers are usable:

- CIFAR-10H: the top answer matches the top human class on 0.9090–0.9180 of images (both configs,
  both splits), at 10 options; raw ECE 0.0060–0.0324.
- POPE: accuracy 0.8839–0.8894 on the yes/no questions, which are rule-derived from COCO boxes, not
  human votes. ECE 0.0576–0.0729, KL 0.3405–0.4375.
- Stock POPE `validation` first failed (HTTP 500 on two grayscale COCO images,
  `results-vision/FAILED.superseded`); the client now converts non-RGB images to RGB PNG (commit
  2c360ef) and the run was redone on 2026-10-31 with the same flags (`rerun-pope-val.log`).

Do temperatures fitted on text carry to images? Read from [`results-repeat/transfer.md`](results-repeat/transfer.md)
(next section; CIFAR-10H and POPE columns, rows other than the image sources themselves):

| Fitted on | `stock-nvfp4` CIFAR-10H KL | `stock-nvfp4` POPE KL | `swift-nvfp4` CIFAR-10H KL | `swift-nvfp4` POPE KL |
|---|---|---|---|---|
| raw | 0.264 [0.264, 0.264] | 0.347 [0.344, 0.348] | 0.320 [0.320, 0.320] | 0.402 [0.401, 0.403] |
| pooled text | 0.386 [0.386, 0.387] | 0.302 [0.301, 0.303] | 0.346 [0.346, 0.346] | 0.327 [0.327, 0.328] |
| image set itself (diagonal) | 0.255 [0.255, 0.255] | 0.270 [0.269, 0.271] | 0.268 [0.268, 0.268] | 0.276 [0.276, 0.277] |

- **POPE (`noul`): yes.** Every text-fitted row beats raw, and the pooled text fit gets about 58% of
  the in-domain KL gain on stock and 60% on Swift (computed from the three numbers per column
  above).
- **CIFAR-10H (`choice`): no.** The pooled text fit is worse than raw in every pass on both
  configs (0.386 vs 0.264 stock, 0.346 vs 0.320 Swift). The text sources' `choice` temperatures
  are 1.35–3.11 (stock) and 1.47–3.71 (Swift), but CIFAR-10H's own are 1.18 and 1.49
  (the image probabilities are closer to calibrated than the text ones, so softening them as much
  as text needs hurts [INFERENCE]). Only the procedural
  fit (stock 0.260, Swift 0.268) and, on Swift, the Calibrated TD fit (0.294) beat raw.
- Provenance of the two columns differs: CIFAR-10H is human votes, POPE is a rule, so the POPE
  column measures agreement with COCO's boxes, where a one-hot gold punishes any hedging.

### Route parity on images (`results-vision-native/`, 2026-10-31)

**Question:** does `/v1/decisions` (the native route) answer the same as `/v1/systemone` when the
state carries an image? Same CIFAR-10H and POPE `validation` and `test` splits as above, both
NVFP4 configs, same server flags, a fresh server launch: one `/v1/systemone` pass, then one
`/v1/decisions` pass over the same cases ([`native-vision.log`](native-vision.log)). Compared per
decision in `summary.json` (`native_vs_systemone`); the answers are in
`results-vision-native/<config>/<set>-<split>/{pass1,native1}.jsonl`. The new image sets below
were run through `/v1/systemone` only.

| Set, split (decisions) | Config | mean / max \|Δp\| | top answers changed |
|---|---|---|---|
| CIFAR-10H `validation` (1,000) | both | 0 / 0 | 0 |
| CIFAR-10H `test` (2,000) | both | 0 / 0 | 0 |
| POPE `validation` (1,800) | stock / Swift | 0.0155 / 0.888, 0.0170 / 0.866 | 19, 27 |
| POPE `test` (3,600) | stock / Swift | 0.0153 / 0.954, 0.0152 / 0.892 | 60, 55 |

On CIFAR-10H the two routes return identical probabilities. On POPE they differ by about as much
as `/v1/systemone` differs from itself between pass 1 and pass 2 in the three-pass run
(`results-vision/<config>/pope-<split>/summary.json`, `pass_drift`: mean 0.0151-0.0187, top
answers changed 21-57 in stock and 29-53 in Swift, max 0.954 in stock `test`). POPE has six
questions per request, and pass 2 can reuse cached prefixes (see "Serving noise across passes"
and the cache ablation); I read the native-vs-systemone difference as the same effect, not as a
difference between routes `[INFERENCE]`; the native pass here is not compared with a second
`/v1/systemone` pass of the same server session. POPE accuracy, `/v1/systemone` pass vs native
(`validation` / `test`): stock 0.8839 / 0.8886 vs 0.8867 / 0.8881, Swift 0.8844 / 0.8892 vs
0.8783 / 0.8850.

### More image sets (`results-vision/`, 2026-10-31 to 2026-11-01)

**Question:** the first two image sets (CIFAR-10H, POPE) said that a temperature fitted on text
carries to one image set and not the other. Does that hold across more image sets and the three
question types (`noul`, `score`, `choice`), and does the model say anything beyond the answer
distribution's base rate on the harder sets? Runs: overnight 2026-10-31 to 2026-11-01, `stock-nvfp4`
and `swift-nvfp4`, same server flags and protocol as above (`/v1/systemone`, concurrency 1), logs
[`tonight2.log`](tonight2.log) (soft-digits, VQAv2, AVA, KonIQ, VizWiz, Rapidata Likert; 3 passes) and
[`tonight3.log`](tonight3.log) (Rapidata pairs; 1 pass), both `failures: 0`. Answers are in
`results-vision/<config>/<set>-<split>/`. Sets, with where each one's answers come from (all labels
are human vote or rating shares, none rule-derived or from a model); `validation` fits, `test` is scored:

- [soft-digits](https://huggingface.co/datasets/majapavlo/image-level-soft-label-digits/tree/3f231e9a1481a8b37f7a47e313221f66d9e22639)
  (`majapavlo/image-level-soft-label-digits` @ `3f231e9a1481a8b37f7a47e313221f66d9e22639`, cc-by-sa-4.0): 28x28
  handwritten digits (MNIST and Ambiguous-MNIST) re-annotated by 3-10 Prolific raters each; gold is
  the equally weighted soft label over 0-9 plus "not a digit" (11-option `choice`). 830 + 830 cases.
- [Rapidata pairs](https://huggingface.co/datasets/Rapidata/human-style-preferences-images/tree/0b9a8d940f4c8e5307019fb207b5707e1e31716c)
  (`Rapidata/human-style-preferences-images` @ `0b9a8d940f4c8e5307019fb207b5707e1e31716c`, cdla-permissive-2.0):
  DALL-E 3 vs Flux 1 pro images for one prompt, about 26 crowd votes per pair on which is preferred;
  2-option `choice` over two images (our wording; the Rapidata pairs card doesn't give the raters' question), first 3 of
  53 shards. 1,000 (`validation`) + 1,500 (`test`) pairs. One pass only.
- [VQAv2 yes/no](https://huggingface.co/datasets/lmms-lab-encoder/VQAv2/tree/32665d35052eb4a6d4414851c3c829a72754915a)
  (`lmms-lab-encoder/VQAv2` @ `32665d35052eb4a6d4414851c3c829a72754915a`, cc-by-4.0): COCO val2014 yes/no
  questions, each question has 10 crowd answers and questions with fewer than 8 yes-or-no answers are dropped; gold the share of `yes` among the yes-or-no answers (`noul`); one case per image. 500 + 1,000
  images (1,207 + 2,349 decisions).
- [VizWiz answerability](https://huggingface.co/datasets/lmms-lab-encoder/VizWiz-VQA/tree/d428a2dae984f79cf1b9d99467dfa883e0c30686)
  (`lmms-lab-encoder/VizWiz-VQA` @ `d428a2dae984f79cf1b9d99467dfa883e0c30686`; licence not stated on the
  mirror, the original is CC BY 4.0): photos taken by blind people with a spoken question; `noul`
  "can the question be answered from the photo", gold the share of 10 crowd answers that are not
  `unanswerable`. 1,000 + 2,000 cases.
- [AVA](https://huggingface.co/datasets/trojblue/AVA-aesthetics-10pct-min50-10bins/tree/ee09ff2b4151b87c61cfb5165bb46c0ebe39b679)
  (`trojblue/AVA-aesthetics-10pct-min50-10bins` @ `ee09ff2b4151b87c61cfb5165bb46c0ebe39b679`; the AVA subset card
  defers to the original AVA terms): photo-contest photos with at least 50 votes each on a 1-10 scale;
  10-level `score`, gold the vote shares. 1,000 + 2,000 images.
- [KonIQ-10k](https://huggingface.co/datasets/xyliu6/koniq10k-1024x768/tree/e06f08f327a532537cdb059a91a07a2cd1f7580f)
  (`xyliu6/koniq10k-1024x768` @ `e06f08f327a532537cdb059a91a07a2cd1f7580f`, cc-by-4.0, each image under
  its own CC licence; shares from [`chaofengc/IQA-PyTorch-Datasets-metainfo`](https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets-metainfo/tree/4ee2b4f85138bac8f875d61dc9b4c50e006f8466) @
  `4ee2b4f85138bac8f875d61dc9b4c50e006f8466`): Flickr photos rated for technical quality, bad to
  excellent, by about 106 crowd workers; 5-level `score`. Official `val` 1,000 fits, a fixed sample of
  2,000 of `test` is scored.
- [Rapidata Likert](https://huggingface.co/datasets/Rapidata/flux1.1-likert-scale-preference/tree/a490293a5614b444b317855c7e9dcdd7da91e124)
  (`Rapidata/flux1.1-likert-scale-preference` @ `a490293a5614b444b317855c7e9dcdd7da91e124`, **no licence
  stated** on the Rapidata Likert card, so only aggregates (the tables here) are published from it, no images or
  per-item rows): Flux 1.1 pro images for 280 benchmark prompts, 31-45 crowd ratings each on how well
  the image matches the prompt (1 not at all ... 5 perfectly); 5-level `score`. Halves by prompt, all
  images: 596 (`validation`) + 528 (`test`).

Raw scores, pass 1 (`summary.json`, `test` split; acc / KL / Brier / ECE; accuracy is the top answer
against the top gold option):

| Set | `stock-nvfp4` | `swift-nvfp4` |
|---|---|---|
| soft-digits | 0.7205 / 0.7980 / 0.2396 / 0.0282 | 0.7795 / 0.7980 / 0.1990 / 0.0338 |
| Rapidata pairs | 0.5913 / 0.4153 / 0.2093 / 0.1865 | 0.6120 / 0.4782 / 0.2300 / 0.2019 |
| VQAv2 yes/no | 0.8740 / 0.3113 / 0.1357 / 0.0533 | 0.8761 / 0.3800 / 0.1415 / 0.0552 |
| VizWiz answerability | 0.7940 / 0.3153 / 0.1848 / 0.0404 | 0.7985 / 0.3965 / 0.2048 / 0.0511 |
| AVA | 0.2695 / 0.8854 / 0.2089 / 0.1487 | 0.1305 / 1.5697 / 0.3725 / 0.4010 |
| KonIQ-10k | 0.3805 / 0.8511 / 0.3373 / 0.2126 | 0.4685 / 0.9518 / 0.3229 / 0.2199 |
| Rapidata Likert | 0.4356 / 1.3528 / 0.2219 / 0.1513 | 0.3258 / 2.3014 / 0.3503 / 0.3654 |

Temperature transfer over the full answers: [`results-vision/transfer.md`](results-vision/transfer.md)
(three passes, every set except Rapidata pairs) and [`results-vision/transfer-pass1.md`](results-vision/transfer-pass1.md)
(pass 1, including Rapidata pairs); they supersede the 300-case quick check
([`results-vision-quick/transfer.md`](results-vision-quick/transfer.md)). Scored KL from gold, mean over
the passes (pairs: pass 1, from `transfer-pass1.md`). `pooled images` is one fit over every other
image source (leave-one-out, so the set is not in its own fit; in the three-pass file the pairs set is
not in the pool, in the pass-1 file it is); `own fit` is the set's own `validation` fit, `pooled text`
the six text sources:

| `stock-nvfp4` | raw | pooled images | own fit | pooled text |
|---|---|---|---|---|
| CIFAR-10H (choice) | 0.264 | 0.255 | 0.255 | 0.386 |
| soft-digits (choice) | 0.798 | 0.767 | 0.768 | 0.917 |
| Rapidata pairs (choice) | 0.415 | 0.329 | 0.046 | 0.137 |
| POPE (noul) | 0.347 | 0.270 | 0.270 | 0.302 |
| VQAv2 (noul) | 0.313 | 0.190 | 0.187 | 0.207 |
| VizWiz (noul) | 0.315 | 0.260 | 0.259 | 0.281 |
| AVA (score) | 0.885 | 0.522 | 0.517 | 0.537 |
| KonIQ (score) | 0.851 | 0.636 | 0.578 | 0.582 |
| Rapidata Likert (score) | 1.353 | 0.220 | 0.087 | 0.357 |

| `swift-nvfp4` | raw | pooled images | own fit | pooled text |
|---|---|---|---|---|
| CIFAR-10H (choice) | 0.320 | 0.267 | 0.268 | 0.346 |
| soft-digits (choice) | 0.798 | 0.655 | 0.654 | 0.743 |
| Rapidata pairs (choice) | 0.478 | 0.261 | 0.047 | 0.134 |
| POPE (noul) | 0.402 | 0.277 | 0.276 | 0.327 |
| VQAv2 (noul) | 0.380 | 0.189 | 0.188 | 0.220 |
| VizWiz (noul) | 0.396 | 0.268 | 0.268 | 0.291 |
| AVA (score) | 1.570 | 0.518 | 0.517 | 0.614 |
| KonIQ (score) | 0.952 | 0.608 | 0.491 | 0.498 |
| Rapidata Likert (score) | 2.301 | 0.268 | 0.092 | 0.568 |

Image to image, by question type (row = where T was fitted, column = scored set; KL ↓, bold =
own fit; Rapidata pairs from the pass-1 file, so `stock-nvfp4` / `swift-nvfp4` in each cell):

| Fit \ scored | CIFAR-10H | soft-digits | Rapidata pairs |
|---|---|---|---|
| raw | 0.264 / 0.320 | 0.798 / 0.798 | 0.415 / 0.478 |
| CIFAR-10H | **0.255 / 0.268** | 0.767 / 0.655 | 0.327 / 0.265 |
| soft-digits | 0.255 / 0.267 | **0.768 / 0.654** | 0.331 / 0.259 |
| Rapidata pairs | 1.711 / 1.537 | 1.737 / 1.598 | **0.046 / 0.047** |

| Fit \ scored | POPE | VQAv2 | VizWiz |
|---|---|---|---|
| raw | 0.347 / 0.402 | 0.313 / 0.380 | 0.315 / 0.396 |
| POPE | **0.270 / 0.276** | 0.189 / 0.189 | 0.259 / 0.268 |
| VQAv2 | 0.272 / 0.278 | **0.187 / 0.188** | 0.263 / 0.269 |
| VizWiz | 0.270 / 0.276 | 0.191 / 0.191 | **0.259 / 0.268** |

| Fit \ scored | AVA | KonIQ | Rapidata Likert |
|---|---|---|---|
| raw | 0.885 / 1.570 | 0.851 / 0.952 | 1.353 / 2.301 |
| AVA | **0.517 / 0.517** | 0.589 / 0.540 | 0.177 / 0.169 |
| KonIQ | 0.522 / 0.557 | **0.578 / 0.491** | 0.257 / 0.398 |
| Rapidata Likert | 0.550 / 0.552 | 0.681 / 0.673 | **0.087 / 0.092** |

- **`noul`: yes, and image to image is as good as in-domain.** POPE, VQAv2 and VizWiz fits are within
  0.004 KL of each other on every column (e.g. VQAv2 scored with POPE's T: 0.189 vs 0.187 own,
  raw 0.313, stock). Their `noul` temperatures are 1.96-2.39 (stock) against 2.92-3.83 for Typed Decisions, procedural,
  Calibrated TD and TD Bench (LeWiDi is 1.58), so the text fit over-softens (pooled text 0.207-0.302 vs own
  0.187-0.270) but still beats raw everywhere.
- **`score`: yes between photo-rating sets, with some loss.** AVA, KonIQ-10k and Rapidata Likert all
  beat raw under any other image fit, by a lot where raw is very overconfident (Swift AVA raw 1.570,
  Swift Likert 2.301). The loss against the own fit is small for AVA (0.522 vs 0.517 pooled) and larger
  for KonIQ (0.636 vs 0.578 stock) and for Likert fits applied to other sets: the Likert `score` T is
  11.85 (stock), against 3.76 for AVA and 2.90 for KonIQ, and it gets 0.550 on AVA and 0.681 on KonIQ
  against 0.517 and 0.578 own. Text `score` fits (pooled text) are close to the image ones on AVA and KonIQ (0.537, 0.582).
- **`choice`: only CIFAR-10H and soft-digits agree.** Their fits are interchangeable (0.255 vs 0.255 on
  CIFAR-10H; 0.767 vs 0.768 on soft-digits, stock) and both improve on raw, which is the case the
  text-fitted `choice` T (0.386) failed on. The Rapidata pairs fit (T 12.5-15.0, nearly uniform) is
  catastrophic on the other two (1.711, 1.737 stock), and fitting on the other two helps pairs only
  part of the way (0.327-0.331 vs 0.046 own). Pairs in the pool also move `pooled images` for CIFAR-10H and soft-digits:
  pass 1 with pairs gives CIFAR-10H 0.289 / 0.290 and soft-digits 0.824 / 0.685 (stock / Swift), against 0.255 / 0.267 and 0.767 / 0.655 without; stock soft-digits is then worse
  than raw (0.798).
- **Text to images, summarised:** pooled text beats raw on every image set except CIFAR-10H (0.386
  vs 0.264 stock, 0.346 vs 0.320 Swift) and soft-digits stock (0.917 vs 0.798; Swift 0.743 vs 0.798
  helps). Both are `choice` sets with a ~10-option answer; the text sets' `choice` T of 1.35-3.7 is
  too high for them. This repeats the CIFAR-10H result above on a second ten-way set. Cause not tested.

Does the model carry signal on the Rapidata sets? A uniform guess and a constant (the `validation`
split's mean gold distribution) as baselines, `test` pass 1 (`baselines.py`, full table in
[`results-vision/baselines.md`](results-vision/baselines.md)); KL from gold ↓; `rho` is Spearman between
the model's expected rating and the gold mean rating, `acc` the top answer against the gold mode:

| Set | config | raw | own fit | uniform | val mean | acc (chance) | rho |
|---|---|---|---|---|---|---|---|
| Rapidata pairs | stock | 0.415 | 0.046 | 0.049 | 0.049 | 0.591 (0.500) | - |
| Rapidata pairs | Swift | 0.478 | 0.047 | 0.049 | 0.049 | 0.612 (0.500) | - |
| Rapidata Likert | stock | 1.353 | 0.087 | 0.107 | 0.085 | 0.436 (0.200) | 0.458 |
| Rapidata Likert | Swift | 2.301 | 0.092 | 0.107 | 0.085 | 0.326 (0.200) | 0.368 |

- **Rapidata pairs: by KL, no.** The own fit (T 14.98 stock, 12.54 Swift) flattens the answers to
  almost uniform: 0.046 / 0.047 against 0.049 for a uniform guess. The gold votes are close to a coin
  flip, so the fit takes KL down from raw (0.415, 0.478) by 0.369 and 0.431, to about the
  uniform guess. What is left is ordering: the top answer matches the majority side on 0.591 /
  0.612 of 1,500 pairs against 0.544 for a random side (see the next bullet: tied gold pairs count as hits for either
  answer) and the order of the two images is randomised per pair.
- **Rapidata Likert: ordering, not distribution.** Own fit 0.087 / 0.092 against 0.107 uniform and 0.085
  for the constant mean distribution, so the fitted model is no better than a constant by KL, while the
  expected rating still ranks images (rho 0.458 stock, 0.368 Swift on 528 images; top answer 0.436 /
  0.326 vs 0.200 chance).
- **Position, ties and mean rating** (`baselines.py`, same file): on Rapidata pairs the stock config answers `Image 2`
  on 0.735 of the 1,500 `test` pairs (mean P(`Image 2`) 0.683) and Swift on 0.448 (0.475), where the gold
  share of `Image 2` is 0.503 (the order is randomised per pair), so the stock config has a side preference that a
  temperature, which only flattens or sharpens, cannot remove [INFERENCE: not tested by swapping the order of the same pair]. The scorer counts a tied gold pair
  (0.087 of the pairs) as a hit for either answer, so a random side scores 0.544 (always `Image 1` 0.549, always `Image 2` 0.539), not
  0.500; the accuracies 0.591 / 0.612 are against that, and the pairs share prompts (halves are split by prompt). The model's mean expected rating
  (`test`; raw / after its own `validation` fit / voters), stock then Swift: AVA 6.71 / 6.05 / 5.37 and 6.75 / 5.94 / 5.37, KonIQ 3.45 / 3.20 / 3.16
  and 3.38 / 3.18 / 3.16, Rapidata Likert 4.10 / 3.20 / 3.23 and 3.94 / 3.17 / 3.23. Raw, the model rates higher than the voters on all three score
  sets; the fit moves the mean most of the way (KonIQ, Likert) or part of the way (AVA) toward the voters', and the fitted distributions
  still lose to the constant on AVA and KonIQ by KL.
- Same check on the other score sets, from the same file: AVA own fit 0.517 vs 0.588 uniform vs 0.162
  constant, KonIQ 0.578 vs 0.764 vs 0.334 (stock). On these two a constant beats the fitted model by KL
  too, while KonIQ's rho is 0.791 / 0.790. The fit moves the model's mean rating most of the way to the voters' (previous bullet),
  so what remains is in the per-image distributions (their width and shape), not the mean [INFERENCE]. On POPE, VQAv2 and VizWiz the constant is 0.693, 0.480 and 0.317 (stock
  table: own fit 0.269, 0.187, 0.259), so the model beats it, though on VizWiz raw (0.315) is at the constant's level.

Serving noise: soft-digits, AVA, KonIQ-10k, VizWiz and Rapidata Likert answers are bit-identical across
the three passes (probabilities equal in `pass1/2/3.jsonl`, both configs, both splits; `pass_drift`
all zero). VQAv2 is the exception (6-12 top answers changed
pass 1 vs 2, mean absolute probability difference 0.003-0.006), as POPE above; those are the sets with
several questions per request. Largest spread of any cell in the file: 0.004 KL (stock POPE raw; Swift Likert
under the LeWiDi fit) for the image sets, 0.07 DecisionScore / proper score on the text sets as before.
Rapidata pairs has one pass, so it has no spread.

Caveats: the soft-digits `raw` KL on all 830 test images (0.798) is higher than on the 300-case
quick check (0.656 stock, 0.578 Swift), which used the first 300 cases. AVA, KonIQ and Rapidata Likert
golds are rating distributions of tens to hundreds of voters, their KL includes the sampling noise
of those votes, which a model can't beat; the Rapidata pairs gold has about 26 votes per pair. MNIST, COCO, Flickr,
and AVA photos are likely in the model's pretraining data [INFERENCE]. Rapidata Likert has no stated
licence and only aggregates are published (the `summary.json` files and the tables; `transfer.py` and `baselines.py` can't reproduce the Likert rows and columns, nor `pooled images`, from the published files). Rapidata pairs has one
pass and is the only set missing from `transfer.md`. The models are NVFP4 configs only.

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
| CIFAR-10H (both splits) | 1,000 / 2,000 | 0 | 0 |
| POPE `validation` | 1,800 | 21 | 29 |
| POPE `test` | 3,600 | 57 | 53 |

ChaosNLI, Calibrated TD and CIFAR-10H have no changed top answers between pass 1 and 2, and
ChaosNLI and CIFAR-10H have identical metrics in all three passes. On POPE, the three passes of a config differ by at most 0.0067
accuracy, 0.0047 KL, 0.0033 Brier and 0.0068 ECE (`pass1`, `pass2`, `pass3` in `summary.json`).
In the transfer tables the spread across passes is at most 0.07 points of DecisionScore or proper
score (Swift TD Bench raw 76.59–76.66), 0.004 KL (stock POPE raw 0.344–0.348) and 0.05 on a fitted
`noul` temperature (stock procedural, 3.35–3.40). Differences between sources smaller than these
are not resolved.

### Temperature transfer across datasets (`results-repeat/transfer.md`)

**Question:** does a per-type temperature fitted on one dataset improve another? Run 2026-10-30 to
31, CPU only, from the stored answers of the three-pass runs above (`transfer.py`), `stock-nvfp4`
and `swift-nvfp4`. The one-pass version over the first five sets is kept as
[`results-real/transfer.md`](results-real/transfer.md) (the earlier run).

Method. Temperatures only (T per question type, as in "Calibration"): the biases are keyed to one
dataset's questions and can't carry. Names in the tables, with where each dataset's answers come
from:

- Typed Decisions: the [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/1415d375cb86b418c53abec12795adf5b194e7fc)
  set, gold from the unnamed ~4B teacher, so its scores are agreement with that teacher. Metric
  KL from gold.
- procedural: Procedural Typed Decisions, answers computed from rules. Metric Typed Decision
  Bench's per-item proper score.
- Calibrated TD: Calibrated Typed Decisions `test`, human labels. Metric log loss.
- PubMedQA: Calibrated Typed Decisions `heldout`, expert labels. Metric log loss.
- TD Bench: Typed Decision Bench, labels from 21 public datasets (human annotations and automated
  verifier outcomes) and one rule-derived generator. Metric DecisionScore.
- ChaosNLI, LeWiDi, CIFAR-10H: human vote shares. Metric KL from gold.
- POPE: rule-derived from COCO boxes. Metric KL from gold.

Sources (rows) are fitted on their fit split: Typed Decisions `train`, procedural, Calibrated TD,
ChaosNLI, LeWiDi and CIFAR-10H `validation`, POPE `validation`, TD Bench on all of its `test` (it
has no other split). PubMedQA is a target only. Targets (columns) are scored on each set's scored
split. A type the source has no questions of keeps T = 1, and a cell is `-` where the source has none
of the target's types (it would equal raw). The diagonal (bold) is the in-domain fit; for TD Bench
it is the 5-fold cross-fit used above (stratified by task), so no target is scored with
temperatures fitted on its own answers. "pooled others" is one fit on every source except the
column's dataset, "pooled text" the same over text sources only; both are out of sample. Each pass's
fit uses the same pass of the fit split; cells are the mean over the three passes with [min, max].

Full tables, both configs, with the fitted temperatures: [`results-repeat/transfer.md`](results-repeat/transfer.md).
Raw, in-domain and pooled text rows, mean over passes (metrics as in the file: KL ↓ for Typed
Decisions, ChaosNLI, LeWiDi, CIFAR-10H, POPE; log loss ↓ for Calibrated TD and PubMedQA; proper
score / DecisionScore ↑ for procedural and TD Bench):

| `stock-nvfp4` | Typed Decisions | procedural | Calibrated TD | PubMedQA | TD Bench | ChaosNLI | LeWiDi | CIFAR-10H | POPE |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.541 | 74.26 | 0.602 | 0.902 | 77.21 | 0.579 | 0.394 | 0.264 | 0.347 |
| in-domain | 0.203 | 76.44 | 0.457 | - | 80.28 | 0.194 | 0.327 | 0.255 | 0.270 |
| pooled text | 0.219 | 75.95 | 0.463 | 0.678 | 79.94 | 0.240 | 0.361 | 0.386 | 0.302 |

| `swift-nvfp4` | Typed Decisions | procedural | Calibrated TD | PubMedQA | TD Bench | ChaosNLI | LeWiDi | CIFAR-10H | POPE |
|---|---|---|---|---|---|---|---|---|---|
| raw | 0.677 | 74.47 | 0.668 | 0.948 | 76.63 | 0.750 | 0.427 | 0.320 | 0.402 |
| in-domain | 0.198 | 77.46 | 0.438 | - | 80.58 | 0.184 | 0.314 | 0.268 | 0.276 |
| pooled text | 0.210 | 76.82 | 0.446 | 0.657 | 80.25 | 0.251 | 0.354 | 0.346 | 0.327 |

- **Text to text mostly transfers:** 16 cells in the file are worse than raw in at least one pass
  (flagged "(worse than raw in a pass)"; stock 9, Swift 7). 11 of them are on CIFAR-10H. Of the 5
  on text targets, 4 come from the ChaosNLI fit; the fifth is Calibrated TD on LeWiDi, stock.
- **Worst cells:** the ChaosNLI fit (`choice` T 3.11 stock, 3.71 Swift, the highest of any source)
  does worse than raw on procedural (73.24 vs 74.26 stock, 72.99 vs 74.47 Swift) and TD Bench
  (76.90 vs 77.21, 76.04 vs 76.63), and badly on CIFAR-10H (0.622 vs 0.264 stock, 0.604 vs 0.320
  Swift). The other CIFAR-10H failures are all text fits (see above). One text-to-text case outside
  ChaosNLI: Calibrated TD on LeWiDi, stock only (0.397 vs raw 0.394).
- **The pooled text fit** gets close to the in-domain fit on the text sets with their own
  temperature types (Typed Decisions 0.219 vs 0.203 in-domain, raw 0.541, stock) but not all
  (ChaosNLI 0.240 vs 0.194, LeWiDi 0.361 vs 0.327 stock). On PubMedQA, which has no fit of its own,
  it gives 0.678 (stock) and 0.657 (Swift) log loss against raw 0.902 and 0.948.
- **Fitted `noul` temperatures:** Typed Decisions, procedural, Calibrated TD and TD Bench give
  2.92–3.83 (stock) and 4.08–5.08 (Swift); LeWiDi (1.58, 2.05) and POPE (2.04, 2.61) are lower.

Rerun (CPU, from this folder):

```sh
uv run --no-project --with pyarrow --with numpy --with scipy transfer.py "$DATASETS_DIR" \
    --td-train results-repeat/train --td-test results-repeat/test --real results-repeat/test \
    --vision results-vision --passes pass1 pass2 pass3 > results-repeat/transfer.md
```

Caveats: CIFAR-10H images are 32x32; SNLI, MNLI, CIFAR-10 and COCO are likely in the model's pretraining data [INFERENCE]; POPE has no human distribution (rule-derived one-hot gold), so its KL rewards confidence and is not comparable with the other columns; stock POPE `validation` was rerun after the grayscale fix. Pooling fits one temperature per question type, weighted by that type's decisions. With all six text sources included, Typed Decision Bench supplies about 60% of `noul`, 34% of `choice` and 10% of `score` decisions; the proportions change when the target is excluded. NVFP4 configs only. Intervals are the range
over three passes of a dataset, not sampling error of the scored split.

### How many labeled cases (`results-repeat/learning_curve.md`)

**Question:** how many labeled cases does "T per type" need? CPU only, from the stored pass-1
answers of the three-pass run (`learning_curve.py`); the earlier run over three text sets is
[`results-real/learning_curve.md`](results-real/learning_curve.md). For each NVFP4 config and each
dataset with a fit split separate from its scored split (Typed Decisions `train`, Procedural Typed
Decisions `validation`, Calibrated Typed Decisions `validation`, and now ChaosNLI, LeWiDi, CIFAR-10H
and POPE `validation`), T per type is fitted on n cases drawn at random from the fit split and
scored on the same scored split as above. Each n is drawn 20 times; cells show the median metric
over draws with [10th, 90th percentile]. "Share" is the median fraction of the full-split fit's
gain over raw, `(m_n − raw) / (m_full − raw)`: 1 = as good as fitting on the whole split, below 0 =
worse than raw; the second number is the worst draw. A case can hold several questions (POPE: 6), so
n counts cases. Answers: Typed Decisions is agreement with the unnamed ~4B teacher (KL from gold);
Procedural Typed Decisions answers are rule-computed (proper score); Calibrated Typed Decisions,
ChaosNLI, LeWiDi and CIFAR-10H are human labels or vote shares; POPE is rule-derived from COCO
boxes.

Median share at n = 25 / 50 / 100 / 200 (full tables in the file):

| Set | `stock-nvfp4` | `swift-nvfp4` |
|---|---|---|
| Typed Decisions | 1.00 / 1.00 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 / 1.00 |
| ChaosNLI | 0.99 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 |
| POPE | 0.98 / 1.00 / 1.00 / 1.00 | 0.98 / 0.99 / 1.00 / 1.00 |
| LeWiDi | 0.91 / 0.92 / 0.98 / 0.98 | 0.94 / 0.98 / 0.98 / 0.99 |
| procedural | 0.85 / 0.92 / 0.97 / 1.00 | 0.87 / 0.94 / 0.99 / 1.00 |
| Calibrated TD | 0.73 / 0.82 / 0.94 / 0.98 | 0.84 / 0.91 / 0.98 / 0.99 |
| CIFAR-10H | 0.64 / 0.86 / 0.88 / 0.99 | 0.95 / 0.99 / 0.98 / 1.00 |

- **25 cases are enough for 0.9 or more on ChaosNLI, POPE, Typed Decisions and LeWiDi**, both
  configs. Procedural needs 50 (0.92 stock, 0.94 Swift), Calibrated TD 100 on stock (0.94; 0.91 at
  50 on Swift), CIFAR-10H 200 on stock (0.99; 0.88 at 100) and 25 on Swift (0.95).
- **Worst draws are the risk at small n.** Calibrated TD at n = 25 has a worst-draw share of −5.00
  (stock) and −6.82 (Swift), still −4.09 and −2.00 at n = 50, and 0.79 at n = 100 on both. LeWiDi
  at n = 25: −31.83 (stock), −13.17 (Swift). From n = 100 every worst draw is at least 0.56
  (stock CIFAR-10H); stock CIFAR-10H falls to 0.04 at n = 200, where the gain over raw is only 0.009
  KL (0.264 → 0.255), so its share is noisy.
- **POPE counts cases, not questions:** 25 cases is 150 yes/no decisions.

Rerun (CPU, from this folder):

```sh
uv run --no-project --with pyarrow --with numpy --with scipy learning_curve.py "$DATASETS_DIR" \
    --td-train results-repeat/train --td-test results-repeat/test --real results-repeat/test \
    --vision results-vision --pass pass1 > results-repeat/learning_curve.md
```

Caveats: draws come from fit splits of 300–2,334 cases (a 500-case draw covers 21–57% of the fit
split; POPE has no n = 500), NVFP4 configs only, intervals are percentiles over draws and say
nothing about the scored split's own sampling error.

### How many labeled cases, image sets (`results-vision/learning_curve.md`)

**Question:** the same as above, on the image sets of "More image sets". CPU only, from the stored pass-1
answers (`learning_curve.py`, same protocol: T per type fitted on n cases drawn at random from the `validation`
split, scored on the set's `test` split, 20 draws per n, pass 1; the same draws as above for CIFAR-10H and POPE). Rapidata pairs and Rapidata Likert
are included; the image sets' answers come from people (POPE: a rule), see "More image sets". n counts
cases: one image each, except POPE (one image with one setting, 6 questions; a draw can hold the same photo under different settings), VQAv2 (one image, about 2.4 questions) and Rapidata pairs (two images). Median share of the
full-split fit's gain over raw at n = 25 / 50 / 100 / 200, then the worst of the 20 draws (full tables, with the metric and its [p10, p90],
in the file):

| Set (fit cases) | `stock-nvfp4` median | `swift-nvfp4` median | `stock-nvfp4` worst draw | `swift-nvfp4` worst draw |
|---|---|---|---|---|
| CIFAR-10H (1000) | 0.64 / 0.86 / 0.88 / 0.99 | 0.95 / 0.99 / 0.98 / 1.00 | -1.67 / -0.21 / 0.56 / 0.04 | 0.55 / 0.75 / 0.93 / 0.82 |
| soft-digits (830) | 0.87 / 0.99 / 0.98 / 0.98 | 0.99 / 1.01 / 1.00 / 1.00 | 0.18 / 0.33 / 0.55 / 0.19 | 0.84 / 0.87 / 0.93 / 0.91 |
| Rapidata pairs (1000) | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 0.98 / 0.99 / 1.00 / 1.00 | 0.97 / 0.99 / 0.99 / 1.00 |
| POPE (300) | 0.98 / 1.00 / 1.00 / 1.00 | 0.98 / 0.99 / 1.00 / 1.00 | 0.87 / 0.95 / 0.97 / 0.99 | 0.84 / 0.90 / 0.98 / 0.99 |
| VQAv2 (500) | 0.99 / 1.00 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 / 1.00 | 0.89 / 0.95 / 0.96 / 0.99 | 0.88 / 0.97 / 0.99 / 1.00 |
| VizWiz (1000) | 0.92 / 0.98 / 0.95 / 1.00 | 0.95 / 0.99 / 0.99 / 1.00 | 0.20 / 0.65 / 0.64 / 0.90 | 0.67 / 0.82 / 0.87 / 0.98 |
| AVA (1000) | 0.99 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 0.95 / 0.99 / 0.99 / 1.00 | 0.99 / 0.99 / 0.99 / 1.00 |
| KonIQ-10k (1000) | 0.99 / 0.99 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 / 1.00 | 0.93 / 0.95 / 0.99 / 0.99 | 0.94 / 0.98 / 0.99 / 0.99 |
| Rapidata Likert (596) | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 |

- **At n = 100 the median is at least 0.88 on every set and config**, and the worst draw is at least
  0.87 on 15 of the 18 set-config cells. The three below: stock CIFAR-10H (0.56), stock soft-digits (0.55) and stock
  VizWiz (0.64).
- **The ten-way `choice` sets are the volatile ones, because their gain is small**: raw to full-split
  fit is 0.264 to 0.255 on stock CIFAR-10H (0.009 KL) and 0.798 to 0.768 on stock soft-digits (0.030), so a share
  swings widely (stock CIFAR-10H worst draw 0.04 at n = 200, 0.90 at n = 500; stock soft-digits 0.19 at n = 200, 0.86 at n = 500).
  At n = 100 the median draw is within 0.0010 KL of the full-split fit on both (stock CIFAR-10H 0.2561 vs 0.2551, stock
  soft-digits 0.7688 vs 0.7681, from the file).
- **`noul` and `score` sets need little**: POPE, VQAv2, AVA and KonIQ-10k have a worst-draw share of at least
  0.84 at n = 25 (both configs). VizWiz is the exception: worst draw 0.20 (stock) and 0.67 (Swift) at n = 25,
  0.90 and 0.98 at n = 200.
- **Rapidata pairs and Likert read 1.00 everywhere because the fit just flattens the answers**
  to about a constant (see "Does the model carry signal on the Rapidata sets?"), which one or two parameters learn from a few cases.

Rerun (CPU, from this folder; the three text roots point at folders that don't exist so only the image sets are computed):

```sh
uv run --no-project --with pyarrow --with numpy --with scipy --with pillow learning_curve.py "$DATASETS_DIR" \
    --td-train none --td-test none --real none --vision results-vision --pass pass1 > results-vision/learning_curve.md
```

Caveats: draws come from fit splits of 300-1,000 cases (a 500-case draw covers 50-84% of the fit split;
POPE and VQAv2 have no n = 500), NVFP4 configs only, intervals are percentiles over draws and say nothing about the scored
split's own sampling error. The shares of sets whose full-split gain is small (CIFAR-10H, soft-digits) are noisy.

## Benchmark

[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/1415d375cb86b418c53abec12795adf5b194e7fc), `all`
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
uv run --no-project --with pyarrow --with numpy --with scipy transfer.py "$DATASETS_DIR" > results-real/transfer.md
```

Overnight run (three passes of every text and image run, both NVFP4 configs; needs `chaosnli/`,
`lewidi/`, `cifar10h/` and `pope/` under `DATASETS_DIR` at the revisions in "Three passes, four
more sets"), then the tables (host, CPU):

```sh
DATASETS_DIR=... MODELS_DIR=... BENCH_CDI=1 ./tonight.sh
uv run --no-project --with pyarrow --with numpy --with scipy transfer.py "$DATASETS_DIR" \
    --td-train results-repeat/train --td-test results-repeat/test --real results-repeat/test \
    --vision results-vision --passes pass1 pass2 pass3 > results-repeat/transfer.md
uv run --no-project --with pyarrow --with numpy --with scipy learning_curve.py "$DATASETS_DIR" \
    --td-train results-repeat/train --td-test results-repeat/test --real results-repeat/test \
    --vision results-vision --pass pass1 > results-repeat/learning_curve.md
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
- `results-repeat/` and `results-vision/`: three passes per run; each dataset's passes share one server session
  (train and test were separate launches; pass 1 precedes the cache-reusing passes 2-3); NVFP4 configs only. CIFAR-10H images are 32x32. SNLI, MNLI,
  CIFAR-10 and COCO are likely in the model's pretraining data [INFERENCE]. POPE has no human
  distribution: its gold is a rule on COCO boxes. 2 of the 300 POPE `validation` images were
  grayscale and crashed the stock run; the client now converts non-RGB images to RGB PNG (commit
  2c360ef) and stock POPE `validation` was rerun with the same flags (the Swift run started after
  the fix, its `pope-validation` files are dated 00:25, after the 00:10 commit).
