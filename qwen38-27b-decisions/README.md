# qwen38-27b-decisions

**Question:** can Qwen3.8-27B, as served for chat, double as a decision model through SGLang's
`/v1/systemone` and `/v1/decisions`, and do Swift 1.5's fine-tune or the quantization
(BF16 / FP8 / NVFP4) hurt it? Answers come from the answer-label log-probs with thinking off
([SGLang decision models docs](https://docs.sglang.io/docs/supported-models/decision_models)).
Follow-up: how much does client-side calibration (fitted on `train`, no weights or server
changes) fix?

Run 2026-10-30, one RTX PRO 6000, one config at a time.

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
decisions per config:

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

Read: quantization doesn't measurably change agreement (every BF16 → FP8/NVFP4 interval
contains 0, every point gap is under 0.01, widest bound +0.021); the drafter changes nothing (decisions are prefill only);
the two APIs give the same answers. Swift agrees with the teacher 0.006–0.018 more than stock
(interval above 0 at BF16 and NVFP4) and sits further from its distributions (KL 0.66–0.71 vs
0.54–0.59): sharper, with only a small agreement lead.

## Calibration (`stock-nvfp4`, `swift-nvfp4`)

`../scripts/decisions/calibrate.py`, fitted on the `train` split's answers (1,200 cases, 6,000
decisions, `results-train/`) by soft cross-entropy against the gold, then scored once on `test`
with the same scorer. Both act on the probabilities the client gets back; no weights, no server
change, chat on the same server untouched.

- **T:** one temperature per question type, `p_i ∝ p_i^(1/T)`. 3 parameters. Can't change any
  argmax, so agreement stays the same.
- **T+B:** the same plus one log-prior bias per workflow, question and option,
  `p_i ∝ p_i^(1/T) e^(b_i)`, L2-penalised, with the temperatures refit alongside the biases. 3 + 71 parameters. Same idea as the dataset card's
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
on `train` and belong with that card's second table ("fitted or fine-tuned on `train`").

## Configs

| Label | Checkpoint | Revision | `--mem-fraction-static` |
|---|---|---|---|
| `stock-bf16` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 0.76 |
| `stock-fp8` | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | 0.50 |
| `stock-nvfp4` | [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) | `319f741cce68d7914884900c138a1fbb70a42f30` | 0.41 |
| `swift-bf16` | [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) | `b4c84d42903a8646b25857eb2827288d92ed85a4` | 0.76 |
| `swift-fp8` | block FP8 of `swift-bf16`, built locally | 128×128 blocks, stock FP8's layout; public equivalent [realderpz/Swift-1.5-Qwen3.8-27B-FP8](https://huggingface.co/realderpz/Swift-1.5-Qwen3.8-27B-FP8) (llmcompressor `FP8_BLOCK`) | 0.50 |
| `swift-nvfp4` | [ukisai/Swift-1.5-Qwen3.8-27b-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4) | `25482027debd5485e8108897ba9fed8d3ba16595` (ModelOpt mixed, no KV scales) | 0.41 |
| `swift-nvfp4-dflash8` | as above + [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, 8 draft tokens | the daily setup | 0.41 |

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
OUT=$PWD/results-train DECISIONS_SPLIT=train DECISIONS_PASSES=1 DECISIONS_NATIVE_PASSES=0 \
  DATASETS_DIR=... MODELS_DIR=... BENCH_CDI=1 ./run.sh stock-nvfp4 swift-nvfp4

# calibration (host, CPU only)
uv run --no-project --with pyarrow --with numpy --with scipy ../scripts/decisions/calibrate.py \
  --train-answers results-train/swift-nvfp4/decisions/pass1.jsonl \
  --train-data "$DATASETS_DIR/typed-decisions/all/train-00000-of-00001.parquet" \
  --test-answers results/swift-nvfp4/decisions/pass1.jsonl \
  --test-data "$DATASETS_DIR/typed-decisions/all/test-00000-of-00001.parquet" \
  --out results/swift-nvfp4/calibration.json
```

`DATASETS_DIR` needs `typed-decisions/all/{train,test}-00000-of-00001.parquet` at the revision
above.

## Caveats

- One run per config. Repeat passes and the two APIs move at most 3 of 2,000 top answers, so
  request-level noise is negligible; the paired intervals above are the uncertainty from the 400
  cases.
- Gold is an unnamed ~4B-class teacher's soft distribution, not human labels. "Calibrated" means calibrated to
  that teacher.
- The biases are per workflow, question and option: they only exist for questions with labeled
  `train` data. The temperatures are per question type; whether they carry over to other
  questions is not tested here.
- Calibration was fitted for the two NVFP4 configs only.
