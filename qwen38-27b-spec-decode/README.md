# Qwen3.8-27B NVFP4: which speculative drafter is fastest?

Post: [DFlash2 is the fastest drafter for Qwen3.8-27B, up to 120k tokens](https://acostanza.com/blog/qwen38-27b-spec-decode/). Hardware: [hardware.md](../hardware.md).

Question: on one RTX PRO 6000 Blackwell (400 W), which drafter gives the highest decode speed for stock Qwen3.8-27B NVFP4 at `reasoning_effort=medium`, and how does acceptance hold as the context grows to 120k tokens?

Output tokens/s (higher is better; one run per config):

| Scenario | nospec | mtp3 | mtp7 | dflash4 | dflash8 | dspark |
|---|---|---|---|---|---|---|
| Chat, 1 user | 72.0 | 143.8 | 143.0 | 143.3 | **178.0** | 148.1 |
| Chat, 4 users | 233.7 | 522.0 | 506.5 | 489.5 | **592.5** | 503.8 |
| 20k prompt, 1 user | 52.9 | 85.1 | 84.3 | 84.9 | **98.4** | 88.8 |
| 20k prompt, 4 users | 94.7 | 129.9 | 127.9 | 131.5 | **137.5** | 131.2 |
| 40k prompt, 1 user | 39.1 | 52.9 | 55.0 | 54.5 | **58.7** | 56.9 |
| 80k prompt, 1 user | 22.5 | 26.4 | 26.8 | 27.1 | **28.5** | 27.5 |
| 120k prompt, 1 user | 14.2 | 15.3 | 15.4 | 15.9 | **16.5** | 15.9 |

Mean accept length (tokens per verify step; sharegpt_c1 / long20k_c1 / long40k_c1 / long80k_c1 / long120k_c1):

| Config | sharegpt_c1 | long20k_c1 | long40k_c1 | long80k_c1 | long120k_c1 |
|---|---|---|---|---|---|
| mtp3 | 2.80 | 2.87 | 2.74 | 2.90 | 2.84 |
| mtp7 | 3.36 | 3.41 | 3.65 | 3.79 | 3.55 |
| dflash4 | 2.81 | 2.79 | 2.76 | 2.70 | 2.66 |
| dflash8 | 3.43 | 3.54 | 3.28 | 3.43 | 3.46 |
| dspark | 2.84 | 2.95 | 3.07 | 2.88 | 2.70 |

BFCL subset (of 205): nospec 171, mtp3 173, mtp7 173, dflash4 176, dflash8 176, dspark 171. Categories have 10–25 cases, so these differences are noise.

Reading the data: `dflash8` is fastest in every scenario. It reaches 2.5x the no-spec speed on chat with one user and 1.16x at 120k tokens. Accept length swings by 0.15–0.43 across contexts for every drafter; only `dflash4` drifts down (2.81 to 2.66), `dflash8` does not. The 20k–120k rows are the SGLang `random` dataset (one ShareGPT prompt repeated to length), so these accept lengths are not representative of real prompts. The shrinking gain comes from prefill, which speculation does not remove: TPOT stays near-halved at every length (about 5.4–6.5 ms vs 13.9–16.5 ms), while the fixed prefill time (TTFT, 28 s at 120k) dominates the 512-token requests. `mtp7` accepts more tokens per step than `dflash8` at 40k–120k but is slower in every scenario; per-step cost was not measured (its TPOT is higher at 1 user). `dspark` accepts about as many tokens as `mtp3` and is 1–5% ahead of `dflash4` in five of seven throughput rows and level in the other two; it is the slowest config in mean BFCL latency (3.24 s per call vs 0.91 s for `dflash8`).

Full tables: `../scripts/compare.py results/nospec results/mtp3 results/mtp7 results/dflash4 results/dflash8 results/dspark`.

## Diagnosis: is the DFlash result a thinking or sampling artifact?

`dflash8` re-run serving-only with thinking off (`enable_thinking=false`) and with temperature 0 (default scenarios):

| Scenario | dflash8 | noreason | greedy |
|---|---|---|---|
| sharegpt_c1 out tok/s | 178.0 | 184.7 | 177.2 |
| sharegpt_c4 out tok/s | 592.5 | 566.0 | 593.2 |
| long20k_c1 out tok/s | 98.4 | 101.2 | 98.5 |
| long20k_c4 out tok/s | 137.5 | 140.0 | 138.5 |
| sharegpt_c1 accept len | 3.43 | 3.57 | 3.41 |
| long20k_c4 accept len | 3.02 | 3.33 | 3.28 |

Neither changes the picture: within about ±5% except `long20k_c4` accept length (single run, 8 prompts).

## Configs

All configs: [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) @ `319f741cce68d7914884900c138a1fbb70a42f30`, `--mem-fraction-static 0.41`, server flags from [scripts/args/qwen38-27b.sh](../scripts/args/qwen38-27b.sh) followed by the speculative flags below. Image `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` for every config (it supports `DFLASH` and `DSPARK`, so one engine for all). The speculative flags come after all other flags here, unlike post 1 where they sit in the middle of the list (same flag set).

| Label | Speculative flags | KV pool (tokens) | Mamba slots |
|---|---|---|---|
| `nospec` | none | 318,592 | 117 |
| `mtp3` | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | 276,224 | 39 |
| `mtp7` | `--speculative-algorithm EAGLE --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8` | 264,256 | 24 |
| `dflash4` | `--speculative-algorithm DFLASH --speculative-draft-model-path /models/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 4` | 244,160 | 44 |
| `dflash8` | same with `--speculative-num-draft-tokens 8` | 236,864 | 28 |
| `dspark` | `--speculative-algorithm DSPARK --speculative-draft-model-path /models/Qwen3.8-27B-DSpark --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer --speculative-dspark-block-size 7 --speculative-num-steps 1 --speculative-eagle-topk 1` | 241,216 | 28 |
| `dflash8-noreason` | as `dflash8`; requests use `{"chat_template_kwargs":{"enable_thinking":false}}`; serving only | 236,864 | 28 |
| `dflash8-greedy` | as `dflash8`; requests add `"temperature":0` to `reasoning_effort=medium`; serving only | 236,864 | 28 |

KV pool is `max_total_num_tokens` and mamba slots is `mamba_cache_size`, both from each `results/<label>/server.log`. The pool differs per config because the draft model and verify buffers take part of the same 0.41 fraction.

DSpark flags follow the [model card](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) (gamma 7, one step, top-k 1). The card also uses other settings (`extra_buffer`, float32 Mamba state, ratio 11.93, `--max-running-requests 48`); this benchmark keeps the shared server flags so only the drafter differs. MTP uses the target's built-in MTP head (no draft path).

Checkpoints:

| | Repo | Revision |
|---|---|---|
| Target | RadixArk/Qwen3.8-27B-NVFP4 | `319f741cce68d7914884900c138a1fbb70a42f30` |
| DFlash2 | z-lab/Qwen3.8-27B-DFlash2 | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| DSpark | RadixArk/Qwen3.8-27B-DSpark | `b9a5dbdf03bc999c6c73c426b19c2d9041cea393` |

```sh
hf download RadixArk/Qwen3.8-27B-NVFP4 --revision 319f741cce68d7914884900c138a1fbb70a42f30 --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4"
hf download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 --local-dir "$MODELS_DIR/Qwen3.8-27B-DFlash2"
hf download RadixArk/Qwen3.8-27B-DSpark --revision b9a5dbdf03bc999c6c73c426b19c2d9041cea393 --local-dir "$MODELS_DIR/Qwen3.8-27B-DSpark"
```

Run: `./run.sh` (all configs, retries a failed config once) or `./run.sh dflash8 dspark`. About two hours.

## Results

`results/<label>/`: `bench-serving.jsonl` (scenarios `sharegpt_c1`, `sharegpt_c4`, `long20k_c1`, `long20k_c4`, plus `long40k_c1`, `long80k_c1`, `long120k_c1` with 4 prompts of 512 output tokens each, except the diagnosis configs), `server.log`, and for the six main configs `bfcl/`. No config failed.

One run per config. The 40k–120k scenarios have 4 prompts each, so their rates carry a few percent of noise, and `dflash8`'s lead over the nearest drafter at 40k (+3.2%), 80k and 120k is inside it.
