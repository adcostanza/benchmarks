# Qwen3.8-27B NVFP4: which speculative drafter is fastest?

Post: [DFlash2 8 led my Qwen3.8-27B drafter tests on one RTX PRO 6000](https://acostanza.com/benchmarks/qwen38-27b-spec-decode/). Hardware: [hardware.md](../hardware.md).

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

BFCL subset (of 205): nospec 171, mtp3 173, mtp7 173, dflash4 176, dflash8 176, dspark 171. Categories have 10–25 cases, so these differences are not enough to establish a difference or equivalence. BFCL ran with the `Qwen/Qwen3-32B-FC` handler at its default effort and temperature 0.001, not `reasoning_effort=medium`.

Reading the data: `dflash8` has the highest output rate in every scenario. It reaches 2.5x the no-spec speed on chat with one user and 1.16x at 120k tokens. Accept length swings by 0.15–0.43 across contexts for every drafter; only `dflash4` drifts down (2.81 to 2.66), `dflash8` does not. The 20k–120k rows are the SGLang `random` dataset (one ShareGPT prompt repeated to length), so these accept lengths are not representative of real prompts. The shrinking gain is consistent with prefill, which speculation does not remove: TPOT stays near-halved at every length (about 5.4–6.5 ms vs 13.8–16.5 ms), while median TTFT (28 s at 120k) dominates the 512-token requests. `mtp7` accepts more tokens per step than `dflash8` at 40k–120k but is slower in every scenario; per-step cost was not measured (its TPOT is higher at 1 user). `dspark` accepts about as many tokens as `mtp3` and is 1–5% ahead of `dflash4` in five of seven throughput rows and level in the other two; it is the slowest config in mean BFCL latency (3.24 s per call vs 0.91 s for `dflash8`).

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

One run per config. The 40k–120k scenarios have 4 prompts each, so their rates carry a few percent of noise, and `dflash8`'s measured leads over the nearest drafter were 3.2% at 40k, 3.8% at 80k and 3.4% at 120k; one run per config can't distinguish leads that small from run-to-run variation.

## DSpark with its card's server flags

Question: the main runs gave `dspark` the card's drafter flags but this benchmark's shared server flags. Does DSpark do better with the [DSpark card](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)'s server flags (@ `b9a5dbdf`), and is that specific to DSpark? `dspark-cardflags` is `dspark` with the card's server flags; `dflash8-cardflags` is `dflash8` with the same server flags, as a control. Run 2026-11-01, one run per label, same image, target and drafters as above, default scenarios plus 40k/80k/120k, no BFCL (`bfcl=0` in `run.sh`; the BFCL table at the end of `cardflags.log` is `compare.py` printing the old labels, and `results/*-cardflags/` has no `bfcl/`). Run log: `cardflags.log`; comparisons: `compare-cardflags.md`. Rerun: `./run.sh dspark-cardflags dflash8-cardflags` at [`72b88ad`](https://github.com/adcostanza/benchmarks/tree/72b88ad), the commit these runs were released in.

Output tokens/s (higher is better):

| Scenario | dspark | dspark-cardflags | dflash8 | dflash8-cardflags |
|---|---|---|---|---|
| Chat, 1 user | 148.1 | 146.3 | 178.0 | 167.8 |
| Chat, 4 users | 503.8 | 474.6 | 592.5 | 520.2 |
| 20k prompt, 1 user | 88.8 | 87.4 | 98.4 | 94.6 |
| 20k prompt, 4 users | 131.2 | 136.1 | 137.5 | 144.3 |
| 40k prompt, 1 user | 56.9 | 56.2 | 58.7 | 59.2 |
| 80k prompt, 1 user | 27.5 | 28.0 | 28.5 | 29.1 |
| 120k prompt, 1 user | 15.9 | **rejected** | 16.5 | **rejected** |

Card flags vs original flags: `dspark` -1.2%, -5.8%, -1.6%, +3.8%, -1.3%, +2.0% (rows in the order above, 120k excluded); `dflash8` -5.7%, -12.2%, -3.9%, +4.9%, +0.8%, +2.2%. DFlash2 8 vs DSpark on the card flags: +14.7%, +9.6%, +8.3%, +6.0%, +5.5%, +4.0%.

Median time per output token (ms) and mean accept length:

| Scenario | TPOT dspark | TPOT dspark-cardflags | TPOT dflash8 | TPOT dflash8-cardflags | accept dspark | accept dspark-cardflags | accept dflash8 | accept dflash8-cardflags |
|---|---|---|---|---|---|---|---|---|
| Chat, 1 user | 6.92 | 6.33 | 5.57 | 5.73 | 2.84 | 2.89 | 3.43 | 3.35 |
| Chat, 4 users | 7.28 | 7.78 | 5.91 | 7.15 | 2.90 | 2.93 | 3.49 | 3.28 |
| 20k prompt, 1 user | 6.89 | 6.34 | 5.37 | 5.73 | 2.95 | 3.01 | 3.54 | 3.39 |
| 20k prompt, 4 users | 16.57 | 17.06 | 15.27 | 15.80 | 2.93 | 2.98 | 3.02 | 3.27 |
| 40k prompt, 1 user | 6.79 | 7.38 | 5.76 | 6.23 | 3.07 | 3.00 | 3.28 | 3.45 |
| 80k prompt, 1 user | 7.49 | 7.70 | 5.88 | 6.20 | 2.88 | 3.01 | 3.43 | 3.61 |

**Reading.** DSpark did not get meaningfully faster with its card's server flags: six rows between -5.8% and +3.8%, no sign pattern, accept length within ±5% (2.84 to 2.89 on chat, 2.88 to 3.01 at 80k). The control moved the same size and mostly the other way: DFlash2 8 lost 5.7% and 12.2% on chat and 3.9% at 20k with one user, and gained 0.8–4.9% on the rest. So the flags don't help either drafter, and may cost a few percent on chat decode (TPOT up 2.8% at 1 user and 21% at 4 users for DFlash2 8, the largest move in the table; one run each, so a real effect is not separated from noise: the main post's same-config reruns of 20k at 4 users differed by about 3%). DFlash2 8 stays ahead of DSpark in every scenario that ran, by 4.0–14.7% on the card flags, and its original-flags lead was 3.2–20.2%. The DSpark-vs-DFlash2 conclusion of the post does not change.

**120k did not fit.** Both card-flags servers rejected every 120k request: "Input length (120010 tokens) exceeds the maximum allowed length (110330 tokens)" for `dspark-cardflags` and "(108026 tokens)" for `dflash8-cardflags` (`server.log`). The `compare.py` 120k rows (about 3,000 tok/s, TTFT 0) are the 0.7 s it took to reject four requests, not throughput, and `bench-serving.sh` counted them as "Successful requests: 4" (the server aborts the request but answers `200 OK`, so the client counts a completed request with no output tokens; `total_output_tokens_retokenized` is 0). Ignore those rows. The 80k scenario (about 80k + 512 tokens) fit. Flags were not changed, as planned.

Memory, from each `server.log` and `resources.json` (`resources.json` only exists for the new labels; the main runs predate it):

| | dspark | dspark-cardflags | dflash8 | dflash8-cardflags |
|---|---|---|---|---|
| KV pool, `max_total_num_tokens` | 241,216 | 110,336 | 236,864 | 108,032 |
| Max input accepted | - | 110,330 | - | 108,026 |
| Mamba slots, `max_mamba_cache_size` | 28 | 34 | 28 | 33 |
| Mamba `ssm_state` (GB) | 2.04 | 4.92 | 2.04 | 4.78 |
| Hierarchical KV host pool (tokens) | 1,159,424 | 465,984 | 1,150,848 | 463,296 |
| Hierarchical Mamba host cache (GB) | 26.04 | 48.80 | 26.36 | 48.95 |
| Peak GPU memory (MiB) | - | 58,407 | - | 58,729 |
| Peak container memory (GiB) | - | 76.65 | - | 76.4 |
| Peak host memory used (GiB) | - | 95.66 | - | 95.93 |
| Startup (s) | - | 705 | - | 678 |

The float32 Mamba state doubles the SSM state and the 11.93 ratio (against 0.9) gives the Mamba pool most of the 0.41 budget, so the KV pool drops by 54% and 120k no longer fits. The card runs `--mem-fraction-static 0.80` on an H200; here the fraction stays at 0.41, as in every other config.

**Startup 705 s and 678 s instead of about 170 s.** `Capture target prefill CUDA graph` took 538.49 s (`dspark-cardflags`) and 518.34 s (`dflash8-cardflags`) against 50.58 s and 50.82 s for `dspark` and `dflash8`: with `--chunked-prefill-size 32768` the breakable prefill graphs are captured for token counts up to 32,768 (106 sizes) instead of up to 2,048 (42 sizes). The rest of `scheduler_e2e` is 115 s for both labels against 83 s for `dspark` (weight loading 46.83 s and 33.67 s against 13.08 s, which the logs don't explain).

Flags that differ from the shared server flags (`scripts/args/qwen38-27b.sh`; `qwen38_27b_card_server_args` rewrites them):

| Flag | Shared (`dspark`, `dflash8`) | Card flags |
|---|---|---|
| `--mamba-ssm-dtype` | `bfloat16` | `float32` |
| `--mamba-radix-cache-strategy` | `extra_buffer_lazy` | `extra_buffer` |
| `--mamba-full-memory-ratio` | `0.9` | `11.93` |
| `--chunked-prefill-size` | `2048` | `32768` |
| `--max-prefill-tokens` | not set (log shows 16384) | `32768` |
| env `PYTORCH_CUDA_ALLOC_CONF` | not set | `expandable_segments:True` (`card-dspark.env`, `card-dflash.env`) |
| env `SGLANG_RAGGED_VERIFY_MODE` | not set | `static` on `dspark-cardflags` only (`card-dspark.env`) |

`SGLANG_RAGGED_VERIFY_MODE=static` is the image's default (`srt/environ.py` in `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48`), so setting it changes nothing; it is there because the card's launch command sets it. Kept as in every other config: `--mem-fraction-static 0.41` (memory budget; the card uses 0.80 on an H200), `--max-running-requests 4` (the card uses 48; at most 4 users here), and the hierarchical cache.

Caveats:

- One run per label. Differences under about 4% (the size of same-config reruns of 20k at 4 users) are not separable from noise; only the DFlash2 8 chat drop at 4 users (-12.2%) is clearly larger, and one run does not establish it.
- The card flags were run with a 0.41 memory fraction, so this is not the card's configuration: the KV pool is less than half of the main runs' and 120k does not fit. A run at the card's 0.80 fraction (an H200 has 141 GB; this card has 96 GB) would need a different memory budget and is not covered.
- No BFCL for the new labels, so nothing here says whether the flags change tool-call accuracy.
- The 20k–120k rows are one repeated prompt (see the main caveats), the same as the main runs.
