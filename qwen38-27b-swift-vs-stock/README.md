# Swift 1.5 NVFP4 vs stock Qwen3.8-27B NVFP4 on one RTX PRO 6000

Question: does Swift 1.5 cost anything against stock Qwen3.8-27B at NVFP4, in quality, in speed or
in DFlash2 acceptance, and does the stock drafter leave room for one fine-tuned on Swift? Answer:
no cost anywhere measured; Swift writes fewer tokens for the same answers; the stock drafter
already matches the DFlash2 card's accept length on both models, so a Swift-tuned drafter has
little headroom. Hardware: [hardware.md](../hardware.md) (400 W). One run per config.

Tables from `../scripts/compare.py results/dflash8 results/swift-dflash8` and
`../scripts/compare.py results/stock-nvfp4-quality results/swift-nvfp4-quality`. compare.py
numbers the accept-sets and SPEED-Bench runs (`ref#2`, `daily#3`); the set or split of each run
is `settings.set` / `settings.split` in its `summary.json`.

## Quality (`*-quality`, reasoning_effort=medium)

| Benchmark | Stock NVFP4 | Swift NVFP4 |
|---|---|---|
| GSM8K, all 1,319 | 96.82% ± 0.48 (1,277) | 97.04% ± 0.47 (1,280) |
| GSM8K mean output tokens | 348.4 | 319.7 (−8.2%) |
| GSM8K mean request time (32 in flight) | 11.38 s | 10.18 s (−10.5%) |
| BFCL v4 subset, of 205 | 168 (82.0% ± 2.7) | 168 (82.0% ± 2.7) |
| BFCL mean output tokens per call | 146.5 | 133.9 (−8.6%) |
| BFCL mean latency per call | 0.93 s | 0.87 s |

BFCL by category (stock / Swift): irrelevance 15/11, live_irrelevance 15/14, live_multiple 19/20,
live_simple 18/18, multi_turn_base 14/16, multi_turn_miss_param 3/3, multiple 24/23, parallel
18/19, parallel_multiple 18/19, simple_python 24/25 (of 20, 20, 25, 20, 20, 10, 25, 20, 20, 25).

## Accept length on the DFlash2 card's five sets (`dflash8`, `swift-dflash8`)

32 prompts per set, concurrency 1, max 4096 new tokens, temperature 1.0, top-p 0.95, top-k 20.
`ref` is the card's protocol (`xhigh`), `daily` is `medium`. Mean accept length ± standard error.
Card: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) @
`50307d4c`, "Acceptance Length" table, DFlash 2 column (one H200, block size 8, `xhigh`, max 4096
new tokens; prompt count not stated). Checked 2026-10-30.

| Set | Card | Stock, ref | Swift, ref | Stock, daily | Swift, daily |
|---|---|---|---|---|---|
| GSM8K | 5.46 | 5.29 ± 0.11 | 5.57 ± 0.11 | 6.18 ± 0.11 | 6.18 ± 0.08 |
| MATH-500 | 5.28 | 5.18 ± 0.14 | 5.22 ± 0.11 | 5.56 ± 0.15 | 5.68 ± 0.11 |
| HumanEval | 4.39 | 4.24 ± 0.13 | 4.45 ± 0.11 | 6.18 ± 0.09 | 6.20 ± 0.13 |
| MBPP | 4.79 | 4.61 ± 0.10 | 4.79 ± 0.12 | 5.39 ± 0.07 | 5.45 ± 0.08 |
| MT-Bench | 4.10 | 4.13 ± 0.18 | 4.33 ± 0.21 | 4.73 ± 0.25 | 4.84 ± 0.25 |

Output tok/s at concurrency 1 (card: DFlash 2, concurrency 1):

| Set | Card | Stock, ref | Swift, ref | Stock, daily | Swift, daily |
|---|---|---|---|---|---|
| GSM8K | 236.1 | 240.0 | 265.1 | 296.0 | 303.3 |
| MATH-500 | 230.7 | 233.7 | 251.3 | 251.3 | 280.6 |
| HumanEval | 214.6 | 199.5 | 209.8 | 305.2 | 303.6 |
| MBPP | 226.9 | 216.6 | 220.5 | 272.1 | 274.6 |
| MT-Bench | 184.0 | 179.9 | 184.7 | 192.4 | 200.0 |

Mean completion tokens per answer (reasoning + answer), share cut off at 4096 in parentheses:

| Set | Stock, ref | Swift, ref | Stock, daily | Swift, daily |
|---|---|---|---|---|
| GSM8K | 438 (0%) | 296 (0%), −32% | 371 (0%) | 368 (0%), −1% |
| MATH-500 | 1,052 (0%) | 864 (3%), −18% | 1,276 (6%) | 833 (0%), −35% |
| HumanEval | 2,042 (34%) | 1,454 (16%), −29% | 1,196 (3%) | 1,131 (0%), −5% |
| MBPP | 1,551 (19%) | 899 (3%), −42% | 977 (0%) | 895 (0%), −8% |
| MT-Bench | 1,475 (22%) | 1,163 (19%), −21% | 1,113 (3%) | 1,002 (3%), −10% |

Cut-off answers count as 4096 tokens, so where stock is cut off more often, the reduction is
understated.

## Thinking vs answer tokens

`completion_tokens` is thinking plus answer. `thinking_tokens.py <tokenizer.json>` splits it: the
saved reasoning text of each response, tokenized with the model's `tokenizer.json` (stock and Swift
share it), on the prompts where both models stopped on their own. Totals here are over that subset,
so they differ from the all-prompt means above.

| Set | Effort | n (both stopped) | Stock thinking | Swift thinking | Δ | Stock total | Swift total | Δ |
|---|---|---|---|---|---|---|---|---|
| GSM8K, all 1,319 | medium | 1319 of 1319 | 214 | 184 | -14% | 348 | 320 | -8% |
| SPEED-Bench qualitative | medium | 125 of 142 | 622 | 502 | -19% | 1,114 | 970 | -13% |
| gsm8k | xhigh | 32 of 32 | 312 | 167 | -47% | 438 | 296 | -32% |
| math500 | xhigh | 31 of 32 | 759 | 517 | -32% | 1,004 | 760 | -24% |
| humaneval | xhigh | 21 of 32 | 845 | 700 | -17% | 966 | 853 | -12% |
| mbpp | xhigh | 26 of 32 | 762 | 545 | -28% | 964 | 724 | -25% |
| mt-bench | xhigh | 24 of 32 | 416 | 243 | -42% | 624 | 391 | -37% |
| gsm8k | medium | 32 of 32 | 204 | 186 | -9% | 371 | 368 | -1% |
| math500 | medium | 30 of 32 | 768 | 478 | -38% | 1,088 | 776 | -29% |
| humaneval | medium | 31 of 32 | 685 | 652 | -5% | 1,102 | 1,079 | -2% |
| mbpp | medium | 32 of 32 | 342 | 306 | -10% | 977 | 895 | -8% |
| mt-bench | medium | 31 of 32 | 499 | 425 | -15% | 1,017 | 902 | -11% |

Swift at `xhigh` vs Swift at `medium`, all-prompt mean completion tokens on the card's sets (table
above): GSM8K 296 vs 368, MATH-500 864 vs 833, MBPP 899 vs 895, MT-Bench 1,163 vs 1,002, HumanEval
1,454 vs 1,131. Stock: 438 vs 371, 1,052 vs 1,276, 1,551 vs 977, 1,475 vs 1,113, 2,042 vs 1,196.
`xhigh` answers are capped at 4096 (up to 34% cut off), and no `xhigh` accuracy was measured.

## SPEED-Bench (`dflash8`, `swift-dflash8`, daily, concurrency 1)

| Split | Stock accept | Swift accept | Stock tok/s | Swift tok/s |
|---|---|---|---|---|
| qualitative (110 rows, 142 turns, max 4096) | 4.35 ± 0.10 | 4.36 ± 0.10 | 203.3 | 203.5 |
| throughput 1k (12, max 512) | 4.31 ± 0.29 | 4.14 ± 0.29 | 198.8 | 190.8 |
| throughput 2k | 4.11 ± 0.21 | 4.30 ± 0.21 | 187.1 | 195.6 |
| throughput 8k | 4.15 ± 0.30 | 4.09 ± 0.28 | 145.8 | 145.0 |
| throughput 16k | 3.91 ± 0.21 | 4.06 ± 0.28 | 109.5 | 110.7 |
| throughput 32k | 4.05 ± 0.33 | 4.14 ± 0.37 | 70.5 | 70.8 |

Qualitative mean completion tokens: 1,455 (stock) vs 1,308 (Swift), −10%; 11% vs 9% cut off at
4096. Wall time 1,016 s vs 913 s. Per-domain accept length (stock / Swift): coding 5.67 / 5.80,
humanities 3.90 / 4.05, math 5.23 / 4.99, multilingual 4.79 / 4.65, qa 3.71 / 3.69, rag 5.66 /
5.64, reasoning 4.09 / 4.18, roleplay 2.95 / 3.03, stem 4.58 / 4.64, summarization 4.64 / 4.42,
writing 3.80 / 3.78.

## Serving (`dflash8`, `swift-dflash8`, fixed output length)

| Scenario | Stock tok/s | Swift tok/s | Stock accept | Swift accept |
|---|---|---|---|---|
| chat, 1 user | 177.8 | 170.4 | 3.44 | 3.27 |
| chat, 4 users | 591.2 | 593.7 | 3.48 | 3.44 |
| 20k, 1 user | 98.4 | 99.2 | 3.54 | 3.57 |
| 20k, 4 users | 138.7 | 139.5 | 3.30 | 3.35 |
| 40k, 1 user | 58.6 | 60.8 | 3.27 | 3.57 |
| 80k, 1 user | 28.5 | 28.7 | 3.43 | 3.57 |
| 120k, 1 user | 16.5 | 16.5 | 3.46 | 3.55 |

## Configs

- Image: `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` for all four.
- Checkpoints: [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) @
  `319f741cce68d7914884900c138a1fbb70a42f30`;
  [ukisai/Swift-1.5-Qwen3.8-27b-NVFP4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4)
  @ `25482027debd5485e8108897ba9fed8d3ba16595` (ukisai's own NVFP4, ModelOpt, no KV scales).
- Drafter: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) @
  `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`, 8 draft tokens, on every config.
- Flags: [../scripts/args/qwen38-27b.sh](../scripts/args/qwen38-27b.sh) with
  `--mem-fraction-static 0.41`, the same as the earlier posts. KV pool 236,864 tokens on all four
  (`server.log`); GPU memory after load 43.8 GB on all four (`resources.json`).
- Steps and protocols: `../scripts/{gsm8k,bfcl,accept-sets,speed-bench,bench-serving}.sh` headers;
  sizes in `run.sh`. GSM8K daily: sgl-eval prompt and grader, the model's own sampling,
  max 32,768 tokens, 32 in flight. BFCL: `bfcl-eval` 2026.3.23, `Qwen/Qwen3-32B-FC` handler, the
  fixed 205-case subset ([../scripts/bfcl-test-ids.json](../scripts/bfcl-test-ids.json)).

## Caveats

- One run per config. Accept-sets have 32 prompts per set and SPEED-Bench 10–26 rows per domain;
  gaps under ~2 standard errors are noise. Every Swift-minus-stock accept gap on the five sets is
  under 2 SE of the gap.
- Token counts include reasoning and answer (`completion_tokens`). The Swift card's claims are
  about thinking tokens only, on other benchmarks, BF16 on vLLM.
- Swift's card at `medium`: 82.22% vs base 84.14%, 24.8% fewer thinking tokens
  ([card](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b) @ `b4c84d42`, "Efficiency across
  reasoning efforts", checked 2026-10-30). The table doesn't name the benchmark; its `xhigh` row
  equals the card's GPQA-Diamond row. Not measured here.
- The card's accept lengths are on one H200; its evaluation section doesn't state the target
  precision (its launch example serves BF16 `Qwen/Qwen3.8-27B`). Ours are NVFP4 on an RTX PRO 6000
  at 400 W, 32 prompts per set.
- The runs went through a suite runner that isn't in this repo; `suite.json` in each result dir is
  its record (step status, settings, image). `run.sh` calls the same step scripts with the same
  sizes and protocols. The harness commit in `suite.json` is from the private working repo.
- Serving uses fixed output lengths (`ignore_eos`) and SGLang's `random` dataset for 20k+ prompts
  (one ShareGPT prompt repeated); tok/s there is speed only, and the accept length on those rows
  is not representative.
