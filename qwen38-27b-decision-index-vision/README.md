# Qwen3.8-27B on the Decision Index vision board

Question: how does plain Qwen3.8-27B NVFP4, served by SGLang with `images`, score on the [Decision Index](https://github.com/apolinario/decision-index) public vision suite, next to the decision fine-tunes and stock references on the board? Same protocol and flags as the image-decisions runs in `../qwen38-27b-decisions/`. Swift 1.5 NVFP4 runs the same way.

Run 2026-11-02. Public half of the board's vision score only; nothing here was submitted to the board.

## Result

Public vision index (the board's public half; the private half is run by the maintainers), all 12,210 rows answered, no errors:

| Config | Public index |
|---|---:|
| Qwen3.8-27B NVFP4 (stock) | **66.09** |
| Swift 1.5 NVFP4 | **65.80** |

Per benchmark: skill = (accuracy − chance) / (1 − chance), ×100; `results/<label>/run/vision-scores.json` has accuracy, chance and counts. Board columns are `public` scores from `data/vision.json` of the Space [`multimodalart/jev-decision-index`](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/c130faf65f82e81d3bdefb01e758c712dc5c5199/data/vision.json) (revision `c130faf65f82e81d3bdefb01e758c712dc5c5199`, last modified 2026-10-29; file generated 2026-10-29T14:04:19Z, edition 0.4; saved as `board-vision.json`, fetched 2026-11-02). Table made by `board.py` (`board-table.md`); the post's table and chart come from `board_range.py` (the 20 tuned 27B-class entries: 26 to 28.5 B parameters, no active-parameter tag, autoregressive, not an inference technique; my grouping from the board file's `params`, `kind` and `technique`).

| Benchmark | stock | Swift | Qwen3.5-397B (stock, API) | Qwen3-VL-8B (stock) | SigLIP 2 | JEV-27B-VL | jev-vision-27b | VisionJEV 1.0.0 (27B) | Sieve-Vision-27B | RSI-Jev v6.1 VL 27B | WeMM-JEV-27B (top) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CV-Bench | 72.80 | 72.15 | 78.31 | 72.72 | -5.39 | 76.85 | 76.77 | 80.17 | 80.90 | 77.82 | 81.38 |
| BLINK | 58.60 | 57.15 | 50.54 | 51.35 | -0.85 | 63.27 | 53.93 | 74.06 | 68.43 | 53.28 | 74.22 |
| RealWorldQA | 63.93 | 60.05 | 67.82 | 56.16 | 5.39 | 72.25 | 72.81 | 74.20 | 75.58 | 65.60 | 74.20 |
| CharXiv | 64.91 | 68.92 | 62.16 | 58.90 | 4.01 | 79.70 | 78.70 | 76.44 | 72.18 | 79.45 | 81.70 |
| InfographicVQA | 89.62 | 89.07 | 95.08 | 84.70 | 7.65 | 96.72 | 97.27 | 98.91 | 97.81 | 93.99 | 98.36 |
| Mind2Web (font deviation) | 76.43 | 75.88 | 76.57 | 70.95 | 8.46 | 78.62 | 84.10 | 84.52 | 94.38 | 75.75 | 85.89 |
| Winoground | 78.25 | 73.25 | 82.25 | 71.75 | 33.50 | 84.50 | 83.50 | 85.75 | 80.25 | 82.25 | 87.00 |
| KIE (CORD+FUNSD) | 91.12 | 90.38 | 94.06 | 89.05 | -2.25 | 92.05 | 91.18 | 93.19 | 97.46 | 91.05 | 93.79 |
| Moderation (Hateful Memes) | 30.09 | 37.02 | 49.22 | 34.21 | -2.23 | 40.31 | 52.51 | 54.33 | 23.17 | 29.60 | 52.51 |
| R-Bench-M (weight 0.5) | 20.59 | 19.33 | 0.38 | 21.85 | 0.74 | 31.06 | 24.38 | 31.60 | 26.73 | 23.48 | 38.46 |
| MMMU-Pro vision (weight 0.25) | 33.24 | 31.91 | 47.43 | 27.87 | 2.68 | 40.13 | 38.01 | 44.64 | 39.40 | 35.96 | 46.23 |
| **Public index** | **66.09** | **65.80** | 68.52 | 62.33 | 6.16 | 72.80 | 73.07 | 76.77 | 73.17 | 68.67 | 77.93 |

Notes on the board rows:
- Qwen3.5-397B and Qwen3-VL-8B are the board's stock references ("read from its generated letter rather than letter probabilities" for the 397B). SigLIP 2 is the contrastive floor.
- The board's JEV-27B-VL row lists its own base, `Qwen/Qwen3.8-27B`, at public 68.54 / private 60.25 (BF16 run through the entrant's code, not this kit's engine; the board's caveat says the base was read through a different, head-free autojev path than the tuned entry). That is the closest stock-27B number on the board, but not an isolated tuning effect; this run's 66.09 is NVFP4 through SGLang. The two are not the same pipeline.
- Only the public half is comparable here. Board "full" scores need the maintainers' private half.
- Raw accuracies (stock / Swift): CV-Bench 83.5/83.1, BLINK 73.3/72.3, RealWorldQA 77.9/75.6, CharXiv 73.7/76.7, InfographicVQA 92.2/91.8, Mind2Web 82.3/81.9, Winoground 89.1/86.6, KIE 94.6/94.2, Hateful Memes 65.0/68.5, R-Bench-M 33.8/32.8, MMMU-Pro vision 41.8/40.6 (%).

## Latency vs the board rule

Rule: median, mean and p80 each ≤ 1,000 ms per request on one RTX PRO 6000, single process. Measured as the kit measures it (`total_wall_ms`: one request, including the kit's encode/resize, no server startup), `--workers 1`, over all 12,210 rows (`results/<label>/latency.txt`, made by `lat.py`).

| Config | median | mean | p80 | max |
|---|---:|---:|---:|---:|
| stock, all rows | 126 ms | 212 ms | 362 ms | 1,087 ms |
| stock, 1 image (11,636 rows) | 119 | 201 | 359 | 1,087 |
| stock, 2 images (574 rows) | 505 | 434 | 613 | 780 |
| Swift, all rows | 127 | 214 | 365 | 1,150 |

Verdict: passes with wide margin (all three within the 1,000 ms limit for each config; the highest per-benchmark p80 is BLINK's 566 ms (stock); the 574 two-image rows have a median of 505 ms). The suite has 1 or 2 images per row, never 3 or 4. The slowest benchmark medians are MMMU-Pro vision (430 ms), Winoground (362) and InfographicVQA (356). The maintainers measure on their own; the server runs with the flags below, including `--sleep-on-idle`.

## Run

- Wall time (server start to stop, incl. ~3 min load): stock 46 min 25 s, Swift 47 min 7 s; single worker, ~0.2 s/row. Smoke (99 rows spread over all 11 benchmarks, 9 each): clean, every row answered, p50 245 ms, p80 354 ms (`results/stock-nvfp4-smoke/`).
- Prefix-cache hit rate (`#cached-token` / (cached + new) over 12,608 prefill batches, `cache-hit.txt`): 13.9% for both configs (about 1.77 M of 12.7 M tokens). The smoke had 0%.
- Errors: none. `status: ok` on 12,210/12,210 rows for both configs; `vision-scores.json` has `complete: true`, `missing_rows: 0`.
- Reproduce: `run.sh` (needs the rebuilt suite, see Suite; `DI_SUITE=<dir>/vision`). Scoring is part of `vision pipeline`.

## Setup

- Server: `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` (digest `sha256:e03dd310dc72e64ac709ceb244a79f20fd5e05448c811ab33cc777d8040061ff`), flags from `../scripts/args/qwen38-27b.sh` with `--mem-fraction-static 0.41` (the same as the image-decisions runs). The launch line is in each `server.log` (`server_args=`). Thinking is not used on this route.
- Checkpoints: stock [`RadixArk/Qwen3.8-27B-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) @ `319f741cce68d7914884900c138a1fbb70a42f30`; Swift [`ukisai/Swift-1.5-Qwen3.8-27b-NVFP4`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-NVFP4) @ `25482027debd5485e8108897ba9fed8d3ba16595`.
- Kit: [apolinario/decision-index](https://github.com/apolinario/decision-index) @ `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` (2026-10-20), vision edition 0.3. Engine `decide`, `protocol=decisions`, `path=/v1/systemone`, `--workers 1`, `max_pixels` 1,638,400 (the kit's cap; images over it are downscaled, others sent as stored), images as data URLs, run in a container of the SGLang image with the kit mounted.
- Image resolution: the checkpoints' `preprocessor_config.json` has `longest_edge` 16,777,216 px (no cap below 1.6 MP), so the run is native resolution up to the kit's 1.6 MP cap. The largest image sent was 1,638,400 px (`effective_image_budget` in each `environment.json`; the endpoint returns no image-token counts).
- Harness: the release commit named in the repo README's Harness column and in the post; compose service `compose/clients/di-vision.yaml`.

## Suite

Built on the CPU into `<datasets>/decision-index-vision` (the suite directory is `DI_SUITE=<datasets>/decision-index-vision/vision`) with `decision-index suite rebuild --vision --work ... --font /usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf` (logs and the `vision verify` report in `suite/`; the dataset directory in those logs is shown as `<datasets>`). Winoground (`facebook/winoground`, gated) was accessible after the account accepted its terms; it is in the run.

## Declared deviations

1. **Endpoint.** The kit's `protocol=decisions` sends `{model, state, questions, images}`. This SGLang image's `/v1/decisions` answers HTTP 400 to that body (it wants `input` and a question list). The same body works on `/v1/systemone`, so the engine runs with `path=/v1/systemone`, the route the image-decisions vision runs used (images are an SGLang extension there). The response (letter probabilities) is what the kit parses. The first smoke attempt on `/v1/decisions` failed on all 9 rows before any scoring; nothing from it is in the results.
2. **Mind2Web font.** Arial Bold is not on this machine; the box letters are drawn in Liberation Sans Bold (metric-compatible with Arial; sha256 `3973aa5054fb467dd5627245d3dc82e37bf16fe075756156a570455871351582`). `vision verify` therefore flags the Mind2Web images and payloads (and only those): `suite/verify.json`. Mind2Web scores are on slightly different pixels than the board's; the kit's `vision run` refuses an unverified suite, so `run.sh` passes `--no-verify`.
3. **Pillow.** The first rebuild used Pillow 13.0.0, which the kit does not list (9.5.0 and 12.3.0 reproduce its digests); InfographicVQA's resized images then failed their digest. Rebuilt with 12.3.0: InfographicVQA verifies. Everything else except Mind2Web verifies (items, images, row counts, run ids, payload hashes).
4. **Readout.** The suite marks R-Bench-M and MMMU-Pro vision rows (665 + 1,730) with a `readout` (`The answer is (`), which the kit's vLLM reference engine continues as an assistant turn. The `decide` engine (`decision_index/vision/engines.py`) drops the whole `prompt` object, so those 2,395 rows ran without the readout, and MMMU-Pro's 1,730 rows also without `options_in_image` (which tells the reference vLLM engine not to repeat the image's options as text). I didn't inspect the resulting server-side prompt. Noticed during post review; the effect is unmeasured.
5. Prefix cache: see Caveats.

## Caveats

- **Prefix cache.** With these flags a warm prefix cache can change answers (`../qwen38-27b-decisions/README.md`, "Prefix cache check"). This run had a 13.9% hit rate with the cache on and rows in suite order, one worker; there was no cache-off rerun, so the effect on these scores is unmeasured. The board rule applies no cache setting.
- Single pass per config, no run-to-run seeds. `uncertainty.md` has a paired row bootstrap (rows treated as independent, so optimistic): stock − Swift +0.29 (−0.82 to +1.52). Per benchmark, Winoground (stock +5.0, interval +0.75 to +9.50) and Hateful Memes (Swift +6.9, −11.21 to −2.31 for stock − Swift) exclude zero; RealWorldQA (stock +3.9, −1.11 to +8.88) and CharXiv (Swift +4.0, −8.52 to +0.25) do not. Eleven comparisons, no correction.
- Answer keys are each source benchmark's own labels as converted by the kit (no model labels anything); I didn't check who annotated them, and scores are agreement with those keys.
- Letter probabilities from SGLang (NVFP4 + FP8 KV cache) versus the board's BF16 / own-engine rows: engine, quantization and precision all differ from many board rows. The board says its Qwen3.5-397B reference was read from its generated letter; it doesn't state the read-out for the other stock reference.
- The board's Qwen3.8-27B base (68.54, inside the JEV-27B-VL entry) is above this run's stock; not investigated which of engine, NVFP4, FP8 KV or resolution handling accounts for it.
- The board's private half is not run; no Full score exists for these runs.
- Nothing was published or submitted. A board submission would need a results dataset, the kit PR, and the maintainers' private run (kit `docs/vision.md`, "Submitting").

## Files

- `run.sh` — server + `vision pipeline` per config; `lat.py`, `cachehit.py`, `board.py`, `board_range.py` — latency, cache-hit, board tables and the post's table/chart. `board-vision.json` (the Space's `data/vision.json` at the pinned revision, sha256 `1f870dff…9b90`) is not included (the Space's file states no licence); fetch it from the pinned revision to rerun `board.py`/`board_range.py`.
- `analysis.py <suite dir>` → `uncertainty.md` and `results/<label>/rows-slim.jsonl` (see Per-row files).
- `results/<label>/run/` — `vision-scores.json`, `environment.json`, `status.json` (`results.jsonl` with per-row answers, payloads and timings stays internal); `server.log`, `resources.json`, `pipeline.log`, `latency.txt`, `cache-hit.txt`. `results/stock-nvfp4-smoke/` is the 99-row smoke.
- `environment.json` — machine, versions, digests; `suite/` — rebuild and verify output.

## Per-row files

`results/<label>/rows-slim.jsonl`: one line per row with `run_id`, `benchmark`, `status`, `total_wall_ms` and, for rows with more than two options (6,972 of 12,210), the chosen `choice` and `correct`. On two-option rows (subsets of CV-Bench, BLINK, RealWorldQA, KIE and MMMU-Pro, and all of Winoground and Hateful Memes) correctness would reveal the answer key, so those two fields are left out. `results.jsonl` holds the kit's request payloads (the benchmarks' question text and options), so it isn't published; `analysis.py` needs it and the rebuilt suite, so `uncertainty.md` can't be regenerated from the published files alone. The suite itself isn't redistributed (source licences).
