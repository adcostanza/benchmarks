# Benchmarks

Scripts and raw results behind the benchmark posts on [acostanza.com](https://acostanza.com/blog/). Each benchmark has its own folder with a `run.sh` that reproduces it from the shared scripts.

| Benchmark | Post | Date | Hardware |
|---|---|---|---|
| [qwen38-27b-nvfp4-vs-fp8](qwen38-27b-nvfp4-vs-fp8/) | [NVFP4 is 32–37% faster than FP8 on Qwen3.8-27B](https://acostanza.com/blog/qwen38-27b-nvfp4-vs-fp8/) | 2026-10-28 | [RTX PRO 6000 Blackwell, 400 W](hardware.md) |
| [qwen38-27b-spec-decode](qwen38-27b-spec-decode/) | [DFlash2 is the fastest drafter for Qwen3.8-27B, up to 120k tokens](https://acostanza.com/blog/qwen38-27b-spec-decode/) | 2026-10-29 | [RTX PRO 6000 Blackwell, 400 W](hardware.md) |
| [qwen38-flash-next-vs-27b](qwen38-flash-next-vs-27b/) | [Qwen3.8-Flash-Next only beats the 27B past 40k tokens on one RTX PRO 6000](https://acostanza.com/blog/qwen38-flash-next-vs-27b/) | 2026-10-29 | [RTX PRO 6000 Blackwell, 400 W](hardware.md) |
| [qwen38-27b-swift-vs-stock](qwen38-27b-swift-vs-stock/) | [Swift 1.5 NVFP4 matches stock Qwen3.8-27B and thinks 14–19% less at medium](https://acostanza.com/blog/qwen38-27b-swift-vs-stock/) | 2026-10-30 | [RTX PRO 6000 Blackwell, 400 W](hardware.md) |

## Layout

- `compose/sglang.yaml`: the model servers (SGLang, or llama.cpp with `ENGINE=llamacpp`), the BFCL client and a benchmark client for non-SGLang servers, configured by env vars.
- `compose/cdi.yaml`: override for rootless Docker (GPU through CDI). Enable with `BENCH_CDI=1`.
- `compose/clients/*.yaml`: client services of the benchmark steps; loaded on top of `sglang.yaml`.
- `scripts/serve.sh up|down`: start the server and wait for `/health`, or stop it. With `LOG_DIR` set, `up` writes `resources.json` (startup seconds, checkpoint size, page-cache fraction of the checkpoint before start, memory after load) and starts `scripts/monitor.sh` (GPU, container and host memory and host memory pressure every 2 s into `resources.csv`); `down` adds the peaks and saves `server.log`.
- `scripts/args/<model>.sh`: model-specific server flags, sourced by `run.sh` to build `SERVER_ARGS`.
- `scripts/bench-serving.sh <out-dir>`: serving speed with `sglang.benchmark.serving` (chat and 20k-token prompts, 1 and 4 users; `SCENARIOS` selects others, up to 120k tokens). Fails as soon as a scenario's client fails, any request fails, or the server stops answering `/health`; finished scenarios are kept.
- `scripts/bfcl.sh <out-dir>`: tool-calling accuracy on a fixed 205-case [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) subset (`scripts/bfcl-test-ids.json`). Starts only on a healthy server and stops when the server stops answering.
- `scripts/speed-bench.sh <out-dir>`: [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench) (NVIDIA): accept length per domain and throughput on real prompts. Prompts resolved once with `scripts/speed/prepare.sh`.
- `scripts/accept-sets.sh <out-dir>`: accept length on the five sets of the [DFlash2 card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (GSM8K, MATH-500, HumanEval, MBPP, MT-Bench), with the card's protocol (`ref`) and at `medium` (`daily`).
- `scripts/gsm8k.sh <out-dir>`: the full GSM8K test split with sgl-eval's prompt and grader.
- `scripts/compare.py <dir-a> <dir-b> ...`: side-by-side tables of serving and BFCL results (BFCL with ± standard error and a "within 2 SE" mark), plus one row per metric of every step's `summary.json`.
- Each step script's header lists its env vars, protocol and reference; each writes `<out-dir>/<step>/summary.json`.
- `<benchmark>/run.sh`: the exact configs for one post. Results go to `<benchmark>/results/<label>/`, or to `$OUT/<label>/`.

## Running

Needs Docker with the NVIDIA Container Toolkit and the checkpoints under one directory:

```sh
export MODELS_DIR=/path/to/models
export BENCH_CDI=1   # rootless Docker only
qwen38-27b-nvfp4-vs-fp8/run.sh               # every config
qwen38-27b-nvfp4-vs-fp8/run.sh stock-nvfp4   # one config
```

The scripts run the benchmark clients inside the containers, so nothing listens on a host port.

Env vars read by the scripts:

| Var | Meaning | Default |
|---|---|---|
| `MODELS_DIR` | host dir with the checkpoints, mounted at `/models` | required |
| `MODEL_DIR` | checkpoint dir under `MODELS_DIR` (set by `run.sh`) | required |
| `SERVER_ARGS` | model-specific server flags incl. `--mem-fraction-static` (set by `run.sh`) | required |
| `SPEC_ARGS` | extra flags appended after `SERVER_ARGS`, e.g. speculative decoding (set by `run.sh`) | none |
| `SHM_SIZE` | container shared memory | 72gb |
| `LOG_DIR` | where `serve.sh` writes `resources.json`, `resources.csv` and `server.log` (set by `run.sh`) | unset |
| `SCENARIOS` | space-separated `bench-serving.sh` scenarios (`sharegpt_c1 sharegpt_c4 long20k_c1 long20k_c4 long40k_c1 long80k_c1 long120k_c1`) | first four |
| `DATASETS_DIR` | host dir with the datasets of `speed-bench.sh`, `accept-sets.sh` and `gsm8k.sh`, mounted read-only at `/datasets` in the clients | required by those steps |
| `PROTOCOL` | `ref`, `daily` or `both` for steps with a published reference (set by `run.sh`) | per script |
| `<STEP>_LIMIT` | first N items of a step, for smoke runs (`GSM8K_LIMIT`, `ACCEPT_LIMIT`, `SPEEDBENCH_LIMIT`, `BFCL_LIMIT`) | none |
| `HF_TOKEN` | only for `scripts/speed/prepare.sh` (SPEED-Bench prompts from the gated `cais/hle`) | none |
| `EXTRA_BODY` | request extra-body JSON for `bench-serving.sh` | `reasoning_effort` medium |
| `ENGINE` | server the scripts drive: `sglang` or `llamacpp` (set by `run.sh`) | `sglang` |
| `TOKENIZER_DIR` | tokenizer dir under `MODELS_DIR` for llama.cpp runs (set by `run.sh`) | unset |
| `LLAMACPP_IMAGE` | llama.cpp server image | pinned `ghcr.io/ggml-org/llama.cpp` digest |
| `SERVER_ENV_FILE` | env file for the server container | none |
| `SGLANG_IMAGE` | SGLang server image, also the benchmark client image | `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` |
| `OUT` | results root | `<benchmark>/results` |
| `BENCH_PROJECT` | compose project name | `bench` |
| `BENCH_CDI` | `1` to request the GPU through CDI | `0` |
| `SERVE_TIMEOUT` | seconds to wait for the server | 900 |

## Adding a benchmark

Make a folder named after the post, write a `run.sh` that sets `SERVER_ARGS` (via `scripts/args/`) and loops over `label checkpoint mem-fraction` configs like the existing one, and commit its `results/`. Each post links to a git tag with the post's slug, so later script changes don't alter what an older post points at.

## License

MIT, see [LICENSE](LICENSE).
