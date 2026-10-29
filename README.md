# Benchmarks

Scripts and raw results behind the benchmark posts on [acostanza.com](https://acostanza.com/blog/). Each benchmark has its own folder with a `run.sh` that reproduces it from the shared scripts.

| Benchmark | Post | Date | Hardware |
|---|---|---|---|
| [qwen38-27b-nvfp4-vs-fp8](qwen38-27b-nvfp4-vs-fp8/) | [NVFP4 is 32–37% faster than FP8 on Qwen3.8-27B](https://acostanza.com/blog/qwen38-27b-nvfp4-vs-fp8/) | 2026-10-28 | [RTX PRO 6000 Blackwell, 400 W](hardware.md) |
| [qwen38-27b-spec-decode](qwen38-27b-spec-decode/) | [DFlash2 is the fastest drafter for Qwen3.8-27B, up to 120k tokens](https://acostanza.com/blog/qwen38-27b-spec-decode/) | 2026-10-29 | [RTX PRO 6000 Blackwell, 400 W](hardware.md) |

## Layout

- `compose/sglang.yaml`: the SGLang server and the BFCL client, configured by env vars.
- `compose/cdi.yaml`: override for rootless Docker (GPU through CDI). Enable with `BENCH_CDI=1`.
- `scripts/serve.sh up|down`: start the server and wait for `/health`, or stop it (saving `server.log` to `$LOG_DIR` first when set).
- `scripts/args/<model>.sh`: model-specific server flags, sourced by `run.sh` to build `SERVER_ARGS`.
- `scripts/bench-serving.sh <out-dir>`: serving speed with `sglang.benchmark.serving` (chat and 20k-token prompts, 1 and 4 users; `SCENARIOS` selects others, up to 120k tokens).
- `scripts/bfcl.sh <out-dir>`: tool-calling accuracy on a fixed 205-case [BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) subset (`scripts/bfcl-test-ids.json`).
- `scripts/compare.py <dir-a> <dir-b> ...`: side-by-side tables of serving and BFCL results.
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
| `LOG_DIR` | where `serve.sh down` saves `server.log` (set by `run.sh`) | unset |
| `SCENARIOS` | space-separated `bench-serving.sh` scenarios (`sharegpt_c1 sharegpt_c4 long20k_c1 long20k_c4 long40k_c1 long80k_c1 long120k_c1`) | first four |
| `EXTRA_BODY` | request extra-body JSON for `bench-serving.sh` | `reasoning_effort` medium |
| `SGLANG_IMAGE` | server image | `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48` |
| `OUT` | results root | `<benchmark>/results` |
| `BENCH_PROJECT` | compose project name | `bench` |
| `BENCH_CDI` | `1` to request the GPU through CDI | `0` |
| `SERVE_TIMEOUT` | seconds to wait for the server | 900 |

## Adding a benchmark

Make a folder named after the post, write a `run.sh` that sets `SERVER_ARGS` (via `scripts/args/`) and loops over `label checkpoint mem-fraction` configs like the existing one, and commit its `results/`. Each post links to a git tag with the post's slug, so later script changes don't alter what an older post points at.

## License

MIT, see [LICENSE](LICENSE).
