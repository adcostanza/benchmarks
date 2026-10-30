#!/bin/bash
# SPEED-Bench (nvidia/SPEED-Bench, NVIDIA Research, ICML 2026): accept length per domain and
# throughput on real prompts, against the running SGLang server. Needs no drafter for the
# throughput numbers; accept length is recorded only when the server runs speculative decoding
# (otherwise the metric is absent from summary.json, not 1.0).
#   qualitative      11 domains (coding, math, humanities, ...), 80 prompts each in the dataset,
#                    multi-turn rows run turn by turn with the model's own answers as history
#   throughput_*     1k/2k/8k/16k/32k-token prompts (low/mixed/high entropy, 512 per tier)
# Writes <out-dir>/speed-bench/{summary.json,requests_<split>_c<N>.jsonl} (one line per request,
# including the generated text). Per run: output tok/s (completion tokens / wall time), TTFT and
# TPOT p50, mean completion tokens, share of requests cut off at the token limit (truncated), accept length overall and per domain (per-request mean as in the DFlash2 card, with its
# standard error, plus the token-weighted ratio sum(tokens)/sum(verify steps)).
#
# Accept length: the client sets return_spec_tokens_details on /v1/chat/completions and reads
# sglext.spec_tokens_details.spec_accept_length (= completion tokens / verification steps, computed
# per request by SGLang) from the last streamed chunk. We use this client instead of NVIDIA's AIPerf:
# AIPerf reads per-request acceptance only from vLLM; for SGLang it scrapes the server-wide gauge
# sglang:spec_accept_length (a time average over the run, one run per category). Prompts are the
# same SPEED-Bench rows either way.
#
# Prompts: the released parquet masks ~half the rows. Run scripts/speed/prepare.sh once; it
# resolves them with NVIDIA's own resolver (pinned commit) into $DATASETS_DIR/SPEED-Bench-resolved.
# This script refuses rows that still hold the placeholder.
#
# Protocol: daily only (reasoning_effort=medium, the server's default sampling, max tokens 4096
# for qualitative and 512 for throughput). SPEED-Bench defines no reasoning-model reference
# beyond "follow the model card" and "max 4096 tokens", so there is no `ref` run; PROTOCOL=ref is
# rejected. Cases are a fixed deterministic subset (first N of each category in dataset order),
# the same at every concurrency; the prefix cache is flushed before each run. The AIPerf examples
# use concurrency 16 (qualitative) and 64 (throughput); set SPEEDBENCH_CONC to add them.
#
# Usage: scripts/speed-bench.sh <out-dir>   (ENGINE=sglang; server already up)
# Env: SPEEDBENCH_CONC           space-separated concurrencies (default: 1 4)
#      SPEEDBENCH_QUAL_PER_CAT   qualitative rows per domain (default 20 of 80)
#      SPEEDBENCH_TP_PER_TIER    throughput rows per entropy tier and bucket (default 8 of 512)
#      SPEEDBENCH_TP_BUCKETS     throughput buckets (default: 1k 2k 8k 16k 32k)
#      SPEEDBENCH_QUAL_MAX_TOKENS / SPEEDBENCH_TP_MAX_TOKENS   (default 4096 / 512)
#      SPEEDBENCH_LIMIT          run only the first N rows of every run, for smoke runs
#      PROTOCOL                  daily (default) or both; ref is not defined for this benchmark
#      MODEL_DIR                 checkpoint dir, used as the request's model name
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
[[ "$server" == sglang ]] || { echo "speed-bench: needs ENGINE=sglang" >&2; exit 2; }
case "${PROTOCOL:-daily}" in
daily|both) ;;
*) echo "speed-bench: PROTOCOL=${PROTOCOL} not defined for this benchmark (daily only)" >&2; exit 2 ;;
esac

mkdir -p "$out/speed-bench"
out="$(cd "$out" && pwd)"
rm -f "$out"/speed-bench/requests_*.jsonl "$out/speed-bench/summary.json"

name="bench-speed-$$"
envs=(-e MODEL_DIR -e "SERVER=$server")
for v in $(compgen -v SPEEDBENCH_); do envs+=(-e "$v"); done
client() {
    dc --profile client run --rm -T --name "$name" -v "$out:/out" -v "$scripts_dir:/scripts:ro" \
        "${envs[@]}" bench-client python3 /scripts/speed/speed_bench.py /out/speed-bench
}

# The client fails on the first failed request; this also stops it when the server stops
# answering /health while requests hang.
healthy || { echo "speed-bench: server not answering /health" >&2; exit 1; }
client &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    if kill -0 "$pid" 2>/dev/null && ! healthy; then
        echo "speed-bench: server stopped answering /health, stopping the client" >&2
        docker rm -f "$name" >/dev/null 2>&1 || true
        wait "$pid" || true
        exit 1
    fi
done
wait "$pid"
[[ -s "$out/speed-bench/summary.json" ]] || { echo "speed-bench: no summary.json written" >&2; exit 1; }
