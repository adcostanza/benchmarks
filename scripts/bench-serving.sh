#!/bin/bash
# Serving speed of whatever model the sglang container is running.
# Runs sglang.benchmark.serving inside the container at reasoning_effort=medium and
# writes one JSON line per scenario to <out-dir>/bench-serving.jsonl.
#
# Usage: scripts/bench-serving.sh <out-dir>
# Env: SCENARIOS  space-separated scenario names (default: sharegpt_c1 sharegpt_c4 long20k_c1 long20k_c4)
#                 also available: long40k_c1 long80k_c1 long120k_c1
#      EXTRA_BODY request extra-body JSON (default: reasoning_effort medium)
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
mkdir -p "$out"

default_extra='{"chat_template_kwargs":{"reasoning_effort":"medium"}}'
extra="${EXTRA_BODY:-$default_extra}"
scenarios="${SCENARIOS:-sharegpt_c1 sharegpt_c4 long20k_c1 long20k_c4}"
common=(--backend sglang-oai-chat --base-url http://localhost:8000 --flush-cache
        --warmup-requests 2 --extra-request-body "$extra" --seed 1
        --output-file /tmp/bench-serving.jsonl)

run() {
    local name="$1"; shift
    [[ " $scenarios " == *" $name "* ]] || return 0
    echo "=== ${name}"
    dc exec -T sglang python3 -m sglang.benchmark.serving "${common[@]}" --tag "$name" "$@" 2>&1 \
        | grep -E '^(Max request concurrency|Successful requests|Benchmark duration|Output token throughput|Median TTFT|Median TPOT|Accept length)' || true
}

dc exec -T sglang rm -f /tmp/bench-serving.jsonl
# Short chat prompts (ShareGPT), outputs capped at 1024 tokens.
run sharegpt_c1 --dataset-name sharegpt --sharegpt-output-len 1024 --num-prompts 12 --max-concurrency 1
run sharegpt_c4 --dataset-name sharegpt --sharegpt-output-len 1024 --num-prompts 32 --max-concurrency 4
# Agent-sized context: 20k-token prompt, 512-token answer.
run long20k_c1 --dataset-name random --random-input-len 20000 --random-output-len 512 --random-range-ratio 1.0 --num-prompts 4 --max-concurrency 1
run long20k_c4 --dataset-name random --random-input-len 20000 --random-output-len 512 --random-range-ratio 1.0 --num-prompts 8 --max-concurrency 4
# Longer agent contexts, 512-token answer.
for k in 40 80 120; do
    run "long${k}k_c1" --dataset-name random --random-input-len "${k}000" --random-output-len 512 --random-range-ratio 1.0 --num-prompts 4 --max-concurrency 1
done
dc cp sglang:/tmp/bench-serving.jsonl "$out/bench-serving.jsonl"
echo "results: $out/bench-serving.jsonl"
