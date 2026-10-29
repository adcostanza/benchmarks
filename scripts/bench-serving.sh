#!/bin/bash
# Serving speed of whatever model the sglang container is running.
# Runs sglang.benchmark.serving inside the container at reasoning_effort=medium and
# writes one JSON line per scenario to <out-dir>/bench-serving.jsonl.
#
# Usage: scripts/bench-serving.sh <out-dir>
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
mkdir -p "$out"

extra='{"chat_template_kwargs":{"reasoning_effort":"medium"}}'
common=(--backend sglang-oai-chat --base-url http://localhost:8000 --flush-cache
        --warmup-requests 2 --extra-request-body "$extra" --seed 1
        --output-file /tmp/bench-serving.jsonl)

run() {
    local name="$1"; shift
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
dc cp sglang:/tmp/bench-serving.jsonl "$out/bench-serving.jsonl"
echo "results: $out/bench-serving.jsonl"
