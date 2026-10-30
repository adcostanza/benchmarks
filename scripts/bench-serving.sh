#!/bin/bash
# Serving speed of whatever model the server container ($ENGINE) is running.
# Runs sglang.benchmark.serving at reasoning_effort=medium and writes one JSON line per scenario
# to <out-dir>/bench-serving.jsonl.
#   sglang    client runs inside the server container; prefix cache flushed after warmup.
#   llamacpp  client runs in the bench-client container (SGLang image, $SGLANG_IMAGE) as an
#             OpenAI chat client (`vllm-chat`, same request code as `sglang-oai-chat`);
#             llama.cpp has no cache-flush endpoint, so every request sends cache_prompt=false;
#             it only streams token usage when asked (stream_options.include_usage).
#             Tokenizer from /models/$TOKENIZER_DIR (GGUF dirs have none).
#
# Usage: scripts/bench-serving.sh <out-dir>
# Env: SCENARIOS  space-separated scenario names (default: sharegpt_c1 sharegpt_c4 long20k_c1 long20k_c4)
#                 also available: long40k_c1 long80k_c1 long120k_c1
#      EXTRA_BODY request extra-body JSON (default: reasoning_effort medium)
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

default_extra='{"chat_template_kwargs":{"reasoning_effort":"medium"}}'
extra="${EXTRA_BODY:-$default_extra}"
scenarios="${SCENARIOS:-sharegpt_c1 sharegpt_c4 long20k_c1 long20k_c4}"
common=(--warmup-requests 2 --seed 1)
case "$server" in
sglang)
    client=(dc exec -T sglang python3 -m sglang.benchmark.serving)
    common+=(--backend sglang-oai-chat --base-url http://localhost:8000 --flush-cache
             --output-file /tmp/bench-serving.jsonl)
    ;;
llamacpp)
    tokenizer="/models/${TOKENIZER_DIR:?set TOKENIZER_DIR}"
    extra="$(python3 -c 'import json, sys; d = json.loads(sys.argv[1]); d["cache_prompt"] = False; d["stream_options"] = {"include_usage": True}; print(json.dumps(d))' "$extra")"
    client=(dc --profile client run --rm -T -v "$out:/out" bench-client python3 -m sglang.benchmark.serving)
    common+=(--backend vllm-chat --base-url http://llamacpp:8000 --model "$tokenizer"
             --tokenizer "$tokenizer" --output-file /out/bench-serving.jsonl)
    ;;
esac
common+=(--extra-request-body "$extra")

# Copies the server's result file out. `exec cat`, not `docker cp`: rootless Docker's cp
# remounts the container's read-only /models bind mount and fails with EPERM when the host
# mount is nosuid/nodev ("remount-ro ... operation not permitted").
save_results() {
    [[ "$server" == sglang ]] || return 0
    dc exec -T sglang cat /tmp/bench-serving.jsonl >"$out/.bench-serving.jsonl"
    mv "$out/.bench-serving.jsonl" "$out/bench-serving.jsonl"
}

# A scenario fails the script when the client exits non-zero, any request fails, or the server
# stops answering /health afterwards. Full client output goes to <out-dir>/bench-serving.log.
run() {
    local name="$1"; shift
    [[ " $scenarios " == *" $name "* ]] || return 0
    echo "=== ${name}"
    local args=("$@") want="" i rc=0
    for (( i = 0; i < ${#args[@]}; i++ )); do
        [[ "${args[i]}" == --num-prompts ]] && want="${args[i+1]}"
    done
    "${client[@]}" "${common[@]}" --tag "$name" "$@" \
        >"$out/.scenario.log" 2>&1 || rc=$?
    cat "$out/.scenario.log" >>"$out/bench-serving.log"
    grep -E '^(Max request concurrency|Successful requests|Benchmark duration|Output token throughput|Median TTFT|Median TPOT|Accept length)' \
        "$out/.scenario.log" || true
    local got
    got="$(sed -nE 's/^Successful requests:[[:space:]]+([0-9]+).*/\1/p' "$out/.scenario.log")"
    rm -f "$out/.scenario.log"
    if (( rc != 0 )) || [[ "$got" != "$want" ]]; then
        echo "scenario ${name} failed: exit ${rc}, ${got:-0}/${want} requests succeeded" >&2
        exit 1
    fi
    if ! healthy; then
        echo "scenario ${name}: server stopped answering /health" >&2
        exit 1
    fi
    # Keep finished scenarios even if a later one fails.
    save_results
}

if [[ "$server" == sglang ]]; then dc exec -T sglang rm -f /tmp/bench-serving.jsonl; fi
rm -f "$out/bench-serving.log" "$out/bench-serving.jsonl"
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
save_results
echo "results: $out/bench-serving.jsonl"
