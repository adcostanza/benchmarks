#!/bin/bash
# BFCL v4 subset (scripts/bfcl-test-ids.json, 205 cases in 10 categories) against the running
# model server ($ENGINE). Results land in <out-dir>/bfcl/{result,score}.
# Handler: Qwen/Qwen3-32B-FC (BFCL's own Qwen3 FC prompt, thinking on, default effort).
# BFCL loads its tokenizer from /models/$TOKENIZER_DIR (default: MODEL_DIR, the served
# checkpoint; set it for GGUF checkpoints, which have no Hugging Face tokenizer files).
#
# Usage: scripts/bfcl.sh <out-dir>
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
tokenizer="${TOKENIZER_DIR:-${MODEL_DIR:?set MODEL_DIR}}"
model="Qwen/Qwen3-32B-FC"
cats="simple_python,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_irrelevance,multi_turn_base,multi_turn_miss_param"

mkdir -p "$out/bfcl"
out="$(cd "$out" && pwd)"
cp "$scripts_dir/bfcl-test-ids.json" "$out/bfcl/test_case_ids_to_generate.json"

name="bench-bfcl-$$"
client() {
    dc --profile client run --rm -T --name "$name" -v "$out:/out" -e BFCL_PROJECT_ROOT=/out/bfcl bfcl "$@"
}

# BFCL retries a dead server for a long time, so it only starts on a healthy server and is
# stopped as soon as the server stops answering.
guarded() {
    healthy || { echo "bfcl: server not answering /health" >&2; exit 1; }
    client "$@" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        sleep 15
        if kill -0 "$pid" 2>/dev/null && ! healthy; then
            echo "bfcl: server stopped answering /health, stopping the client" >&2
            docker rm -f "$name" >/dev/null 2>&1 || true
            wait "$pid" || true
            exit 1
        fi
    done
    wait "$pid"
}

start=$(date +%s)
guarded bfcl generate --model "$model" --run-ids --skip-server-setup \
    --local-model-path "/models/${tokenizer}" --num-threads 4 --allow-overwrite
echo "generate wall: $(( $(date +%s) - start ))s"
client bfcl evaluate --model "$model" --test-category "$cats" --partial-eval
