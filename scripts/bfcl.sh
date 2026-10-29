#!/bin/bash
# BFCL v4 subset (scripts/bfcl-test-ids.json, 205 cases in 10 categories) against the running
# sglang server. Results land in <out-dir>/bfcl/{result,score}.
# Handler: Qwen/Qwen3-32B-FC (BFCL's own Qwen3 FC prompt, thinking on, default effort).
# MODEL_DIR must name the served checkpoint; BFCL loads its tokenizer from /models/$MODEL_DIR.
#
# Usage: scripts/bfcl.sh <out-dir>
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
: "${MODEL_DIR:?set MODEL_DIR}"
model="Qwen/Qwen3-32B-FC"
cats="simple_python,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_irrelevance,multi_turn_base,multi_turn_miss_param"

mkdir -p "$out/bfcl"
out="$(cd "$out" && pwd)"
cp "$scripts_dir/bfcl-test-ids.json" "$out/bfcl/test_case_ids_to_generate.json"

client() {
    dc --profile client run --rm -T -v "$out:/out" -e BFCL_PROJECT_ROOT=/out/bfcl bfcl "$@"
}

start=$(date +%s)
client bfcl generate --model "$model" --run-ids --skip-server-setup \
    --local-model-path "/models/${MODEL_DIR}" --num-threads 4 --allow-overwrite
echo "generate wall: $(( $(date +%s) - start ))s"
client bfcl evaluate --model "$model" --test-category "$cats" --partial-eval
