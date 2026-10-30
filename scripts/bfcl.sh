#!/bin/bash
# BFCL v4 subset (scripts/bfcl-test-ids.json, 205 cases in 10 categories) against the running
# model server ($ENGINE), repeated BFCL_REPEATS times (default 3). Each repeat is a full
# generate + evaluate with its own BFCL project root: <out-dir>/bfcl/rep<i>/{result,score}.
# Then <out-dir>/bfcl/summary.json (suite v1 contract): per-category and overall accuracy as
# fractions, mean over repeats, and two standard errors: "binomial" sqrt(p(1-p)/205) and
# "repeat" (sample std of the repeat accuracies / sqrt(repeats), only with 2+ repeats).
# Older results with a single run at <dir>/bfcl/{result,score} are still read by compare.py.
# Protocol: daily only (no published reference). Settings recorded in summary.json: BFCL's
# default temperature 0.001 (--temperature is not passed), thinking on, default effort, so
# repeats differ only by server nondeterminism and expose little sampling noise.
# Handler: Qwen/Qwen3-32B-FC (BFCL's own Qwen3 FC prompt, thinking on, default effort).
# BFCL loads its tokenizer from /models/$TOKENIZER_DIR (default: MODEL_DIR, the served
# checkpoint; set it for GGUF checkpoints, which have no Hugging Face tokenizer files).
#
# Usage: scripts/bfcl.sh <out-dir>
# Env:   BFCL_REPEATS  number of full runs (default 3)
#        BFCL_LIMIT    run only the first N of the 205 case ids (smoke runs; the summary's n shrinks)
#        TOKENIZER_DIR see above
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
tokenizer="${TOKENIZER_DIR:-${MODEL_DIR:?set MODEL_DIR}}"
model="Qwen/Qwen3-32B-FC"
cats="simple_python,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_irrelevance,multi_turn_base,multi_turn_miss_param"

repeats="${BFCL_REPEATS:-3}"
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "BFCL_REPEATS must be a positive integer, got: $repeats" >&2; exit 2; }

mkdir -p "$out/bfcl"
out="$(cd "$out" && pwd)"

name="bench-bfcl-$$"
rep=1
client() {
    dc --profile client run --rm -T --name "$name" -v "$out:/out" -e "BFCL_PROJECT_ROOT=/out/bfcl/rep$rep" bfcl "$@"
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

for (( rep = 1; rep <= repeats; rep++ )); do
    echo "=== bfcl repeat $rep/$repeats"
    rm -rf "$out/bfcl/rep$rep"
    mkdir -p "$out/bfcl/rep$rep"
    python3 - "$scripts_dir/bfcl-test-ids.json" "$out/bfcl/rep$rep/test_case_ids_to_generate.json" "${BFCL_LIMIT:-}" <<'PY'
import json, sys
src, dest, limit = sys.argv[1:]
ids = json.load(open(src))
if limit:  # first N case ids overall, in file order
    left, kept = int(limit), {}
    for k, v in ids.items():
        if left > 0:
            kept[k] = v[:left]
            left -= len(kept[k])
    ids = kept
json.dump(ids, open(dest, "w"), indent=1)
PY
    start=$(date +%s)
    guarded bfcl generate --model "$model" --run-ids --skip-server-setup \
        --local-model-path "/models/${tokenizer}" --num-threads 4 --allow-overwrite
    echo "generate wall: $(( $(date +%s) - start ))s"
    client bfcl evaluate --model "$model" --test-category "$cats" --partial-eval
done

python3 "$scripts_dir/compare.py" --bfcl-summary "$out/bfcl" --repeats "$repeats" \
    --limit "${BFCL_LIMIT:-0}" --model "$model" --temperature 0.001
