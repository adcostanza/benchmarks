#!/bin/bash
# Typed Decisions (LocalLLaMA/typed-decisions `all` test split: 400 cases, 2,000 decisions) on
# the running server's /v1/systemone and /v1/decisions: the chat model used as a classifier from its answer-label
# log-probs (SGLang decision models docs). Raw answers: <out-dir>/decisions/pass<k>.jsonl,
# metrics: <out-dir>/decisions/summary.json. Needs DATASETS_DIR with
# typed-decisions/all/test-00000-of-00001.parquet (revision in scripts/decisions/run.py).
#
# Scoring is the dataset's own scorer (bench/score_with_jobs.py): accuracy against the gold
# argmax, KL from gold, Brier, ECE; plus label mass, latency and pass-to-pass drift.
# Not a suite v1 step: it runs on its own (qwen38-27b-decisions/run.sh).
#
# Usage: scripts/decisions.sh <out-dir>
# Env: DECISIONS_LIMIT        first N cases only (smoke runs)
#      DECISIONS_CONCURRENCY  in-flight requests (default 1, so latency is per request)
#      DECISIONS_PASSES       passes over /v1/systemone (default 2; pass 2 is prefix-cached)
#      DECISIONS_NATIVE_PASSES passes over native /v1/decisions (default 1)
#      DECISIONS_SPLIT        test (default) or train (train answers fit scripts/decisions/calibrate.py)
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
: "${DATASETS_DIR:?set DATASETS_DIR}"

mkdir -p "$out/decisions"
out="$(cd "$out" && pwd)"

name="bench-decisions-$$"
healthy || { echo "decisions: server not answering /health" >&2; exit 1; }
dc --profile client run --rm -T --name "$name" -v "$out:/out" decisions python3 /app/run.py \
    --base-url "http://${server}:8000" --data "/datasets/typed-decisions/all/${DECISIONS_SPLIT:-test}-00000-of-00001.parquet" --split "${DECISIONS_SPLIT:-test}" \
    --out /out --limit "${DECISIONS_LIMIT:-0}" --concurrency "${DECISIONS_CONCURRENCY:-1}" \
    --passes "${DECISIONS_PASSES:-2}" --native-passes "${DECISIONS_NATIVE_PASSES:-1}" &
pid=$!
# Stop the client as soon as the server stops answering (its requests would otherwise hang).
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    if kill -0 "$pid" 2>/dev/null && ! healthy; then
        echo "decisions: server stopped answering /health, stopping the client" >&2
        docker rm -f "$name" >/dev/null 2>&1 || true
        wait "$pid" || true
        exit 1
    fi
done
wait "$pid"
