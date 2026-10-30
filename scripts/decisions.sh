#!/bin/bash
# A decision dataset on the running server's /v1/systemone and /v1/decisions: the chat model used
# as a classifier from its answer-label log-probs (SGLang decision models docs). Sets, their
# files under DATASETS_DIR, revisions and scorers: scripts/decisions/sets.py. Default: Typed
# Decisions (LocalLLaMA/typed-decisions `all` test split: 400 cases, 2,000 decisions).
#
# Raw answers: <out-dir>/<sub>/pass<k>.jsonl, metrics: <out-dir>/<sub>/summary.json, where <sub>
# is `decisions` for Typed Decisions and `<set>-<split>` for the others.
# Not a suite v1 step: it runs on its own (qwen38-27b-decisions/run.sh).
#
# Usage: scripts/decisions.sh <out-dir>
# Env: DECISIONS_SET          typed-decisions (default), typed-decision-bench,
#                             procedural-typed-decisions, calibrated-typed-decisions
#      DECISIONS_SPLIT        test (default); the set's other splits fit scripts/decisions/calibrate.py
#      DECISIONS_LIMIT        first N cases only (smoke runs)
#      DECISIONS_CONCURRENCY  in-flight requests (default 1, so latency is per request)
#      DECISIONS_PASSES       passes over /v1/systemone (default 2; pass 2 is prefix-cached)
#      DECISIONS_NATIVE_PASSES passes over native /v1/decisions (default 1)
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
: "${DATASETS_DIR:?set DATASETS_DIR}"
set_name="${DECISIONS_SET:-typed-decisions}"
split="${DECISIONS_SPLIT:-test}"
sub=decisions
[[ "$set_name" == typed-decisions ]] || sub="$set_name-$split"

mkdir -p "$out/$sub"
out="$(cd "$out/$sub" && pwd)"

name="bench-decisions-$$"
healthy || { echo "decisions: server not answering /health" >&2; exit 1; }
dc --profile client run --rm -T --name "$name" -v "$out:/out" decisions python3 /app/run.py \
    --base-url "http://${server}:8000" --set "$set_name" --data-dir /datasets --split "$split" \
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
