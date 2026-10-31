#!/bin/bash
# Overnight GPU run: repeat passes of every text run for the serving-noise spread of the temperature
# transfer matrix (transfer.py). Both NVFP4 configs, same server flags as run.sh. A failed step is
# listed in <out>/FAILED (run.sh) and the rest continues.
#
#   results-repeat/train/<config>/decisions/pass{1..3}.jsonl  Typed Decisions train
#   results-repeat/test/<config>/...                          every other text run
# Pass 1 is cold; passes 2-3 run on the prefix cache of the same server session.
# Rough time (2026-10-30 one-pass walls): text ~21 min per pass per config for the sets already
# run, plus ChaosNLI and LeWiDi (~10,600 short cases, [INFERENCE] ~6 min), so ~3 h for both
# configs at 3 passes plus server starts.
#
# Usage: DATASETS_DIR=... MODELS_DIR=... BENCH_CDI=1 ./tonight.sh
# Env: PASSES (default 3)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${DATASETS_DIR:?set DATASETS_DIR}" "${MODELS_DIR:?set MODELS_DIR}"
passes="${PASSES:-3}"
configs=(stock-nvfp4 swift-nvfp4)
export DECISIONS_NATIVE_PASSES=0
text_runs="chaosnli:validation chaosnli:test lewidi:validation lewidi:test typed-decision-bench:test procedural-typed-decisions:validation procedural-typed-decisions:test calibrated-typed-decisions:validation calibrated-typed-decisions:test calibrated-typed-decisions:heldout typed-decisions:test"

OUT="$here/results-repeat/train" DECISIONS_RUNS="typed-decisions:train" DECISIONS_PASSES="$passes" \
    "$here/run.sh" "${configs[@]}"
OUT="$here/results-repeat/test" DECISIONS_RUNS="$text_runs" DECISIONS_PASSES="$passes" \
    "$here/run.sh" "${configs[@]}"
echo "done; failures: $(cat "$here"/results-repeat/*/FAILED 2>/dev/null | wc -l)"
