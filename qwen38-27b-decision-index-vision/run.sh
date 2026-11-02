#!/bin/bash
# Decision Index public vision suite (11 benchmarks, 12,210 rows) on SGLang's /v1/decisions with
# images, for stock Qwen3.8-27B NVFP4 and Swift 1.5 NVFP4. The kit (apolinario/decision-index) is
# cloned at the pinned commit and run in a container on the compose network; the server flags are
# the image-decisions post's (scripts/args/qwen38-27b.sh, same image, same 0.41 memory fraction).
# Endpoint: the kit's `protocol=decisions` body (model, state, questions, images) is what this SGLang image
# accepts on /v1/systemone; its /v1/decisions wants `input` and a question list (HTTP 400 otherwise), so
# the engine is pointed at /v1/systemone, the route the image-decisions post's vision runs also used.
#
# Needs: MODELS_DIR with the checkpoints below, DI_SUITE = the rebuilt suite
# (decision-index suite rebuild --vision --work <dir>; DI_SUITE=<dir>/vision), BENCH_CDI=1 on
# rootless Docker. Rebuild + verify commands and the Mind2Web font: README.md.
#   Qwen3.8-27B-NVFP4                    RadixArk/Qwen3.8-27B-NVFP4 @ 319f741c
#   Swift-1.5-Qwen3.8-27B-NVFP4-ukisai   ukisai/Swift-1.5-Qwen3.8-27b-NVFP4 @ 25482027
#
# Usage: ./run.sh [label ...]   default: stock-nvfp4 swift-nvfp4
# Env: SUFFIX     appended to the result dir name (smoke runs: SUFFIX=-smoke with DI_SUITE pointing at a
#                 copy of the suite whose vision-rows.jsonl is the smoke subset, see README.md)
#      WORKERS    concurrent requests (default 1: the board's latency rule is single-process)
#      OUT        results dir (default: results/ next to this script)
# --no-verify: the Mind2Web images/payloads fail the kit's hash check by design (Liberation Sans Bold
# instead of Arial Bold, README.md "Deviations"); the rebuild report and verify.json show everything else matches.
# Resumable: rerunning a label retries only rows that errored or are missing.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
scripts="$here/../scripts"
: "${DI_SUITE:?set DI_SUITE}"
KIT_COMMIT=25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d
export DI_KIT="${DI_KIT:-$here/decision-index}"
if [[ ! -d "$DI_KIT/.git" ]]; then
    git clone -q https://github.com/apolinario/decision-index "$DI_KIT"
fi
git -C "$DI_KIT" checkout -q "$KIT_COMMIT"
export DI_SUITE
. "$scripts/env.sh"
# The development compose file mounts ${SGLANG_SRC:-/var/empty} (an optional patch mount, unused here); rootless Docker
# can't create /var/empty where it is missing. The released compose/sglang.yaml has no such mount, so this is a no-op there.
[[ -d /var/empty ]] || { mkdir -p /tmp/bench-empty; export SGLANG_SRC=/tmp/bench-empty; }
. "$scripts/args/qwen38-27b.sh"

configs=(
    "stock-nvfp4|Qwen3.8-27B-NVFP4|0.41"
    "swift-nvfp4|Swift-1.5-Qwen3.8-27B-NVFP4-ukisai|0.41"
)
wanted=("$@")
out="${OUT:-$here/results}"
for c in "${configs[@]}"; do
    IFS='|' read -r label model mem <<<"$c"
    if (( ${#wanted[@]} )); then [[ " ${wanted[*]} " == *" $label "* ]] || continue; fi
    dir="$out/$label${SUFFIX:-}"
    export MODEL_DIR="$model" SERVER_ARGS="$(qwen38_27b_server_args "$mem")" SPEC_ARGS=""
    export LOG_DIR="$dir"
    if ! "$scripts/serve.sh" up; then
        "$scripts/serve.sh" down || true
        echo "$label: server failed to start, see server.log" >>"$out/FAILED"
        continue
    fi
    mkdir -p "$dir/run"
    dc --profile client run --rm -T --name "bench-di-vision-$$" -v "$dir/run:/out" di-vision \
        vision pipeline --dir /suite --engine decide \
        --option "base_url=http://${server}:8000" --option protocol=decisions --option path=/v1/systemone \
        --workers "${WORKERS:-1}" --no-verify --out /out \
        >>"$dir/pipeline.log" 2>&1 \
        || echo "$label: vision pipeline failed, see pipeline.log" >>"$out/FAILED"
    "$scripts/serve.sh" down
done
