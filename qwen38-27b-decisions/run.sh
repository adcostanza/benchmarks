#!/bin/bash
# Typed Decisions on /v1/systemone: Swift 1.5 vs stock Qwen3.8-27B at BF16, FP8 and NVFP4, i.e.
# the chat model used as a classifier from its answer-label log-probs. Question: does Swift's
# fine-tune or the quantization (or neither) hurt decisions, and by how much.
#
# Every config uses the same server flags (scripts/args/qwen38-27b.sh, FP8 KV cache included), so
# the weights are the only difference. No speculative decoding (decisions only prefill; SGLang
# validates them without spec or with NEXTN only), except `swift-nvfp4-dflash8`, the daily setup.
#
# Needs MODELS_DIR containing:
#   Qwen3.8-27B                          Qwen/Qwen3.8-27B (BF16)
#   Qwen3.8-27B-FP8                      Qwen/Qwen3.8-27B-FP8
#   Qwen3.8-27B-NVFP4                    RadixArk/Qwen3.8-27B-NVFP4
#   Swift-1.5-Qwen3.8-27B                ukisai/Swift-1.5-Qwen3.8-27b (BF16)
#   Swift-1.5-Qwen3.8-27B-FP8            block FP8 of the above (built locally, see README.md)
#   Swift-1.5-Qwen3.8-27B-NVFP4-ukisai   ukisai/Swift-1.5-Qwen3.8-27b-NVFP4
#   Qwen3.8-27B-DFlash2                  z-lab/Qwen3.8-27B-DFlash2 (daily-setup row only)
# and DATASETS_DIR with typed-decisions/ (scripts/decisions.sh), plus the sets named in
# DECISIONS_RUNS (files per set: scripts/decisions/sets.py).
#
# Usage: ./run.sh [label ...]   default: every config below, in order
# Env: DECISIONS_RUNS  space-separated <set>:<split> runs per config, in order (default
#                      typed-decisions:test); other DECISIONS_* settings as in scripts/decisions.sh
#      SERVER_ARGS_DROP  space-separated flags removed from the shared flags, each with its value
#                        if it has one (flag-ablation runs, e.g. the prefix cache check variants)
#      SERVER_ARGS_EXTRA flags appended after the shared flags
# Results go to $OUT/<label>/ (default: results/ next to this script). A config whose server
# fails to start or whose decisions step fails is listed in $OUT/FAILED and the run continues.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
scripts="$here/../scripts"
out="${OUT:-$here/results}"
: "${DATASETS_DIR:?set DATASETS_DIR}"
. "$scripts/args/qwen38-27b.sh"

# Shared flags for one config, minus SERVER_ARGS_DROP, plus SERVER_ARGS_EXTRA.
server_args() {
    local -a args kept=()
    read -ra args <<<"$(qwen38_27b_server_args "$1")"
    local i=0
    while (( i < ${#args[@]} )); do
        if [[ " ${SERVER_ARGS_DROP:-} " == *" ${args[i]} "* ]]; then
            (( i + 1 < ${#args[@]} )) && [[ ${args[i+1]} != --* ]] && (( i++ )) || true
        else
            kept+=("${args[i]}")
        fi
        (( i++ )) || true
    done
    echo "${kept[*]} ${SERVER_ARGS_EXTRA:-}"
}
dflash8="--speculative-algorithm DFLASH --speculative-draft-model-path /models/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8"

# label | checkpoint | mem fraction (as in suite-v1) | speculative flags
configs=(
    "stock-bf16|Qwen3.8-27B|0.76|"
    "stock-fp8|Qwen3.8-27B-FP8|0.50|"
    "stock-nvfp4|Qwen3.8-27B-NVFP4|0.41|"
    "swift-bf16|Swift-1.5-Qwen3.8-27B|0.76|"
    "swift-fp8|Swift-1.5-Qwen3.8-27B-FP8|0.50|"
    "swift-nvfp4|Swift-1.5-Qwen3.8-27B-NVFP4-ukisai|0.41|"
    "swift-nvfp4-dflash8|Swift-1.5-Qwen3.8-27B-NVFP4-ukisai|0.41|$dflash8"
)

wanted=("$@")
mkdir -p "$out"
for c in "${configs[@]}"; do
    IFS='|' read -r label model mem spec <<<"$c"
    if (( ${#wanted[@]} )) && [[ ! " ${wanted[*]} " == *" $label "* ]]; then
        continue
    fi
    export MODEL_DIR="$model" SERVER_ARGS="$(server_args "$mem")" SPEC_ARGS="$spec"
    export LOG_DIR="$out/$label"
    if ! "$scripts/serve.sh" up; then
        "$scripts/serve.sh" down || true
        echo "$label: server failed to start, see server.log" >>"$out/FAILED"
        continue
    fi
    for run in ${DECISIONS_RUNS:-typed-decisions:test}; do
        DECISIONS_SET="${run%%:*}" DECISIONS_SPLIT="${run#*:}" "$scripts/decisions.sh" "$out/$label" \
            || echo "$label: decisions $run failed, see the log" >>"$out/FAILED"
    done
    "$scripts/serve.sh" down
done
