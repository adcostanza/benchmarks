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
# and DATASETS_DIR with typed-decisions/ (scripts/decisions.sh).
#
# Usage: ./run.sh [label ...]   default: every config below, in order
# Results go to $OUT/<label>/ (default: results/ next to this script). A config whose server
# fails to start or whose decisions step fails is listed in $OUT/FAILED and the run continues.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
scripts="$here/../scripts"
out="${OUT:-$here/results}"
: "${DATASETS_DIR:?set DATASETS_DIR}"
. "$scripts/args/qwen38-27b.sh"
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
    export MODEL_DIR="$model" SERVER_ARGS="$(qwen38_27b_server_args "$mem")" SPEC_ARGS="$spec"
    export LOG_DIR="$out/$label"
    if ! "$scripts/serve.sh" up; then
        "$scripts/serve.sh" down || true
        echo "$label: server failed to start, see server.log" >>"$out/FAILED"
        continue
    fi
    "$scripts/decisions.sh" "$out/$label" || echo "$label: decisions failed, see the log" >>"$out/FAILED"
    "$scripts/serve.sh" down
done
