#!/bin/bash
# Reproduces https://acostanza.com/blog/qwen38-27b-nvfp4-vs-fp8/
# Stock Qwen3.8-27B FP8 vs NVFP4 on SGLang with DFlash2 speculative decoding (8 draft tokens).
#
# Needs MODELS_DIR containing (see README.md for the download commands):
#   Qwen3.8-27B-FP8      Qwen/Qwen3.8-27B-FP8
#   Qwen3.8-27B-NVFP4    RadixArk/Qwen3.8-27B-NVFP4
#   Qwen3.8-27B-DFlash2  z-lab/Qwen3.8-27B-DFlash2
#
# Usage: ./run.sh [label ...]   default: every config below
# Results go to $OUT/<label>/ (default: results/ next to this script).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
scripts="$here/../scripts"
out="${OUT:-$here/results}"

export SPEC_ARGS="--speculative-algorithm DFLASH --speculative-draft-model-path /models/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8"

# label  checkpoint dir  mem fraction (sized so both get about the same KV pool)
configs=(
    "stock-fp8   Qwen3.8-27B-FP8   0.50"
    "stock-nvfp4 Qwen3.8-27B-NVFP4 0.41"
)

wanted=("$@")
for c in "${configs[@]}"; do
    read -r label model mem <<<"$c"
    if (( ${#wanted[@]} )) && [[ ! " ${wanted[*]} " == *" $label "* ]]; then
        continue
    fi
    export MODEL_DIR="$model" MEM_FRACTION="$mem"
    "$scripts/serve.sh" up
    "$scripts/bench-serving.sh" "$out/$label"
    "$scripts/bfcl.sh" "$out/$label"
    "$scripts/serve.sh" down
done

if [[ -d "$out/stock-fp8" && -d "$out/stock-nvfp4" ]]; then
    "$scripts/compare.py" "$out/stock-fp8" "$out/stock-nvfp4"
fi
