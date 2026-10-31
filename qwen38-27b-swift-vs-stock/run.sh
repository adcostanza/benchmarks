#!/bin/bash
# Reproduces https://acostanza.com/benchmarks/qwen38-27b-swift-vs-stock/
# Swift 1.5 NVFP4 (ukisai's official build) vs stock Qwen3.8-27B NVFP4 (RadixArk), both with
# DFlash2 speculative decoding (8 draft tokens), same flags and memory fraction.
#   *-quality   GSM8K (all 1,319, daily) and the BFCL subset (1 run)
#   *dflash8    serving (seven scenarios), SPEED-Bench (c1, 10 per domain, 4 per entropy tier
#               and length bucket, daily), the DFlash2 card's five sets (32 per set, ref + daily)
#   Measured: about 2.3 h for all four (suite.json step times sum to 2.1 h)
#
# Needs MODELS_DIR containing:
#   Qwen3.8-27B-NVFP4                    RadixArk/Qwen3.8-27B-NVFP4
#   Swift-1.5-Qwen3.8-27B-NVFP4-ukisai   ukisai/Swift-1.5-Qwen3.8-27b-NVFP4
#   Qwen3.8-27B-DFlash2                  z-lab/Qwen3.8-27B-DFlash2
# and DATASETS_DIR prepared as the step scripts' headers describe (GSM8K, SPEED-Bench resolved
# with scripts/speed/prepare.sh, the five sets of the DFlash2 card).
#
# Usage: ./run.sh [label ...]   default: every config below
# Results go to $OUT/<label>/ (default: results/ next to this script).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
scripts="$here/../scripts"
out="${OUT:-$here/results}"
: "${DATASETS_DIR:?set DATASETS_DIR}"

. "$scripts/args/qwen38-27b.sh"
spec="--speculative-algorithm DFLASH --speculative-draft-model-path /models/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8"

# label  checkpoint dir  steps
configs=(
    "swift-nvfp4-quality Swift-1.5-Qwen3.8-27B-NVFP4-ukisai quality"
    "stock-nvfp4-quality Qwen3.8-27B-NVFP4                  quality"
    "swift-dflash8       Swift-1.5-Qwen3.8-27B-NVFP4-ukisai drafter"
    "dflash8             Qwen3.8-27B-NVFP4                  drafter"
)

quality() {
    PROTOCOL=daily "$scripts/gsm8k.sh" "$1"
    BFCL_REPEATS=1 "$scripts/bfcl.sh" "$1"
}
drafter() {
    SCENARIOS="sharegpt_c1 sharegpt_c4 long20k_c1 long20k_c4 long40k_c1 long80k_c1 long120k_c1" \
        "$scripts/bench-serving.sh" "$1"
    PROTOCOL=daily SPEEDBENCH_CONC=1 SPEEDBENCH_QUAL_PER_CAT=10 SPEEDBENCH_TP_PER_TIER=4 \
        "$scripts/speed-bench.sh" "$1"
    PROTOCOL=both ACCEPT_NUM_PROMPTS=32 "$scripts/accept-sets.sh" "$1"
}

wanted=("$@")
for c in "${configs[@]}"; do
    read -r label model steps <<<"$c"
    if (( ${#wanted[@]} )) && [[ ! " ${wanted[*]} " == *" $label "* ]]; then
        continue
    fi
    export MODEL_DIR="$model" SERVER_ARGS="$(qwen38_27b_server_args 0.41 "$spec")" LOG_DIR="$out/$label"
    "$scripts/serve.sh" up
    "$steps" "$out/$label"
    "$scripts/serve.sh" down
done

for pair in "stock-nvfp4-quality swift-nvfp4-quality" "dflash8 swift-dflash8"; do
    read -r a b <<<"$pair"
    if [[ -d "$out/$a" && -d "$out/$b" ]]; then
        "$scripts/compare.py" "$out/$a" "$out/$b"
    fi
done
