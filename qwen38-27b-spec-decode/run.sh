#!/bin/bash
# Speculative-decoding drafters on stock Qwen3.8-27B NVFP4 (SGLang), reasoning_effort=medium.
# Same target, flags and engine for every config; only the speculative flags differ.
#
# Needs MODELS_DIR containing (see ../README.md for download commands):
#   Qwen3.8-27B-NVFP4    RadixArk/Qwen3.8-27B-NVFP4
#   Qwen3.8-27B-DFlash2  z-lab/Qwen3.8-27B-DFlash2
#   Qwen3.8-27B-DSpark   RadixArk/Qwen3.8-27B-DSpark
#
# Usage: ./run.sh [label ...]   default: every config below
# DRY_RUN=1 prints each selected config's resolved flags and starts nothing.
# Results go to $OUT/<label>/ (default: results/ next to this script).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
scripts="$here/../scripts"
out="${OUT:-$here/results}"
. "$scripts/args/qwen38-27b.sh"

export MODEL_DIR=Qwen3.8-27B-NVFP4
server_args="$(qwen38_27b_server_args 0.41)"
# Server flags of the DSpark card, for the *-cardflags labels (see the README).
card_args="$(qwen38_27b_card_server_args 0.41)"

d2="--speculative-algorithm DFLASH --speculative-draft-model-path /models/Qwen3.8-27B-DFlash2"
dspark="--speculative-algorithm DSPARK --speculative-draft-model-path /models/Qwen3.8-27B-DSpark"
dspark+=" --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer"
dspark+=" --speculative-dspark-block-size 7 --speculative-num-steps 1 --speculative-eagle-topk 1"
mtp="--speculative-algorithm EAGLE --speculative-eagle-topk 1"

default_scenarios="sharegpt_c1 sharegpt_c4 long20k_c1 long20k_c4"
long_scenarios="$default_scenarios long40k_c1 long80k_c1 long120k_c1"
medium='{"chat_template_kwargs":{"reasoning_effort":"medium"}}'
nothink='{"chat_template_kwargs":{"enable_thinking":false}}'
greedy='{"chat_template_kwargs":{"reasoning_effort":"medium"},"temperature":0}'

# Config i: label | speculative flags | scenarios | request extra body | run BFCL (1/0)
labels=(nospec mtp3 mtp7 dflash4 dflash8 dspark dflash8-noreason dflash8-greedy dspark-cardflags dflash8-cardflags)
declare -A spec scenarios body bfcl sargs senv
set_config() {
    spec[$1]="$2"; scenarios[$1]="$3"; body[$1]="$4"; bfcl[$1]="$5"
    sargs[$1]="${6:-$server_args}"; senv[$1]="${7:-}"
}
set_config nospec ""                                                           "$long_scenarios" "$medium" 1
set_config mtp3 "$mtp --speculative-num-steps 3 --speculative-num-draft-tokens 4" "$long_scenarios" "$medium" 1
set_config mtp7 "$mtp --speculative-num-steps 7 --speculative-num-draft-tokens 8" "$long_scenarios" "$medium" 1
set_config dflash4 "$d2 --speculative-num-draft-tokens 4"                      "$long_scenarios" "$medium" 1
set_config dflash8 "$d2 --speculative-num-draft-tokens 8"                      "$long_scenarios" "$medium" 1
set_config dspark "$dspark"                                                    "$long_scenarios" "$medium" 1
# Diagnosis on dflash8: no thinking, and greedy sampling. Serving only.
set_config dflash8-noreason "$d2 --speculative-num-draft-tokens 8" "$default_scenarios" "$nothink" 0
set_config dflash8-greedy   "$d2 --speculative-num-draft-tokens 8" "$default_scenarios" "$greedy"  0
# Same speculative flags as dspark / dflash8, server flags and environment of the DSpark card.
set_config dspark-cardflags "$dspark" "$long_scenarios" "$medium" 0 "$card_args" "$here/card-dspark.env"
set_config dflash8-cardflags "$d2 --speculative-num-draft-tokens 8" "$long_scenarios" "$medium" 0 "$card_args" "$here/card-dflash.env"

wanted=("$@")

# Each step returns on failure explicitly: callers run this as an `if !` condition, where bash
# ignores `set -e`, so a failed scenario would otherwise still go on to BFCL.
run_config() {
    local label="$1"
    export SPEC_ARGS="${spec[$label]}" SCENARIOS="${scenarios[$label]}" EXTRA_BODY="${body[$label]}"
    export LOG_DIR="$out/$label" SERVER_ARGS="${sargs[$label]}" SERVER_ENV_FILE="${senv[$label]}"
    "$scripts/serve.sh" up || return
    "$scripts/bench-serving.sh" "$out/$label" || return
    if (( ${bfcl[$label]} )); then "$scripts/bfcl.sh" "$out/$label"; fi
}

for label in "${labels[@]}"; do
    if (( ${#wanted[@]} )) && [[ ! " ${wanted[*]} " == *" $label "* ]]; then
        continue
    fi
    if [[ -n "${DRY_RUN:-}" ]]; then
        export SPEC_ARGS="${spec[$label]}"
        echo "== $label"; echo "SERVER_ARGS: ${sargs[$label]}"; echo "SPEC_ARGS: $SPEC_ARGS"
        echo "SERVER_ENV_FILE: ${senv[$label]} -> $(tr '\n' ' ' <"${senv[$label]:-/dev/null}")"
        echo "SCENARIOS: ${scenarios[$label]}  BFCL: ${bfcl[$label]}"
        continue
    fi
    # One retry; a config that fails twice is recorded in FAILED and the run goes on.
    if ! run_config "$label"; then
        "$scripts/serve.sh" down || true
        rm -rf "$out/$label"
        if ! run_config "$label"; then
            mkdir -p "$out/$label"
            echo "$label failed twice, see server.log" >>"$out/FAILED"
        fi
    fi
    "$scripts/serve.sh" down
done

if [[ -n "${DRY_RUN:-}" ]]; then exit 0; fi

have() { for l in "$@"; do [[ -f "$out/$l/bench-serving.jsonl" ]] && echo "$out/$l"; done; }
for group in "nospec mtp3 mtp7 dflash4 dflash8 dspark" "dflash8 dflash8-noreason dflash8-greedy" \
    "dspark dspark-cardflags dflash8 dflash8-cardflags"; do
    mapfile -t dirs < <(have $group)
    if (( ${#dirs[@]} > 1 )); then "$scripts/compare.py" "${dirs[@]}"; fi
done
