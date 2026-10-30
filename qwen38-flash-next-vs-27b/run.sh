#!/bin/bash
# Qwen3.8-Flash-Next on one RTX PRO 6000, reasoning_effort=medium: NVFP4 (RadixArk) on SGLang and
# UD-Q4_K_XL GGUF (unsloth) on llama.cpp.
# Compared against the 27B results of ../qwen38-27b-spec-decode and ../qwen38-27b-nvfp4-vs-fp8
# (same harness and scenarios); the 27B is not rerun here.
#
# Needs MODELS_DIR containing (see README.md for the download commands):
#   Qwen3.8-Flash-Next-NVFP4    RadixArk/Qwen3.8-Flash-Next-NVFP4 (also the tokenizer for the GGUF runs)
#   Qwen3.8-Flash-Next-DFlash2  incoai/Qwen3.8-Flash-Next-DFlash2
#   Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL  unsloth/Qwen3.8-Flash-Next-GGUF, UD-Q4_K_XL shards
#
# Usage: ./run.sh [label ...]   default: every config below
# Results go to $OUT/<label>/ (default: results/ next to this script).
# Env: CONTEXT_LENGTH (default 262144, SGLang configs), BASE_IMAGE.
# Stock SGLang caps the Flash-Next draft block at 4 tokens (QSA compress ratio), for MTP and DFLASH.
#
# Whole card: fn-nospec, fn-mtp, fn-dflash2 (SGLang, --mem-fraction-static 0.88).
# Daily-driver budget: the 27B NVFP4 + DFlash2 8 server (../qwen38-27b-spec-decode, dflash8) uses
# 43.8 GB of GPU memory after load, 44.0 GB peak. fn-gguf-q4 (llama.cpp, experts of 30 of 48
# layers in host RAM, 44.3 GB after load) is sized to it. SGLang has no config at this budget:
# it rejects --cpu-offload-gb and --offload-group-size together with --ple-offload-embedding
# (srt/arg_groups/memory_hook.py), and KTransformers expert offload has no qwen4_exp support.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
scripts="$here/../scripts"
out="${OUT:-$here/results}"
. "$scripts/args/qwen38-flash-next.sh"
. "$scripts/args/qwen38-flash-next-gguf.sh"

export SERVER_ENV_FILE="$here/server.env"
base_image="${BASE_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20261029-c10e0e17}"
# DFLASH on Flash-Next needs the draft repo's target-feature capture patch, see dflash2.Dockerfile.
d2_image="bench-sglang-fn-dflash2:${base_image##*-}"
ctx="${CONTEXT_LENGTH:-262144}"

mtp="--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
d2="--speculative-algorithm DFLASH --speculative-draft-model-path /models/Qwen3.8-Flash-Next-DFlash2"
d2+=" --speculative-draft-model-quantization unquant --linear-attn-verify-backend triton --speculative-num-draft-tokens 4"
# Prefill CUDA graph off: leaves GPU memory for long prefills (see the plan in README.md).
lean="--cuda-graph-backend-prefill disabled"
gguf_ctx=153600  # the daily driver's context length

export SCENARIOS="sharegpt_c1 sharegpt_c4 long20k_c1 long20k_c4 long40k_c1 long80k_c1 long120k_c1"
export EXTRA_BODY='{"chat_template_kwargs":{"reasoning_effort":"medium"}}'

# label  engine  mem fraction (SGLang) or --n-cpu-moe layers (llama.cpp)  image  extra flags
labels=(fn-nospec fn-mtp fn-dflash2 fn-gguf-q4)
declare -A engine mem image extra
engine[fn-nospec]=sglang;    mem[fn-nospec]=0.88;  image[fn-nospec]="$base_image";  extra[fn-nospec]="$lean"
engine[fn-mtp]=sglang;       mem[fn-mtp]=0.88;     image[fn-mtp]="$base_image";     extra[fn-mtp]="$lean $mtp"
engine[fn-dflash2]=sglang;   mem[fn-dflash2]=0.88; image[fn-dflash2]="$d2_image";   extra[fn-dflash2]="$lean $d2"
engine[fn-gguf-q4]=llamacpp; mem[fn-gguf-q4]=30;   image[fn-gguf-q4]="";            extra[fn-gguf-q4]=""

wanted=("$@")
selected() { (( ${#wanted[@]} == 0 )) || [[ " ${wanted[*]} " == *" $1 "* ]]; }

if selected fn-dflash2 && ! docker image inspect "$d2_image" >/dev/null 2>&1; then
    docker build --build-arg BASE="$base_image" -f "$here/dflash2.Dockerfile" -t "$d2_image" "$here"
fi

config_env() {
    local label="$1"
    export ENGINE="${engine[$label]}" LOG_DIR="$out/$label"
    # SGLang image: the server for SGLang configs, the benchmark client for llama.cpp configs.
    export SGLANG_IMAGE="${image[$label]:-$base_image}"
    if [[ "$ENGINE" == llamacpp ]]; then
        export MODEL_DIR=Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL TOKENIZER_DIR=Qwen3.8-Flash-Next-NVFP4
        export SERVER_ARGS="$(qwen38_flash_next_gguf_server_args "${mem[$label]}" "$gguf_ctx" "${extra[$label]}")"
    else
        export MODEL_DIR=Qwen3.8-Flash-Next-NVFP4 TOKENIZER_DIR=
        export SERVER_ARGS="$(qwen38_flash_next_server_args "${mem[$label]}" "$ctx" "${extra[$label]}")"
    fi
}

# Each step returns on failure explicitly: the caller runs this as an `if !` condition, where bash
# ignores `set -e`, so a failed scenario would otherwise still go on to BFCL.
run_config() {
    local label="$1"
    rm -rf "$LOG_DIR"
    "$scripts/serve.sh" up || return
    "$scripts/bench-serving.sh" "$out/$label" || return
    "$scripts/bfcl.sh" "$out/$label"
}

for label in "${labels[@]}"; do
    selected "$label" || continue
    config_env "$label"
    # A failed config keeps its server.log and resources, is recorded in FAILED, and the run goes on.
    if ! run_config "$label"; then
        echo "$label failed, see $label/server.log" >>"$out/FAILED"
    fi
    "$scripts/serve.sh" down
done

# Flash-Next vs the 27B NVFP4 + DFlash2 block 8 (spec-decode benchmark, all scenarios) and
# vs the 27B stock-nvfp4 run of the first post (default four scenarios only).
dirs=()
for d in "$here/../qwen38-27b-spec-decode/results/dflash8" "$here/../qwen38-27b-nvfp4-vs-fp8/results/stock-nvfp4" \
         "$out/fn-nospec" "$out/fn-mtp" "$out/fn-dflash2" "$out/fn-gguf-q4"; do
    [[ -f "$d/bench-serving.jsonl" ]] && dirs+=("$d")
done
if (( ${#dirs[@]} > 1 )); then "$scripts/compare.py" "${dirs[@]}"; fi
