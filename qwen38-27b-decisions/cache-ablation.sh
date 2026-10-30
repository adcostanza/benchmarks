#!/bin/bash
# Prefix cache ablation: which server flag makes a prefix-cache hit change top answers?
# Reruns the prefix cache check (first 50 Typed Decisions test cases, two /v1/systemone passes then
# one native pass) on one config, changing one shared flag at a time (README "Prefix cache
# ablation"). Baseline numbers: results-cache/.
#
# Usage: ./cache-ablation.sh [variant ...]   default: every variant below, in order
# Env: CONFIG (default swift-nvfp4), DATASETS_DIR, MODELS_DIR, BENCH_CDI as for run.sh
# Results: results-cache-ablation/<variant>/<config>/; table: ./cache_check.py results-cache-ablation/*/*
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
config="${CONFIG:-swift-nvfp4}"
hicache="--enable-hierarchical-cache --hicache-size --hicache-write-policy"

# variant | flags dropped | flags added
variants=(
    "no-radix-cache||--disable-radix-cache"
    "kv-auto|--kv-cache-dtype|--kv-cache-dtype auto"
    "ssm-fp32|--mamba-ssm-dtype|--mamba-ssm-dtype float32"
    "no-hicache|$hicache|"
    "no-buffer|--mamba-radix-cache-strategy|--mamba-radix-cache-strategy no_buffer"
    "no-overlap||--disable-overlap-schedule"
    "deterministic-triton|--attention-backend|--attention-backend triton --enable-deterministic-inference"
    # no-hicache had no cache hits at all; GPU-only radix hits need the eager extra buffer
    "gpu-radix|$hicache --mamba-radix-cache-strategy|--mamba-radix-cache-strategy extra_buffer"
    "gpu-radix-deterministic|$hicache --mamba-radix-cache-strategy --attention-backend|--mamba-radix-cache-strategy extra_buffer --attention-backend triton --enable-deterministic-inference"
    "hicache-deterministic-kv-auto|--attention-backend --kv-cache-dtype|--attention-backend triton --enable-deterministic-inference --kv-cache-dtype auto"
)

wanted=("$@")
for v in "${variants[@]}"; do
    IFS='|' read -r name drop extra <<<"$v"
    if (( ${#wanted[@]} )) && [[ ! " ${wanted[*]} " == *" $name "* ]]; then
        continue
    fi
    echo "== $name: drop [$drop] add [$extra]"
    OUT="$here/results-cache-ablation/$name" DECISIONS_LIMIT=50 \
        SERVER_ARGS_DROP="$drop" SERVER_ARGS_EXTRA="$extra" "$here/run.sh" "$config"
done
