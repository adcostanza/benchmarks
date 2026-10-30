#!/bin/bash
# One-time: resolve the masked SPEED-Bench prompts (see prepare.py) into
# $DATASETS_DIR/SPEED-Bench-resolved/<config>/test.parquet. Needs network, a Hugging Face token in
# $HF_TOKEN whose account accepted the terms of the gated dataset cais/hle, and ~40 GB of
# download cache in the speed-prep-hf volume (drop it afterwards: docker volume rm
# ${BENCH_PROJECT:-bench}_speed-prep-hf). Takes tens of minutes. Uses no GPU.
#
# Usage: scripts/speed/prepare.sh [config ...]   (default: qualitative + the five throughput configs)
set -euo pipefail
. "$(dirname "$0")/../env.sh"
: "${DATASETS_DIR:?set DATASETS_DIR}"
: "${HF_TOKEN:?set HF_TOKEN (cais/hle is gated)}"
mkdir -p "$DATASETS_DIR/SPEED-Bench-resolved"
dc --profile client run --rm -T speed-prep sh -c \
    'pip install -q --no-cache-dir datasets==4.4.1 tiktoken==0.14.0 pandas==3.1.0 numpy==2.5.4 && python /prepare.py "$@"' sh "$@"
