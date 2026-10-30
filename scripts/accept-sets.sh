#!/bin/bash
# Accept length on the five sets of the DFlash2 card (https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2):
# GSM8K, MATH-500, HumanEval, MBPP, MT-Bench, against the running SGLang server. Meaningful only
# when the server runs speculative decoding; without it the accept-length metrics are absent
# from summary.json (not 1.0) and only tok/s is recorded. Generated text is never executed.
#
# Matches the card: "Acceptance length is the per-request mean of completion tokens divided by
# verification steps", throughput = total output tokens / end-to-end wall time, temperature 1.0,
# top-p 0.95, top-k 20, max 4096 new tokens, `xhigh` reasoning. Prompt formats, the shuffle (Random(42)
# over the dataset indices) and the request fields follow z-lab/dflash, dflash/benchmark.py at
# commit 07ebd93db9f472af339b644bb70221ad8428328a (DATASETS, _select_dataset, _run_openai),
# reimplemented in scripts/accept/accept_sets.py. As there, MT-Bench sends only the first turn,
# MBPP is the `sanitized` test config (257), and a pool of `concurrency` warmup requests with 64
# max tokens runs first.
#   ref    the card's protocol: reasoning_effort=xhigh (dflash sends enable_thinking,
#          reasoning_effort and reasoning_strength)
#   daily  reasoning_effort=medium, same sampling
# Writes <out-dir>/accept-sets/{summary.json,requests_<protocol>_<set>_c<N>.jsonl}. One run per
# protocol x set x concurrency: accept_length (per-request mean, with standard error;
# accept_length_token_weighted is sum(tokens)/sum(steps)), output_tok_s, TTFT/TPOT p50, mean
# completion tokens, and the share of requests cut off at the token limit.
#
# Caveats: the card does not state its prompt count; the dflash README example uses
# --num-prompts 128 (the CLI default is 1024) and the shuffle cycles, so 128 repeats MT-Bench's 80
# prompts (as dflash does). Prompts are read from local copies of the same HF datasets
# (revisions in summary.json), not downloaded. The card's numbers are on an H200.
#
# Usage: scripts/accept-sets.sh <out-dir>   (ENGINE=sglang; server already up)
# Env: PROTOCOL        ref | daily | both (default both)
#      ACCEPT_CONC     space-separated concurrencies (default: 1; the card also reports 8 and 32)
#      ACCEPT_NUM_PROMPTS  prompts per set (default 128)
#      ACCEPT_SETS     subset of: gsm8k math500 humaneval mbpp mt-bench (default: all)
#      ACCEPT_LIMIT    run only the first N prompts of every set, for smoke runs
#      MODEL_DIR       checkpoint dir, used as the request's model name
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
[[ "$server" == sglang ]] || { echo "accept-sets: needs ENGINE=sglang" >&2; exit 2; }
case "${PROTOCOL:-both}" in
ref|daily|both) ;;
*) echo "accept-sets: PROTOCOL must be ref, daily or both" >&2; exit 2 ;;
esac

mkdir -p "$out/accept-sets"
out="$(cd "$out" && pwd)"
rm -f "$out"/accept-sets/requests_*.jsonl "$out/accept-sets/summary.json"

name="bench-accept-$$"
envs=(-e MODEL_DIR -e "PROTOCOL=${PROTOCOL:-both}" -e "SERVER=$server")
for v in $(compgen -v ACCEPT_); do envs+=(-e "$v"); done
client() {
    dc --profile client run --rm -T --name "$name" -v "$out:/out" -v "$scripts_dir:/scripts:ro" \
        "${envs[@]}" bench-client python3 /scripts/accept/accept_sets.py /out/accept-sets
}

# The client fails on the first failed request; this also stops it when the server stops
# answering /health while requests hang.
healthy || { echo "accept-sets: server not answering /health" >&2; exit 1; }
client &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    if kill -0 "$pid" 2>/dev/null && ! healthy; then
        echo "accept-sets: server stopped answering /health, stopping the client" >&2
        docker rm -f "$name" >/dev/null 2>&1 || true
        wait "$pid" || true
        exit 1
    fi
done
wait "$pid"
[[ -s "$out/accept-sets/summary.json" ]] || { echo "accept-sets: no summary.json written" >&2; exit 1; }
