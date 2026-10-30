#!/bin/bash
# GSM8K full test split (1,319 questions, openai/gsm8k `main`) against the running model
# server ($ENGINE), thinking on. Raw responses and extracted answers: <out-dir>/gsm8k/<protocol>.jsonl,
# metrics: <out-dir>/gsm8k/summary.json. Needs DATASETS_DIR with gsm8k/main/test-*.parquet.
#
# Reference: RadixArk Qwen3.8-27B-NVFP4 card, "GSM8K: full 1,319-example split, thinking mode,
# sgl-eval", 97.27% (1,283/1,319), temperature 1.0, top_p 0.95, top_k 20, text only. sgl-eval is
# https://github.com/sgl-project/sgl-eval (PyPI sgl-eval, 0.1.3 used here), which SGLang's CI
# calls through sglang.test.sgl_eval_utils. Its gsm8k: NeMo-Skills generic/math prompt, zero-shot,
# "put the answer (and only answer) inside \boxed{}", answer = last \boxed{} of the message
# content, graded with math_verify; it sends min_p 0 and repetition_penalty 1 on every request.
# Everything the card does not state is sgl-eval's default: no max_tokens (server limit), no seed,
# 64 threads. The card gives no reasoning effort, so `ref` leaves the template default.
#   ref    the settings above, thinking switched on through chat_template_kwargs, no max_tokens.
#   daily  reasoning_effort=medium, the model's own sampling defaults (none sent), same prompt
#          and grader, max_tokens $GSM8K_DAILY_MAX_TOKENS so answers are not cut off.
# Both score the last \boxed{} even when the generation was cut off (finish_reason length), as
# sgl-eval does; the truncated count is reported next to the accuracy.
#
# Usage: scripts/gsm8k.sh <out-dir>
# Env: PROTOCOL               ref | daily | both (default both)
#      GSM8K_LIMIT            run only the first N questions (smoke runs)
#      GSM8K_CONCURRENCY      in-flight requests (default 32; sgl-eval defaults to 64)
#      GSM8K_DAILY_MAX_TOKENS max_tokens for daily (default 32768)
#      GSM8K_REF_MAX_TOKENS   max_tokens for ref (default unset: the server's limit)
# Caveats: sampling is unseeded, so a rerun moves the score by about the standard error
# (sqrt(p(1-p)/n), 0.4 points at 97%). Grading is sgl-eval's own, run on the dataset's parquet
# file instead of sgl-eval's download of the original JSONL. llama.cpp (ENGINE=llamacpp) uses the
# same chat endpoint but is untested here.
set -euo pipefail
. "$(dirname "$0")/env.sh"
out="${1:?out dir required}"
: "${DATASETS_DIR:?set DATASETS_DIR}"

case "${PROTOCOL:-both}" in
both) protocols="ref,daily" ;;
ref|daily) protocols="$PROTOCOL" ;;
*) echo "PROTOCOL must be ref, daily or both, got: $PROTOCOL" >&2; exit 2 ;;
esac

mkdir -p "$out/gsm8k"
out="$(cd "$out" && pwd)"

name="bench-gsm8k-$$"
client() {
    dc --profile client run --rm -T --name "$name" -v "$out:/out" gsm8k python3 /app/run.py "$@"
}

# The client starts only on a healthy server and is stopped as soon as the server stops
# answering /health (its requests would otherwise retry and hang for hours).
guarded() {
    healthy || { echo "gsm8k: server not answering /health" >&2; exit 1; }
    client "$@" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        sleep 15
        if kill -0 "$pid" 2>/dev/null && ! healthy; then
            echo "gsm8k: server stopped answering /health, stopping the client" >&2
            docker rm -f "$name" >/dev/null 2>&1 || true
            wait "$pid" || true
            exit 1
        fi
    done
    wait "$pid"
}

guarded --base-url "http://${server}:8000" --out /out --protocols "$protocols" \
    --limit "${GSM8K_LIMIT:-0}" --concurrency "${GSM8K_CONCURRENCY:-32}" \
    --daily-max-tokens "${GSM8K_DAILY_MAX_TOKENS:-32768}" --ref-max-tokens "${GSM8K_REF_MAX_TOKENS:-0}"
