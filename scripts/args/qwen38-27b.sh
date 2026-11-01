# Sourced by run.sh scripts. Server flags for the stock Qwen3.8-27B checkpoints.
# Usage: SERVER_ARGS="$(qwen38_27b_server_args <mem-fraction> [spec-args])"
# spec-args are placed inside the flag list; pass them here or through SPEC_ARGS
# (compose appends SPEC_ARGS after SERVER_ARGS), not both.
qwen38_27b_server_args() {
    local mem="${1:?mem fraction required}" spec="${2:-}"
    echo "--mem-fraction-static $mem --max-running-requests 4 --context-length 153600" \
        "--dtype auto --attention-backend flashinfer --reasoning-parser qwen3" \
        "--tool-call-parser qwen3_coder $spec" \
        "--mamba-full-memory-ratio 0.9 --mamba-ssm-dtype bfloat16" \
        "--mamba-radix-cache-strategy extra_buffer_lazy --chunked-prefill-size 2048" \
        "--page-size 64 --enable-hierarchical-cache --hicache-size 64" \
        "--hicache-write-policy write_through --kv-cache-dtype fp8_e4m3" \
        "--schedule-policy lpm --enable-session-radix-cache --sleep-on-idle --trust-remote-code"
}

# Same flags with the server flags of the DSpark card (RadixArk/Qwen3.8-27B-DSpark, "Serving with
# SGLang") in place of the shared ones; memory fraction and --max-running-requests stay as given.
# The card also sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which is environment, not a
# flag: use a SERVER_ENV_FILE (qwen38-27b-spec-decode/card-*.env).
qwen38_27b_card_server_args() {
    qwen38_27b_server_args "$@" | sed \
        -e 's/--mamba-full-memory-ratio 0.9/--mamba-full-memory-ratio 11.93/' \
        -e 's/--mamba-ssm-dtype bfloat16/--mamba-ssm-dtype float32/' \
        -e 's/--mamba-radix-cache-strategy extra_buffer_lazy/--mamba-radix-cache-strategy extra_buffer/' \
        -e 's/--chunked-prefill-size 2048/--chunked-prefill-size 32768 --max-prefill-tokens 32768/'
}
