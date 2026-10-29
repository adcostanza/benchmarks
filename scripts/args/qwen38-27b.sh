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
