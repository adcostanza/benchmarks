# Sourced by run.sh scripts. Server flags for RadixArk/Qwen3.8-Flash-Next-NVFP4 on one 96 GB
# RTX PRO 6000 (TP1). The 51B-parameter n-gram (PLE) table, ~48 GiB in FP8, stays in pinned
# host RAM (--ple-offload-embedding); the other ~78 GiB of the checkpoint lives on the GPU.
# Usage: SERVER_ARGS="$(qwen38_flash_next_server_args <mem-fraction> <context-length> [extra-args])"
# extra-args (speculative decoding, memory tuning) are placed inside the flag list; pass spec
# args here or through SPEC_ARGS (compose appends SPEC_ARGS after SERVER_ARGS), not both.
# Prefill chunk 2048: at 4096 the first 20k-token prefill ran out of GPU memory at 0.92–0.96.
qwen38_flash_next_server_args() {
    local mem="${1:?mem fraction required}" ctx="${2:?context length required}" extra="${3:-}"
    echo "--mem-fraction-static $mem --max-running-requests 4 --max-mamba-cache-size 12" \
        "--context-length $ctx --dtype bfloat16 --quantization modelopt_fp4 --tp 1" \
        "--reasoning-parser qwen3 --tool-call-parser qwen3_coder $extra" \
        "--ple-offload-embedding --mamba-ssm-dtype bfloat16" \
        "--mamba-radix-cache-strategy extra_buffer_lazy --mamba-track-interval 64" \
        "--chunked-prefill-size 2048 --page-size 64 --kv-cache-dtype fp8_e4m3 --sleep-on-idle"
}
