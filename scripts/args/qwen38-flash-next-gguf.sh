# Sourced by run.sh scripts. llama.cpp server flags for unsloth/Qwen3.8-Flash-Next-GGUF
# (UD-Q4_K_XL, 4 shards) on one RTX PRO 6000 plus host RAM, as in the published llama.cpp recipes
# (lukaLLM/Qwen3.8-Flash-Next-VRAM-Benchmark and others):
#   -ot per_layer_token_embd=CPU   n-gram (PLE) table in host RAM
#   --n-cpu-moe N                  expert weights of the first N of 48 layers in host RAM;
#                                  the knob that sets GPU memory
#   -fit off                       no automatic resizing, so GPU memory follows N
# -lm none (no mmap): weights are read into process memory instead of being mapped from the page cache,
# so host RAM use shows up in the container's memory like SGLang's pinned buffers.
# -t 16: one thread per physical core (Ryzen 9 7950X). No MTP (loses when experts are in RAM).
# Usage: SERVER_ARGS="$(qwen38_flash_next_gguf_server_args <n-cpu-moe> <context-length> [extra-args])"
qwen38_flash_next_gguf_server_args() {
    local ncmoe="${1:?n-cpu-moe required}" ctx="${2:?context length required}" extra="${3:-}"
    echo "-m /models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" \
        "-ngl 999 -ot per_layer_token_embd=CPU --n-cpu-moe $ncmoe -fit off" \
        "-c $ctx -kvu --parallel 4 -fa on --jinja -b 4096 -ub 4096 -t 16 -lm none $extra"
}
