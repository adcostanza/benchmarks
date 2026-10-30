#!/bin/bash
# Start the model server ($ENGINE, default SGLang) and wait until it answers /health, or stop it.
# Usage: scripts/serve.sh up | down
# `up` reads MODELS_DIR, MODEL_DIR, SERVER_ARGS, SPEC_ARGS, ENGINE (see compose/sglang.yaml) and
# SERVE_TIMEOUT (seconds to wait for /health, default 900).
# With LOG_DIR set:
#   `up` records startup time, checkpoint size, how much of the checkpoint was already in the page
#   cache (1.0 = warm start) and memory after load in $LOG_DIR/resources.json, then samples
#   memory every 2 s (scripts/monitor.sh) until `down`.
#   `down` stops the sampler, adds the peaks to resources.json and saves the server log to
#   $LOG_DIR/server.log.
set -euo pipefail
. "$(dirname "$0")/env.sh"

gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

case "${1:?up or down}" in
up)
    if [[ -n "${LOG_DIR:-}" ]]; then
        mkdir -p "$LOG_DIR"
        rm -f "$LOG_DIR/resources.json" "$LOG_DIR/resources.csv"
        ckpt="${MODELS_DIR:?set MODELS_DIR}/${MODEL_DIR:?set MODEL_DIR}"
        "$scripts_dir/resources.py" summary "$LOG_DIR" \
            "model_dir=${MODEL_DIR}" \
            "checkpoint_gib=$(du -sb "$ckpt" | awk '{printf "%.1f", $1 / 1073741824}')" \
            "page_cache_fraction=$("$scripts_dir/resources.py" cached "$ckpt")" \
            "gpu_mem_before_mib=$(gpu_mib)"
    fi
    start=$(date +%s)
    dc up -d "$server"
    deadline=$(( start + ${SERVE_TIMEOUT:-900} ))
    while (( $(date +%s) < deadline )); do
        if healthy; then
            echo "$server ready: ${MODEL_DIR} in $(( $(date +%s) - start ))s"
            if [[ -n "${LOG_DIR:-}" ]]; then
                ctr="$(dc ps -q "$server")"
                mem="$(docker stats --no-stream --format '{{.MemUsage}}' "$ctr" | cut -d/ -f1 | tr -d ' ')"
                "$scripts_dir/resources.py" summary "$LOG_DIR" \
                    "startup_s=$(( $(date +%s) - start ))" \
                    "gpu_mem_after_load_mib=$(gpu_mib)" \
                    "container_mem_after_load=\"${mem}\""
                setsid "$scripts_dir/monitor.sh" "$LOG_DIR" "$ctr" >/dev/null 2>&1 < /dev/null &
                echo $! >"$LOG_DIR/.monitor.pid"
            fi
            exit 0
        fi
        if [[ "$(dc ps -a --format '{{.State}}' "$server")" =~ ^(exited|dead)$ ]]; then
            dc logs --tail 50 "$server"
            exit 1
        fi
        sleep 2
    done
    echo "$server not ready after ${SERVE_TIMEOUT:-900}s" >&2
    dc logs --tail 50 "$server"
    exit 1
    ;;
down)
    if [[ -n "${LOG_DIR:-}" ]]; then
        mkdir -p "$LOG_DIR"
        if [[ -f "$LOG_DIR/.monitor.pid" ]]; then
            kill "$(cat "$LOG_DIR/.monitor.pid")" 2>/dev/null || true
            rm -f "$LOG_DIR/.monitor.pid"
            "$scripts_dir/resources.py" summary "$LOG_DIR"
        fi
        # Container IPs are not part of the result.
        dc logs --no-color "$server" 2>&1 | sed -E 's/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+/<client>/g' >"$LOG_DIR/server.log" || true
    fi
    dc rm -sf "$server"
    ;;
*)
    echo "usage: $0 up | down" >&2
    exit 2
    ;;
esac
