#!/bin/bash
# Start the SGLang server and wait until it answers /health, or stop it.
# Usage: scripts/serve.sh up | down
# `up` reads MODELS_DIR, MODEL_DIR, SERVER_ARGS, SPEC_ARGS (see compose/sglang.yaml) and
# SERVE_TIMEOUT (seconds to wait for /health, default 900).
# `down` first saves the server log to $LOG_DIR/server.log when LOG_DIR is set.
set -euo pipefail
. "$(dirname "$0")/env.sh"

case "${1:?up or down}" in
up)
    dc up -d sglang
    deadline=$(( $(date +%s) + ${SERVE_TIMEOUT:-900} ))
    while (( $(date +%s) < deadline )); do
        if dc exec -T sglang curl -sf localhost:8000/health >/dev/null 2>&1; then
            echo "sglang ready: ${MODEL_DIR}"
            exit 0
        fi
        if [[ "$(dc ps -a --format '{{.State}}' sglang)" == exited ]]; then
            dc logs --tail 50 sglang
            exit 1
        fi
        sleep 5
    done
    echo "sglang not ready after ${SERVE_TIMEOUT:-900}s" >&2
    dc logs --tail 50 sglang
    exit 1
    ;;
down)
    if [[ -n "${LOG_DIR:-}" ]]; then
        mkdir -p "$LOG_DIR"
        # Container IPs are not part of the result.
        dc logs --no-color sglang 2>&1 | sed -E 's/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+/<client>/g' >"$LOG_DIR/server.log" || true
    fi
    dc rm -sf sglang
    ;;
*)
    echo "usage: $0 up | down" >&2
    exit 2
    ;;
esac
