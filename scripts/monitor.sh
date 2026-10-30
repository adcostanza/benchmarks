#!/bin/bash
# Samples GPU memory, the sglang container's memory, host memory and host memory pressure
# (PSI "some"/"full" avg10 from /proc/pressure/memory, in %, empty without PSI) every 2 s into
# <run-dir>/resources.csv until killed. Started and stopped by serve.sh.
# Usage: scripts/monitor.sh <run-dir> <container>
set -uo pipefail
dir="${1:?run dir required}"
ctr="${2:?container required}"
csv="$dir/resources.csv"
echo "t,gpu_mem_mib,container_mem_gib,host_mem_used_gib,mem_psi_some_avg10,mem_psi_full_avg10" >"$csv"
to_gib() {
    awk -v v="$1" 'BEGIN {
        n = v + 0; u = v; sub(/^[0-9.]+/, "", u)
        f = (u == "GiB") ? 1 : (u == "MiB") ? 1/1024 : (u == "KiB") ? 1/1048576 : (u == "TiB") ? 1024 : 1/1073741824
        printf "%.3f", n * f }'
}
while [[ "$(docker inspect -f '{{.State.Running}}' "$ctr" 2>/dev/null)" == true ]]; do
    gpu="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    mem="$(docker stats --no-stream --format '{{.MemUsage}}' "$ctr" 2>/dev/null | cut -d/ -f1 | tr -d ' ')"
    host="$(awk '/^MemTotal/ {t=$2} /^MemAvailable/ {a=$2} END {printf "%.3f", (t-a)/1048576}' /proc/meminfo)"
    psi="$(awk '{sub(/avg10=/, "", $2); v[$1] = $2} END {print v["some"] "," v["full"]}' /proc/pressure/memory 2>/dev/null || echo ,)"
    echo "$(date +%s),${gpu:-0},$(to_gib "${mem:-0B}"),${host},${psi}" >>"$csv"
    sleep 2
done
