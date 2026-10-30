# Sourced by the other scripts and by each benchmark's run.sh.
#   BENCH_PROJECT  compose project name (default: bench)
#   BENCH_CDI=1    request the GPU through CDI (rootless Docker), see compose/cdi.yaml
#   ENGINE         server the scripts drive: sglang (default) or llamacpp (compose service name)
scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$scripts_dir")"

dc() {
    local files=(-f "$repo_dir/compose/sglang.yaml")
    [[ "${BENCH_CDI:-0}" == 1 ]] && files+=(-f "$repo_dir/compose/cdi.yaml")
    docker compose -p "${BENCH_PROJECT:-bench}" "${files[@]}" "$@"
}

server="${ENGINE:-sglang}"
case "$server" in
sglang|llamacpp) ;;
*) echo "ENGINE must be sglang or llamacpp, got: $server" >&2; exit 2 ;;
esac

healthy() { dc exec -T "$server" curl -sf -m 10 localhost:8000/health >/dev/null 2>&1; }
