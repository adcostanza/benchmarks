"""Resolve the masked SPEED-Bench prompts into complete parquet files (runs in the speed-prep
container, see scripts/speed/prepare.sh).

nvidia/SPEED-Bench ships placeholder text instead of the prompt for rows whose sources do not
permit redistribution (494 of 880 qualitative rows, ~40-50% of each throughput config). NVIDIA's
resolver, examples/specdec_bench/specdec_bench/datasets/speed.py in NVIDIA/Model-Optimizer at the
pinned commit below, refetches them from the public source datasets (revisions pinned in that
file) and pads/truncates the throughput prompts. This script downloads that file and its base
class at the pinned commit, checks their sha256, and calls SPEEDBench.prepare_data for each config.
Loaders are datasets.load_dataset on parquet/JSON sources; no trust_remote_code.

Output: /datasets/SPEED-Bench-resolved/<config>/test.parquet
Usage (inside the container): python /prepare.py [config ...]   (default: all six)
"""
import hashlib
import os
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

COMMIT = "2802302bf93d7fb0a6d6809d77ed4a1bb99affdc"
BASE = ("https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/" + COMMIT +
        "/examples/specdec_bench/specdec_bench/datasets/")
FILES = {
    "base.py": "316f88cd093ce9f02f1fa82a30cb25b73ed9186484f01c6fd99dd65ac0fe9d3c",
    "speed.py": "a6db7cfaceae3b973c41df4e8ceab1f48e51a5c867fc37ad232e287c8d4066e3",
}
CONFIGS = ["qualitative", "throughput_1k", "throughput_2k", "throughput_8k", "throughput_16k",
           "throughput_32k"]

pkg = Path("/tmp/specdec_bench/datasets")
pkg.mkdir(parents=True, exist_ok=True)
for name, want in FILES.items():
    data = urllib.request.urlopen(BASE + name, timeout=60).read()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        sys.exit(f"{name}: sha256 {got} != pinned {want}")
    (pkg / name).write_bytes(data)
(pkg / "__init__.py").write_text("")
(pkg.parent / "__init__.py").write_text("")
sys.path.insert(0, "/tmp")
from specdec_bench.datasets.speed import SPEEDBench  # noqa: E402

# cais/hle is gated and needed by every config; fail now rather than after the first downloads.
req = urllib.request.Request(
    "https://huggingface.co/datasets/cais/hle/resolve/main/data/test-00000-of-00001.parquet",
    headers={"Authorization": "Bearer " + os.environ.get("HF_TOKEN", ""), "Range": "bytes=0-0"})
try:
    urllib.request.urlopen(req, timeout=60).close()
except urllib.error.HTTPError as e:
    sys.exit(f"cais/hle: HTTP {e.code}; accept its terms at https://huggingface.co/datasets/cais/hle "
             "with the account that owns HF_TOKEN")

configs = sys.argv[1:] or CONFIGS
for config in configs:
    if config not in CONFIGS:
        sys.exit(f"unknown config {config}")
    final = Path("/datasets/SPEED-Bench-resolved") / config / "test.parquet"
    if final.exists():
        print(f"{config}: already resolved")
        continue
    print(f"=== {config}", flush=True)
    path = SPEEDBench.prepare_data(output_dir=Path("/tmp/out") / config, config_name=config)
    final.parent.mkdir(parents=True, exist_ok=True)
    path.replace(final) if path.stat().st_dev == final.parent.stat().st_dev else \
        final.write_bytes(path.read_bytes())
    print(f"{config}: wrote {final}", flush=True)
