"""SPEED-Bench runner (inside the bench-client container, see scripts/speed-bench.sh)."""
import asyncio
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from specclient import die, run_metrics, run_pool, write_summary  # noqa: E402

PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED"
RESOLVED = Path("/datasets/SPEED-Bench-resolved")
REPO, REVISION = "nvidia/SPEED-Bench", "fbd6dc64353a3153bf24252a113ebd6d2cff9dec"
RESOLVER_COMMIT = "2802302bf93d7fb0a6d6809d77ed4a1bb99affdc"

env = os.environ.get
out_dir = Path(sys.argv[1])
base_url = "http://" + env("SERVER", "sglang") + ":8000"
model = "/models/" + env("MODEL_DIR", "model")
concs = [int(c) for c in env("SPEEDBENCH_CONC", "1 4").split()]
per_cat = int(env("SPEEDBENCH_QUAL_PER_CAT", "20"))
per_tier = int(env("SPEEDBENCH_TP_PER_TIER", "8"))
buckets = env("SPEEDBENCH_TP_BUCKETS", "1k 2k 8k 16k 32k").split()
qual_max = int(env("SPEEDBENCH_QUAL_MAX_TOKENS", "4096"))
tp_max = int(env("SPEEDBENCH_TP_MAX_TOKENS", "512"))
limit = int(env("SPEEDBENCH_LIMIT", "0")) or None
effort = "medium"


def load(config):
    path = RESOLVED / config / "test.parquet"
    if not path.exists():
        die(f"{path} missing: run scripts/speed/prepare.sh once (resolves the masked prompts)")
    rows = pq.read_table(path).to_pylist()
    bad = [r["question_id"] for r in rows if any(t.startswith(PLACEHOLDER) for t in r["turns"])]
    if bad:
        die(f"{config}: {len(bad)} rows still hold the placeholder (e.g. {bad[0]}); rerun prepare")
    return rows


def pick(rows, k):
    """First k rows of every category, in dataset order."""
    seen, out = {}, []
    for r in rows:
        if seen.get(r["category"], 0) < k:
            seen[r["category"]] = seen.get(r["category"], 0) + 1
            out.append({"id": r["question_id"], "category": r["category"], "turns": list(r["turns"])})
    return out[:limit] if limit else out


def body_for(max_tokens):
    # Daily protocol: reasoning_effort=medium, the server's default sampling (model generation
    # config), no sampling parameters sent.
    return lambda messages: {"model": model, "messages": messages, "max_tokens": max_tokens,
                             "chat_template_kwargs": {"reasoning_effort": effort}}


async def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    # Two short requests so one-time server work is not charged to the first measured run.
    warm = pick(load("qualitative"), 1)[:2]
    await run_pool(base_url, warm, 1, body_for(64), os.devnull)
    runs = []

    async def one(split, config, items, conc, max_tokens, categories):
        print(f"=== {split} c{conc}: {len(items)} prompts", flush=True)
        tag = f"{split}_c{conc}"
        part = out_dir / f"requests_{tag}.jsonl"
        part.unlink(missing_ok=True)
        records, wall = await run_pool(base_url, items, conc, body_for(max_tokens), part)
        metrics, stderr = run_metrics(records, wall, categories)
        runs.append({"protocol": "daily", "n": len(records), "metrics": metrics, "stderr": stderr,
                     "settings": {"split": split, "config": config, "concurrency": conc,
                                  "prompts": len(items), "max_tokens": max_tokens,
                                  "reasoning_effort": effort, "sampling": "server default",
                                  "per_category": per_cat if categories else per_tier}})
        print(f"    {metrics['output_tok_s']:.1f} tok/s, ttft p50 {metrics['ttft_p50_ms']:.0f} ms, "
              f"accept {metrics.get('accept_length', 'n/a')}", flush=True)

    qual = pick(load("qualitative"), per_cat)
    for conc in concs:
        await one("qualitative", "qualitative", qual, conc, qual_max, True)
    for bucket in buckets:
        items = pick(load(f"throughput_{bucket}"), per_tier)
        for conc in concs:
            await one(f"throughput_{bucket}", f"throughput_{bucket}", items, conc, tp_max, True)

    write_summary(out_dir, "speed-bench",
                  {"repo": REPO, "revision": REVISION, "resolver_commit": RESOLVER_COMMIT}, runs)


asyncio.run(main())
