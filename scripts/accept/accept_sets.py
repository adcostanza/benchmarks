"""Accept-length sets runner (inside the bench-client container, see scripts/accept-sets.sh).

Prompt formats, sample selection and request fields follow z-lab/dflash dflash/benchmark.py at
commit 07ebd93db9f472af339b644bb70221ad8428328a (DATASETS, _select_dataset, _run_openai,
_reasoning_kwargs), reimplemented here.
"""
import asyncio
import glob
import os
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "speed"))
from specclient import die, run_metrics, run_pool, write_summary  # noqa: E402

BOXED = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DATA = Path("/datasets")
# name -> (HF repo, revision, files glob under /datasets, prompt formatter)
SETS = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
              "gsm8k/main/test-*.parquet", lambda x: x["question"] + BOXED),
    "math500": ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
                "MATH-500/test.jsonl", lambda x: x["problem"] + BOXED),
    "humaneval": ("openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                  "openai_humaneval/openai_humaneval/test-*.parquet",
                  lambda x: "Write a solution to the following problem and make sure that it "
                            "passes the tests:\n```python\n" + x["prompt"] + "\n```"),
    "mbpp": ("google-research-datasets/mbpp", "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
             "mbpp/sanitized/test-*.parquet", lambda x: x["prompt"]),
    # MT-Bench: dflash's OpenAI client sends only the first of the two turns.
    "mt-bench": ("HuggingFaceH4/mt_bench_prompts", "e3a795c5e9a82ee40611c416b8a7786c73198991",
                 "mt_bench_prompts/data/train-*.parquet", lambda x: x["prompt"][0]),
}

env = os.environ.get
out_dir = Path(sys.argv[1])
base_url = "http://" + env("SERVER", "sglang") + ":8000"
model = "/models/" + env("MODEL_DIR", "model")
protocols = ["ref", "daily"] if env("PROTOCOL", "both") == "both" else [env("PROTOCOL")]
concs = [int(c) for c in env("ACCEPT_CONC", "1").split()]
num_prompts = int(env("ACCEPT_NUM_PROMPTS", "128"))
limit = int(env("ACCEPT_LIMIT", "0")) or None
only = env("ACCEPT_SETS", " ".join(SETS)).split()
if any(p not in ("ref", "daily") for p in protocols):
    die("PROTOCOL must be ref, daily or both", 2)
if any(s not in SETS for s in only):
    die(f"ACCEPT_SETS must be among {' '.join(SETS)}", 2)

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
MAX_TOKENS = 4096
CHAT_KWARGS = {
    # dflash _reasoning_kwargs("xhigh") without a template sends all three keys.
    "ref": {"enable_thinking": True, "reasoning_effort": "xhigh", "reasoning_strength": "xhigh"},
    "daily": {"reasoning_effort": "medium"},
}


def load(name):
    repo, rev, pattern, fmt = SETS[name]
    files = sorted(glob.glob(str(DATA / pattern)))
    if len(files) != 1:
        die(f"{name}: expected one file for {pattern}, found {files}")
    if files[0].endswith(".jsonl"):
        import json
        rows = [json.loads(line) for line in open(files[0], encoding="utf-8") if line.strip()]
    else:
        rows = pq.read_table(files[0]).to_pylist()
    return [fmt(r) for r in rows]


def select(prompts, count):
    """dflash _select_dataset(..., repeat=True): shuffle indices with Random(42), cycle."""
    order = list(range(len(prompts)))
    random.Random(42).shuffle(order)
    return [prompts[order[i % len(order)]] for i in range(count)]


def body_for(protocol, max_tokens):
    return lambda messages: {"model": model, "messages": messages, "max_tokens": max_tokens,
                             **SAMPLING, "chat_template_kwargs": CHAT_KWARGS[protocol]}


async def main():
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for protocol in protocols:
        for conc in concs:
            for name in only:
                prompts = select(load(name), num_prompts + conc)
                measured = [{"id": f"{name}-{i}", "category": name, "turns": [p]}
                            for i, p in enumerate(prompts[:num_prompts])]
                if limit:
                    measured = measured[:limit]
                warm = [{"id": f"{name}-warm{i}", "turns": [p]}
                        for i, p in enumerate(prompts[num_prompts:])]
                print(f"=== {protocol} {name} c{conc}: {len(measured)} prompts", flush=True)
                # dflash warms up with `concurrency` requests of at most 64 new tokens.
                await run_pool(base_url, warm, conc, body_for(protocol, 64), os.devnull)
                part = out_dir / f"requests_{protocol}_{name}_c{conc}.jsonl"
                part.unlink(missing_ok=True)
                records, wall = await run_pool(base_url, measured, conc,
                                               body_for(protocol, MAX_TOKENS), part)
                metrics, stderr = run_metrics(records, wall)
                runs.append({"protocol": protocol, "n": len(records), "metrics": metrics,
                             "stderr": stderr,
                             "settings": {"set": name, "concurrency": conc, "prompts": len(measured),
                                          "max_tokens": MAX_TOKENS, **SAMPLING,
                                          "chat_template_kwargs": CHAT_KWARGS[protocol],
                                          "shuffle_seed": 42}})
                print(f"    {metrics['output_tok_s']:.1f} tok/s, "
                      f"accept {metrics.get('accept_length', 'n/a')}", flush=True)
    dataset = {"repo": "z-lab/dflash", "revision": "07ebd93db9f472af339b644bb70221ad8428328a",
               "sets": {n: {"repo": SETS[n][0], "revision": SETS[n][1]} for n in only}}
    write_summary(out_dir, "accept-sets", dataset, runs)


asyncio.run(main())
