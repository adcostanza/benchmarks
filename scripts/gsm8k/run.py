#!/usr/bin/env python3
"""GSM8K test split against an OpenAI-compatible chat endpoint (client side of scripts/gsm8k.sh).

Prompt, answer extraction and grading are sgl-eval's (the tool behind the RadixArk card's
GSM8K number): NeMo-Skills' generic/math prompt (zero-shot, answer in \\boxed{}), the last
\\boxed{...} of the message content as the prediction, and math_verify-based equivalence.
Only the dataset source differs: the parquet file instead of sgl-eval's download of the
original grade-school-math JSONL (same 1,319 questions and ####-answers).

Writes <out>/gsm8k/<protocol>.jsonl (one row per question) and <out>/gsm8k/summary.json.
Exits non-zero on any failed request.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import sgl_eval
import yaml
from sgl_eval._vendored.nemo_skills.math_grader import extract_answer, math_equal

SUITE_VERSION = 1
DATASET = {"repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866"}
PROMPT_YAML = Path(sgl_eval.__file__).parent / "_vendored/nemo_skills/prompts/math.yaml"
RETRIES = 3

# Request fields per protocol. None = field not sent (server/model default).
PROTOCOLS = {
    # RadixArk NVFP4 card: thinking mode, temperature 1.0, top_p 0.95, top_k 20, sgl-eval.
    # sgl-eval also sends min_p 0 and repetition_penalty 1 on every request and no max_tokens
    # (server limit) unless told otherwise; the thinking switch it sets is
    # chat_template_kwargs.thinking, the Qwen template reads enable_thinking, so both are sent.
    "ref": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"thinking": True, "enable_thinking": True},
    },
    # Our daily setting: reasoning_effort medium, the model's own sampling defaults.
    "daily": {"chat_template_kwargs": {"reasoning_effort": "medium"}},
}


def render_prompt(problem):
    user = yaml.safe_load(PROMPT_YAML.read_text())["user"]
    return user.format(problem=problem, examples="")


def load_cases(path, limit):
    table = pq.read_table(path).to_pylist()
    cases = []
    for i, row in enumerate(table[:limit] if limit else table):
        # sgl-eval's prepare.py: text after '####', commas removed, int when integral.
        gold = float(row["answer"].split("####")[1].strip().replace(",", ""))
        cases.append({"idx": i, "question": row["question"], "gold": int(gold) if gold == int(gold) else gold})
    return cases


def post(url, body):
    """POST JSON, retry connection errors and 5xx/429; any other HTTP error is fatal."""
    data = json.dumps(body).encode()
    for attempt in range(RETRIES):
        req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=14400) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == RETRIES - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if attempt == RETRIES - 1:
                raise RuntimeError(f"request failed: {e}") from e
        time.sleep(2 ** attempt)


def generate(case, base_url, model, fields, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": render_prompt(case["question"])}], **fields}
    if max_tokens:
        body["max_tokens"] = max_tokens
    start = time.time()
    resp = post(f"{base_url}/v1/chat/completions", body)
    latency = time.time() - start
    choice = resp["choices"][0]
    msg = choice["message"]
    usage = resp.get("usage") or {}
    if usage.get("completion_tokens") is None:
        raise RuntimeError(f"case {case['idx']}: response has no usage.completion_tokens")
    return {
        "content": msg.get("content") or "",
        "reasoning_content": msg.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage["completion_tokens"],
        "prompt_tokens": usage.get("prompt_tokens"),
        "latency_s": round(latency, 3),
    }


def run_protocol(name, cases, args, model, out_dir):
    fields = PROTOCOLS[name]
    max_tokens = args.ref_max_tokens if name == "ref" else args.daily_max_tokens
    settings = {
        "model": model,
        "prompt": "sgl-eval generic/math, zero-shot, answer in \\boxed{}",
        "extraction": "last \\boxed{} of message content; sgl-eval math_equal (math_verify)",
        "request_fields": fields,
        "max_tokens": max_tokens or "unset (server limit)",
        "concurrency": args.concurrency,
        "seed": None,
    }
    rows = []
    path = out_dir / f"{name}.jsonl"
    stop = threading.Event()
    tic = time.time()
    with open(path, "w") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        def work(case):
            if stop.is_set():
                raise RuntimeError("aborted")
            return generate(case, args.base_url, model, fields, max_tokens)

        futures = {pool.submit(work, c): c for c in cases}
        done = 0
        try:
            # Grading stays on the main thread: math_verify's timeout uses SIGALRM.
            for fut in as_completed(futures):
                case = futures[fut]
                gen = fut.result()
                pred = extract_answer(gen["content"])
                correct = bool(math_equal(case["gold"], pred, numeric_precision=15, timeout_seconds=10))
                row = {**case, "protocol": name, "extracted": pred, "correct": correct, **gen}
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
                done += 1
                if done % 100 == 0 or done == len(cases):
                    ok = sum(r["correct"] for r in rows)
                    print(f"gsm8k {name}: {done}/{len(cases)} acc {ok / done:.4f}", flush=True)
        except BaseException as e:
            print(f"gsm8k {name}: failed: {e}", file=sys.stderr, flush=True)
            stop.set()
            fh.flush()
            os._exit(1)
    wall = time.time() - tic

    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    p = correct / n
    metrics = {
        "accuracy": p,
        "correct": correct,
        "truncated": sum(r["finish_reason"] == "length" for r in rows),
        "no_answer": sum(r["extracted"] is None for r in rows),
        "mean_output_tokens": sum(r["completion_tokens"] for r in rows) / n,
        "mean_request_seconds": sum(r["latency_s"] for r in rows) / n,
        "wall_seconds": round(wall, 3),
        "seconds_per_question": round(wall / n, 4),
    }
    return {
        "protocol": name,
        "n": n,
        "metrics": metrics,
        "stderr": {"accuracy": math.sqrt(p * (1 - p) / n)},
        "settings": settings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocols", required=True, help="comma-separated: ref, daily")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--ref-max-tokens", type=int, default=0, help="0 = do not send max_tokens")
    ap.add_argument("--daily-max-tokens", type=int, required=True)
    ap.add_argument("--data", default="/datasets/gsm8k/main/test-00000-of-00001.parquet")
    args = ap.parse_args()

    out_dir = Path(args.out) / "gsm8k"
    out_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{args.base_url}/v1/models", timeout=30) as resp:
        model = json.load(resp)["data"][0]["id"]
    cases = load_cases(args.data, args.limit)
    runs = [run_protocol(name, cases, args, model, out_dir) for name in args.protocols.split(",")]
    summary = {"benchmark": "gsm8k", "suite_version": SUITE_VERSION, "dataset": DATASET, "runs": runs}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in runs:
        m = r["metrics"]
        print(f"gsm8k {r['protocol']}: {m['correct']}/{r['n']} = {m['accuracy']:.4f} "
              f"+- {r['stderr']['accuracy']:.4f}, truncated {m['truncated']}, "
              f"{m['mean_output_tokens']:.0f} tok/q, {m['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
