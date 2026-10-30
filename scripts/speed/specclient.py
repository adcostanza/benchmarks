"""Shared pieces of speed-bench.sh and accept-sets.sh (run inside the bench-client container):
a streaming chat client that reads SGLang's per-request speculative-decoding stats, a
fixed-concurrency request pool with raw JSONL output, and summary statistics.

Accept length is the one SGLang computes per request and returns when the request sets
return_spec_tokens_details=true: usage completion tokens / verification steps, as
sglext.spec_tokens_details.{spec_accept_length,spec_verify_ct} on the last streamed chunk
(tokenizer_manager.py, "meta_info["spec_accept_length"] = completion_tokens / spec_verify_ct",
and serving_chat.py, SglExt, in image lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48).
A server without speculative decoding sends no such object; then the metric is missing, never 1.0.
"""
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path

import aiohttp

SUITE_VERSION = 1


def die(msg, code=1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def spec_details(ev):
    """The spec_tokens_details dict of one streamed chunk or response body, or None."""
    d = (ev.get("sglext") or {}).get("spec_tokens_details")
    if isinstance(d, list):
        d = d[0] if d else None
    return d or None


async def chat(session, url, body, timeout_s):
    """One streaming chat completion. Raises on any HTTP/stream error or missing usage."""
    body = dict(body, stream=True, stream_options={"include_usage": True},
                return_spec_tokens_details=True)
    t0 = time.perf_counter()
    ttft = None
    content, reasoning = [], []
    usage = spec = finish = None
    async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
        async for raw in r.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("error"):
                raise RuntimeError(f"stream error: {ev['error']}")
            if ev.get("usage"):
                usage = ev["usage"]
            spec = spec_details(ev) or spec
            for ch in ev.get("choices") or []:
                delta = ch.get("delta") or {}
                c = delta.get("content") or ""
                rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if c or rc:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    (content if c else reasoning).append(c or rc)
                finish = ch.get("finish_reason") or finish
    latency = time.perf_counter() - t0
    if usage is None or not usage.get("completion_tokens"):
        raise RuntimeError("response without usage or with zero completion tokens")
    if finish not in ("stop", "length"):
        raise RuntimeError(f"unexpected finish_reason {finish!r}")
    out = {
        "latency_s": latency, "ttft_s": ttft,
        "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
        "finish_reason": finish, "content": "".join(content), "reasoning": "".join(reasoning),
    }
    if spec and spec.get("spec_verify_ct"):
        out["spec_accept_length"] = spec["spec_accept_length"]
        out["spec_verify_ct"] = spec["spec_verify_ct"]
        out["spec_accept_rate"] = spec.get("spec_accept_rate")
    return out


async def post_ok(session, url, what):
    async with session.post(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
        if r.status != 200:
            raise RuntimeError(f"{what}: HTTP {r.status}")


async def run_pool(base_url, items, concurrency, make_body, out_file, timeout_s=3600,
                   flush_cache=True):
    """Run items (dicts with id, category, turns: [user text, ...]) at fixed concurrency.
    Turns of one item run in order, each fed the earlier answers (content only) as history.
    Every finished request is appended to out_file as one JSON line. Stops at the first failure.
    Returns (records, wall_seconds); records are in completion order."""
    records = []
    sem = asyncio.Semaphore(concurrency)
    failure = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        if flush_cache:
            # SGLang answers 400 while the previous run's last request is still being closed out.
            for attempt in range(10):
                try:
                    await post_ok(session, base_url + "/flush_cache", "flush_cache")
                    break
                except RuntimeError:
                    if attempt == 9:
                        raise
                    await asyncio.sleep(2)
        with open(out_file, "a", encoding="utf-8") as f:

            async def conversation(item):
                async with sem:
                    if failure:
                        return
                    messages = []
                    for turn, text in enumerate(item["turns"]):
                        messages.append({"role": "user", "content": text})
                        try:
                            res = await chat(session, base_url + "/v1/chat/completions",
                                             make_body(messages), timeout_s)
                        except Exception as e:  # noqa: BLE001 - reported, then run stops
                            failure.append(f"{item['id']} turn {turn}: {type(e).__name__}: {e}")
                            return
                        rec = {"id": item["id"], "category": item.get("category"), "turn": turn,
                               **res}
                        records.append(rec)
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                        messages.append({"role": "assistant", "content": res["content"]})

            t0 = time.perf_counter()
            await asyncio.gather(*(conversation(i) for i in items))
            wall = time.perf_counter() - t0
    if failure:
        die(f"request failed: {failure[0]}")
    expected = sum(len(i["turns"]) for i in items)
    if len(records) != expected:
        die(f"{len(records)}/{expected} requests completed")
    return records, wall


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean_sem(xs):
    m = statistics.fmean(xs)
    sem = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return m, sem


def run_metrics(records, wall, categories=False):
    """metrics, stderr dicts for one pool run. Latency metrics in ms, throughput in tokens/s."""
    metrics, stderr = {}, {}
    toks = sum(r["completion_tokens"] for r in records)
    metrics["output_tok_s"] = toks / wall
    metrics["completion_tokens_total"] = toks
    metrics["wall_s"] = wall
    metrics["completion_tokens_mean"] = toks / len(records)
    metrics["truncated"] = sum(r["finish_reason"] == "length" for r in records) / len(records)
    ttft = [r["ttft_s"] * 1000 for r in records]
    tpot = [(r["latency_s"] - r["ttft_s"]) * 1000 / (r["completion_tokens"] - 1)
            for r in records if r["completion_tokens"] > 1]
    metrics["ttft_p50_ms"] = percentile(ttft, 0.5)
    if tpot:
        metrics["tpot_p50_ms"] = percentile(tpot, 0.5)

    def accept(prefix, rs):
        al = [r["spec_accept_length"] for r in rs if "spec_accept_length" in r]
        if not al:
            return
        metrics[prefix], stderr[prefix] = mean_sem(al)
        metrics[prefix + "_n"] = len(al)
        ws = [r for r in rs if "spec_verify_ct" in r]
        metrics[prefix + "_token_weighted"] = (
            sum(r["completion_tokens"] for r in ws) / sum(r["spec_verify_ct"] for r in ws))

    accept("accept_length", records)
    if categories:
        for cat in sorted({r["category"] for r in records}):
            accept(f"accept_length.{cat}", [r for r in records if r["category"] == cat])
    return metrics, stderr


def write_summary(out_dir, benchmark, dataset, runs):
    path = Path(out_dir) / "summary.json"
    path.write_text(json.dumps({"benchmark": benchmark, "suite_version": SUITE_VERSION,
                                "dataset": dataset, "runs": runs}, indent=2) + "\n")
    print(f"wrote {path}")
