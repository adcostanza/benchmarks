# Qwen3.8-Flash-Next vs Qwen3.8-27B on one RTX PRO 6000

Question: is Qwen3.8-Flash-Next a better daily driver than the 27B NVFP4 + DFlash2 8 (the daily setup when these runs were made; since 2026-10-30: Swift 1.5 NVFP4 + DFlash2 8) on one 96 GB
card, at the whole card and at the daily driver's 44 GB GPU budget? Answer: no. Hardware:
[hardware.md](../hardware.md) (400 W). No more Flash-Next 3.8 runs are planned; the tables below
are final, including the partial llama.cpp row.

The 27B numbers come from [../qwen38-27b-spec-decode](../qwen38-27b-spec-decode/) (same harness,
scenarios and BFCL subset); the 27B is not rerun here.

## Whole card (one run per config)

Output tok/s, median time to first token in parentheses:

| Scenario | 27B NVFP4, no drafter | 27B NVFP4 + DFlash2 8 | Flash-Next NVFP4, no drafter (`fn-nospec`) |
|---|---|---|---|
| chat, 1 user | 72.0 | 178.0 | 83.9 (81 ms) |
| chat, 4 users | 233.7 | 592.5 | 305.2 (168 ms) |
| 20k, 1 user | 52.9 (2.33 s) | 98.4 (2.35 s) | 72.6 (1.69 s) |
| 20k, 4 users | 94.7 (5.77 s) | 137.5 (5.73 s) | 134.2 (4.35 s) |
| 40k, 1 user | 39.1 (5.53 s) | 58.7 (5.57 s) | 58.3 (3.41 s) |
| 80k, 1 user | 22.5 (14.7 s) | 28.5 (14.8 s) | 41.4 (6.98 s) |
| 120k, 1 user | 14.2 (27.6 s) | 16.5 (27.8 s) | 31.9 (10.6 s) |
| BFCL subset (of 205) | 171 | 176 (other runs of the same config: 168, 174) | 136, see caveats (not used in the post) |

Resources for `fn-nospec` (`results/fn-nospec/resources.json`): startup 159 s with the checkpoint
fully in the page cache (warm); GPU memory 87.7 GB after load, 88.1 GB peak; container RAM 70.0 GiB
after load, 71.9 GiB peak (the n-gram table is ~48 GiB of it); checkpoint 126.0 GiB on disk.

## 44 GB GPU budget (partial run)

The daily driver (27B NVFP4 + DFlash2 8, `--mem-fraction-static 0.41`) uses 43.8 GB of GPU memory
after load, 44.0 GB peak (`../qwen38-27b-swift-vs-stock/results/dflash8/resources.json`). SGLang won't start Flash-Next at that budget: `--ple-offload-embedding` is
rejected together with `--cpu-offload-gb` and `--offload-group-size`
(`srt/arg_groups/memory_hook.py` in the image), and KTransformers expert offload has no `qwen4_exp`
support ([ktransformers#2179](https://github.com/kvcache-ai/ktransformers/issues/2179), open on
2026-10-29). So `fn-gguf-q4` runs on llama.cpp with the expert weights of 30 of 48 layers in host
RAM. Output tok/s, median time to first token in parentheses:

| Scenario | 27B NVFP4 + DFlash2 8 | Flash-Next UD-Q4_K_XL, llama.cpp (`fn-gguf-q4`) |
|---|---|---|
| chat, 1 user | 178.0 (48 ms) | 30.2 (1.11 s) |
| chat, 4 users | 592.5 (73 ms) | 37.2 (4.40 s) |
| 20k, 1 user | 98.4 (2.35 s) | 19.3 (9.81 s) |
| 20k, 4 users | 137.5 (5.73 s) | 21.6 (15.9 s) |
| 40k, 1 user | 58.7 (5.57 s) | 14.2 (19.7 s) |
| 80k, 1 user | 28.5 (14.8 s) | failed, see caveats |

Resources (`results/fn-gguf-q4/resources.json`): startup 32 s (checkpoint 92.5% in the page cache);
GPU memory 44.3 GB after load, 44.5 GB peak; container RAM 47.3 GiB after load, 64.0 GiB peak;
checkpoint 103.7 GiB on disk.

## Configs

- Checkpoint: [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) @ `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- Image: `lmsysorg/sglang:nightly-dev-cu13-20261029-c10e0e17`. `server.env` sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`.
- Flags: [../scripts/args/qwen38-flash-next.sh](../scripts/args/qwen38-flash-next.sh). The 51B
  n-gram (PLE) table sits in pinned host RAM (`--ple-offload-embedding`), as in the SGLang cookbook's
  RTX PRO 6000 recipe; prefill chunk 2048, prefill CUDA graph off, context 262,144,
  4 running requests, 12 mamba slots, FP8 KV cache.
- `fn-nospec`: `--mem-fraction-static 0.88`, no drafter. KV pool 611,520 tokens, 12 mamba slots
  (`results/fn-nospec/server.log`).
- `fn-mtp`, `fn-dflash2` (in `run.sh`, never run at 0.88, and not planned): native MTP and
  [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) @
  `a22a1744e85c17364dcc54a6a420af7135d4c7f6`, both at 4 draft tokens (stock SGLang caps the
  Flash-Next block at 4). DFLASH needs the capture patch from that repo (`dflash2.Dockerfile`).
- `fn-gguf-q4`: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
  @ `766911a6b7369840a91dbcd95f9f997acaab6cd6`, `UD-Q4_K_XL`, on
  `ghcr.io/ggml-org/llama.cpp@sha256:0283744a35a6e7698b4f7a2d63360233dc769ece51e84bdba7aa187a33978276`
  (b11977). Flags: [../scripts/args/qwen38-flash-next-gguf.sh](../scripts/args/qwen38-flash-next-gguf.sh)
  with `--n-cpu-moe 30`, context 153,600 (the daily driver's), 4 slots, unified KV. The benchmark
  client runs in the SGLang image with `--backend vllm-chat` and the NVFP4 tokenizer; each request
  sends `cache_prompt: false` (llama.cpp has no cache-flush endpoint).

## Caveats

- **BFCL 136/205 is not comparable with the 27B rows.** `parallel` scored 4/20 (27B: 18–19/20),
  `parallel_multiple` 10/20. The scorer isn't at fault: none of the 16 failed `parallel` responses in
  `results/fn-nospec/bfcl/result/` has two or more `<tool_call>` blocks; the model made one call
  or answered in prose. BFCL's `Qwen/Qwen3-32B-FC` handler builds its own prompt instead of the
  model's chat template, so this says how Flash-Next does under that prompt, not with its own
  template. Left out of the post.
- **`fn-gguf-q4` is partial.** 80k failed in warmup: the unified KV pool (153,600 tokens) ran out
  because slot 0 still held ~80k tokens from the first warmup request when the second started; the
  client then failed on a non-JSON stream chunk. Two back-to-back 78k requests on a fresh server
  worked, so it depends on state from earlier scenarios. Not fixed. 120k and BFCL scoring never ran
  (`bfcl/result/` has the generated responses, unscored).
- Scenarios of 20k tokens and longer use SGLang's `random` dataset: one ShareGPT prompt repeated to
  length. The speed is real; the content isn't a real long document.
- Startup is warm (`page_cache_fraction` 1.0). A cold start reads 126 GiB from NVMe.
- `attempts/2026-10-29-fn-mtp-mem0.96/`: the first attempt (MTP, `--mem-fraction-static 0.96`,
  prefill chunk 4096) served chat at 163.3 / 423.0 tok/s, then ran out of GPU memory on the first
  20k-token prefill.
