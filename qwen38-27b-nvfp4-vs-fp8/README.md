# Qwen3.8-27B: NVFP4 vs FP8 on SGLang

Post: [RadixArk's NVFP4 Qwen3.8-27B served 32–37% more tokens/s than Qwen's FP8 on SGLang](https://acostanza.com/benchmarks/qwen38-27b-nvfp4-vs-fp8/) (2026-10-28). Hardware: [hardware.md](../hardware.md).

| | FP8 | NVFP4 | Change |
|---|---|---|---|
| Chat, 1 user (tok/s) | 133.8 | 176.3 | +32% |
| Chat, 4 users (tok/s) | 449.9 | 596.5 | +33% |
| 20k prompt, 1 user (tok/s) | 73.3 | 98.5 | +34% |
| 20k prompt, 4 users (tok/s) | 104.1 | 142.3 | +37% |
| Time to first token, 20k, 1 user (s) | 3.20 | 2.34 | −27% |
| Time to first token, 20k, 4 users (s) | 8.48 | 5.78 | −32% |
| BFCL subset (of 205), one run | 167 | 174 | +7 |
| BFCL mean latency per call (544 vs 529 calls; different multi-turn paths, not a controlled speed comparison) | 1.22 s | 0.92 s | −24% |

Full tables: `../scripts/compare.py results/stock-fp8 results/stock-nvfp4`.

## Configs

| Label | Checkpoint | Revision | `--mem-fraction-static` | KV pool (tokens) |
|---|---|---|---|---|
| `stock-fp8` | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | 0.50 | 240,512 |
| `stock-nvfp4` | [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) | `319f741cce68d7914884900c138a1fbb70a42f30` | 0.41 | 236,864 |

Both use an FP8 E4M3 KV cache (`--kv-cache-dtype fp8_e4m3`) and DFlash2 speculative decoding with 8 draft tokens: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Image `lmsysorg/sglang:nightly-dev-cu13-20261010-961bcf48`. Server flags are in `scripts/args/qwen38-27b.sh` on main (`compose/sglang.yaml` at `d4d7869`); each line of `bench-serving.jsonl` also records the full server args.

Download:

```sh
hf download Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a --local-dir "$MODELS_DIR/Qwen3.8-27B-FP8"
hf download RadixArk/Qwen3.8-27B-NVFP4 --revision 319f741cce68d7914884900c138a1fbb70a42f30 --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4"
hf download z-lab/Qwen3.8-27B-DFlash2 --revision 50307d4c4cde6860d4eee73e2547cd786fe8e8a4 --local-dir "$MODELS_DIR/Qwen3.8-27B-DFlash2"
```

Run: `./run.sh` (both configs) or `./run.sh stock-nvfp4`.

## Results

`results/<label>/`:

- `bench-serving.jsonl`: one JSON line per scenario (`sharegpt_c1`, `sharegpt_c4`, `long20k_c1`, `long20k_c4`).
- `bfcl/score/`: BFCL accuracy per category; `bfcl/result/`: every model response.

One run per config. The 20k scenarios use SGLang's `random` dataset (one ShareGPT prompt repeated to length), not real prompts. Speed requests use `reasoning_effort=medium`; BFCL runs the `Qwen/Qwen3-32B-FC` handler at its default effort, temperature 0.001. BFCL categories have 10–25 cases; 167 vs 174 of 205 in one run does not establish a gain or equivalence. One NVFP4 case (`multi_turn_base_60`) hit an inference error. Tokens/s is aggregate output tokens per benchmark second, prefill included. Results are for these two checkpoints on this stack, not NVFP4 or FP8 in general.
