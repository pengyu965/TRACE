# Architecture

```
events.json  +  videos/<id>.mp4
        │
        │  ──────────────  PART 1  (preprocessing)  ─────────────
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  extract_audio            extract_frames                     │
  │  mp4 → wav (parallel)     mp4 → 1-fps jpg (parallel)         │
  │     │                          │                             │
  │     ▼                          ├──► yolo_batch  ──► yolo.jsonl
  │  whisper_asr                   │     (N GPUs)                │
  │  (N GPUs, transcribe           │                             │
  │   + translate)                 └──► hunyuan_ocr ──► ocr.jsonl │
  │     │                                (N GPUs)                │
  │     ▼                                                        │
  │  asr.jsonl                                                   │
  │                                                              │
  │       ─────────────── merge ───────────────                  │
  │                       │                                      │
  │                       ▼                                      │
  │                merged.jsonl  ← Part-2-compatible schema      │
  └──────────────────────────────────────────────────────────────┘
                       │
        │  ──────────────  PART 2  (claim generation)  ──────────
                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  claim_step1                                                 │
  │  Qwen3-30B-A3B-Instruct  (BF16, GPUs 0-3)                    │
  │  per (event, video): chunked LLM relevance filter            │
  │     → claim_guidance/q_<event_slug>_guidance.json            │
  │                                                              │
  │  ── LLM fully unloaded — VRAM flushed before LVLM loads ──   │
  │                                                              │
  │  claim_step2                                                 │
  │  Qwen3-VL-30B-A3B-Instruct (BF16/FP8, GPUs 0-3)              │
  │  per (event, video): full video + ASR + Step-1 guidance      │
  │  frame_mode ∈ {guided, uniform}                              │
  │     → claim_results/<video_id>_results.json                  │
  │     → claim_results/all_results.jsonl                        │
  └──────────────────────────────────────────────────────────────┘
                       │
        │  ──────────────  PART 3  (cross-video aggregation)  ───
                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  aggregate  —  Method A (paper headline)                     │
  │                                                              │
  │   stage 1   split per-video → per-query                      │
  │             (claim_results/*_results.json + events.json      │
  │              -> aggregate_queries.jsonl)                     │
  │   stage 2   Qwen3-Embedding-8B  →  greedy single-link        │
  │             clustering at τ=0.9                              │
  │   stage 3a  Qwen3-30B-Instruct  per-cluster LLM verify       │
  │             (re-prompted up to 3 times on validator failure; │
  │              every attempt persisted under raw_llm/)         │
  │   stage 4   assemble MAGMaR submission JSON, run hard checks │
  │             (verbatim text, video coverage, dedup citations) │
  │   stage 5   markdown diff report (A vs B if both ran)        │
  │                                                              │
  │     → aggregate/submission/penkil_method_a.jsonl  (MAGMaR)   │
  └──────────────────────────────────────────────────────────────┘
                       │
                       ▼
              MAGMaR submission file
```

## Step-by-step

| step | inputs | outputs | GPU? | resumable on |
|---|---|---|---|---|
| `audio`       | `events.json`, `videos/`         | `audio/<slug>/<vid>.wav`         | no  | wav exists & non-empty |
| `frames`      | `events.json`, `videos/`         | `frames/<slug>/<vid>/frame_*.jpg`| no  | folder has frames |
| `asr`         | `audio/`, `events.json`          | `asr.jsonl`, `asr/<slug>/<vid>.json` | yes (1+ GPU) | `video_id` in jsonl |
| `yolo`        | `frames/`                        | `yolo.jsonl`, `yolo_frames/...`  | yes | `video_id` in jsonl |
| `ocr`         | `frames/`                        | `ocr.jsonl`, `ocr_frames/...`    | yes | `video_id` in jsonl |
| `merge`       | all `.jsonl` + `events.json`     | `merged.jsonl`                   | no  | always overwrites |
| `claim_step1` | `merged.jsonl`, `events.json`    | `claim_guidance/q_<slug>_guidance.json`, `all_guidances.json` | yes (4 GPUs recommended) | per-event guidance JSON exists |
| `claim_step2` | guidance, `merged.jsonl`, video files, `asr.jsonl` | `claim_results/<vid>_results.json`, `all_results.jsonl` | yes (4 GPUs recommended) | (always reruns) |
| `aggregate`   | `claim_results/*.json`, `events.json` | `aggregate/per_query/`, `clusters/`, `agent_io/`, `raw_llm/`, `submission/penkil_method_a.jsonl` | yes (1+ GPU; H100 80 GB fits TP=1) | (always reruns; LLM traces under `raw_llm/` allow no-GPU `revalidate`) |

Every step except `merge`, `claim_step2`, and `aggregate` is fully resumable;
those three rebuild their aggregate output from scratch each run.

## Parallelism and GPU layout

| stage | model | precision | per-GPU VRAM | GPU shard |
|---|---|---|---|---|
| `whisper_asr`      | faster-whisper large-v3       | float16 | ~5 GB  | video-level shard over N GPUs |
| `yolo_batch`       | yolo12x.pt                    | fp32    | ~3 GB  | video-level shard over N GPUs |
| `hunyuan_ocr`      | tencent/HunyuanOCR 1B         | bf16    | ~6 GB  | video-level shard over N GPUs |
| `claim_step1`      | Qwen3-30B-A3B-Instruct-2507   | bf16    | ~60 GB total (vLLM tp=4)       | 4 GPUs |
| `claim_step2`      | Qwen3-VL-30B-A3B-Instruct-FP8 | fp8/auto| ~30 GB total (vLLM tp=4)       | 4 GPUs |
| `aggregate` stage2 | Qwen3-Embedding-8B            | fp16    | ~17 GB                          | 1 GPU |
| `aggregate` stage3 | Qwen3-30B-A3B-Instruct-2507   | bf16    | ~60 GB on single GPU, less per-card with tp>1 | configurable via `--aggregator-tensor-parallel` |

Claim Step 1's LLM is fully unloaded before Step 2's LVLM is loaded. The
aggregator likewise loads the embedder for stage 2, releases it, then loads
the LLM once for stage 3 (shared across Method A and Method B if both run).

## Resumability — how

Each per-step JSONL or per-event JSON is the source of truth for "what's
done". Re-running any step:

1. Scans the existing artifacts.
2. Collects already-done IDs.
3. Skips those, processes the rest.
4. Appends new rows (Part 1) or per-event files (claim_step1).

`claim_step2` and `aggregate` are the exceptions — they currently rebuild
from scratch on every run. For partial reruns of Part 2, restrict events via
`--event-keys`. For partial reruns of Part 3 without re-invoking the LLM,
use `python -m pipeline.aggregator.cli revalidate ...` which replays the
validators against the saved `raw_llm/` traces.

## Crash tolerance

Part-1 multi-GPU steps (asr, yolo, ocr) use a polling pattern instead of
blocking on `q.get()` so a CUDA OOM on one GPU drops that shard but the
other GPU's results are still written.

Part-2 claim steps use vLLM tensor-parallel internally; a GPU dying mid-run
will surface the error and the whole step bails. Per-event guidance cache
from Step 1 is preserved.

Part 3 persists every LLM attempt (prompt + raw output + parse / validation
error) to `aggregate/raw_llm/method_*/query_<qid>.json`. If validators or
the prompt template change later, `revalidate` rebuilds outputs without a
GPU. On generation-time validator failure the model is re-prompted up to
`--aggregator-max-retries` times (default 3); a safe singleton fallback is
used as a last resort so a submission file always materialises.

## What this pipeline does NOT do

- No Qwen-Omni ASR. The test-side `04_omni_asr.py` + `05_merge_factcards.py`
  are intentionally left out — too heavy for general use. Whisper-only.
- No baseline / ablation experiments from the original repos (`baseline.py`,
  `baseline_wikivideo.py`) — only the production guided-grounding path.
- No retrieval / external evidence — the only knowledge sources are the
  video, its ASR, YOLO, and OCR.
