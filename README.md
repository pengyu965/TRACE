# 🎯 TRACE: Evidence Grounding-Guided Multi-Video Event Understanding and Claim Generation [ACL 2026 MAGMaR]

[Pengyu Yan*](https://scholar.google.com/citations?user=q2QMx5gAAAAJ&hl=en)<sup>1</sup>, [Akhil Gorugantu*](https://scholar.google.com/citations?user=ust_T20AAAAJ&hl=en)<sup>1</sup>, [Mahesh Bhosale](https://bhosalems.github.io/)<sup>1</sup>, [Abdul Wasi](https://scholar.google.com/citations?user=_2friTYAAAAJ&hl=en)<sup>1</sup>, [Vishvesh Trivedi](https://github.com/NerdyVisky)<sup>2</sup>, [David Doermann](https://scholar.google.com/citations?user=RoGOW9AAAAAJ&hl=en)<sup>1</sup>.

<sup>1</sup>**University at Buffalo**  |  <sup>2</sup>**New York University**

<sup>\*</sup> Equal Contribution. Correspondence: pyan4@buffalo.edu

[Paper (arXiv)](https://arxiv.org/abs/2605.16740) · [🤗 MAGMaR dataset](https://huggingface.co/datasets/akhilvssg/TRACE-MAGMaR-2026) · [🤗 WikiVideo dataset](https://huggingface.co/datasets/akhilvssg/TRACE-WikiVideo)

---

## Overview

<p align="center">
  <img src="figures/framework.png" alt="TRACE pipeline overview" width="92%"/>
</p>

Multi-video event understanding demands models that can locate and attribute query-relevant evidence scattered across long, heterogeneous video corpora. Existing large vision–language models (LVLMs) often underperform in this regime because they quickly exhaust their context budget and struggle to precisely localize evidentially important segments, frequently missing dense informational cues such as broadcast graphics, subtitles, and scoreboards. We introduce **TRACE**, an evidence grounding guided framework that follows a **ground-before-reasoning** strategy for multi-video event reasoning. Our approach first builds a structured, text-searchable timeline for each video using OCR and object detection. A text-only LLM then conducts query-aware evidence localization, selecting relevant moments prior to any downstream visual reasoning. The retrieved frames and their grounding summaries are subsequently used to steer LVLM-based claim generation and cross-video citation consolidation. Experiments on MAGMaR 2026 and WikiVideo demonstrate that structured grounding markedly boosts factual completeness and attribution fidelity. On the MAGMaR validation split, TRACE raises macro-average **MiRAGE F1 from 0.705 to 0.811** compared to an unguided Qwen3-VL-30B baseline, with especially strong improvements in **citation recall (0.440 → 0.628)**. The method also attains **state-of-the-art** results on the official MAGMaR 2026 leaderboard.

---

## 🚀 Quick Start

```bash
git clone https://github.com/pengyu965/TRACE.git && cd TRACE
conda create -n trace python=3.13 -y && conda activate trace
pip install -r requirements.txt
sudo apt-get install -y ffmpeg
```

> **Note — HunyuanOCR requires a specific `transformers` build.**
> The OCR model (`tencent/HunyuanOCR`) uses the `hunyuan_vl` architecture, which is absent from standard PyPI releases of `transformers`.
> Install it by `pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4` (This is already included in requirement.txt but in case you need it separately.)

Then prepare data ([📦 Datasets](#-datasets)) and run the pipeline ([🏃 Running TRACE](#-running-trace)).

## 📦 Datasets

We release two companion datasets on the Hugging Face Hub that ship **every artefact produced by the TRACE pipeline** — official inputs, Part-1 preprocessing outputs (ASR + OCR + object detection), Part-2 LVLM claim files, Part-3 aggregation traces, and the final submission JSONLs. Both releases are AGPL-3.0 and mirror the same folder shape, so anything you learn from one transfers directly to the other.

### MAGMaR-2026 — [🤗 `akhilvssg/TRACE-MAGMaR-2026`](https://huggingface.co/datasets/akhilvssg/TRACE-MAGMaR-2026)

Companion artefacts for the **MAGMaR 2026 Oracle Track test set** — the configuration that produces our paper's headline number (Avg MiRAGE F1 = 0.811). Ships both frame-selection variants (`uniform/` ↔ Additional Key Frames ✗, `guided/` ↔ Additional Key Frames ✓) and both Stage-3 methods (`stage3_io/embed_sim/` ↔ Method A, `stage3_io/llm/` ↔ Method B). Includes 90 mp4s packed in `videos.tar` (3.8 GB) and `frames_annotated.tar.gz` (5.4 GB) of YOLO + OCR overlay JPGs, so the pipeline can be reproduced from the release alone.

```bash
hf download akhilvssg/TRACE-MAGMaR-2026 --repo-type dataset \
    --local-dir ./data/TRACE-MAGMaR-2026

# unpack the bundled videos + annotated frames
cd data/TRACE-MAGMaR-2026
tar  -xf videos.tar                    # → videos/<video_id>.mp4
tar -xzf frames_annotated.tar.gz       # → frames_annotated/{yolo,ocr}/...
```

The headline paper result lives at
[`part3_aggregation/guided/submission/penkil_v2_method_a.jsonl`](https://huggingface.co/datasets/akhilvssg/TRACE-MAGMaR-2026/blob/main/part3_aggregation/guided/submission/penkil_v2_method_a.jsonl).

### WikiVideo (MAGMaR-style) — [🤗 `akhilvssg/TRACE-WikiVideo`](https://huggingface.co/datasets/akhilvssg/TRACE-WikiVideo)

A **52-query WikiVideo evaluation set** built specifically for TRACE, converted to the MAGMaR input format so the **exact same pipeline** runs end-to-end on WikiVideo. We start from 56 WikiVideo events, drop the 4 that overlap the MAGMaR 2026 test set, then have an LLM agent generate a `<persona_title, background, query>` triplet per event in the MAGMaR persona-query format. Each generated triplet is scored on a 5-point scale across four axes — `persona_grounding`, `query_answerability`, `article_angle_alignment`, `overall_grounding` — flagged triplets are rewritten and re-scored, and only events with `overall_grounding ≥ 4` survive. The complete audit trail (per-event scores + free-text reasoning) ships under [`inputs/audit/`](https://huggingface.co/datasets/akhilvssg/TRACE-WikiVideo/tree/main/inputs/audit) for full transparency.

Only the `uniform/` variant is shipped (matching the paper's WikiVideo configuration). The raw `.mp4` files are **not** redistributed — download them from the upstream [🤗 `hltcoe/wikivideo`](https://huggingface.co/datasets/hltcoe/wikivideo) release at `data/full/videos/en/`. All `video_id`s match upstream verbatim.

```bash
# our MAGMaR-style queries + Part-1/2/3 artefacts
hf download akhilvssg/TRACE-WikiVideo --repo-type dataset \
    --local-dir ./data/TRACE-WikiVideo

# upstream videos (used as-is, no re-encoding)
hf download hltcoe/wikivideo --repo-type dataset \
    --local-dir ./data/wikivideo
```

### Input format

You supply one JSON file describing the events you want to process, plus a directory of videos. The events file looks like:

```json
{
  "events": [
    {
      "event_key":     "2018_anchorage_earthquake",
      "query":         "What happened in the 2018 Anchorage, Alaska earthquake?",
      "persona_title": "Civil engineer",
      "background":    "I monitor seismic damage assessments and economic and humanitarian impact ...",
      "language":      "english",
      "videos":        ["0vKs4-EZ_D0", "abc123def45"]
    }
  ]
}
```

`persona_title`, `background`, and `language` are **optional** (used only by Part-2 prompts). The pipeline resolves `<videos-dir>/<video_id>.{mp4,mkv,webm,mov,m4v}` (first match wins). See [`examples/events.example.json`](examples/events.example.json) for a worked example.

The two HF releases let you skip Part 1 entirely: their `part1_preprocessing/{yolo,ocr,asr_whisper}.jsonl` files are drop-in replacements for the merged-timeline outputs the pipeline would compute locally, so you can jump straight to Part 2 / Part 3.


---

## 🏃 Running TRACE

Overall, TRACE has three parts showing as follows.
```
events.json + videos/
     │
     ▼
  ┌──────────────────────────────────────────────────────┐
  │  PART 1 — Structured Grounding                       │
  │  YOLOv12 + HunyuanOCR + Whisper large-v3             │
  │  → merged.jsonl (frame-level OCR + objects + ASR)    │
  └──────────────────────────────────────────────────────┘
                      ▼
  ┌──────────────────────────────────────────────────────┐
  │  PART 2 — Grounding-Guided Claim Generation          │
  │     Step 1: Query-Conditioned Evidence Localization. │
  │         (Query+Persona <--> detection and OCR result)│
  │         (Qwen3-30B-A3B-Instruct)                     │
  │      Step 2: Grounding-Guided Claim Generation       │
  │        (uniform + keyframes, Guided Evidence, ASR)   │
  │        (Qwen3-VL-30B-A3B-Instruct)                   │
  │  → claim_results/<video>_results.json                │
  └──────────────────────────────────────────────────────┘
                      ▼
  ┌──────────────────────────────────────────────────────┐
  │  PART 3 — Cross-Video Claim Consolidation            │
  │  Qwen3-Embedding-8B + greedy single-link clustering  │
  │  Qwen3-30B per-cluster LLM verify + canonicalise     │
  └──────────────────────────────────────────────────────┘
```

### Part 1 only — Structured Grounding

```bash
PART=1 INPUT=./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  VIDEOS=/path/to/videos OUT=out/ GPUS=0,1 \
    bash scripts/run_pipeline.sh
```

Equivalent direct invocation of the underlying modules:

```bash
python -m pipeline.extract_audio        --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --videos-dir videos/ --output-dir out/
python -m pipeline.extract_frames       --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --videos-dir videos/ --output-dir out/
python -m pipeline.whisper_asr          --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --output-dir out/ --gpus 0
python -m pipeline.yolo_batch           --output-dir out/ --gpus 0,1
python -m pipeline.hunyuan_ocr_batch    --output-dir out/ --gpus 0,1
python -m pipeline.merge                --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --output-dir out/
```

### Part 2 only — Claim Generation

```bash
PART=2 INPUT=./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  VIDEOS=/path/to/videos OUT=out/ GPUS=0,1,2,3 \
    bash scripts/run_pipeline.sh
```

Or:

```bash
python -m pipeline.claim_gen.run_claim_gen \
    --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --videos-dir videos/ --output-dir out/ \
    --gpus 0,1,2,3 --frame-mode guided
```

`--frame-mode guided` (default) uses the hybrid strategy from §3.3 of the paper — 100 uniform frames augmented with guidance-targeted keyframes. Pass `--frame-mode uniform` to disable the guidance-extra frames (lighter, comparable to the WikiVideo ablation).

### Part 3 only — Cross-Video Consolidation

```bash
PART=3 INPUT=./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  VIDEOS=/path/to/videos OUT=out/ AGG_TP=4 \
    bash scripts/run_pipeline.sh
```

Or:

```bash
python -m pipeline.aggregator.run_aggregate \
    --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --output-dir out/ --tensor-parallel 4
```

Method A (embedding-similarity clustering + per-cluster LLM verify) runs by default — this is the headline configuration from the paper. Add `AGG_METHOD_B=1` (env var) or `--aggregator-run-method-b` (CLI) to also run Method B (pure-LLM clustering) as an ablation; per Table 4 in the paper, Method B trails Method A by ~0.006 Avg F1.

### Configuration defaults

| Flag / env var | Default | MAGMaR config (paper) | Controls |
|---|---|---|---|
| `--fps` | `1.0` | `1.0` | frame extraction rate for Part 1 |
| `--whisper-model` | `large-v3` | `large-v3` | ASR backbone (faster-whisper) |
| `--yolo-model` | `yolo12x.pt` | `yolo12x.pt` | object detector for Part 1 |
| `--ocr-model` | `tencent/HunyuanOCR` | `tencent/HunyuanOCR` | scene-text OCR for Part 1 |
| `--llm-model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | same | Part-2 Step-1 relevance filter |
| `--lvlm-model` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | same (BF16) | Part-2 Step-2 claim generation |
| `--frame-mode` | `guided` | `guided` | Part-2 Step-2 frame selection (§3.3) |
| `--aggregator-model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | same | Part-3 Method A / B LLM |
| `--aggregator-embed-model` | `Qwen/Qwen3-Embedding-8B` | same | Part-3 Method A clusterer |
| `--aggregator-cluster-tau` | `0.9` | `0.9` | cosine threshold for greedy single-link clustering |
| `--aggregator-tensor-parallel` | `1` (assumes 1×80 GB) | `4` (4×48 GB used in paper) | vLLM tensor-parallel size for Part 3 |
| `LVLM_VIDEO_NFRAMES` | `100` | `100` | uniform-frame budget passed to the LVLM |
| `MAX_FRAMES_IN_GUIDANCE` | `30` | `30` | max guidance-targeted extra frames |
| `--aggregator-team-id` | `123456` | `(your team id)` | MAGMaR submission metadata.team_id |
| `--aggregator-run-id-prefix` | `penkil` | (your prefix) | submission filename prefix |

### Hardware recipes

The MAGMaR paper experiments used **4× RTX 6000 Ada (192 GB total)**. For other hardware, here are the tested recipes:

**1× H100 / A100 80 GB**

```bash
python -m pipeline.run \
    --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --videos-dir videos/ --output-dir out/ \
    --gpus 0 --claim-gpus 0 \
    --aggregator-tensor-parallel 1
```
Everything runs on one GPU with vLLM tp=1.

**4× 48 GB (RTX 6000 Ada — paper config)**

```bash
python -m pipeline.run \
    --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --videos-dir videos/ --output-dir out/ \
    --gpus 0,1,2,3 --claim-gpus 0,1,2,3 \
    --aggregator-tensor-parallel 4
```
All defaults work; both 30B models distribute via tp=4. Matches the configuration in §4.1 of the paper.

**4× 24 GB (RTX 3090 / A5000)**

```bash
python -m pipeline.run \
    --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --videos-dir videos/ --output-dir out/ \
    --gpus 0,1,2,3 --claim-gpus 0,1,2,3 \
    --aggregator-tensor-parallel 4 \
    --frame-mode uniform
```

⚠️ **Important on 24 GB cards**:
- You **must** pass `--aggregator-tensor-parallel 4` (default `1` assumes 80 GB and will OOM on 24 GB).
- `--frame-mode uniform` caps the vision-token budget; without it Part-2 Step-2 may OOM during decode.
- If Part-2 Step-1 or aggregator stage 3 still OOMs, edit `VLLM_GPU_MEM_UTIL` in [`pipeline/claim_gen/config.py`](pipeline/claim_gen/config.py) (0.90 → 0.85) and `llm_gpu_mem_util` in [`pipeline/aggregator/config.py`](pipeline/aggregator/config.py) (0.92 → 0.85).
- If Part-2 Step-2 still OOMs even with `--frame-mode uniform`, fall back to the 8B LVLM: `--lvlm-model Qwen/Qwen3-VL-8B-Instruct`. Lower quality but trivially fits.

**No GPU — wiring smoke**

```bash
python scripts/dry_smoke.py \
    --input ./data/TRACE-MAGMaR-2026/inputs/MAGMaR2026_queries.jsonl  --output-dir out/
```

Runs in ~3 seconds. Skips every model call but exercises every other code path (file I/O, schemas, stage 1, stage 4 validator, stage 5 diff report). Produces a real `aggregate/submission/penkil_method_a.jsonl` with synthesised claim text. Useful for CI and for sanity-checking the wiring before committing to a real GPU run.

### Disk

Each 30B model checkpoint is ~60 GB; the embedder is ~16 GB. Total HF cache needed: **~140 GB**. Set `HF_HOME` to a partition with that much free:

```bash
export HF_HOME=/path/with/big/disk/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
```

### Resumability and partial reruns

Every step is idempotent. Per-modality JSONLs (`asr.jsonl`, `yolo.jsonl`, `ocr.jsonl`) and per-event guidance files (`claim_guidance/q_<slug>_guidance.json`) are the source of truth for "what's done". Re-runs scan them, skip already-done IDs, and append.

For partial reruns of Part 3 **without re-invoking the LLM** (e.g. after editing the validator or prompt template), use the standalone CLI's `revalidate` command — it replays validators against the persisted `aggregate/raw_llm/` per-attempt traces:

```bash
python -m pipeline.aggregator.cli revalidate --work-dir out/aggregate
python -m pipeline.aggregator.cli stage4     --work-dir out/aggregate
```

---

## Evaluation

We score TRACE with the **MiRAGE** judge (Martin et al., 2025b) on Info F1 and Cite F1. The final submission file lands at:

```
out/aggregate/submission/penkil_method_a.jsonl
```

with one JSON line per query in the MAGMaR 2026 generation-task schema:

```json
{
  "metadata":   {"run_id": "penkil_method_a", "query_id": "<event_slug>",
                 "team_id": "123456", "task": "oracle"},
  "responses":  [{"text": "<verbatim claim>", "citations": ["<vid1>", "<vid2>"]}],
  "references": ["<vid1>", "<vid2>"]
}
```

Hard invariants enforced by aggregator stage 4 (validation failure → stage 4 exits non-zero):
- every `response.text` is a verbatim copy of one input claim,
- citations are deduped `video_id`s drawn from that cluster's input,
- every video that contributed at least one claim appears in some response's citations,
- total citations across responses do not exceed total input claims.

<!-- TODO(authors): if the paper version is updated, re-verify the numbers in
     the two tables below match the camera-ready manuscript before pushing. -->

### Headline results (from the paper)

| Method | Avg F1 | InfoF1 | CiteR | CiteF1 |
|---|---|---|---|---|
| Qwen3.5-9B (text baseline) | 0.472 | 0.554 | 0.251 | 0.390 |
| Qwen3-VL-8B (uniform sampling) | 0.723 | 0.835 | 0.452 | 0.608 |
| Qwen3-VL-30B (uniform sampling) | 0.705 | 0.800 | 0.440 | 0.609 |
| **TRACE (Ours)** | **0.811** | **0.869** | **0.628** | **0.753** |

MAGMaR 2026 Oracle Track validation set (8 topics). See Table 2 in the paper. TRACE achieves the highest scores across every metric, with the largest gain in **citation recall** (+42.7 % relative over the strongest unguided baseline) — exactly where multi-video evidence localisation is hardest.

### Ablation (Method A vs Method B, with vs without guided keyframes)

| Guided keyframes | Aggregation | Avg F1 | InfoF1 | CiteF1 |
|---|---|---|---|---|
| ✗ | LLM (Method B) | 0.802 | 0.859 | 0.745 |
| ✗ | Embed-Sim (Method A) | 0.808 | 0.868 | 0.748 |
| ✓ | LLM (Method B) | 0.804 | 0.867 | 0.741 |
| ✓ | Embed-Sim (Method A) | **0.811** | **0.869** | **0.753** |

Table 4 in the paper. Embedding-based aggregation (Method A) and guided keyframe augmentation provide complementary gains; their combination is the configuration this repository runs by default.

<!-- TODO(authors): leaderboard screenshot or rendered table figure.
     Suggested: `figures/magmar_leaderboard.png`. -->
<p align="center">
  <em>[ MAGMaR 2026 official leaderboard figure — to be added ]</em>
</p>

---

## Acknowledgements

- **MAGMaR 2026 Workshop** and the **MultiVENT** team (Sanders et al., 2023; Kriz et al., 2025) for the multi-video event understanding benchmark.
- **WikiVideo** ([repo](https://github.com/alexmartin1722/wikivideo), [paper](https://arxiv.org/abs/2504.00939), Martin et al., 2025a) for the multi-video article-generation dataset distributed at [🤗 hltcoe/wikivideo](https://huggingface.co/datasets/hltcoe/wikivideo).
- **MiRAGE** (Martin et al., 2025b) for the multimodal RAG evaluation framework used as our scorer.
- **Qwen** team for [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (relevance filter + aggregator LLM), [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) (claim-generation LVLM), and [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) (semantic clusterer).
- **YOLOv12** (Tian et al., [arXiv:2502.12524](https://arxiv.org/abs/2502.12524)) — object detector used in Part 1.
- **Tencent Hunyuan** team for [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) — multilingual scene-text OCR used in Part 1.
- **OpenAI Whisper** (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)) — ASR for the Part-1 audio stream.
- **vLLM** for high-throughput offline LLM/LVLM inference, and **xgrammar** for constrained JSON generation in the aggregator.

---

## Citation

```bibtex
@inproceedings{yan2026trace,
  title     = {TRACE: Evidence Grounding-Guided Multi-Video Event Understanding and Claim Generation},
  author    = {Yan, Pengyu and Gorugantu, Akhil and Bhosale, Mahesh and Wasi, Abdul and Trivedi, Vishvesh and Doermann, David},
  booktitle = {Proceedings of the MAGMaR Workshop at ACL 2026},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.16740},
  note      = {Equal contribution: Pengyu Yan and Akhil Gorugantu}
}
```

---

## License

GNU AGPL-3.0 — see [LICENSE](LICENSE).
