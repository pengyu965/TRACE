# Output schemas

All aggregate outputs are line-delimited JSON (JSONL) — one record per line.
Per-video copies are also written under `asr/`, `yolo_frames/`, `ocr_frames/`,
`claim_guidance/`, and `claim_results/` for easy spot-checking.

---

## `events.json` (input)

```json
{
  "events": [
    {
      "event_key":     "russia_ukraine_war",
      "query":         "Tell me about ...",
      "persona_title": "International Affairs Reporter",       // optional, default ""
      "background":    "I cover geopolitical conflicts ...",   // optional, default ""
      "language":      "english",                              // optional, default "english"
      "videos":        ["abc123", "def456"]
    }
  ]
}
```

- `event_key` — any string; slugified internally for paths.
- `query` — free-text retrieval query, passed through into `merged.jsonl` and used in Part-2 prompts.
- `persona_title` / `background` — used by Part 2 (claim generation). Empty defaults work fine if you only need Part 1.
- `videos` — list of video stems. The pipeline resolves `<videos-dir>/<video_id>.{mp4,mkv,webm,mov,m4v}`.

---

## `asr.jsonl`

```json
{
  "video_id":          "abc123",
  "event_slug":        "russia_ukraine_war",
  "model":             "large-v3",
  "language_detected": "ru",
  "language_prob":     0.998,
  "duration_sec":      314.2,
  "native":            "Полная транскрипция ...",
  "english":           "Full English translation ...",
  "segments_native":   [{"start": 0.0, "end": 4.8, "text": "..."}, ...],
  "segments_english":  [{"start": 0.0, "end": 4.8, "text": "..."}, ...],
  "wall_time_sec":     38.4
}
```

On error: `{"video_id", "event_slug", "error": "ErrType: msg"}`.

---

## `yolo.jsonl`

```json
{
  "video_id":      "abc123",
  "event_slug":    "russia_ukraine_war",
  "model":         "yolo12x.pt",
  "fps":           1.0,
  "image_size":    [1280, 720],
  "num_frames":    314,
  "conf_threshold": 0.25,
  "wall_time_sec": 102.1,
  "frames": [
    {"frame_idx": 1, "time_sec": 0.0, "num_detections": 3,
     "class_counts": {"person": 2, "car": 1},
     "detections": [
       {"class_id": 0, "class_name": "person",
        "confidence": 0.91, "bbox_xyxy": [12.3, 45.6, 200.1, 480.0]}
     ]}
  ]
}
```

---

## `ocr.jsonl`

```json
{
  "video_id":        "abc123",
  "event_slug":      "russia_ukraine_war",
  "model":           "tencent/HunyuanOCR",
  "fps":             1.0,
  "image_size":      [1280, 720],
  "num_frames":      314,
  "languages_found": ["eng_Latn", "zho_Hans"],
  "wall_time_sec":   415.7,
  "frames": [
    {"frame_idx": 1, "time_sec": 0.0, "num_detections": 2,
     "frame_languages": ["eng_Latn"], "elapsed_sec": 1.3, "error": null,
     "detections": [
       {"text": "BREAKING NEWS", "src_lang": "eng_Latn",
        "bbox_xyxy_norm": [50, 800, 450, 900],
        "bbox_xyxy":      [64, 576, 576, 648]}
     ]}
  ]
}
```

`bbox_xyxy_norm` is the model's native 0..1000 coordinate space;
`bbox_xyxy` is rescaled to original frame pixels.

---

## `merged.jsonl`  *(handoff between Part 1 and Part 2)*

One record per video. The top-level shape is **directly compatible** with the
claim-gen `data_loader` — no adapter needed. Extra fields (`event_key`,
`query`, `persona_title`, `background`, `asr`, `*_model`) are carried alongside
for downstream consumers.

```json
{
  "video_id":   "abc123",
  "image_size": [1280, 720],
  "frames": [
    {
      "frame_idx": 1,
      "time_sec":  0.0,
      "yolo": {
        "detections": [
          {"class_name": "person", "confidence": 0.91,
           "bbox_xyxy": [12.3, 45.6, 200.1, 480.0]}
        ],
        "class_counts": {"person": 2, "car": 1}
      },
      "ocr": {
        "detections": [
          {"text": "BREAKING NEWS", "src_lang": "eng_Latn",
           "bbox_xyxy": [64, 576, 576, 648]}
        ],
        "languages": ["eng_Latn"],
        "error":     null
      }
    }
  ],

  "event_key":     "russia_ukraine_war",
  "event_slug":    "russia_ukraine_war",
  "query":         "Tell me about ...",
  "persona_title": "International Affairs Reporter",
  "background":    "I cover geopolitical conflicts ...",
  "fps":           1.0,
  "num_frames":    314,

  "yolo_model":     "yolo12x.pt",
  "ocr_model":      "tencent/HunyuanOCR",
  "ocr_languages":  ["eng_Latn", "zho_Hans"],

  "asr": {
    "language_detected": "ru",
    "language_prob":     0.998,
    "duration":          314.2,
    "duration_sec":      314.2,
    "native":            "...",
    "english":           "...",
    "segments_native":   [...],
    "segments_english":  [...]
  }
}
```

If a modality is missing for a video, the corresponding sub-block carries an
`"error"` key instead of data (e.g. `"asr": {"error": "asr-missing"}`).

---

## `claim_guidance/q_<event_slug>_guidance.json`  *(Part 2, Step 1 output)*

One file per event. Top-level keys are `video_id`s. Each video's value is the
relevance-filter output for that event's query.

```json
{
  "abc123": {
    "relevant_frames": [
      {
        "time_sec": 2.0,
        "objects": [
          {"label": "person",  "confidence": 0.94, "bbox_xyxy": [120, 80, 640, 720]},
          {"label": "tie",     "confidence": 0.84, "bbox_xyxy": [310, 320, 420, 480]}
        ],
        "ocr": [
          {"text": "Mark Carney",    "lang": "eng_Latn", "bbox_xyxy": [200, 700, 560, 740]},
          {"text": "LIBERAL LEADER", "lang": "eng_Latn", "bbox_xyxy": [200, 745, 560, 775]}
        ]
      }
    ],
    "summary": "This election-night broadcast shows Liberal leader Mark Carney ..."
  },
  "def456": { ... }
}
```

`claim_guidance/all_guidances.json` is the same content concatenated across all events.

---

## `claim_results/{video_id}_results.json` and `claim_results/all_results.jsonl`  *(Part 2, Step 2 output)*

```json
{
  "video_id": "abc123",
  "queries": [
    {
      "query_id":      "russia_ukraine_war",       // event_slug
      "event_key":     "russia_ukraine_war",
      "query":         "Tell me about ...",
      "persona_title": "International Affairs Reporter",
      "persona":       "I cover geopolitical conflicts ...",  // == background
      "generated_claims": [
        "Russian forces launched a full-scale invasion of Ukraine on Feb 24, 2022.",
        "The on-screen graphic shows over 8 million Ukrainians displaced as refugees.",
        "..."
      ]
    }
  ]
}
```

`claim_results/all_results.jsonl` is the same content as one JSON line per video. The per-video `*_results.json` files are what **Part 3 (aggregator)** reads.

---

## Part 3 — Aggregator outputs

### `aggregate/aggregate_queries.jsonl`  *(internal, built from events.json)*

```json
{"query_id": "russia_ukraine_war", "query_type": "factoid", "language": "en",
 "title": "russia_ukraine_war", "persona_title": "International Affairs Reporter",
 "background": "I cover geopolitical conflicts ...",
 "query": "Tell me about ...", "persona": "I cover geopolitical conflicts ..."}
```

### `aggregate/per_query/query_<qid>.json`  *(stage 1)*

```json
{
  "query_id":      "russia_ukraine_war",
  "query_type":    "factoid", "language": "en",
  "title":         "russia_ukraine_war",
  "persona_title": "International Affairs Reporter",
  "background":    "...",
  "query":         "...",
  "videos": [
    {"video_id": "abc123", "claims": ["claim 1", "claim 2", ...]},
    {"video_id": "def456", "claims": ["claim 3", ...]}
  ]
}
```

### `aggregate/clusters/query_<qid>_clusters.json`  *(stage 2, Method A)*

Greedy single-link clustering output. Each cluster carries the input claim
indices that fall within cosine distance ≤ `cluster_tau`.

### `aggregate/agent_io/method_a/output/query_<qid>.json`  *(stage 3a)*

```json
{
  "query_id":  "russia_ukraine_war",
  "responses": [
    {"text": "Verbatim claim from cluster representative", "citations": ["abc123", "def456"]},
    {"text": "Another verbatim claim", "citations": ["abc123"]}
  ]
}
```

### `aggregate/raw_llm/method_a/query_<qid>.json`  *(per-attempt trace)*

Every prompt + raw model output + parse / validation error for the cluster,
preserved so the offline `revalidate` command can re-run validators without
re-invoking the LLM.

### `aggregate/submission/penkil_method_a.jsonl`  *(stage 4 — FINAL OUTPUT)*

MAGMaR 2026 generation-task submission. One JSON line per query:

```json
{
  "metadata":   {"run_id": "penkil_method_a", "query_id": "russia_ukraine_war",
                 "team_id": "123456", "task": "oracle"},
  "responses":  [
    {"text": "Verbatim claim text",  "citations": ["abc123", "def456"]},
    {"text": "Another claim",        "citations": ["abc123"]}
  ],
  "references": ["abc123", "def456"]
}
```

Hard invariants enforced by `stage4` validator (validation failure → stage 4 exits non-zero):
- every `response.text` is a verbatim copy of one input claim,
- citations are deduped `video_id`s drawn from the cluster's input,
- every video that contributed at least one claim appears in some response's citations,
- total citations across responses do not exceed total input claims.

### `aggregate/submission/diff_report.md`  *(stage 5)*

Human-readable markdown summarising counts per query and (when Method B also
ran) a side-by-side comparison of Method A vs Method B output sizes.
