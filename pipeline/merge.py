"""Merge per-video ASR + YOLO + OCR outputs into a single merged.jsonl.

One row per video. Top-level shape is compatible with the claim_gen
data_loader (`frames[].yolo.detections`, `frames[].ocr.detections`,
`image_size`), so Part 2 reads this file directly — no adapter needed.

Extra fields (`event_key`, `event_slug`, `query`, `persona_title`,
`background`, `asr {...}`) are also carried at the top level so the merged
file is a complete handoff record for downstream pipelines.

Inputs:
    <out>/asr.jsonl
    <out>/yolo.jsonl
    <out>/ocr.jsonl
    events.json  (event_key + query + optional persona_title/background)

Output:
    <out>/merged.jsonl

Usage:
    python -m pipeline.merge --input events.json --output-dir out/
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .io_utils import load_events


def load_jsonl_by_vid(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        print(f"[warn] {path} missing — skipping")
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = rec.get("video_id")
            if vid:
                out[vid] = rec   # last-wins on duplicate rows
    return out


def join_frames(yolo_rec: dict | None, ocr_rec: dict | None) -> list[dict]:
    """Merge YOLO + OCR per-frame records on frame_idx.

    Output shape per frame matches the claim_gen data_loader expectation:
        {time_sec, frame_idx,
         yolo: {detections: [{class_name, confidence, bbox_xyxy}, ...]},
         ocr:  {detections: [{text, src_lang, bbox_xyxy}, ...]}}
    """
    by_idx: dict[int, dict] = defaultdict(dict)

    if yolo_rec:
        for fr in yolo_rec.get("frames", []):
            entry = by_idx[fr["frame_idx"]]
            entry["frame_idx"] = fr["frame_idx"]
            entry["time_sec"]  = fr["time_sec"]
            entry["yolo"] = {
                "detections": [
                    {
                        "class_name": d.get("class_name", ""),
                        "confidence": d.get("confidence", 0.0),
                        "bbox_xyxy":  d.get("bbox_xyxy", []),
                    }
                    for d in fr.get("detections", [])
                ],
                "class_counts": fr.get("class_counts", {}),
            }
    if ocr_rec:
        for fr in ocr_rec.get("frames", []):
            entry = by_idx[fr["frame_idx"]]
            entry["frame_idx"] = fr["frame_idx"]
            entry["time_sec"]  = fr["time_sec"]
            entry["ocr"] = {
                "detections": [
                    {
                        "text":      d.get("text", ""),
                        "src_lang":  d.get("src_lang", ""),
                        "bbox_xyxy": d.get("bbox_xyxy", []),
                    }
                    for d in fr.get("detections", [])
                ],
                "languages": fr.get("frame_languages", []),
                "error":     fr.get("error"),
            }

    # Fill in empty yolo / ocr blocks for any frame missing one of the two
    for entry in by_idx.values():
        entry.setdefault("yolo", {"detections": [], "class_counts": {}})
        entry.setdefault("ocr",  {"detections": [], "languages": [], "error": None})

    return [by_idx[k] for k in sorted(by_idx.keys())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    asr  = load_jsonl_by_vid(out_root / "asr.jsonl")
    yolo = load_jsonl_by_vid(out_root / "yolo.jsonl")
    ocr  = load_jsonl_by_vid(out_root / "ocr.jsonl")
    events = load_events(args.input)

    out_path = out_root / "merged.jsonl"
    n_written = 0
    missing = {"asr": 0, "yolo": 0, "ocr": 0}

    with open(out_path, "w") as fout:
        for ev in events:
            for vid in ev["videos"]:
                yolo_rec = yolo.get(vid)
                ocr_rec  = ocr.get(vid)
                asr_rec  = asr.get(vid)
                if not asr_rec:  missing["asr"]  += 1
                if not yolo_rec: missing["yolo"] += 1
                if not ocr_rec:  missing["ocr"]  += 1

                visual_meta = yolo_rec or ocr_rec or {}
                row = {
                    # Required by claim_gen.data_loader
                    "video_id":   vid,
                    "image_size": visual_meta.get("image_size", [1920, 1080]),
                    "frames":     join_frames(yolo_rec, ocr_rec),

                    # Top-level extras carried through for downstream stages
                    "event_key":     ev["event_key"],
                    "event_slug":    ev["slug"],
                    "query":         ev["query"],
                    "persona_title": ev.get("persona_title", ""),
                    "background":    ev.get("background", ""),
                    "fps":           visual_meta.get("fps"),
                    "num_frames":    visual_meta.get("num_frames"),

                    # Visual model metadata
                    "yolo_model":      yolo_rec.get("model") if yolo_rec else None,
                    "ocr_model":       ocr_rec.get("model")  if ocr_rec  else None,
                    "ocr_languages":   ocr_rec.get("languages_found", []) if ocr_rec else [],

                    # Full ASR block (compatible with claim_gen's separate asr_whisper.jsonl
                    # consumer — fields `english`, `segments_english`, `duration` line up).
                    "asr": ({
                        "language_detected": asr_rec.get("language_detected"),
                        "language_prob":     asr_rec.get("language_prob"),
                        "duration":          asr_rec.get("duration_sec"),
                        "duration_sec":      asr_rec.get("duration_sec"),
                        "native":            asr_rec.get("native"),
                        "english":           asr_rec.get("english"),
                        "segments_native":   asr_rec.get("segments_native",  []),
                        "segments_english":  asr_rec.get("segments_english", []),
                    } if asr_rec else {"error": "asr-missing"}),
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"Wrote {n_written} merged records → {out_path}")
    print(f"Missing per modality: {missing}")


if __name__ == "__main__":
    main()
