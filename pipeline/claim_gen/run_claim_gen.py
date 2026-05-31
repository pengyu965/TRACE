"""Unified Step 1 + Step 2 entry point for claim generation.

Step 1 caches one guidance file per event (q{event_slug}_guidance.json) so
reruns skip already-done events. Step 2 reads those caches, runs the LVLM,
writes per-video claim files plus a consolidated all_results.jsonl.

Usage
-----
    python -m pipeline.claim_gen.run_claim_gen \
        --input        events.json \
        --videos-dir   videos/ \
        --output-dir   out/ \
        --gpus         0,1,2,3

    # Just Step 1:
    python -m pipeline.claim_gen.run_claim_gen ... --step 1

    # Just Step 2 (reads guidance from --guidance-dir):
    python -m pipeline.claim_gen.run_claim_gen ... --step 2

    # Restrict to specific events (matches by event_key or its slug):
    python -m pipeline.claim_gen.run_claim_gen ... --event-keys russia_ukraine_war

    # Switch frame-selection mode for Step 2:
    python -m pipeline.claim_gen.run_claim_gen ... --frame-mode uniform
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from . import config
from .data_loader import DatasetIndex
from ..io_utils import slugify


def _bar(title: str) -> None:
    print("\n" + "═" * 65)
    print(f"  {title}")
    print("═" * 65)


def _guidance_path(guidance_dir: Path, qid: str) -> Path:
    return guidance_dir / f"q_{qid}_guidance.json"


def _load_guidance_cache(guidance_dir: Path, qid: str) -> dict | None:
    p = _guidance_path(guidance_dir, qid)
    return json.loads(p.read_text()) if p.exists() else None


def _enrich_guidance(guidance: dict, frames: list[dict]) -> dict:
    """Re-inflate Step-1's compact relevant_frames with full bbox/conf from merged.jsonl."""
    frame_by_time = {fr["time_sec"]: fr for fr in frames}
    enriched = []
    for rf in guidance.get("relevant_frames", []):
        t    = rf.get("time_sec")
        orig = frame_by_time.get(t)
        if orig is None:
            enriched.append(rf)
            continue
        enriched.append({
            "time_sec": t,
            "objects": [
                {"label": d["class_name"], "confidence": round(d["confidence"], 3),
                 "bbox_xyxy": d.get("bbox_xyxy", [])}
                for d in orig["yolo"]["detections"][:config.MAX_DETECTIONS_PER_FRAME]
            ],
            "ocr": [
                {"text": d["text"], "lang": d.get("src_lang", ""),
                 "bbox_xyxy": d.get("bbox_xyxy", [])}
                for d in orig["ocr"]["detections"][:config.MAX_OCR_PER_FRAME]
            ],
        })
    return {**guidance, "relevant_frames": enriched}


def _resolve_event_ids(ds: DatasetIndex, requested: list[str] | None) -> list[str]:
    """Map user-supplied --event-keys to canonical slugs that exist in the index."""
    all_ids = [r["metadata"]["query_id"] for r in ds.queries]
    if not requested:
        return all_ids
    requested_slugs = [slugify(r) for r in requested]
    ids = [s for s in requested_slugs if s in all_ids]
    missing = [r for r, s in zip(requested, requested_slugs) if s not in all_ids]
    if missing:
        print(f"Warning: event keys {missing} not in events.json — skipping.",
              file=sys.stderr)
    return ids


# ── Step 1 ────────────────────────────────────────────────────────────────────

def run_step1(ds: DatasetIndex, query_ids: list[str], guidance_dir: Path) -> dict:
    _bar(f"STEP 1 — Relevance Filter   [{config.LLM_MODEL_ID}]")
    guidance_dir.mkdir(parents=True, exist_ok=True)

    guidances: dict[str, dict] = {}
    pending: list[str] = []
    for qid in query_ids:
        cached = _load_guidance_cache(guidance_dir, qid)
        if cached is not None:
            guidances[qid] = cached
            print(f"  Q{qid} — cache hit ({len(cached)} videos), skipping")
        else:
            pending.append(qid)

    if not pending:
        print("  All guidance cached — nothing to run.")
        return guidances

    from .step1_filter import load_llm, unload_llm, filter_video
    load_llm()

    for qid in tqdm(pending, desc="Step1 events", unit="event"):
        row  = ds.query_map[qid]
        m    = row["metadata"]
        refs = ds.get_topic_videos(qid) or row["references"]
        guidances[qid] = {}

        if not refs:
            tqdm.write(f"  WARNING: Q{qid} — no videos, skipping.")
            _guidance_path(guidance_dir, qid).write_text(json.dumps({}, indent=2))
            continue

        n_annot = sum(1 for v in refs if ds.get_merged_frames(v))
        tqdm.write(f"\n  Q{qid} — {m['title']}  [{m.get('language','english')}]")
        tqdm.write(f"  Persona : {m['persona_title']}")
        tqdm.write(f"  Videos  : {len(refs)} total  |  {n_annot} with YOLO/OCR annotations")

        for vid_id in tqdm(refs, desc=f"  Q{qid} videos", unit="video", leave=False):
            frames = ds.get_merged_frames(vid_id)
            if not frames:
                tqdm.write(f"    [skip] {vid_id}: no YOLO/OCR data")
                guidances[qid][vid_id] = {"relevant_frames": [], "summary": ""}
                continue
            g = filter_video(
                query=m["query"], persona_title=m["persona_title"],
                background=m["background"], frames=frames,
            )
            g = _enrich_guidance(g, frames)
            guidances[qid][vid_id] = g
            tqdm.write(f"    {vid_id}  ({len(frames)} frames) → "
                       f"{len(g.get('relevant_frames', []))} relevant frames")

        _guidance_path(guidance_dir, qid).write_text(json.dumps(guidances[qid], indent=2))
        tqdm.write(f"  ✓ saved → {_guidance_path(guidance_dir, qid).name}")

    unload_llm()
    (guidance_dir / "all_guidances.json").write_text(json.dumps(guidances, indent=2))
    print("\n  ✓ merged guidance → all_guidances.json")
    return guidances


# ── Step 2 ────────────────────────────────────────────────────────────────────

def run_step2(
    ds: DatasetIndex,
    query_ids: list[str],
    guidance_dir: Path,
    results_dir: Path,
) -> list[dict]:
    _bar(f"STEP 2 — Claim Generation   [{config.LVLM_MODEL_ID}, "
         f"frame_mode={config.FRAME_MODE}]")
    results_dir.mkdir(parents=True, exist_ok=True)

    guidances: dict[str, dict] = {}
    missing = []
    for qid in query_ids:
        cached = _load_guidance_cache(guidance_dir, qid)
        if cached is None:
            missing.append(qid)
        else:
            guidances[qid] = cached
    if missing:
        print(f"\n  ERROR: guidance missing for {missing}.\n"
              f"  Guidance dir: {guidance_dir}\n"
              f"  Run Step 1 first.", file=sys.stderr)
        sys.exit(1)

    from .step2_generate import load_lvlm, unload_lvlm, generate_claims
    load_lvlm()

    video_results: dict[str, dict] = defaultdict(lambda: {"video_id": None, "queries": []})

    pairs = []
    for qid in query_ids:
        row = ds.query_map.get(qid)
        if row is None:
            continue
        topic_vids = set(ds.get_topic_videos(qid))
        for vid_id, guidance in guidances[qid].items():
            if vid_id in topic_vids:
                pairs.append((qid, vid_id, guidance))

    for qid, vid_id, guidance in tqdm(pairs, desc="Step2 pairs", unit="pair"):
        row = ds.query_map[qid]
        m   = row["metadata"]
        full_asr = ds.get_full_asr(vid_id)
        n_frames = len(guidance.get("relevant_frames", []))
        tqdm.write(f"  Q{qid} × {vid_id}  ({n_frames} relevant frames | asr={len(full_asr)} chars)")

        claims = generate_claims(
            query=m["query"], persona_title=m["persona_title"],
            background=m["background"], title=m["title"],
            guidance=guidance, full_asr_text=full_asr,
            video_id=vid_id,
        )
        tqdm.write(f"  → {len(claims)} claims")

        video_results[vid_id]["video_id"] = vid_id
        video_results[vid_id]["queries"].append({
            "query_id":         qid,
            "event_key":        m.get("event_key", qid),
            "query":            m["query"],
            "persona_title":    m["persona_title"],
            "persona":          m["background"],
            "generated_claims": claims,
        })

    unload_lvlm()

    all_results = list(video_results.values())
    for entry in all_results:
        (results_dir / f"{entry['video_id']}_results.json").write_text(
            json.dumps(entry, indent=2, ensure_ascii=False)
        )
    combined = results_dir / "all_results.jsonl"
    with open(combined, "w", encoding="utf-8") as f:
        for entry in all_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n  ✓ {len(all_results)} videos → {combined}")
    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Claim generation (Step 1 + Step 2).")
    ap.add_argument("--input",       required=True,  help="events.json")
    ap.add_argument("--videos-dir",  required=True,  help="folder containing <video_id>.mp4")
    ap.add_argument("--output-dir",  required=True,  help="pipeline output root (where merged.jsonl lives)")
    ap.add_argument("--step",        type=int, choices=[1, 2], default=None,
                    help="1 = filter only, 2 = generate only, omit = both back-to-back")
    ap.add_argument("--guidance-dir", default=None,
                    help="Step-1 guidance directory (default: <output-dir>/claim_guidance)")
    ap.add_argument("--results-dir",  default=None,
                    help="Step-2 results directory (default: <output-dir>/claim_results)")
    ap.add_argument("--event-keys",   nargs="*", metavar="KEY", default=None,
                    help="Restrict to these event_keys (or their slugs). Defaults to all.")
    ap.add_argument("--gpus",         default="0,1,2,3",
                    help="GPU shard for both Step 1 and Step 2 (same set, sequential)")
    ap.add_argument("--llm-model",    default=None, help="override LLM model id")
    ap.add_argument("--lvlm-model",   default=None, help="override LVLM model id")
    ap.add_argument("--frame-mode",   default="guided", choices=["guided", "uniform"],
                    help="Step-2 frame selection: 'guided' = uniform + guidance-extra frames; "
                         "'uniform' = uniform only but keep summary in prompt")
    ap.add_argument("--no-vllm",      action="store_true",
                    help="Use HuggingFace transformers instead of vLLM")
    args = ap.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    out_root = Path(args.output_dir)
    guidance_dir = Path(args.guidance_dir) if args.guidance_dir else out_root / "claim_guidance"
    results_dir  = Path(args.results_dir)  if args.results_dir  else out_root / "claim_results"

    config.configure(
        output_dir=out_root,
        videos_dir=args.videos_dir,
        events_file=args.input,
        llm_gpus=gpu_ids,
        lvlm_gpus=gpu_ids,
        use_vllm=not args.no_vllm,
        llm_model=args.llm_model,
        lvlm_model=args.lvlm_model,
        frame_mode=args.frame_mode,
    )

    ds = DatasetIndex()
    query_ids = _resolve_event_ids(ds, args.event_keys)

    print(ds)
    print(f"  events to process : {query_ids}")
    print(f"  guidance dir      : {guidance_dir}")
    print(f"  results dir       : {results_dir}")
    print(f"  gpus              : {gpu_ids}")
    print(f"  frame_mode        : {config.FRAME_MODE}")
    print(f"  backend           : {'vLLM' if config.USE_VLLM else 'HF transformers'}")

    if args.step == 1:
        run_step1(ds, query_ids, guidance_dir)
    elif args.step == 2:
        run_step2(ds, query_ids, guidance_dir, results_dir)
    else:
        run_step1(ds, query_ids, guidance_dir)
        run_step2(ds, query_ids, guidance_dir, results_dir)


if __name__ == "__main__":
    main()
