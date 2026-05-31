"""faster-whisper ASR sharded across N GPUs at the video level.

For each .wav under <out>/audio/<event_slug>/<video_id>.wav, runs two passes:
  - task='transcribe' → native-language transcript (preserves source language)
  - task='translate'  → English translation

Outputs:
    <out>/asr/<event_slug>/<video_id>.json          per-video record
    <out>/asr.jsonl                                  one row per video (aggregate)

Resumable: any video_id already present in asr.jsonl with no `error` is skipped.

Usage:
    python -m pipeline.whisper_asr --input events.json --output-dir out/ --gpus 0
    python -m pipeline.whisper_asr --input events.json --output-dir out/ --gpus 0,1
    python -m pipeline.whisper_asr --input events.json --output-dir out/ --model large-v3
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch.multiprocessing as mp

from .io_utils import load_events, append_jsonl


def load_done(path: Path) -> set[str]:
    """Return video_ids whose most-recent row is successful (no `error`)."""
    last: dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vid = obj.get("video_id")
                if vid:
                    last[vid] = obj
    return {vid for vid, obj in last.items() if not obj.get("error")}


def transcribe_one(model, wav_path: Path) -> dict:
    """Two-pass transcribe + translate."""
    segments_native, info = model.transcribe(
        str(wav_path),
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,
    )
    segs_native = [{"start": s.start, "end": s.end, "text": s.text.strip()}
                   for s in segments_native]
    text_native = " ".join(s["text"] for s in segs_native).strip()
    lang = info.language

    segments_en, _ = model.transcribe(
        str(wav_path),
        task="translate",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,
    )
    segs_en = [{"start": s.start, "end": s.end, "text": s.text.strip()}
               for s in segments_en]
    text_en = " ".join(s["text"] for s in segs_en).strip()

    return {
        "language_detected": lang,
        "language_prob":     float(info.language_probability) if info.language_probability is not None else None,
        "duration_sec":      float(info.duration) if info.duration is not None else None,
        "native":            text_native,
        "english":           text_en,
        "segments_native":   segs_native,
        "segments_english":  segs_en,
    }


def worker(gpu_id: int, items: list[tuple[str, str, str]], asr_root: str,
           model_id: str, compute_type: str, return_q):
    """Process a shard of (slug, video_id, wav_path) on a single GPU."""
    import traceback
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        from faster_whisper import WhisperModel
        print(f"[gpu {gpu_id}] loading {model_id} ({compute_type})", flush=True)
        model = WhisperModel(model_id, device="cuda", compute_type=compute_type)
        print(f"[gpu {gpu_id}] ready, {len(items)} videos", flush=True)
    except Exception:
        print(f"[gpu {gpu_id}] FATAL during model load:\n{traceback.format_exc()}",
              flush=True)
        return_q.put([]); return

    asr_root_p = Path(asr_root)
    out_rows: list[dict] = []
    try:
        for slug, vid, wav_path in items:
            t0 = time.time()
            wav = Path(wav_path)
            if not wav.exists():
                row = {"video_id": vid, "event_slug": slug,
                       "error": f"wav-missing: {wav_path}"}
                out_rows.append(row); continue
            try:
                res = transcribe_one(model, wav)
                row = {"video_id": vid, "event_slug": slug,
                       "model": model_id, **res,
                       "wall_time_sec": round(time.time() - t0, 2)}
                per_video = asr_root_p / slug / f"{vid}.json"
                per_video.parent.mkdir(parents=True, exist_ok=True)
                json.dump(row, open(per_video, "w"),
                          ensure_ascii=False, indent=2)
                print(f"[gpu {gpu_id}] {slug}/{vid}  lang={res['language_detected']}  "
                      f"dur={res.get('duration_sec', 0) or 0:.0f}s  "
                      f"{time.time()-t0:.1f}s", flush=True)
            except Exception as e:
                row = {"video_id": vid, "event_slug": slug,
                       "error": f"{type(e).__name__}: {e}",
                       "wall_time_sec": round(time.time() - t0, 2)}
                print(f"[gpu {gpu_id}] {slug}/{vid}  FAIL  {e}",
                      flush=True, file=sys.stderr)
            out_rows.append(row)
    except Exception:
        print(f"[gpu {gpu_id}] FATAL in loop:\n{traceback.format_exc()}", flush=True)
    finally:
        return_q.put(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--gpus",       default="0",
                    help="comma-separated GPU ids (e.g. '0' or '0,1')")
    ap.add_argument("--model",      default="large-v3")
    ap.add_argument("--compute-type", default="float16",
                    help="faster-whisper compute_type: float16 (default) / int8_float16 / int8")
    ap.add_argument("--limit",      type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    audio_root = out_root / "audio"
    asr_root   = out_root / "asr"
    jsonl_path = out_root / "asr.jsonl"
    asr_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise SystemExit("--gpus must list at least one GPU id")

    events = load_events(args.input)
    all_items: list[tuple[str, str, str]] = []
    for ev in events:
        for vid in ev["videos"]:
            wav = audio_root / ev["slug"] / f"{vid}.wav"
            all_items.append((ev["slug"], vid, str(wav)))

    done = load_done(jsonl_path)
    todo = [x for x in all_items if x[1] not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Events     : {len(events)}")
    print(f"Total vids : {len(all_items)}")
    print(f"Already done: {len(done)}")
    print(f"To process : {len(todo)}")
    print(f"GPUs       : {gpu_ids}")
    print(f"Model      : {args.model} ({args.compute_type})\n")
    if not todo:
        print("Nothing to do.")
        return

    shards: list[list] = [[] for _ in gpu_ids]
    for i, item in enumerate(todo):
        shards[i % len(gpu_ids)].append(item)
    for gid, sh in zip(gpu_ids, shards):
        print(f"  gpu {gid}: {len(sh)} videos")

    t_start = time.time()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for gid, sh in zip(gpu_ids, shards):
        p = ctx.Process(target=worker,
                        args=(gid, sh, str(asr_root), args.model,
                              args.compute_type, q))
        p.start(); procs.append(p)

    collected: list[list[dict]] = []
    expected = len(procs)
    while len(collected) < expected:
        try:
            collected.append(q.get(timeout=60))
        except Exception:
            dead_with_err = [p for p in procs if not p.is_alive() and (p.exitcode or 0) != 0]
            if dead_with_err and len(collected) + len(dead_with_err) >= expected:
                print(f"WARN: {len(dead_with_err)} worker(s) crashed; continuing")
                break
            if not any(p.is_alive() for p in procs):
                break
    flat = [r for lst in collected for r in lst]
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate(); p.join(timeout=5)

    flat.sort(key=lambda r: (r.get("event_slug", ""), r["video_id"]))
    append_jsonl(jsonl_path, flat)

    print(f"\nWall: {time.time()-t_start:.1f}s")
    print(f"Wrote {len(flat)} records → {jsonl_path}")


if __name__ == "__main__":
    main()
