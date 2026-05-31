"""YOLO12x on pre-extracted frames, sharded across N GPUs at the video level.

Inputs:
    <out>/frames/<event_slug>/<video_id>/frame_*.jpg

Outputs:
    <out>/yolo_frames/<event_slug>/<video_id>/frame_*.jpg   annotated copies
    <out>/yolo.jsonl                                         aggregate (one row / video)

Resumable: any video_id already in yolo.jsonl is skipped.

Usage:
    python -m pipeline.yolo_batch --output-dir out/ --gpus 0,1
    python -m pipeline.yolo_batch --output-dir out/ --gpus 0,1 --model yolo12m.pt --conf 0.4
"""
from __future__ import annotations
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch.multiprocessing as mp

from .io_utils import load_done_jsonl, append_jsonl


def gather_videos(frames_root: Path) -> list[tuple[str, str]]:
    out = []
    if not frames_root.is_dir():
        return out
    for slug in sorted(os.listdir(frames_root)):
        sd = frames_root / slug
        if not sd.is_dir(): continue
        for vid in sorted(os.listdir(sd)):
            vd = sd / vid
            if vd.is_dir() and any(vd.glob("frame_*.jpg")):
                out.append((slug, vid))
    return out


def worker(gpu_id: int, items, frames_root: str, annot_root: str,
           model_name: str, conf: float, fps: float, return_q):
    """Process a shard of videos on one GPU."""
    import traceback
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import cv2
        from ultralytics import YOLO
        print(f"[gpu {gpu_id}] loading {model_name}", flush=True)
        model = YOLO(model_name)
        print(f"[gpu {gpu_id}] ready, {len(items)} videos", flush=True)
    except Exception:
        print(f"[gpu {gpu_id}] FATAL during model load:\n{traceback.format_exc()}",
              flush=True)
        return_q.put([]); return

    frames_root_p = Path(frames_root)
    annot_root_p  = Path(annot_root)
    out = []
    try:
        for slug, vid in items:
            t0 = time.time()
            frames_dir = frames_root_p / slug / vid
            annot_dir  = annot_root_p  / slug / vid
            annot_dir.mkdir(parents=True, exist_ok=True)
            image_size = None
            frame_records = []

            for fp in sorted(frames_dir.glob("frame_*.jpg")):
                i = int(fp.stem.split("_")[-1])
                time_sec = (i - 1) / fps

                results = model(str(fp), conf=conf, verbose=False)
                r = results[0]
                names = r.names
                dets = []
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    dets.append({
                        "class_id":   cls,
                        "class_name": names[cls],
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox_xyxy":  [round(x1, 2), round(y1, 2),
                                       round(x2, 2), round(y2, 2)],
                    })
                class_counts = dict(Counter(d["class_name"] for d in dets))

                img = cv2.imread(str(fp))
                if img is None:
                    continue
                if image_size is None:
                    h, w = img.shape[:2]
                    image_size = [w, h]
                for d in dets:
                    x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    lbl = f"{d['class_name']} {d['confidence']:.2f}"
                    cv2.putText(img, lbl, (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(annot_dir / fp.name), img)

                frame_records.append({
                    "frame_idx":      i,
                    "time_sec":       time_sec,
                    "num_detections": len(dets),
                    "class_counts":   class_counts,
                    "detections":     dets,
                })

            elapsed = time.time() - t0
            rec = {
                "video_id":       vid,
                "event_slug":     slug,
                "num_frames":     len(frame_records),
                "fps":            fps,
                "image_size":     image_size,
                "wall_time_sec":  round(elapsed, 2),
                "model":          model_name,
                "conf_threshold": conf,
                "frames":         frame_records,
            }
            out.append(rec)
            total_dets = sum(fr["num_detections"] for fr in frame_records)
            print(f"[gpu {gpu_id}] {slug}/{vid}  frames={len(frame_records)}  "
                  f"dets={total_dets}  {elapsed:.1f}s", flush=True)
    except Exception:
        print(f"[gpu {gpu_id}] FATAL in loop:\n{traceback.format_exc()}", flush=True)
    finally:
        return_q.put(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model",      default="yolo12x.pt")
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--gpus",       default="0,1")
    ap.add_argument("--fps",        type=float, default=1.0,
                    help="must match the FPS used by extract_frames")
    ap.add_argument("--limit",      type=int,   default=0)
    args = ap.parse_args()

    out_root    = Path(args.output_dir)
    frames_root = out_root / "frames"
    annot_root  = out_root / "yolo_frames"
    jsonl_path  = out_root / "yolo.jsonl"
    annot_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise SystemExit("--gpus must list at least one GPU id")

    all_videos = gather_videos(frames_root)
    done = load_done_jsonl(jsonl_path)
    todo = [(s, v) for s, v in all_videos if v not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Found {len(all_videos)} (slug, video_id) pairs with frames")
    print(f"Already done in JSONL: {len(done)}")
    print(f"To process           : {len(todo)}")
    print(f"GPUs                 : {gpu_ids}\n")

    if not todo:
        print("Nothing to do.")
        return

    shards: list[list[tuple[str, str]]] = [[] for _ in gpu_ids]
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
                        args=(gid, sh, str(frames_root), str(annot_root),
                              args.model, args.conf, args.fps, q))
        p.start(); procs.append(p)

    collected = []
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

    flat.sort(key=lambda r: (r["event_slug"], r["video_id"]))
    append_jsonl(jsonl_path, flat)

    print(f"\nWall: {time.time()-t_start:.1f}s")
    print(f"Wrote {len(flat)} records → {jsonl_path}")


if __name__ == "__main__":
    main()
