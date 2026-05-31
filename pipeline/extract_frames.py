"""Extract 1-fps frames per video into event-wise folders.

Layout:
    <out>/frames/<event_slug>/<video_id>/frame_<NNNNN>.jpg

Resumable: any video_id whose frame folder already has frames is skipped.

Usage:
    python -m pipeline.extract_frames \
        --input      events.json \
        --videos-dir /path/to/videos \
        --output-dir out/ \
        --workers    8 \
        --fps        1.0
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

from .io_utils import load_events, resolve_video_path


def extract_one(args) -> tuple[str, str, int, str]:
    slug, vid, src, dst, fps = args
    dst_dir = Path(dst)
    if dst_dir.is_dir() and any(dst_dir.glob("frame_*.jpg")):
        return (slug, vid, len(list(dst_dir.glob("frame_*.jpg"))), "skip-already-done")
    if not src or not Path(src).exists():
        return (slug, vid, 0, "missing-on-disk")
    dst_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(dst_dir / "frame_%05d.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-vf", f"fps={fps}", "-q:v", "2", pattern],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return (slug, vid, 0, f"ffmpeg-failed: {e.returncode}")
    n = len(list(dst_dir.glob("frame_*.jpg")))
    return (slug, vid, n, "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fps",        type=float, default=1.0)
    ap.add_argument("--workers",    type=int,   default=8)
    ap.add_argument("--limit",      type=int,   default=0)
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    frames_root = out_root / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    events = load_events(args.input)
    work = []
    for ev in events:
        for vid in ev["videos"]:
            src = resolve_video_path(args.videos_dir, vid)
            dst = frames_root / ev["slug"] / vid
            work.append((ev["slug"], vid, str(src) if src else "", str(dst), args.fps))
    if args.limit:
        work = work[:args.limit]

    print(f"Events       : {len(events)}")
    print(f"Videos total : {len(work)}")
    print(f"FPS          : {args.fps}")
    print(f"Out          : {frames_root}/<event_slug>/<video_id>/frame_*.jpg")
    print(f"Workers      : {args.workers}\n")

    t0 = time.time()
    statuses = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(extract_one, work), 1):
            statuses.append(res)
            slug, vid, n, st = res
            print(f"[{i:4d}/{len(work)}] {slug}/{vid}  n={n}  {st}", flush=True)

    by_status: dict[str, int] = {}
    for _, _, _, st in statuses:
        k = st.split(":")[0]
        by_status[k] = by_status.get(k, 0) + 1
    stats_path = out_root / "frames_stats.json"
    json.dump({
        "fps":          args.fps,
        "total_videos": len(work),
        "elapsed_sec":  round(time.time() - t0, 1),
        "by_status":    by_status,
        "details":      [{"slug": s, "video_id": v, "n_frames": n, "status": st}
                         for s, v, n, st in statuses],
    }, open(stats_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"Status : {by_status}")
    print(f"Stats  : {stats_path}")


if __name__ == "__main__":
    main()
