"""Extract 16 kHz mono WAV per video from an events.json input.

Layout:
    <out>/audio/<event_slug>/<video_id>.wav

Resumable: any (slug, video_id) whose .wav already exists is skipped.

Usage:
    python -m pipeline.extract_audio \
        --input      events.json \
        --videos-dir /path/to/videos \
        --output-dir out/ \
        --workers    8
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

from .io_utils import load_events, resolve_video_path


SAMPLE_RATE = 16_000


def extract_one(args) -> tuple[str, str, str]:
    slug, vid, src, dst = args
    dst_path = Path(dst)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return (slug, vid, "skip-already-done")
    if not src or not Path(src).exists():
        return (slug, vid, "missing-on-disk")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn", str(dst_path)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        return (slug, vid, f"ffmpeg-failed: {e.returncode}")
    return (slug, vid, "ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True, help="events.json input")
    ap.add_argument("--videos-dir", required=True, help="directory containing <video_id>.{mp4,mkv,...}")
    ap.add_argument("--output-dir", required=True, help="pipeline output root")
    ap.add_argument("--workers",    type=int, default=8)
    ap.add_argument("--limit",      type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    audio_root = out_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    events = load_events(args.input)
    work = []
    for ev in events:
        for vid in ev["videos"]:
            src = resolve_video_path(args.videos_dir, vid)
            dst = audio_root / ev["slug"] / f"{vid}.wav"
            work.append((ev["slug"], vid, str(src) if src else "", str(dst)))
    if args.limit:
        work = work[:args.limit]

    print(f"Events       : {len(events)}")
    print(f"Videos total : {len(work)}")
    print(f"Out          : {audio_root}/<event_slug>/<video_id>.wav")
    print(f"Workers      : {args.workers}\n")

    t0 = time.time()
    statuses = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(extract_one, work), 1):
            statuses.append(res)
            slug, vid, st = res
            print(f"[{i:4d}/{len(work)}] {slug}/{vid}  {st}", flush=True)

    by_status: dict[str, int] = {}
    for _, _, st in statuses:
        by_status[st.split(":")[0]] = by_status.get(st.split(":")[0], 0) + 1
    stats_path = out_root / "audio_stats.json"
    json.dump({
        "sample_rate_hz": SAMPLE_RATE,
        "total_videos":   len(work),
        "elapsed_sec":    round(time.time() - t0, 1),
        "by_status":      by_status,
        "details":        [{"slug": s, "video_id": v, "status": st} for s, v, st in statuses],
    }, open(stats_path, "w"), indent=2, ensure_ascii=False)

    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"Status : {by_status}")
    print(f"Stats  : {stats_path}")


if __name__ == "__main__":
    main()
