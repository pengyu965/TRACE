"""Shared helpers: input parsing, event-slug canonicalisation, JSONL resume."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def load_events(input_path: str) -> list[dict]:
    """Parse the events.json input file.

    Expected schema:
        {"events": [{
            "event_key":     "...",        // required
            "query":         "...",        // required
            "videos":        ["vid1", ...],// required
            "persona_title": "...",        // optional (default "")
            "background":    "...",        // optional (default "")
            "language":      "english"     // optional (default "english")
         }, ...]}

    Returns a list of dicts with keys event_key, slug, query, videos,
    persona_title, background, language.
    """
    data = json.load(open(input_path))
    if "events" not in data or not isinstance(data["events"], list):
        raise ValueError(f"{input_path}: expected a top-level 'events' list")
    out = []
    seen_slugs = set()
    for i, ev in enumerate(data["events"]):
        for k in ("event_key", "query", "videos"):
            if k not in ev:
                raise ValueError(f"events[{i}]: missing required field '{k}'")
        if not isinstance(ev["videos"], list) or not ev["videos"]:
            raise ValueError(f"events[{i}] ({ev['event_key']}): 'videos' must be a non-empty list")
        slug = slugify(ev["event_key"])
        if slug in seen_slugs:
            raise ValueError(f"events[{i}]: slug collision on '{slug}' (from event_key '{ev['event_key']}')")
        seen_slugs.add(slug)
        out.append({
            "event_key":     ev["event_key"],
            "slug":          slug,
            "query":         ev["query"],
            "videos":        list(ev["videos"]),
            "persona_title": ev.get("persona_title", ""),
            "background":    ev.get("background", ""),
            "language":      ev.get("language", "english"),
        })
    return out


def video_pairs(events: list[dict]) -> list[tuple[str, str]]:
    """Flatten to [(slug, video_id), ...] in event order."""
    pairs = []
    for ev in events:
        for vid in ev["videos"]:
            pairs.append((ev["slug"], vid))
    return pairs


def load_done_jsonl(path: Path, key: str = "video_id") -> set[str]:
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)[key])
                except Exception:
                    pass
    return done


def append_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def resolve_video_path(videos_dir: str, video_id: str) -> Path | None:
    """Locate <videos_dir>/<video_id>.{mp4,mkv,webm,mov,m4v} — first match wins."""
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4v"):
        p = Path(videos_dir) / f"{video_id}{ext}"
        if p.exists():
            return p
    return None
