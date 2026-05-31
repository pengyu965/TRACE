"""Load and index dataset modalities by video_id, plus build queries from
events.json.

    ds = DatasetIndex()
    frames    = ds.get_merged_frames("abc123")
    asr_segs  = ds.get_timed_asr("abc123")
    queries   = ds.queries          # one entry per event
"""
from __future__ import annotations
import json
from pathlib import Path

from . import config
from ..io_utils import load_events


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class DatasetIndex:
    """Indexes merged.jsonl + asr.jsonl + events.json by video_id / event_key.

    The `queries` list mirrors what the original MAGMaR pipeline expected:
        {"metadata": {query_id, title, persona_title, background, query, language},
         "references": [video_id, ...],
         "responses":  []}
    `query_id` and `title` both default to the event_slug. The list of videos
    is carried as `references` so `topic_video_mapping` lookups still work.
    """

    def __init__(self) -> None:
        self.merged: dict[str, dict] = {
            r["video_id"]: r for r in _load_jsonl(config.MERGED_ANNOTS)
        }
        self.whisper: dict[str, dict] = {
            r["video_id"]: r for r in _load_jsonl(config.ASR_WHISPER)
        }
        # No second-modality Omni in this pipeline; alias to Whisper so calls
        # to get_omni_* keep working in case anyone references them.
        self.omni: dict[str, dict] = dict(self.whisper)

        events = load_events(config.EVENTS_FILE)
        self.queries: list[dict] = []
        self.topic_map: dict[str, list[str]] = {}
        for ev in events:
            slug = ev["slug"]
            self.topic_map[slug] = list(ev["videos"])
            self.queries.append({
                "metadata": {
                    "query_id":      slug,
                    "title":         slug,
                    "event_key":     ev["event_key"],
                    "persona_title": ev.get("persona_title", ""),
                    "background":    ev.get("background", ""),
                    "query":         ev["query"],
                    "language":      ev.get("language", "english"),
                },
                "references": list(ev["videos"]),
                "responses":  [],
            })
        self.query_map: dict[str, dict] = {
            r["metadata"]["query_id"]: r for r in self.queries
        }

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_topic_videos(self, title: str) -> list[str]:
        return self.topic_map.get(title, [])

    def video_ids(self) -> list[str]:
        return list(self.merged.keys())

    def get_merged_frames(self, video_id: str) -> list[dict]:
        return self.merged.get(video_id, {}).get("frames", [])

    def get_timed_asr(self, video_id: str) -> list[dict]:
        return self.whisper.get(video_id, {}).get("segments_english", [])

    def get_full_asr(self, video_id: str) -> str:
        return self.whisper.get(video_id, {}).get("english", "") or ""

    def get_image_size(self, video_id: str) -> list[int]:
        return self.merged.get(video_id, {}).get("image_size", [1920, 1080])

    def get_video_duration(self, video_id: str) -> float:
        rec = self.whisper.get(video_id, {})
        return rec.get("duration") or rec.get("duration_sec") or 0.0

    def build_asr_lookup(self, video_id: str) -> dict[int, str]:
        lookup: dict[int, str] = {}
        for seg in self.get_timed_asr(video_id):
            for t in range(int(seg["start"]), int(seg["end"]) + 1):
                lookup[t] = seg["text"]
        return lookup

    def __repr__(self) -> str:
        return (f"DatasetIndex(videos_with_annot={len(self.merged)}, "
                f"asr={len(self.whisper)}, queries={len(self.queries)})")
