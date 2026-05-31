"""Bridge: build a Part-3-compatible queries.jsonl from the unified events.json.

The aggregator's `stage1_split` expects one record per query with these fields:
    query_id, query_type, language, title, persona_title, background, query, persona

This script materialises that file from our events.json (which carries
event_key, query, persona_title, background, language, videos) so the rest of
the aggregator runs unchanged.

Mapping:
    query_id       <- event_slug             (matches what Part 2 wrote in claim_results)
    query_type     <- "factoid"              (no slot in events.json; default)
    language       <- ev.language ("en")      (normalised: english->en, russian->ru, ...)
    title          <- ev.event_key
    persona_title  <- ev.persona_title or ""
    background     <- ev.background    or ""
    query          <- ev.query
    persona        <- ev.background    or ""  (Part 3 uses this field; equal to background)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ..io_utils import load_events


_LANG_TO_ISO = {
    "english": "en", "en": "en",
    "russian": "ru", "ru": "ru",
    "chinese": "zh", "zh": "zh", "zh-cn": "zh",
    "japanese": "ja", "ja": "ja",
    "korean": "ko", "ko": "ko",
    "spanish": "es", "es": "es",
    "french": "fr", "fr": "fr",
    "german": "de", "de": "de",
    "hindi": "hi", "hi": "hi",
    "arabic": "ar", "ar": "ar",
    "nepali": "ne", "ne": "ne",
    "burmese": "my", "my": "my",
    "thai": "th", "th": "th",
}


def _to_iso(lang: str) -> str:
    return _LANG_TO_ISO.get((lang or "english").strip().lower(), (lang or "en").strip().lower())


def build_queries_jsonl(events_path: Path, out_path: Path) -> int:
    events = load_events(str(events_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in events:
            rec = {
                "query_id":      ev["slug"],
                "query_type":    "factoid",
                "language":      _to_iso(ev.get("language", "english")),
                "title":         ev["event_key"],
                "persona_title": ev.get("persona_title", ""),
                "background":    ev.get("background", ""),
                "query":         ev["query"],
                "persona":       ev.get("background", ""),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(events)


def main():
    ap = argparse.ArgumentParser(
        description="Materialise queries.jsonl from events.json for the aggregator stage."
    )
    ap.add_argument("--input",      required=True, help="events.json")
    ap.add_argument("--output",     required=True, help="where to write queries.jsonl")
    args = ap.parse_args()
    n = build_queries_jsonl(Path(args.input), Path(args.output))
    print(f"Wrote {n} query records → {args.output}")


if __name__ == "__main__":
    main()
