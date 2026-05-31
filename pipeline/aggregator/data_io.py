"""Loaders for the upstream per-video result schema and queries.jsonl."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def load_queries(path: Path) -> dict[str, dict]:
    """queries.jsonl -> {query_id: query_record}"""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = str(r["query_id"])
            r["query_id"] = qid
            out[qid] = r
    return out


def load_per_video(per_video_dir: Path) -> list[dict]:
    """Read every *.json under per_video_dir. Each file is expected to look like:
       {video_id, queries: [{query_id, query, persona_title, persona, generated_claims:[...]}, ...]}
    """
    records = []
    for p in sorted(per_video_dir.glob("*.json")):
        with open(p) as f:
            records.append(json.load(f))
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
