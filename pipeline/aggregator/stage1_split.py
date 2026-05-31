"""Stage 1: Split per-video result JSONs into per-query JSONs.

Input shape per video file (under cfg.per_video_dir):
{
  "video_id": "abc.mp4",
  "queries": [
    {"query_id": "1", "query": "...", "persona_title": "...", "persona": "...",
     "generated_claims": ["claim 1", "claim 2", ...]},
    ...
  ]
}

queries.jsonl (cfg.queries_file): one record per query carrying
  {query_id, query_type, language, title, persona_title, background, query}

Output (cfg.per_query_dir / query_<qid>.json):
{
  "query_id": "1",
  "query_type": ..., "language": ..., "title": ..., "persona_title": ...,
  "background": ..., "query": ...,
  "videos": [{"video_id": "...", "claims": ["...", ...]}, ...]
}
"""
from __future__ import annotations
from collections import defaultdict
from . import data_io


def run(cfg):
    cfg.ensure_dirs()
    queries = data_io.load_queries(cfg.queries_file)
    per_video = data_io.load_per_video(cfg.per_video_dir)

    # qid -> video_id -> list[claim]
    bucket: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for v in per_video:
        vid = v["video_id"]
        for qrec in v.get("queries", []):
            qid = str(qrec["query_id"])
            claims = list(qrec.get("generated_claims", []))
            bucket[qid][vid].extend(claims)

    written = 0
    for qid, query in queries.items():
        videos_block = []
        # Stable order: per the per-video file order found on disk.
        for v in per_video:
            vid = v["video_id"]
            if vid in bucket.get(qid, {}):
                videos_block.append({"video_id": vid, "claims": bucket[qid][vid]})
        out = {
            "query_id": qid,
            "query_type": query.get("query_type"),
            "language": query.get("language"),
            "title": query.get("title"),
            "persona_title": query.get("persona_title"),
            "background": query.get("background"),
            "query": query.get("query"),
            "videos": videos_block,
        }
        data_io.write_json(cfg.per_query_dir / f"query_{qid}.json", out)
        written += 1
    print(f"[stage1] wrote {written} per-query files to {cfg.per_query_dir}")
