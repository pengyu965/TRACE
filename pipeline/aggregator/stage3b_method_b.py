"""Stage 3B: pure-LLM clustering with Qwen3-30B Instruct.

One prompt per query: the model sees the full flat list of (video_id, claim)
and produces a list of {text, citations} groups. We batch all queries into one
vLLM call.

Output: cfg.method_b_out / query_<qid>.json
{
  "query_id": "...",
  "method": "B",
  "n_input_claims": N,
  "n_output_responses": K,
  "responses": [{"text": "...", "citations": [...]}, ...],
  "diagnostics": {"status": ..., "attempts": ..., "error": ...}
}
"""
from __future__ import annotations
import json
from pathlib import Path

from . import data_io
from .prompts import build_method_b_user
from .validate import check_method_b, ValidationError
from ._llm_runner import run_with_retry
from .qwen_client import QwenAggregator


def _query_ctx_from_per_query(d: dict) -> dict:
    return {
        "query_id": d["query_id"],
        "title": d.get("title") or "",
        "persona_title": d.get("persona_title") or "",
        "background": d.get("background") or "",
        "query": d.get("query") or "",
    }


def _flatten(pq: dict) -> list[dict]:
    out = []
    for v in pq["videos"]:
        for c in v["claims"]:
            out.append({"video_id": v["video_id"], "claim": c})
    return out


def run(cfg, client: QwenAggregator | None = None):
    cfg.ensure_dirs()
    if client is None:
        client = QwenAggregator(cfg)

    per_query_files = sorted(cfg.per_query_dir.glob("query_*.json"))
    items: list[dict] = []
    routing: list[tuple[str, list[dict]]] = []  # (qid, flat_claims)

    for pq_path in per_query_files:
        with open(pq_path) as f:
            pq = json.load(f)
        qid = pq["query_id"]
        flat = _flatten(pq)
        if not flat:
            routing.append((qid, flat))
            items.append(None)  # type: ignore[arg-type]
            continue
        ctx = _query_ctx_from_per_query(pq)
        user = build_method_b_user(ctx, flat)

        def _validate(responses, _flat=flat):
            return check_method_b(responses, _flat)

        def _fallback(_flat=flat):
            return [{"text": c["claim"], "citations": [c["video_id"]]} for c in _flat]

        items.append({"user": user, "validate": _validate, "fallback": _fallback})
        routing.append((qid, flat))

    # Strip empty queries before sending to the model, but keep their position.
    real_indices = [i for i, it in enumerate(items) if it is not None]
    real_items = [items[i] for i in real_indices]
    print(f"[stage3b] sending {len(real_items)} per-query prompts to {cfg.llm_model}")
    real_results = run_with_retry(client, real_items, max_retries=cfg.llm_max_retries)

    results: list[dict | None] = [None] * len(items)
    for i, r in zip(real_indices, real_results):
        results[i] = r

    n_fb = 0
    for (qid, flat), res in zip(routing, results):
        if res is None:
            out = {
                "query_id": qid, "method": "B",
                "n_input_claims": 0, "n_output_responses": 0,
                "responses": [],
                "diagnostics": {"status": "skipped-empty", "n_attempts": 0, "error": None},
            }
            raw = {"query_id": qid, "status": "skipped-empty", "flat_claims": flat, "attempts": []}
        else:
            out = {
                "query_id": qid, "method": "B",
                "n_input_claims": len(flat),
                "n_output_responses": len(res["responses"]),
                "responses": res["responses"],
                "diagnostics": {
                    "status": res["status"],
                    "n_attempts": res["n_attempts"],
                    "error": res["last_error"],
                },
            }
            raw = {
                "query_id": qid,
                "flat_claims": flat,
                "status": res["status"],
                "n_attempts": res["n_attempts"],
                "last_error": res["last_error"],
                "final_responses": res["responses"],
                "attempts": res["attempts"],
            }
            if res["status"] == "fallback":
                n_fb += 1
        data_io.write_json(cfg.method_b_out / f"query_{qid}.json", out)
        data_io.write_json(cfg.raw_b_dir / f"query_{qid}.json", raw)

    print(f"[stage3b] done. fallback queries: {n_fb}/{len(real_results)}")
    print(f"[stage3b] raw model traces written to {cfg.raw_b_dir}")
