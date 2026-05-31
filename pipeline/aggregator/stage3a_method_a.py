"""Stage 3A: per-cluster verify/refine with Qwen3-30B Instruct.

For every cluster across all queries, send one prompt asking the model to:
  - keep as one same-fact group, or
  - split into multiple sub-groups, or
  - return the singleton untouched.

We batch ALL clusters across ALL queries into one vLLM call for throughput,
then re-group the per-cluster responses back into per-query output files.

Output: cfg.method_a_out / query_<qid>.json
{
  "query_id": "...",
  "method": "A",
  "n_input_claims": N,
  "n_output_responses": K,
  "responses": [{"text": "...", "citations": [...]}, ...],
  "diagnostics": {"n_clusters": ..., "n_fallback_clusters": ..., "per_cluster_attempts": [...]}
}
"""
from __future__ import annotations
import json
from pathlib import Path

from . import data_io
from .prompts import build_method_a_user
from .validate import check_method_a, ValidationError
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


def run(cfg, client: QwenAggregator | None = None):
    cfg.ensure_dirs()
    if client is None:
        client = QwenAggregator(cfg)

    per_query_files = sorted(cfg.per_query_dir.glob("query_*.json"))
    work_items: list[dict] = []
    # parallel list: (qid, cluster_id, members) for reassembly
    routing: list[tuple[str, int, list[dict]]] = []

    for pq_path in per_query_files:
        with open(pq_path) as f:
            pq = json.load(f)
        qid = pq["query_id"]
        ctx = _query_ctx_from_per_query(pq)
        cl_path = cfg.clusters_dir / f"query_{qid}_clusters.json"
        with open(cl_path) as f:
            cl = json.load(f)
        for c in cl["clusters"]:
            members = c["members"]
            user = build_method_a_user(ctx, members, cfg.cluster_tau)

            # Closures pick up these names per-iteration.
            def _validate(responses, _members=members):
                return check_method_a(responses, _members)

            def _fallback(_members=members):
                # One singleton per member, verbatim.
                return [
                    {"text": m["claim"], "citations": [m["video_id"]]} for m in _members
                ]
            work_items.append({"user": user, "validate": _validate, "fallback": _fallback})
            routing.append((qid, c["cluster_id"], members))

    print(f"[stage3a] sending {len(work_items)} cluster prompts to {cfg.llm_model}")
    results = run_with_retry(client, work_items, max_retries=cfg.llm_max_retries)

    # Reassemble per-query outputs in cluster_id order, and save raw model
    # traces (every attempt's prompt + thinking + json + errors) to disk so
    # the run can be re-validated later without re-invoking the model.
    per_query_out: dict[str, dict] = {}
    diag_per_q: dict[str, list[dict]] = {}
    raw_per_q: dict[str, list[dict]] = {}
    for (qid, cid, members), res in zip(routing, results):
        per_query_out.setdefault(qid, {"responses": []})
        per_query_out[qid]["responses"].extend(res["responses"])
        diag_per_q.setdefault(qid, []).append({
            "cluster_id": cid, "size": len(members),
            "status": res["status"], "n_attempts": res["n_attempts"],
            "error": res["last_error"],
        })
        raw_per_q.setdefault(qid, []).append({
            "cluster_id": cid,
            "members": members,
            "status": res["status"],
            "n_attempts": res["n_attempts"],
            "last_error": res["last_error"],
            "final_responses": res["responses"],
            "attempts": res["attempts"],
        })

    # Sum input claims per query for diagnostics.
    for pq_path in per_query_files:
        with open(pq_path) as f:
            pq = json.load(f)
        qid = pq["query_id"]
        n_in = sum(len(v["claims"]) for v in pq["videos"])
        responses = per_query_out.get(qid, {"responses": []})["responses"]
        diag = diag_per_q.get(qid, [])
        out = {
            "query_id": qid,
            "method": "A",
            "n_input_claims": n_in,
            "n_output_responses": len(responses),
            "responses": responses,
            "diagnostics": {
                "n_clusters": len(diag),
                "n_fallback_clusters": sum(1 for d in diag if d["status"] == "fallback"),
                "per_cluster_attempts": diag,
            },
        }
        data_io.write_json(cfg.method_a_out / f"query_{qid}.json", out)
        data_io.write_json(
            cfg.raw_a_dir / f"query_{qid}.json",
            {"query_id": qid, "clusters": raw_per_q.get(qid, [])},
        )

    n_fb = sum(1 for r in results if r["status"] == "fallback")
    print(f"[stage3a] done. fallback clusters: {n_fb}/{len(results)}")
    print(f"[stage3a] raw model traces written to {cfg.raw_a_dir}")
