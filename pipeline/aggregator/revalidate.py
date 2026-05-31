"""Re-run validators against saved raw LLM traces.

When the validator rules change (e.g. tighter / looser coverage) or you want
to try a different fallback strategy, you can re-evaluate the same model
attempts that were captured at run time without re-invoking the model.

Reads:
    cfg.raw_a_dir / query_<qid>.json   # produced by stage 3a
    cfg.raw_b_dir / query_<qid>.json   # produced by stage 3b

Rewrites:
    cfg.method_a_out / query_<qid>.json
    cfg.method_b_out / query_<qid>.json

You typically follow up with `stage4` and `stage5`.
"""
from __future__ import annotations
import json

from . import data_io
from .validate import check_method_a, check_method_b, ValidationError


def _replay_a_cluster(cluster_record: dict) -> tuple[list[dict], str, int, str | None]:
    members = cluster_record["members"]
    attempts = cluster_record.get("attempts", [])
    last_err: str | None = cluster_record.get("last_error")
    chosen_n = 0
    for att in attempts:
        chosen_n = att["n"]
        parsed = None
        try:
            parsed = json.loads(att.get("json_text") or "") if att.get("json_text") else None
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            last_err = f"JSON parse failed: {att.get('parse_error')}"
            continue
        try:
            normalised = check_method_a(parsed.get("responses", []), members)
            return normalised, "ok", chosen_n, None
        except ValidationError as e:
            last_err = str(e)
            continue
    fallback = [{"text": m["claim"], "citations": [m["video_id"]]} for m in members]
    return fallback, "fallback", chosen_n or len(attempts), last_err


def _replay_b_query(record: dict) -> tuple[list[dict], str, int, str | None]:
    if record.get("status") == "skipped-empty":
        return [], "skipped-empty", 0, None
    flat = record["flat_claims"]
    attempts = record.get("attempts", [])
    last_err: str | None = record.get("last_error")
    chosen_n = 0
    for att in attempts:
        chosen_n = att["n"]
        parsed = None
        try:
            parsed = json.loads(att.get("json_text") or "") if att.get("json_text") else None
        except json.JSONDecodeError:
            parsed = None
        if parsed is None:
            last_err = f"JSON parse failed: {att.get('parse_error')}"
            continue
        try:
            normalised = check_method_b(parsed.get("responses", []), flat)
            return normalised, "ok", chosen_n, None
        except ValidationError as e:
            last_err = str(e)
            continue
    fallback = [{"text": c["claim"], "citations": [c["video_id"]]} for c in flat]
    return fallback, "fallback", chosen_n or len(attempts), last_err


def run(cfg):
    cfg.ensure_dirs()
    n_a_files = n_b_files = 0
    n_a_fb_before = n_a_fb_after = 0
    n_b_fb_before = n_b_fb_after = 0

    for p in sorted(cfg.raw_a_dir.glob("query_*.json")):
        with open(p) as f:
            d = json.load(f)
        qid = d["query_id"]
        rebuilt_responses: list[dict] = []
        diag: list[dict] = []
        for cluster_record in d.get("clusters", []):
            if cluster_record["status"] == "fallback":
                n_a_fb_before += 1
            responses, status, n_attempts, err = _replay_a_cluster(cluster_record)
            rebuilt_responses.extend(responses)
            if status == "fallback":
                n_a_fb_after += 1
            diag.append({
                "cluster_id": cluster_record["cluster_id"],
                "size": len(cluster_record["members"]),
                "status": status, "n_attempts": n_attempts, "error": err,
            })
        out = {
            "query_id": qid, "method": "A",
            "n_input_claims": sum(len(c["members"]) for c in d.get("clusters", [])),
            "n_output_responses": len(rebuilt_responses),
            "responses": rebuilt_responses,
            "diagnostics": {
                "n_clusters": len(diag),
                "n_fallback_clusters": sum(1 for x in diag if x["status"] == "fallback"),
                "per_cluster_attempts": diag,
            },
        }
        data_io.write_json(cfg.method_a_out / f"query_{qid}.json", out)
        n_a_files += 1

    for p in sorted(cfg.raw_b_dir.glob("query_*.json")):
        with open(p) as f:
            d = json.load(f)
        qid = d["query_id"]
        if d.get("status") == "fallback":
            n_b_fb_before += 1
        responses, status, n_attempts, err = _replay_b_query(d)
        if status == "fallback":
            n_b_fb_after += 1
        n_in = len(d.get("flat_claims", []))
        out = {
            "query_id": qid, "method": "B",
            "n_input_claims": n_in,
            "n_output_responses": len(responses),
            "responses": responses,
            "diagnostics": {"status": status, "n_attempts": n_attempts, "error": err},
        }
        data_io.write_json(cfg.method_b_out / f"query_{qid}.json", out)
        n_b_files += 1

    print(f"[revalidate] Method A: rewrote {n_a_files} files. "
          f"Fallback clusters: {n_a_fb_before} -> {n_a_fb_after}")
    print(f"[revalidate] Method B: rewrote {n_b_files} files. "
          f"Fallback queries:   {n_b_fb_before} -> {n_b_fb_after}")
