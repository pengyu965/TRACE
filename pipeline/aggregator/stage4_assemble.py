"""Stage 4: assemble final MAGMaR submission JSONL files and validate.

For each method (A, B):
  - Read per-query agent output.
  - Build the submission line: {metadata, responses, references}.
  - Run the same per-query checks the upstream pipeline used.
  - Write cfg.submission_dir / {run_id_prefix}_method_{a,b}.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path

from . import data_io


def _norm(s: str) -> str:
    return " ".join(s.split())


def _load_per_query(pq_dir: Path, qid: str):
    with open(pq_dir / f"query_{qid}.json") as f:
        d = json.load(f)
    expected_videos: list[str] = []
    seen: set[str] = set()
    input_claims: set[tuple[str, str]] = set()
    for v in d["videos"]:
        if v["video_id"] not in seen and v["claims"]:
            expected_videos.append(v["video_id"])
            seen.add(v["video_id"])
        for c in v["claims"]:
            input_claims.add((v["video_id"], _norm(c)))
    n_claims = sum(len(v["claims"]) for v in d["videos"])
    return expected_videos, input_claims, n_claims


def _assemble_one(method: str, qid: str, agent_out_dir: Path, expected_videos, input_claims,
                  n_input_claims, run_id: str, team_id: str, task: str):
    with open(agent_out_dir / f"query_{qid}.json") as f:
        d = json.load(f)
    responses = d["responses"]
    issues: list[str] = []
    warns: list[str] = []

    refs: list[str] = []
    refs_seen: set[str] = set()
    citations_total = 0
    expected_video_set = set(expected_videos)
    cited: set[str] = set()
    norm_input = {nc for _, nc in input_claims}

    for i, r in enumerate(responses):
        text = r.get("text", "")
        cits = r.get("citations", [])
        if not text or not text.strip():
            issues.append(f"  response[{i}] empty text")
        if not cits:
            issues.append(f"  response[{i}] empty citations")
        if len(cits) != len(set(cits)):
            issues.append(f"  response[{i}] duplicate citations: {cits}")
        for v in cits:
            if v not in expected_video_set:
                issues.append(f"  response[{i}] cites unknown video {v!r}")
            if v not in refs_seen:
                refs.append(v)
                refs_seen.add(v)
            cited.add(v)
        citations_total += len(cits)
        if _norm(text) not in norm_input:
            warns.append(f"  response[{i}] text not verbatim: {text[:80]!r}")

    missing = expected_video_set - cited
    if missing:
        issues.append(f"  videos with claims but not cited: {sorted(missing)}")
    if citations_total != n_input_claims:
        warns.append(
            f"  citations_total={citations_total} != n_input_claims={n_input_claims}"
        )

    line = {
        "metadata": {
            "run_id": run_id,
            "query_id": qid,
            "team_id": team_id,
            "task": task,
        },
        "responses": [
            {"text": r["text"], "citations": list(dict.fromkeys(r["citations"]))}
            for r in responses
        ],
        "references": refs,
    }
    return line, issues, warns, citations_total


def run(cfg) -> bool:
    cfg.ensure_dirs()
    qids = sorted(p.stem.replace("query_", "") for p in cfg.per_query_dir.glob("query_*.json"))
    overall_ok = True

    for method, agent_dir in (("A", cfg.method_a_out), ("B", cfg.method_b_out)):
        if method == "A" and not cfg.run_method_a:
            continue
        if method == "B" and not cfg.run_method_b:
            continue
        run_id = f"{cfg.run_id_prefix}_method_{method.lower()}"
        out_path = cfg.submission_dir / f"{run_id}.jsonl"
        lines = []
        rows = []
        print(f"\n[stage4] === Method {method} ===")
        for qid in qids:
            exp, inp, nin = _load_per_query(cfg.per_query_dir, qid)
            line, issues, warns, cit_total = _assemble_one(
                method, qid, agent_dir, exp, inp, nin, run_id, cfg.team_id, cfg.task,
            )
            status = "OK"
            if issues:
                status = "FAIL"
                overall_ok = False
                print(f"  qid {qid}: FAIL")
                for iss in issues:
                    print(iss)
            for w in warns:
                print(f"  qid {qid} WARN: {w}")
            rows.append((qid, nin, len(line["responses"]), len(line["references"]),
                         cit_total, status))
            lines.append(line)
        data_io.write_jsonl(out_path, lines)
        print(f"[stage4] wrote {out_path}")
        print(f"  {'qid':>6} {'in':>5} {'resp':>5} {'refs':>5} {'cits':>5}  status")
        for qid, nin, nr, nref, cit, st in rows:
            print(f"  {qid:>6} {nin:>5} {nr:>5} {nref:>5} {cit:>5}  {st}")

    print(f"\n[stage4] overall: {'OK' if overall_ok else 'FAILED'}")
    return overall_ok
