"""Stage 5: Markdown report.

When both Method A and Method B are present, writes a side-by-side diff.
When only Method A is present, writes a Method-A-only run summary.
"""
from __future__ import annotations
import json
from pathlib import Path


def _load_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def run(cfg):
    sub_a = cfg.submission_dir / f"{cfg.run_id_prefix}_method_a.jsonl"
    sub_b = cfg.submission_dir / f"{cfg.run_id_prefix}_method_b.jsonl"
    a = {l["metadata"]["query_id"]: l for l in _load_jsonl(sub_a)}
    b = {l["metadata"]["query_id"]: l for l in _load_jsonl(sub_b)}
    qids = sorted(p.stem.replace("query_", "") for p in cfg.per_query_dir.glob("query_*.json"))
    has_b = bool(b)

    rows = []
    big_diffs = []
    for qid in qids:
        with open(cfg.per_query_dir / f"query_{qid}.json") as f:
            pq = json.load(f)
        topic = pq.get("title") or ""
        n_in = sum(len(v["claims"]) for v in pq["videos"])
        ar = a.get(qid, {}).get("responses", [])
        br = b.get(qid, {}).get("responses", [])
        a_multi = sum(1 for r in ar if len(r["citations"]) > 1)
        b_multi = sum(1 for r in br if len(r["citations"]) > 1)
        rows.append({"qid": qid, "topic": topic, "n_in": n_in,
                     "a_resp": len(ar), "b_resp": len(br),
                     "a_multi": a_multi, "b_multi": b_multi})
        if has_b and abs(len(ar) - len(br)) >= 2:
            big_diffs.append((abs(len(ar) - len(br)), qid, len(ar), len(br)))

    md: list[str] = []
    if has_b:
        md += [
            "# Method A vs Method B Diff Report", "",
            f"Method A (headline): Qwen3-Embedding-8B greedy single-link at tau={cfg.cluster_tau}, "
            f"then per-cluster {cfg.llm_model} verify/refine.",
            f"Method B (ablation): pure {cfg.llm_model} clustering from the flat claim list.",
            "",
            "## Per-query summary", "",
            "| qid | topic | in | A resp | B resp | A multi-cit | B multi-cit |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            md.append(f"| {r['qid']} | {r['topic']} | {r['n_in']} | {r['a_resp']} | "
                      f"{r['b_resp']} | {r['a_multi']} | {r['b_multi']} |")
    else:
        md += [
            "# Method A Run Report", "",
            f"Method A: Qwen3-Embedding-8B greedy single-link at tau={cfg.cluster_tau}, "
            f"then per-cluster {cfg.llm_model} verify/refine.",
            "",
            "## Per-query summary", "",
            "| qid | topic | in | A resp | A multi-cit |",
            "|---|---|---:|---:|---:|",
        ]
        for r in rows:
            md.append(f"| {r['qid']} | {r['topic']} | {r['n_in']} | {r['a_resp']} | {r['a_multi']} |")

    a_total_in = sum(r["n_in"] for r in rows)
    a_total_resp = sum(r["a_resp"] for r in rows)
    md += ["", "## Aggregate",
           f"- Total input claims: **{a_total_in}**",
           f"- Total responses Method A: **{a_total_resp}** ({a_total_in - a_total_resp} merged)"]
    if has_b:
        b_total_resp = sum(r["b_resp"] for r in rows)
        md.append(f"- Total responses Method B: **{b_total_resp}** ({a_total_in - b_total_resp} merged)")
    md.append("")

    if has_b:
        md.append("## Biggest A vs B divergences")
        if not big_diffs:
            md.append("_No queries with |A-B| >= 2._")
        else:
            md.append("| qid | A resp | B resp | abs diff |")
            md.append("|---|---:|---:|---:|")
            big_diffs.sort(reverse=True)
            for diff, qid, ar, br in big_diffs[:10]:
                md.append(f"| {qid} | {ar} | {br} | {diff} |")

    out_path = cfg.submission_dir / "diff_report.md"
    with open(out_path, "w") as f:
        f.write("\n".join(md))
    print(f"[stage5] wrote {out_path}")
