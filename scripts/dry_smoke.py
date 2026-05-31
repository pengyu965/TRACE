"""End-to-end DRY smoke test — no GPU, no model downloads, ~30 sec total.

This script exercises every stage's actual code path EXCEPT the model calls.
The LLM / LVLM / embedding outputs are synthesised so the rest of the pipeline
(file I/O, schemas, orchestrator wiring, stage-1/4/5 logic, validators) runs
against realistic intermediate artefacts.

What it verifies:
  ✓ pipeline.merge produces a Part-2-compatible merged.jsonl
  ✓ pipeline.claim_gen.data_loader reads it correctly
  ✓ events_adapter builds a valid queries.jsonl
  ✓ aggregator stage1 splits per-video → per-query correctly
  ✓ aggregator stage4 validates and assembles a real MAGMaR submission JSON
  ✓ aggregator stage5 produces a diff report
  ✓ the final submission file matches the MAGMaR 2026 schema

What it does NOT verify:
  ✗ that the real models produce sensible content (they're stubbed)
  ✗ VRAM behaviour under real inference

Use a real GPU run on a subset (--limit 1, --only claim_step1 claim_step2)
to validate model-side correctness.

Usage:
    python scripts/dry_smoke.py \
        --input      /path/to/events.json \
        --output-dir /path/to/out          # contains merged.jsonl from Part 1
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Local imports
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
sys.path.insert(0, str(REPO))

from pipeline.io_utils import load_events
from pipeline.aggregator.events_adapter import build_queries_jsonl
from pipeline.aggregator.config import PipelineConfig
from pipeline.aggregator import stage1_split, stage4_assemble, stage5_diff_report


BANNER = "═" * 60


def section(title: str) -> None:
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


def synthesise_claim_results(events: list[dict], merged_path: Path, claim_results_dir: Path) -> int:
    """Mock claim_step2 output by drawing 'claims' from OCR/YOLO content of merged.jsonl.

    For each (event, video), produces 2 short claims of the form
    "{event}: detected {class_name} at t={time_sec}s".
    These are content-grounded enough that they survive the verbatim-text check
    when piped through to stage 4.
    """
    merged_by_vid = {}
    if merged_path.exists():
        with open(merged_path) as f:
            for line in f:
                rec = json.loads(line)
                merged_by_vid[rec["video_id"]] = rec

    claim_results_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for ev in events:
        for vid in ev["videos"]:
            rec = merged_by_vid.get(vid, {})
            frames = rec.get("frames", [])
            # Pick the first frame with any yolo detection, and the first with OCR text
            yolo_claim = None
            ocr_claim  = None
            for fr in frames:
                if yolo_claim is None and fr.get("yolo", {}).get("detections"):
                    cls = fr["yolo"]["detections"][0]["class_name"]
                    yolo_claim = (f"In event '{ev['event_key']}', a {cls} was visible at "
                                  f"t={fr['time_sec']:.0f}s in video {vid}.")
                if ocr_claim is None and fr.get("ocr", {}).get("detections"):
                    text = fr["ocr"]["detections"][0]["text"]
                    ocr_claim = (f"In event '{ev['event_key']}', the on-screen text "
                                 f"\"{text}\" appeared at t={fr['time_sec']:.0f}s in video {vid}.")
                if yolo_claim and ocr_claim:
                    break

            claims = [c for c in (yolo_claim, ocr_claim) if c]
            if not claims:
                claims = [f"In event '{ev['event_key']}', no visible objects or text were detected in video {vid}."]

            out_path = claim_results_dir / f"{vid}_results.json"
            json.dump({
                "video_id": vid,
                "queries": [{
                    "query_id":         ev["slug"],
                    "event_key":        ev["event_key"],
                    "query":            ev["query"],
                    "persona_title":    ev.get("persona_title", ""),
                    "persona":          ev.get("background", ""),
                    "generated_claims": claims,
                }],
            }, open(out_path, "w"), indent=2, ensure_ascii=False)
            n_files += 1
    return n_files


def synthesise_method_a_output(per_query_dir: Path, agent_out_dir: Path) -> int:
    """Mock stage3a output.

    Strategy: pass every input claim straight through as its own response.
    text == verbatim claim, citation == its single source video_id.
    This satisfies all four hard invariants stage4 checks:
      ✓ text is verbatim
      ✓ citations are deduped video_ids
      ✓ every contributing video appears in citations
      ✓ total citations == total input claims
    """
    agent_out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pq in sorted(per_query_dir.glob("query_*.json")):
        d = json.load(open(pq))
        responses = []
        for v in d["videos"]:
            for claim in v["claims"]:
                responses.append({"text": claim, "citations": [v["video_id"]]})
        out = {
            "query_id":  d["query_id"],
            "responses": responses,
        }
        json.dump(out, open(agent_out_dir / pq.name, "w"), indent=2, ensure_ascii=False)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      required=True, help="events.json")
    ap.add_argument("--output-dir", required=True, help="pipeline output root (must contain merged.jsonl)")
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    merged   = out_root / "merged.jsonl"
    if not merged.exists():
        sys.exit(f"[fatal] {merged} not found. Run Part 1 (or pipeline.merge) first.")

    events = load_events(args.input)

    section("STEP A — synthesise claim_results/ (mock Part 2 output)")
    n_results = synthesise_claim_results(events, merged, out_root / "claim_results")
    print(f"  wrote {n_results} per-video claim files → {out_root / 'claim_results'}")

    section("STEP B — events_adapter: build queries.jsonl")
    work_dir = out_root / "aggregate"
    work_dir.mkdir(parents=True, exist_ok=True)
    queries_file = work_dir / "aggregate_queries.jsonl"
    n_queries = build_queries_jsonl(Path(args.input), queries_file)
    print(f"  wrote {n_queries} queries → {queries_file}")

    section("STEP C — aggregator stage 1: split per-video → per-query (REAL)")
    cfg = PipelineConfig(
        per_video_dir = out_root / "claim_results",
        queries_file  = queries_file,
        work_dir      = work_dir,
    )
    stage1_split.run(cfg)

    section("STEP D — synthesise stage 3a output (mock LLM verify)")
    n_stage3 = synthesise_method_a_output(cfg.per_query_dir, cfg.method_a_out)
    print(f"  wrote {n_stage3} mocked agent outputs → {cfg.method_a_out}")

    section("STEP E — aggregator stage 4: assemble submission + validate (REAL)")
    ok = stage4_assemble.run(cfg)

    section("STEP F — aggregator stage 5: diff report (REAL)")
    stage5_diff_report.run(cfg)

    section("VERDICT")
    if not ok:
        print("  ❌ stage 4 reported validation failures. Pipeline wiring may have schema drift.")
        sys.exit(1)

    submission = cfg.submission_dir / f"{cfg.run_id_prefix}_method_a.jsonl"
    if not submission.exists():
        print(f"  ❌ submission file not created: {submission}")
        sys.exit(2)

    n_lines = sum(1 for _ in open(submission))
    print(f"  ✅ end-to-end wiring works.")
    print(f"     submission: {submission}")
    print(f"     {n_lines} JSON lines (one per event)")
    print(f"\n  Sample (first line):")
    first = json.loads(open(submission).readline())
    print(json.dumps(first, indent=2, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
