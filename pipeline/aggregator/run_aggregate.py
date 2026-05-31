"""Orchestrator-facing wrapper around the aggregator CLI.

Builds the queries.jsonl from events.json, then invokes the same code path as

    python -m pipeline.aggregator.cli run --per-video-dir ... --queries-file ... --work-dir ...

so the underlying aggregator logic (stages 1..5) is shared verbatim with the
standalone CLI.

Inputs assumed already produced by earlier pipeline stages:
    <output-dir>/claim_results/<video_id>_results.json   (Part 2, Step 2 output)
    events.json                                          (the original input)

Outputs (under <output-dir>/aggregate/):
    aggregate_queries.jsonl                              (built from events.json)
    per_query/query_<qid>.json                           (stage 1)
    clusters/query_<qid>_clusters.json                   (stage 2)
    agent_io/method_a/output/query_<qid>.json            (stage 3a)
    raw_llm/method_a/query_<qid>.json                    (stage 3a — every attempt's trace)
    submission/penkil_method_a.jsonl                     (stage 4 — FINAL MAGMaR submission)
    submission/diff_report.md                            (stage 5)

Usage
-----
    python -m pipeline.aggregator.run_aggregate \
        --input        events.json \
        --output-dir   out/ \
        --tensor-parallel 1

    # Bigger model with vLLM tensor parallel:
    python -m pipeline.aggregator.run_aggregate ... --tensor-parallel 4

    # Also run the Method B ablation:
    python -m pipeline.aggregator.run_aggregate ... --run-method-b
"""
from __future__ import annotations
import argparse
from pathlib import Path

from .config import PipelineConfig
from .events_adapter import build_queries_jsonl
from . import (
    stage1_split, stage2_embed_cluster,
    stage3a_method_a, stage3b_method_b,
    stage4_assemble, stage5_diff_report,
)


def main():
    ap = argparse.ArgumentParser(description="Cross-video claim aggregation (Part 3).")
    ap.add_argument("--input",        required=True, help="events.json")
    ap.add_argument("--output-dir",   required=True, help="pipeline output root")
    # Aggregator-specific knobs (forwarded to PipelineConfig)
    ap.add_argument("--team-id",         default="123456")
    ap.add_argument("--task",            default="oracle")
    ap.add_argument("--run-id-prefix",   default="penkil")
    ap.add_argument("--embed-model",     default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--cluster-tau",     type=float, default=0.9)
    ap.add_argument("--llm-model",       default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--llm-max-model-len", type=int, default=32768)
    ap.add_argument("--llm-max-new-tokens", type=int, default=8000)
    ap.add_argument("--llm-gpu-mem-util",   type=float, default=0.92)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--max-retries",     type=int, default=3)
    ap.add_argument("--skip-method-a",   action="store_true")
    ap.add_argument("--run-method-b",    action="store_true")
    args = ap.parse_args()

    out_root      = Path(args.output_dir)
    per_video_dir = out_root / "claim_results"
    work_dir      = out_root / "aggregate"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: synthesise queries.jsonl from events.json
    queries_file = work_dir / "aggregate_queries.jsonl"
    n = build_queries_jsonl(Path(args.input), queries_file)
    print(f"[aggregate] built {n} queries → {queries_file}")

    if not per_video_dir.is_dir() or not any(per_video_dir.glob("*.json")):
        raise SystemExit(
            f"[fatal] no per-video files under {per_video_dir}.\n"
            f"        Run Part 2 (claim_step1 + claim_step2) first."
        )

    cfg = PipelineConfig(
        per_video_dir       = per_video_dir,
        queries_file        = queries_file,
        work_dir            = work_dir,
        team_id             = args.team_id,
        task                = args.task,
        run_id_prefix       = args.run_id_prefix,
        embed_model         = args.embed_model,
        cluster_tau         = args.cluster_tau,
        llm_model           = args.llm_model,
        llm_max_model_len   = args.llm_max_model_len,
        llm_max_new_tokens  = args.llm_max_new_tokens,
        llm_gpu_mem_util    = args.llm_gpu_mem_util,
        llm_tensor_parallel = args.tensor_parallel,
        llm_max_retries     = args.max_retries,
        run_method_a        = not args.skip_method_a,
        run_method_b        = args.run_method_b,
    )

    stage1_split.run(cfg)
    stage2_embed_cluster.run(cfg)

    # Load the LLM once, share across 3a + 3b
    from .qwen_client import QwenAggregator
    client = QwenAggregator(cfg)
    if cfg.run_method_a:
        stage3a_method_a.run(cfg, client=client)
    if cfg.run_method_b:
        stage3b_method_b.run(cfg, client=client)

    ok = stage4_assemble.run(cfg)
    stage5_diff_report.run(cfg)

    print(f"\n[aggregate] submission → {cfg.submission_dir / (cfg.run_id_prefix + '_method_a.jsonl')}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
