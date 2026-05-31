"""CLI entry point.

End-to-end run (Method A only, paper config):

    python -m magmar_aggregator.cli run \\
        --per-video-dir /path/to/per_video \\
        --queries-file  /path/to/queries.jsonl \\
        --work-dir      ./run

Also run the Method B ablation (pure-LLM clustering):

    python -m magmar_aggregator.cli run \\
        --per-video-dir ... --queries-file ... --work-dir ./run \\
        --run-method-b

Individual stages (debug / partial reruns):

    python -m magmar_aggregator.cli stage1 --per-video-dir ... --queries-file ... --work-dir ./run
    python -m magmar_aggregator.cli stage2  --work-dir ./run
    python -m magmar_aggregator.cli stage3a --work-dir ./run
    python -m magmar_aggregator.cli stage3b --work-dir ./run     # only if ablation enabled
    python -m magmar_aggregator.cli stage4  --work-dir ./run
    python -m magmar_aggregator.cli stage5  --work-dir ./run

Replay validators against saved raw model traces (no GPU):

    python -m magmar_aggregator.cli revalidate --work-dir ./run
"""
from __future__ import annotations
import argparse
from pathlib import Path

from .config import PipelineConfig


def _add_common_args(p: argparse.ArgumentParser, require_inputs: bool):
    p.add_argument("--per-video-dir", type=Path, required=require_inputs)
    p.add_argument("--queries-file", type=Path, required=require_inputs)
    p.add_argument("--topic-video-mapping", type=Path, default=None)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--team-id", default="123456")
    p.add_argument("--task", default="oracle")
    p.add_argument("--run-id-prefix", default="penkil")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-8B")
    p.add_argument("--cluster-tau", type=float, default=0.9)
    p.add_argument("--llm-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    p.add_argument("--llm-max-model-len", type=int, default=32768)
    p.add_argument("--llm-max-new-tokens", type=int, default=8000)
    p.add_argument("--llm-gpu-mem-util", type=float, default=0.92)
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--skip-method-a", action="store_true",
                   help="Skip the headline Method A (embed + per-cluster LLM verify).")
    p.add_argument("--run-method-b", action="store_true",
                   help="Also run the Method B ablation (pure-LLM clustering over flat claims). Off by default.")


def _cfg_from_args(args, require_inputs: bool) -> PipelineConfig:
    return PipelineConfig(
        per_video_dir=args.per_video_dir if require_inputs else (args.per_video_dir or Path(".")),
        queries_file=args.queries_file if require_inputs else (args.queries_file or Path(".")),
        topic_video_mapping=args.topic_video_mapping,
        work_dir=args.work_dir,
        team_id=args.team_id,
        task=args.task,
        run_id_prefix=args.run_id_prefix,
        embed_model=args.embed_model,
        cluster_tau=args.cluster_tau,
        llm_model=args.llm_model,
        llm_max_model_len=args.llm_max_model_len,
        llm_max_new_tokens=args.llm_max_new_tokens,
        llm_gpu_mem_util=args.llm_gpu_mem_util,
        llm_tensor_parallel=args.tensor_parallel,
        llm_max_retries=args.max_retries,
        run_method_a=not args.skip_method_a,
        run_method_b=args.run_method_b,
    )


def cmd_stage1(args):
    from . import stage1_split
    stage1_split.run(_cfg_from_args(args, require_inputs=True))


def cmd_stage2(args):
    from . import stage2_embed_cluster
    stage2_embed_cluster.run(_cfg_from_args(args, require_inputs=False))


def cmd_stage3a(args):
    from . import stage3a_method_a
    stage3a_method_a.run(_cfg_from_args(args, require_inputs=False))


def cmd_stage3b(args):
    from . import stage3b_method_b
    stage3b_method_b.run(_cfg_from_args(args, require_inputs=False))


def cmd_stage4(args):
    from . import stage4_assemble
    ok = stage4_assemble.run(_cfg_from_args(args, require_inputs=False))
    raise SystemExit(0 if ok else 1)


def cmd_stage5(args):
    from . import stage5_diff_report
    stage5_diff_report.run(_cfg_from_args(args, require_inputs=False))


def cmd_revalidate(args):
    """Re-run validators against saved raw LLM traces. No GPU needed."""
    from . import revalidate
    revalidate.run(_cfg_from_args(args, require_inputs=False))


def cmd_run(args):
    """End-to-end. Loads the LLM exactly once and shares it across stage 3a + 3b."""
    from . import stage1_split, stage2_embed_cluster, stage3a_method_a, stage3b_method_b
    from . import stage4_assemble, stage5_diff_report
    from .qwen_client import QwenAggregator

    cfg = _cfg_from_args(args, require_inputs=True)
    stage1_split.run(cfg)
    stage2_embed_cluster.run(cfg)

    # The embedder is released at the end of stage 2 (see explicit cleanup
    # there). Now load the aggregator once for stage 3.
    client = QwenAggregator(cfg)
    if cfg.run_method_a:
        stage3a_method_a.run(cfg, client=client)
    if cfg.run_method_b:
        stage3b_method_b.run(cfg, client=client)

    ok = stage4_assemble.run(cfg)
    stage5_diff_report.run(cfg)
    raise SystemExit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(prog="magmar_aggregator")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn, ri in [
        ("run", cmd_run, True),
        ("stage1", cmd_stage1, True),
        ("stage2", cmd_stage2, False),
        ("stage3a", cmd_stage3a, False),
        ("stage3b", cmd_stage3b, False),
        ("stage4", cmd_stage4, False),
        ("stage5", cmd_stage5, False),
        ("revalidate", cmd_revalidate, False),
    ]:
        sp = sub.add_parser(name)
        _add_common_args(sp, require_inputs=ri)
        sp.set_defaults(func=fn)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
