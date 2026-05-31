"""End-to-end orchestrator (3-part pipeline).

  Part 1 — preprocessing
    audio       mp4 → 16 kHz mono wav
    frames      mp4 → 1-fps jpg
    asr         Whisper transcribe + translate
    yolo        YOLO12x detections
    ocr         HunyuanOCR text + bbox
    merge       join per-video → merged.jsonl  (Part-2-compatible schema)

  Part 2 — claim generation  (heavy: needs Qwen3-30B + Qwen3-VL-30B)
    claim_step1 Relevance filter LLM   → claim_guidance/q_<slug>_guidance.json
    claim_step2 Claim generation LVLM  → claim_results/<vid>_results.json

  Part 3 — cross-video aggregation  (heavy: needs Qwen3-30B + Qwen3-Embedding-8B)
    aggregate   Embed + cluster + LLM verify   → aggregate/submission/penkil_method_a.jsonl
                                                  (= the MAGMaR submission file)

The orchestrator runs each step as a subprocess so failures are contained and
each step remains individually rerunnable.

Usage
-----
    # Full 3-part pipeline:
    python -m pipeline.run \
        --input        events.json \
        --videos-dir   /path/to/videos \
        --output-dir   out/ \
        --gpus         0,1,2,3 \
        --aggregator-tensor-parallel 1

    # Part 1 only (preprocessing):
    python -m pipeline.run ... --skip claim_step1 claim_step2 aggregate

    # Part 2 only (claim generation, after Part 1 produced merged.jsonl):
    python -m pipeline.run ... --only claim_step1 claim_step2

    # Part 3 only (aggregation, after Part 2 produced claim_results/):
    python -m pipeline.run ... --only aggregate

    # Uniform-only LVLM frame sampling (lighter, the WikiVideo variant):
    python -m pipeline.run ... --frame-mode uniform
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    "audio", "frames", "asr", "yolo", "ocr", "merge",
    "claim_step1", "claim_step2",
    "aggregate",
]


def run_step(name: str, cmd: list[str], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    print(f"\n{'='*60}\n  STEP: {name}\n  log : {log_path}\n{'='*60}", flush=True)
    print("  cmd : " + " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log_path, "a") as flog:
        flog.write(f"\n\n===== {name} @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        flog.write("cmd: " + " ".join(cmd) + "\n")
        flog.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            flog.write(line);       flog.flush()
        rc = proc.wait()
    print(f"  done: {name}  exit={rc}  ({time.time()-t0:.1f}s)", flush=True)
    if rc != 0:
        raise SystemExit(f"[fatal] step '{name}' failed (exit {rc}); see {log_path}")


def main():
    ap = argparse.ArgumentParser()
    # ── Required ──────────────────────────────────────────────────────────────
    ap.add_argument("--input",       required=True)
    ap.add_argument("--videos-dir",  required=True)
    ap.add_argument("--output-dir",  required=True)

    # ── Part 1 knobs ──────────────────────────────────────────────────────────
    ap.add_argument("--gpus",          default="0,1",
                    help="GPU shard for YOLO + OCR (also ASR by default)")
    ap.add_argument("--asr-gpus",      default=None,
                    help="override GPUs for ASR (default: same as --gpus)")
    ap.add_argument("--fps",           type=float, default=1.0)
    ap.add_argument("--workers",       type=int,   default=8)
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--yolo-model",    default="yolo12x.pt")
    ap.add_argument("--ocr-model",     default="tencent/HunyuanOCR")
    ap.add_argument("--conf",          type=float, default=0.25)

    # ── Part 2 knobs ──────────────────────────────────────────────────────────
    ap.add_argument("--claim-gpus",  default=None,
                    help="GPU shard for claim_step1 + claim_step2 (default: --gpus). "
                         "Recommended: 4 GPUs (0,1,2,3) since the 30B models need it.")
    ap.add_argument("--llm-model",   default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                    help="claim_step1 LLM")
    ap.add_argument("--lvlm-model",  default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                    help="claim_step2 LVLM")
    ap.add_argument("--frame-mode",  default="guided", choices=["guided", "uniform"],
                    help="Step 2 frame selection: 'guided' = uniform + extra at guidance times; "
                         "'uniform' = uniform only but keep summary in prompt")
    ap.add_argument("--no-vllm",     action="store_true",
                    help="Use HuggingFace transformers backend instead of vLLM")
    ap.add_argument("--event-keys",  nargs="*", default=None,
                    help="Restrict claim-gen to these event keys/slugs (Part 2 only)")

    # ── Part 3 (aggregator) knobs ─────────────────────────────────────────────
    ap.add_argument("--aggregator-model",        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
                    help="aggregator LLM (Method A + B)")
    ap.add_argument("--aggregator-embed-model",  default="Qwen/Qwen3-Embedding-8B",
                    help="embedder for stage 2 (Method A clustering)")
    ap.add_argument("--aggregator-tensor-parallel", type=int, default=1,
                    help="vLLM tensor-parallel size for the aggregator (>=1)")
    ap.add_argument("--aggregator-cluster-tau",  type=float, default=0.9,
                    help="cosine threshold for greedy single-link clustering (Method A)")
    ap.add_argument("--aggregator-max-retries",  type=int, default=3,
                    help="per-prompt retry budget when validation fails")
    ap.add_argument("--aggregator-team-id",      default="123456",
                    help="MAGMaR submission team_id")
    ap.add_argument("--aggregator-task",         default="oracle",
                    help="MAGMaR submission task slot (e.g. 'oracle')")
    ap.add_argument("--aggregator-run-id-prefix", default="penkil",
                    help="submission filename prefix: <prefix>_method_a.jsonl")
    ap.add_argument("--aggregator-run-method-b", action="store_true",
                    help="also run the Method B ablation (pure-LLM clustering)")

    # ── Step selection ────────────────────────────────────────────────────────
    ap.add_argument("--limit",  type=int, default=0,
                    help="cap videos per step (smoke testing)")
    ap.add_argument("--skip",   nargs="*", default=[], choices=STEPS)
    ap.add_argument("--only",   nargs="*", default=[], choices=STEPS)
    args = ap.parse_args()

    out_root = Path(args.output_dir)
    log_dir  = out_root / "logs"
    asr_gpus   = args.asr_gpus   or args.gpus
    claim_gpus = args.claim_gpus or args.gpus
    py = sys.executable

    def enabled(step: str) -> bool:
        if args.only:
            return step in args.only
        return step not in args.skip

    common_in  = ["--input", args.input, "--output-dir", str(out_root)]
    common_vid = ["--videos-dir", args.videos_dir]
    limit      = ["--limit", str(args.limit)] if args.limit else []
    event_keys = (["--event-keys", *args.event_keys] if args.event_keys else [])

    claim_cmd_base = [
        py, "-m", "pipeline.claim_gen.run_claim_gen",
        *common_in, *common_vid,
        "--gpus",       claim_gpus,
        "--llm-model",  args.llm_model,
        "--lvlm-model", args.lvlm_model,
        "--frame-mode", args.frame_mode,
        *event_keys,
    ]
    if args.no_vllm:
        claim_cmd_base.append("--no-vllm")

    aggregate_cmd = [
        py, "-m", "pipeline.aggregator.run_aggregate",
        "--input",            args.input,
        "--output-dir",       str(out_root),
        "--llm-model",        args.aggregator_model,
        "--embed-model",      args.aggregator_embed_model,
        "--tensor-parallel",  str(args.aggregator_tensor_parallel),
        "--cluster-tau",      str(args.aggregator_cluster_tau),
        "--max-retries",      str(args.aggregator_max_retries),
        "--team-id",          args.aggregator_team_id,
        "--task",             args.aggregator_task,
        "--run-id-prefix",    args.aggregator_run_id_prefix,
    ]
    if args.aggregator_run_method_b:
        aggregate_cmd.append("--run-method-b")

    cmds = {
        "audio":  [py, "-m", "pipeline.extract_audio",     *common_in, *common_vid,
                   "--workers", str(args.workers), *limit],
        "frames": [py, "-m", "pipeline.extract_frames",    *common_in, *common_vid,
                   "--workers", str(args.workers), "--fps", str(args.fps), *limit],
        "asr":    [py, "-m", "pipeline.whisper_asr",       *common_in,
                   "--gpus", asr_gpus, "--model", args.whisper_model, *limit],
        "yolo":   [py, "-m", "pipeline.yolo_batch",        "--output-dir", str(out_root),
                   "--gpus", args.gpus, "--model", args.yolo_model,
                   "--conf", str(args.conf), "--fps", str(args.fps), *limit],
        "ocr":    [py, "-m", "pipeline.hunyuan_ocr_batch", "--output-dir", str(out_root),
                   "--gpus", args.gpus, "--model", args.ocr_model,
                   "--fps", str(args.fps), *limit],
        "merge":  [py, "-m", "pipeline.merge",             *common_in],
        "claim_step1": [*claim_cmd_base, "--step", "1"],
        "claim_step2": [*claim_cmd_base, "--step", "2"],
        "aggregate":   aggregate_cmd,
    }

    t_all = time.time()
    for step in STEPS:
        if enabled(step):
            run_step(step, cmds[step], log_dir)
        else:
            print(f"\n[skip] {step}", flush=True)

    print(f"\nALL DONE in {time.time()-t_all:.1f}s")
    print(f"Outputs under: {out_root}")


if __name__ == "__main__":
    main()
