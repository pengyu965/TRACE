"""Mutable run-time configuration for claim_gen.

All paths get filled in by `configure(...)` at orchestrator start-up. The
module-level constants are sensible defaults — override them via the
`pipeline.run` CLI flags or by editing this file.
"""
from __future__ import annotations
from pathlib import Path


# ── Paths (overwritten at runtime by configure()) ─────────────────────────────
MERGED_ANNOTS: Path = Path("out/merged.jsonl")
ASR_WHISPER:   Path = Path("out/asr.jsonl")
ASR_OMNI:      Path = Path("out/asr.jsonl")          # no Omni; reuse Whisper
VIDEO_DIR:     Path = Path("videos/")
EVENTS_FILE:   Path = Path("events.json")
OUTPUT_DIR:    Path = Path("out/")

# ── Inference backend ─────────────────────────────────────────────────────────
USE_VLLM:          bool  = True
VLLM_GPU_MEM_UTIL: float = 0.90

# ── Models ────────────────────────────────────────────────────────────────────
LLM_MODEL_ID:  str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LVLM_MODEL_ID: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# ── GPU layout (Step 1 unloaded before Step 2; both get all four cards) ──────
LLM_GPU_IDS:  list[int] = [0, 1, 2, 3]
LVLM_GPU_IDS: list[int] = [0, 1, 2, 3]

# Quantisation knobs (only used when USE_VLLM=False, HF backend)
LLM_LOAD_IN_4BIT:  bool = True
LVLM_LOAD_IN_4BIT: bool = True

# ── Generation parameters ─────────────────────────────────────────────────────
LLM_MAX_NEW_TOKENS:    int   = 2048
LVLM_MAX_NEW_TOKENS:   int   = 4096
LVLM_VIDEO_MAX_PIXELS: int   = 448 * 448
LVLM_VIDEO_NFRAMES:    int   = 100
LLM_TEMPERATURE:       float = 0.0
LVLM_TEMPERATURE:      float = 0.6

# ── Step 2 frame-selection mode ──────────────────────────────────────────────
# "guided"  — uniform N + extra frames at guidance timestamps (production)
# "uniform" — uniform N only, but keep the textual summary (wikivideo variant)
FRAME_MODE: str = "guided"

# ── Guidance caps ─────────────────────────────────────────────────────────────
STEP1_CHUNK_SIZE:         int = 60
MAX_FRAMES_IN_GUIDANCE:   int = 30
MAX_DETECTIONS_PER_FRAME: int = 10
MAX_OCR_PER_FRAME:        int = 10


def configure(
    output_dir:    str | Path,
    videos_dir:    str | Path,
    events_file:   str | Path,
    llm_gpus:      list[int] | None = None,
    lvlm_gpus:     list[int] | None = None,
    use_vllm:      bool | None      = None,
    llm_model:     str | None       = None,
    lvlm_model:    str | None       = None,
    frame_mode:    str | None       = None,
) -> None:
    """Patch the module-level constants. Called once at orchestrator start."""
    global OUTPUT_DIR, VIDEO_DIR, EVENTS_FILE, MERGED_ANNOTS, ASR_WHISPER, ASR_OMNI
    global LLM_GPU_IDS, LVLM_GPU_IDS, USE_VLLM
    global LLM_MODEL_ID, LVLM_MODEL_ID, FRAME_MODE

    OUTPUT_DIR    = Path(output_dir)
    VIDEO_DIR     = Path(videos_dir)
    EVENTS_FILE   = Path(events_file)
    MERGED_ANNOTS = OUTPUT_DIR / "merged.jsonl"
    ASR_WHISPER   = OUTPUT_DIR / "asr.jsonl"
    ASR_OMNI      = OUTPUT_DIR / "asr.jsonl"

    if llm_gpus  is not None: LLM_GPU_IDS  = list(llm_gpus)
    if lvlm_gpus is not None: LVLM_GPU_IDS = list(lvlm_gpus)
    if use_vllm  is not None: USE_VLLM     = use_vllm
    if llm_model is not None: LLM_MODEL_ID  = llm_model
    if lvlm_model is not None: LVLM_MODEL_ID = lvlm_model
    if frame_mode is not None:
        if frame_mode not in ("guided", "uniform"):
            raise ValueError(f"frame_mode must be 'guided' or 'uniform', got {frame_mode!r}")
        FRAME_MODE = frame_mode
