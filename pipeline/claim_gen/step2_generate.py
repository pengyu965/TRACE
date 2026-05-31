"""Step 2 — Claim generation with LVLM.

For a single (query, video) pair, takes:
  - persona + query
  - visual guidance from Step 1 (relevant frame metadata as text context)
  - the full video file — frame sampling handled internally
  - full ASR transcript as a plain text block

Frame selection respects config.FRAME_MODE:
  "guided"   — uniform N frames + extra frames at guidance timestamps
  "uniform"  — uniform N frames only, but keep the textual summary

Public API
----------
    load_lvlm()
    claims = generate_claims(query, persona_title, background, title,
                             guidance, full_asr_text, video_id)
    unload_lvlm()
"""
from __future__ import annotations
import gc
import json
import os

import numpy as np
import torch

from . import config


# ── Lazy model handles ────────────────────────────────────────────────────────
_processor = None
_model     = None


def load_lvlm() -> None:
    global _processor, _model
    if _model is not None:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.LVLM_GPU_IDS)

    from transformers import AutoProcessor
    _processor = AutoProcessor.from_pretrained(
        config.LVLM_MODEL_ID, trust_remote_code=True
    )

    if config.USE_VLLM:
        from vllm import LLM
        print(f"[Step2] Loading {config.LVLM_MODEL_ID} via vLLM "
              f"(tp={len(config.LVLM_GPU_IDS)}) …")
        _model = LLM(
            model=config.LVLM_MODEL_ID,
            tensor_parallel_size=len(config.LVLM_GPU_IDS),
            dtype="auto",
            max_model_len=32768,
            gpu_memory_utilization=config.VLLM_GPU_MEM_UTIL,
            limit_mm_per_prompt={"video": 1},
            trust_remote_code=True,
            enforce_eager=False,
        )
    else:
        from transformers import BitsAndBytesConfig
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
        except ImportError:
            from transformers import AutoModelForVision2Seq as _VLModel

        bnb_cfg = None
        if config.LVLM_LOAD_IN_4BIT:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        quant_label = "NF4 4-bit" if config.LVLM_LOAD_IN_4BIT else "BF16"
        print(f"[Step2] Loading {config.LVLM_MODEL_ID} via HF "
              f"({quant_label}, GPUs {config.LVLM_GPU_IDS}) …")
        _model = _VLModel.from_pretrained(
            config.LVLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        _model.eval()

    print("[Step2] LVLM ready.")


def unload_lvlm() -> None:
    global _processor, _model
    del _model, _processor
    _model = _processor = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[Step2] LVLM unloaded.")


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert video analyst generating factual claims for an academic report.

You will receive:
1. A persona and background describing who you are writing for.
2. A query stating what information is needed.
3. The full video — this is your primary source of evidence.
4. Key frame annotations produced by a prior automated analysis, including
   detected objects, on-screen text (OCR), and a brief summary. These
   annotations serve as grounding hints to direct your attention to potentially
   relevant moments in the video. They may be incomplete, noisy, or redundant —
   treat them as a supplementary reference only, not as ground truth.
5. Speech transcript: the Whisper ASR transcript of the video's spoken audio.

Your task is to produce a list of concise, factual claims that directly answer
the query from the persona's perspective.

Output a JSON array of strings — nothing else:
[
  "<one-sentence factual claim>",
  "<one-sentence factual claim>",
  ...
]

Rules:
- Ground every claim in what you directly observe in the video or hear in the
  transcript. Do not rely solely on the frame annotations.
- Each claim must be a single sentence.
- Do not repeat the same claim twice.
- Prefer specific facts (numbers, names, dates) over vague summaries.
Output valid JSON only. No prose outside the JSON array."""


# ── Frame extraction ──────────────────────────────────────────────────────────

def _extract_guided_frames(
    video_path: str,
    guidance: dict,
    n_uniform: int,
    max_pixels: int,
) -> tuple[np.ndarray, list[int]]:
    """Return (frames_array, frame_indices). Frames are resized so W*H ≤ max_pixels."""
    import av
    from PIL import Image

    container = av.open(video_path)
    stream = container.streams.video[0]
    fps = float(stream.average_rate or 25)

    total_frames = stream.frames
    if not total_frames and stream.duration is not None:
        total_frames = int(stream.duration * float(stream.time_base) * fps)
    total_frames = max(int(total_frames), 1)

    n = min(n_uniform, total_frames)
    uniform_idx: set[int] = set(
        np.linspace(0, total_frames - 1, n).round().astype(int).tolist()
    )
    guided_idx: set[int] = {
        min(int(round(fr["time_sec"] * fps)), total_frames - 1)
        for fr in guidance.get("relevant_frames", [])
        if "time_sec" in fr
    }
    all_idx = sorted(uniform_idx | guided_idx)
    idx_set = set(all_idx)

    frames_dict: dict[int, np.ndarray] = {}
    for i, frame in enumerate(container.decode(video=0)):
        if i in idx_set:
            img = frame.to_image().convert("RGB")
            w, h = img.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
            frames_dict[i] = np.array(img)
        if i >= max(all_idx):
            break
    container.close()

    present = [i for i in all_idx if i in frames_dict]
    return np.stack([frames_dict[i] for i in present]), present


# ── Prompt builder ────────────────────────────────────────────────────────────

def _format_guidance_text(guidance: dict) -> str:
    lines = ["[Key Frame Annotations]"]
    summary = guidance.get("summary", "")
    if summary:
        lines.append(f"Summary: {summary}")
        lines.append("")

    for fr in guidance.get("relevant_frames", [])[:config.MAX_FRAMES_IN_GUIDANCE]:
        t      = fr.get("time_sec", "?")
        objs   = fr.get("objects", [])
        texts  = fr.get("ocr", [])
        obj_s  = ", ".join(o["label"] for o in objs)
        text_s = ", ".join(f'"{t_["text"]}"' for t_ in texts)
        line   = f"  t={t:.0f}s"
        if obj_s:
            line += f"  objects=[{obj_s}]"
        if text_s:
            line += f"  text=[{text_s}]"
        lines.append(line)

    return "\n".join(lines)


def _build_messages(
    query: str,
    persona_title: str,
    background: str,
    title: str,
    guidance: dict,
    full_asr_text: str,
    video_path: str,
) -> list[dict]:
    text_body = "\n".join([
        f"Event: {title}",
        f"Persona: {persona_title}",
        f"Background: {background[:500]}",
        "",
        f"Query: {query}",
        "",
        _format_guidance_text(guidance),
        "",
        "[Speech Transcript — full audio]",
        full_asr_text.strip() if full_asr_text.strip() else "(no speech detected)",
        "",
        f"Generate claims that directly answer the query for the {persona_title}.",
    ])

    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": config.LVLM_VIDEO_MAX_PIXELS,
                    "nframes": config.LVLM_VIDEO_NFRAMES,
                },
                {"type": "text", "text": text_body},
            ],
        },
    ]


# ── Generation backends ───────────────────────────────────────────────────────

def _generate_one(messages: list[dict], frames: np.ndarray, frame_indices: list[int]) -> str:
    text_prompt = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if config.USE_VLLM:
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=config.LVLM_TEMPERATURE,
            max_tokens=config.LVLM_MAX_NEW_TOKENS,
        )
        video_data = [(frames, {
            "fps": 1.0,
            "total_num_frames": len(frames),
            "frames_indices": frame_indices,
        })]
        outputs = _model.generate(
            {"prompt": text_prompt, "multi_modal_data": {"video": video_data}},
            sp,
        )
        return outputs[0].outputs[0].text.strip()
    else:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = _processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(next(_model.parameters()).device)
        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=config.LVLM_MAX_NEW_TOKENS,
                temperature=config.LVLM_TEMPERATURE,
                do_sample=True,
            )
        return _processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()


# ── Public API ────────────────────────────────────────────────────────────────

def _video_path_for(video_id: str) -> str | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4v"):
        p = config.VIDEO_DIR / f"{video_id}{ext}"
        if p.exists():
            return str(p)
    return None


def generate_claims(
    query: str,
    persona_title: str,
    background: str,
    title: str,
    guidance: dict,
    full_asr_text: str,
    video_id: str,
) -> list[str]:
    """Generate claims for a single (query, video) pair.

    Honours config.FRAME_MODE:
        "guided"  → uniform N + extra frames at guidance timestamps
        "uniform" → uniform N only (relevant_frames stripped before sampling),
                    but the textual summary is still injected into the prompt
    """
    assert _model is not None, "Call load_lvlm() before generate_claims()."

    video_path = _video_path_for(video_id)
    if video_path is None:
        print(f"    [warn] video file not found for {video_id} — skipping")
        return []

    # Decide frame-extraction guidance based on config.FRAME_MODE
    extract_guidance = guidance
    if config.FRAME_MODE == "uniform":
        extract_guidance = {"relevant_frames": [], "summary": guidance.get("summary", "")}

    frames, frame_indices = _extract_guided_frames(
        video_path, extract_guidance, config.LVLM_VIDEO_NFRAMES, config.LVLM_VIDEO_MAX_PIXELS
    )
    n_extra = max(0, len(frames) - config.LVLM_VIDEO_NFRAMES)
    print(f"    [frames] {len(frames)} total ({config.LVLM_VIDEO_NFRAMES} uniform + "
          f"{n_extra} guidance-only)  mode={config.FRAME_MODE}")

    # The prompt always gets the full guidance text (with relevant_frames + summary)
    messages = _build_messages(
        query, persona_title, background, title, guidance, full_asr_text, video_path
    )
    raw = _generate_one(messages, frames, frame_indices)

    start = raw.find("[")
    end   = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return [raw[:300]] if raw else []

    try:
        result = json.loads(raw[start:end])
        claims = []
        for item in result:
            if isinstance(item, str):
                claims.append(item)
            elif isinstance(item, dict):
                claims.append(item.get("text", str(item)))
        return claims
    except json.JSONDecodeError:
        return [raw[start:end][:300]]
