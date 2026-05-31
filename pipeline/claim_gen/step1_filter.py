"""Step 1 — Relevance-guided grounding filter.

For each (query, video) pair, call Qwen3-30B-A3B-Instruct to identify only
the frames whose detected objects and OCR text are relevant to the query and
persona.  ASR is not used here — speech is handled exclusively in Step 2.

The timeline is processed in chunks of STEP1_CHUNK_SIZE rows.  With vLLM all
chunks for a video are batched into a single engine call; with HuggingFace
they run sequentially.  A final summary call is made after aggregation.

Public API
----------
    load_llm()
    guidance = filter_video(query, persona_title, background, frames)
    unload_llm()
"""
from __future__ import annotations
import gc
import json
import os

import torch
from tqdm import tqdm

from . import config


# ── Lazy model handles ────────────────────────────────────────────────────────
_model     = None   # vLLM LLM  or  HF AutoModelForCausalLM
_tokenizer = None


def load_llm() -> None:
    global _model, _tokenizer
    if _model is not None:
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.LLM_GPU_IDS)

    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(
        config.LLM_MODEL_ID, trust_remote_code=True
    )

    if config.USE_VLLM:
        from vllm import LLM
        print(f"[Step1] Loading {config.LLM_MODEL_ID} via vLLM "
              f"(tp={len(config.LLM_GPU_IDS)}) …")
        _model = LLM(
            model=config.LLM_MODEL_ID,
            tensor_parallel_size=len(config.LLM_GPU_IDS),
            dtype="bfloat16",
            max_model_len=32768,
            gpu_memory_utilization=config.VLLM_GPU_MEM_UTIL,
            trust_remote_code=True,
            enforce_eager=False,
        )
    else:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_cfg = None
        if config.LLM_LOAD_IN_4BIT:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        quant_label = "NF4 4-bit" if config.LLM_LOAD_IN_4BIT else "BF16"
        print(f"[Step1] Loading {config.LLM_MODEL_ID} via HF "
              f"({quant_label}, GPUs {config.LLM_GPU_IDS}) …")
        _model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_cfg,
            trust_remote_code=True,
        )
        _model.eval()

    print("[Step1] LLM ready.")


def unload_llm() -> None:
    global _model, _tokenizer
    del _model, _tokenizer
    _model = _tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    print("[Step1] LLM unloaded.")


# ── System prompts ────────────────────────────────────────────────────────────

_SYSTEM_FILTER = """\
You are a precise video analysis assistant.

Given a query and persona, and a segment of a 1-fps video timeline showing
detected objects and on-screen text (OCR), identify ONLY the frames that
contain elements directly relevant to answering the query for that persona and query.

The object labels come from a COCO-80 detector and may be coarse (e.g.
"person", "tv"). Try to build up the connection with the coarse labels with the persona and query content. For example, if the query is "What does the main character do in the video?" and the persona is "The main character is a chef who loves cooking shows.", then a frame with a "person" label and "knife" label might be relevant, while a frame with a "cooking" text might also be relevant.

Output a JSON object — nothing else — with this exact schema:
{
  "relevant_frames": [
    {
      "time_sec": <float>,
      "objects":  [<label_str>, ...],
      "ocr":      [<text_str>, ...]
    }
  ]
}

Rules:
- Include a frame only if at least one detected element is relevant to the query.
- List only the relevant object labels with bounding boxes and OCR strings
- If no frames in this segment are relevant, return {"relevant_frames": []}.
- Output valid JSON only. No prose outside the JSON block."""

_SYSTEM_SUMMARY = """\
You are a analysis assistant trying to build the connection between the detected elements in video and the query/persona.

Given a query, persona, and a set of objects in the relevant frames extracted from a video,
write a concise summary describing what these key objects from the relevant frames contribute to answering the query.

Output a JSON object — nothing else:
{"summary": "<single paragraph, ≤ 120 words>"}

Output valid JSON only. No prose outside the JSON block."""


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_timeline(frames: list[dict]) -> str:
    lines: list[str] = []
    for fr in frames:
        t     = fr["time_sec"]
        obj_s = ", ".join(
            d["class_name"]
            for d in fr["yolo"]["detections"][:config.MAX_DETECTIONS_PER_FRAME]
        )
        ocr_s = ", ".join(
            f'"{d["text"]}"'
            for d in fr["ocr"]["detections"][:config.MAX_OCR_PER_FRAME]
        )
        line = f"t={t:.0f}s"
        if obj_s:
            line += f"  objs=[{obj_s}]"
        if ocr_s:
            line += f"  text=[{ocr_s}]"
        lines.append(line)
    return "\n".join(lines)


def _build_relevant_summary(relevant_frames: list[dict]) -> str:
    lines: list[str] = []
    for fr in relevant_frames:
        t     = fr.get("time_sec", "?")
        objs  = ", ".join(fr.get("objects", []))
        texts = ", ".join(f'"{x}"' for x in fr.get("ocr", []))
        line  = f"t={t:.0f}s"
        if objs:
            line += f"  objects=[{objs}]"
        if texts:
            line += f"  text=[{texts}]"
        lines.append(line)
    return "\n".join(lines)


def _make_chunk_messages(
    query: str,
    persona_title: str,
    background: str,
    chunk: list[dict],
    chunk_idx: int,
    n_chunks: int,
) -> list[dict]:
    timeline = _build_timeline(chunk)
    user_msg = (
        f"Persona: {persona_title}\n"
        f"Background: {background[:400]}\n\n"
        f"Query: {query}\n\n"
        f"Timeline segment {chunk_idx + 1}/{n_chunks} "
        f"({len(chunk)} rows, t={chunk[0]['time_sec']:.0f}s – t={chunk[-1]['time_sec']:.0f}s):\n"
        f"{timeline}"
    )
    return [
        {"role": "system", "content": _SYSTEM_FILTER},
        {"role": "user",   "content": user_msg},
    ]


def _make_summary_messages(
    query: str,
    persona_title: str,
    background: str,
    relevant_frames: list[dict],
) -> list[dict]:
    frame_text = _build_relevant_summary(relevant_frames[:config.MAX_FRAMES_IN_GUIDANCE])
    user_msg = (
        f"Persona: {persona_title}\n"
        f"Background: {background[:400]}\n\n"
        f"Query: {query}\n\n"
        f"Relevant frames identified in the video:\n"
        f"{frame_text if frame_text else '(none found)'}"
    )
    return [
        {"role": "system", "content": _SYSTEM_SUMMARY},
        {"role": "user",   "content": user_msg},
    ]


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict | None:
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(raw[start:end])
    except json.JSONDecodeError:
        return None


# ── Generation backends ───────────────────────────────────────────────────────

def _to_prompt(messages: list[dict]) -> str:
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _generate_batch(messages_list: list[list[dict]]) -> list[str]:
    prompts = [_to_prompt(m) for m in messages_list]

    if config.USE_VLLM:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0, max_tokens=config.LLM_MAX_NEW_TOKENS)
        outputs = _model.generate(prompts, sp)
        return [o.outputs[0].text.strip() for o in outputs]
    else:
        results = []
        for prompt in tqdm(prompts, desc="HF chunks", unit="chunk", leave=False):
            inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
            with torch.no_grad():
                out_ids = _model.generate(
                    **inputs,
                    max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=_tokenizer.eos_token_id,
                )
            text = _tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            results.append(text)
        return results


# ── Public API ────────────────────────────────────────────────────────────────

def filter_video(
    query: str,
    persona_title: str,
    background: str,
    frames: list[dict],
) -> dict:
    """Run Step 1 for a single (query, video) pair using chunked LLM calls.

    With vLLM all chunk prompts are batched into one engine call.
    Returns: {"relevant_frames": [...], "summary": "..."}
    """
    assert _model is not None, "Call load_llm() before filter_video()."

    chunk_size = config.STEP1_CHUNK_SIZE
    chunks     = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]
    n_chunks   = len(chunks)

    chunk_messages = [
        _make_chunk_messages(query, persona_title, background, chunk, i, n_chunks)
        for i, chunk in enumerate(chunks)
    ]

    backend = "vLLM" if config.USE_VLLM else "HF"
    print(f"{n_chunks} chunks [{backend}] … ", end="", flush=True)

    chunk_outputs = _generate_batch(chunk_messages)

    all_relevant: list[dict] = []
    for raw in chunk_outputs:
        parsed = _parse_json(raw)
        if parsed:
            all_relevant.extend(parsed.get("relevant_frames", []))

    print(f"{len(all_relevant)} hits | summary… ", end="", flush=True)

    summary_messages = _make_summary_messages(query, persona_title, background, all_relevant)
    summary_raw = _generate_batch([summary_messages])[0]
    parsed_summary = _parse_json(summary_raw)
    summary = parsed_summary.get("summary", "") if parsed_summary else summary_raw[:200]

    print("done")

    return {
        "relevant_frames": all_relevant,
        "summary": summary,
    }
