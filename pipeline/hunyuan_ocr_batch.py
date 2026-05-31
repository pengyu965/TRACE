"""HunyuanOCR on pre-extracted frames, sharded across N GPUs at the video level.

Inputs:
    <out>/frames/<event_slug>/<video_id>/frame_*.jpg

Outputs:
    <out>/ocr_frames/<event_slug>/<video_id>/frame_*.jpg   annotated copies
    <out>/ocr.jsonl                                         aggregate (one row / video)

Each detection carries:
    text             — raw OCR string
    src_lang         — NLLB script tag from lang_detect (zho_Hans, eng_Latn, ...)
    bbox_xyxy_norm   — model's native 0..1000 coords
    bbox_xyxy        — rescaled to original frame pixel coords

Resumable: any video_id already in ocr.jsonl is skipped.

Usage:
    python -m pipeline.hunyuan_ocr_batch --output-dir out/ --gpus 0,1
    python -m pipeline.hunyuan_ocr_batch --output-dir out/ --gpus 0,1 --limit 3
"""
from __future__ import annotations
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

from .io_utils import load_done_jsonl, append_jsonl
from .lang_detect import detect_lang


DEFAULT_MODEL = "tencent/HunyuanOCR"
PROMPT        = "检测并识别图片中的文字，将文本坐标格式化输出。"
COORD_RANGE   = 1000


def gather_videos(frames_root: Path) -> list[tuple[str, str]]:
    out = []
    if not frames_root.is_dir():
        return out
    for slug in sorted(os.listdir(frames_root)):
        sd = frames_root / slug
        if not sd.is_dir(): continue
        for vid in sorted(os.listdir(sd)):
            vd = sd / vid
            if vd.is_dir() and any(vd.glob("frame_*.jpg")):
                out.append((slug, vid))
    return out


def clean_repeated_substrings(text: str) -> str:
    """Trim tail repetition (a common HunyuanOCR failure mode on noisy frames)."""
    n = len(text)
    if n < 8000: return text
    for length in range(2, n // 10 + 1):
        cand = text[-length:]
        count = 0; i = n - length
        while i >= 0 and text[i:i + length] == cand:
            count += 1; i -= length
        if count >= 10:
            return text[:n - length * (count - 1)]
    return text


def parse_detections(raw: str, img_size):
    """Parse the model's "text(x1,y1),(x2,y2)" output back into structured boxes."""
    W, H = img_size
    sx, sy = W / COORD_RANGE, H / COORD_RANGE
    pat = re.compile(r"([^()]+?)\((\d+),(\d+)\),\((\d+),(\d+)\)")
    out = []
    for m in pat.finditer(raw):
        text, x1, y1, x2, y2 = m.groups()
        text = text.strip()
        if not text:
            continue
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        out.append({
            "text":           text,
            "src_lang":       detect_lang(text),
            "bbox_xyxy_norm": [x1i, y1i, x2i, y2i],
            "bbox_xyxy":      [int(round(x1i * sx)), int(round(y1i * sy)),
                               int(round(x2i * sx)), int(round(y2i * sy))],
        })
    return out


def draw_annotations(img, dets):
    import cv2
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox_xyxy"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lbl = f"[{d['src_lang'].split('_')[0]}] {d['text']}"
        cv2.putText(img, lbl[:60], (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


def worker(gpu_id: int, items, frames_root: str, annot_root: str,
           model_name: str, max_new_tokens: int, fps: float, return_q):
    import traceback
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        import cv2
        from PIL import Image
        from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

        print(f"[gpu {gpu_id}] loading {model_name}", flush=True)
        processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        model = HunYuanVLForConditionalGeneration.from_pretrained(
            model_name, attn_implementation="eager", dtype=torch.bfloat16,
        ).to("cuda:0")
        print(f"[gpu {gpu_id}] ready, {len(items)} videos", flush=True)
    except Exception:
        print(f"[gpu {gpu_id}] FATAL during model load:\n{traceback.format_exc()}",
              flush=True)
        return_q.put([]); return

    frames_root_p = Path(frames_root)
    annot_root_p  = Path(annot_root)
    out = []
    try:
        for slug, vid in items:
            t0 = time.time()
            frames_dir = frames_root_p / slug / vid
            annot_dir  = annot_root_p  / slug / vid
            annot_dir.mkdir(parents=True, exist_ok=True)
            image_size = None
            frame_records = []
            langs_in_video: set[str] = set()

            for fp in sorted(frames_dir.glob("frame_*.jpg")):
                ti = time.time()
                i = int(fp.stem.split("_")[-1])
                time_sec = (i - 1) / fps
                err = None; dets: list[dict] = []
                try:
                    image = Image.open(fp).convert("RGB")
                    if image_size is None:
                        image_size = [image.size[0], image.size[1]]
                    messages = [[
                        {"role": "system", "content": ""},
                        {"role": "user", "content": [
                            {"type": "image", "image": str(fp)},
                            {"type": "text",  "text":  PROMPT},
                        ]},
                    ]]
                    texts = [processor.apply_chat_template(
                                m, tokenize=False, add_generation_prompt=True)
                             for m in messages]
                    inputs = processor(text=texts, images=image,
                                       padding=True, return_tensors="pt")
                    inputs = inputs.to("cuda:0")
                    with torch.no_grad():
                        generated = model.generate(**inputs,
                                                   max_new_tokens=max_new_tokens,
                                                   do_sample=False)
                    input_ids = inputs.input_ids if "input_ids" in inputs else inputs.inputs
                    trimmed = [o[len(i):] for i, o in zip(input_ids, generated)]
                    decoded = processor.batch_decode(
                        trimmed, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False)
                    raw = clean_repeated_substrings(decoded[0])
                    dets = parse_detections(raw, image.size)
                except Exception as e:
                    err = repr(e)
                for d in dets:
                    langs_in_video.add(d["src_lang"])

                img = cv2.imread(str(fp))
                if img is not None and dets:
                    draw_annotations(img, dets)
                if img is not None:
                    cv2.imwrite(str(annot_dir / fp.name), img)

                frame_records.append({
                    "frame_idx":       i,
                    "time_sec":        time_sec,
                    "num_detections":  len(dets),
                    "frame_languages": sorted({d["src_lang"] for d in dets}),
                    "detections":      dets,
                    "elapsed_sec":     round(time.time() - ti, 2),
                    "error":           err,
                })

            elapsed = time.time() - t0
            rec = {
                "video_id":        vid,
                "event_slug":      slug,
                "num_frames":      len(frame_records),
                "fps":             fps,
                "image_size":      image_size,
                "wall_time_sec":   round(elapsed, 2),
                "model":           model_name,
                "languages_found": sorted(langs_in_video),
                "frames":          frame_records,
            }
            out.append(rec)
            total_dets = sum(fr["num_detections"] for fr in frame_records)
            print(f"[gpu {gpu_id}] {slug}/{vid}  frames={len(frame_records)}  "
                  f"dets={total_dets}  langs={sorted(langs_in_video)}  "
                  f"{elapsed:.1f}s", flush=True)
    except Exception:
        print(f"[gpu {gpu_id}] FATAL in loop:\n{traceback.format_exc()}", flush=True)
    finally:
        return_q.put(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir",     required=True)
    ap.add_argument("--model",          default=DEFAULT_MODEL)
    ap.add_argument("--gpus",           default="0,1")
    ap.add_argument("--fps",            type=float, default=1.0,
                    help="must match the FPS used by extract_frames")
    ap.add_argument("--max-new-tokens", type=int,   default=4096)
    ap.add_argument("--limit",          type=int,   default=0)
    args = ap.parse_args()

    out_root    = Path(args.output_dir)
    frames_root = out_root / "frames"
    annot_root  = out_root / "ocr_frames"
    jsonl_path  = out_root / "ocr.jsonl"
    annot_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise SystemExit("--gpus must list at least one GPU id")

    all_videos = gather_videos(frames_root)
    done = load_done_jsonl(jsonl_path)
    todo = [(s, v) for s, v in all_videos if v not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Found {len(all_videos)} (slug, video_id) pairs with frames")
    print(f"Already done in JSONL: {len(done)}")
    print(f"To process           : {len(todo)}")
    print(f"GPUs                 : {gpu_ids}\n")
    if not todo:
        print("Nothing to do.")
        return

    shards: list[list[tuple[str, str]]] = [[] for _ in gpu_ids]
    for i, item in enumerate(todo):
        shards[i % len(gpu_ids)].append(item)
    for gid, sh in zip(gpu_ids, shards):
        print(f"  gpu {gid}: {len(sh)} videos")

    t_start = time.time()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for gid, sh in zip(gpu_ids, shards):
        p = ctx.Process(target=worker,
                        args=(gid, sh, str(frames_root), str(annot_root),
                              args.model, args.max_new_tokens, args.fps, q))
        p.start(); procs.append(p)

    collected = []
    expected = len(procs)
    while len(collected) < expected:
        try:
            collected.append(q.get(timeout=60))
        except Exception:
            dead_with_err = [p for p in procs if not p.is_alive() and (p.exitcode or 0) != 0]
            if dead_with_err and len(collected) + len(dead_with_err) >= expected:
                print(f"WARN: {len(dead_with_err)} worker(s) crashed; continuing")
                break
            if not any(p.is_alive() for p in procs):
                break
    flat = [r for lst in collected for r in lst]
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate(); p.join(timeout=5)

    flat.sort(key=lambda r: (r["event_slug"], r["video_id"]))
    append_jsonl(jsonl_path, flat)

    print(f"\nWall: {time.time()-t_start:.1f}s")
    print(f"Wrote {len(flat)} records → {jsonl_path}")


if __name__ == "__main__":
    main()
