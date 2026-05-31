"""Part 2 — grounding-guided claim generation.

Two stages:
  step1_filter   — Qwen3-30B LLM relevance filter over YOLO+OCR timeline
  step2_generate — Qwen3-VL-30B claim generation with video + ASR + guidance

Entry point: pipeline.claim_gen.run_claim_gen
"""
