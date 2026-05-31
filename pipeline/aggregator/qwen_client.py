"""Wrap a Qwen3 LLM (default Qwen3-30B-A3B-Instruct-2507) via vLLM offline mode.

Loads the model once, exposes `generate_json(prompts, schema)` which:
  1. Builds the chat template (system + user).
  2. Runs batched generation with greedy decoding.
  3. Splits any <think>...</think> block (present only when running a thinking
     variant) from the final JSON.
  4. Parses the JSON and returns (raw_text, thinking, parsed_dict).

Validation against the pipeline's hard invariants happens in validate.py, not here.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any

from vllm import LLM, SamplingParams


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class QwenOutput:
    raw: str
    thinking: str
    json_text: str
    parsed: dict[str, Any] | None
    parse_error: str | None = None


def _split_thinking(raw: str) -> tuple[str, str]:
    m = THINK_RE.search(raw)
    if not m:
        return "", raw.strip()
    thinking = m.group(1).strip()
    rest = THINK_RE.sub("", raw, count=1).strip()
    return thinking, rest


def _strip_json_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # remove ```json or ``` fence
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    return s.strip()


def _try_parse(json_text: str) -> tuple[dict | None, str | None]:
    try:
        return json.loads(_strip_json_fence(json_text)), None
    except json.JSONDecodeError as e:
        # Fall back: try to extract the largest {...} block.
        m = re.search(r"\{.*\}", json_text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), None
            except json.JSONDecodeError as e2:
                return None, str(e2)
        return None, str(e)


class QwenAggregator:
    def __init__(self, cfg):
        from .config import PipelineConfig  # for type hint only
        self.cfg = cfg
        self.llm = LLM(
            model=cfg.llm_model,
            max_model_len=cfg.llm_max_model_len,
            gpu_memory_utilization=cfg.llm_gpu_mem_util,
            tensor_parallel_size=cfg.llm_tensor_parallel,
            trust_remote_code=True,
            enforce_eager=False,
            seed=cfg.llm_seed,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def _build_chat_text(self, system: str, user: str) -> str:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    def generate_json(
        self,
        system_prompt: str,
        user_prompts: list[str],
        json_schema: dict | None = None,
    ) -> list[QwenOutput]:
        """Batched generation. Returns one QwenOutput per user prompt.

        json_schema is accepted but intentionally NOT applied as a vLLM
        guided-decoding constraint: doing so forces the model to emit JSON
        from the first token, which suppresses the <think>...</think> block a
        thinking model needs to reason about clustering. Instead we rely on
        the prompt + post-hoc parse + retry loop. The schema is kept in the
        API for documentation and possible future use.
        """
        del json_schema
        prompts = [self._build_chat_text(system_prompt, u) for u in user_prompts]
        params = SamplingParams(
            temperature=self.cfg.llm_temperature,
            top_p=self.cfg.llm_top_p,
            max_tokens=self.cfg.llm_max_new_tokens,
            seed=self.cfg.llm_seed,
        )
        outs = self.llm.generate(prompts, params)
        results: list[QwenOutput] = []
        for o in outs:
            raw = o.outputs[0].text
            thinking, body = _split_thinking(raw)
            parsed, err = _try_parse(body)
            results.append(QwenOutput(raw=raw, thinking=thinking, json_text=body,
                                      parsed=parsed, parse_error=err))
        return results
