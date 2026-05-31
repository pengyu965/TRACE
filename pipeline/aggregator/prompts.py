"""Prompt templates for Method A (per-cluster verify) and Method B (flat clustering).

These are the exact rubrics that were used with Claude subagents during the MAGMaR
2026 development runs, lifted verbatim and parametrised for Qwen3-30B Instruct.
The hard invariants below are also enforced by `validate.py` after generation.
"""
from __future__ import annotations
import json
from typing import Any


SYSTEM_PROMPT = """You are an aggregation expert for multi-video question answering.

Your job: given a query, a persona, and a set of free-text claims extracted from videos by an upstream model, decide which claims describe the SAME factual proposition and group them. For each group, return one canonical response with citations to every video that supported it.

You must follow these rules WITHOUT exception:

1. SAME-FACT EQUIVALENCE. Two claims are equivalent only if they assert the same numbers, actors, outcome, and relation. Two claims about the same general topic but with different specifics are NOT equivalent and must be returned as separate responses.

2. VERBATIM CANONICAL TEXT. The `text` field of every response MUST be a character-for-character copy of one of the input claims in that group. You may NOT paraphrase, summarise, combine sentences, or alter punctuation. If a group contains multiple equivalent claims, pick the most informative one (more named entities, numbers, dates; longer wins ties).

3. CITATIONS. Each response's `citations` array MUST contain the deduplicated `video_id`s of the input claims in that group, in first-appearance order. No duplicates within a citations array.

4. COVERAGE. Every input claim MUST contribute to exactly one response. No claim may be dropped, no claim may appear in two groups.

5. JSON ONLY. After your reasoning, output ONLY a single JSON object matching the schema. No prose before or after the JSON.
"""


METHOD_A_USER_TEMPLATE = """# Query context
Query ID: {query_id}
Title: {title}
Persona: {persona_title}
Background: {background}
Question: {query}

# Candidate cluster (pre-grouped by embedding similarity at cosine tau={tau})
{cluster_block}

# Your task
Decide whether this candidate cluster is:
  (a) A single same-fact group: return ONE response whose `text` is the most informative member (verbatim) and `citations` is the deduped union of member video_ids.
  (b) Multiple sub-groups that should be split: return ONE response per sub-group, each with verbatim text and its own citations.
  (c) A singleton: return exactly the one member (this is the answer for size-1 clusters).

Reason step by step inside <think>...</think>. Then output JSON ONLY in this schema:

{{
  "responses": [
    {{"text": "<verbatim from one member>", "citations": ["<video_id>", ...]}},
    ...
  ]
}}

Every member claim must contribute to exactly one response. Do not invent text. Do not add fields.
"""


METHOD_B_USER_TEMPLATE = """# Query context
Query ID: {query_id}
Title: {title}
Persona: {persona_title}
Background: {background}
Question: {query}

# All extracted claims for this query (flat list)
{claim_block}

# Your task
Cluster the claims above by SAME factual proposition (rule 1 in the system prompt). For each cluster, emit one response whose `text` is the verbatim most-informative member and whose `citations` is the deduped union of source video_ids in first-appearance order.

Reason step by step inside <think>...</think>. Then output JSON ONLY in this schema:

{{
  "responses": [
    {{"text": "<verbatim from one input claim>", "citations": ["<video_id>", ...]}},
    ...
  ]
}}

EVERY input claim listed above must contribute to exactly one response. Do not drop any claim. Do not invent text or video_ids. Do not add fields.
"""


RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "responses": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["responses"],
    "additionalProperties": False,
}


def render_cluster_block(members: list[dict]) -> str:
    lines = []
    for i, m in enumerate(members):
        lines.append(f"  [{i}] video_id={m['video_id']!r}")
        lines.append(f"      claim: {m['claim']}")
    return "\n".join(lines)


def render_claim_block(claims: list[dict]) -> str:
    lines = []
    for i, c in enumerate(claims):
        lines.append(f"  [{i}] video_id={c['video_id']!r}")
        lines.append(f"      claim: {c['claim']}")
    return "\n".join(lines)


def build_method_a_user(query_ctx: dict, cluster_members: list[dict], tau: float) -> str:
    return METHOD_A_USER_TEMPLATE.format(
        query_id=query_ctx["query_id"],
        title=query_ctx.get("title", ""),
        persona_title=query_ctx.get("persona_title", ""),
        background=query_ctx.get("background", ""),
        query=query_ctx.get("query", ""),
        tau=tau,
        cluster_block=render_cluster_block(cluster_members),
    )


def build_method_b_user(query_ctx: dict, flat_claims: list[dict]) -> str:
    return METHOD_B_USER_TEMPLATE.format(
        query_id=query_ctx["query_id"],
        title=query_ctx.get("title", ""),
        persona_title=query_ctx.get("persona_title", ""),
        background=query_ctx.get("background", ""),
        query=query_ctx.get("query", ""),
        claim_block=render_claim_block(flat_claims),
    )


def build_retry_user(prev_user: str, error_msg: str, prev_output: str) -> str:
    """Wraps the previous user prompt with the validation failure for a retry."""
    return (
        prev_user
        + "\n\n# Previous attempt failed validation\n"
        + f"Validator error: {error_msg}\n\n"
        + "Your previous JSON was:\n"
        + prev_output[:4000]
        + "\n\nFix the JSON so it satisfies every rule in the system prompt. "
        + "In particular: verbatim text, deduped citations, every input claim in exactly one response. "
        + "Output the corrected JSON only."
    )
