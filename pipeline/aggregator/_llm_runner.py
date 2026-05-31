"""Shared retry-on-invariant-fail loop used by Method A and Method B.

`run_with_retry` takes a list of work items. Each item is a dict carrying:
    - 'user': the initial user prompt,
    - 'validate': a callable (parsed_dict) -> normalised_responses, raises ValidationError,
    - 'fallback': a callable () -> list[dict], used after max retries are exhausted.

It returns a list parallel to `items` of {responses, status, attempts (full history),
last_error, last_thinking}. The attempts history is a list of dicts (one per try):
    {n, user, raw, thinking, json_text, parse_error, validation_error}
This lets us re-validate later from disk without re-running the model.
"""
from __future__ import annotations
from .prompts import SYSTEM_PROMPT, RESPONSE_JSON_SCHEMA, build_retry_user
from .validate import ValidationError


def run_with_retry(client, items: list[dict], max_retries: int) -> list[dict]:
    n_items = len(items)
    results: list[dict | None] = [None] * n_items
    history: list[list[dict]] = [[] for _ in range(n_items)]
    pending = list(range(n_items))
    user_for: dict[int, str] = {i: items[i]["user"] for i in pending}

    while pending:
        prompts = [user_for[i] for i in pending]
        outs = client.generate_json(SYSTEM_PROMPT, prompts, json_schema=RESPONSE_JSON_SCHEMA)
        next_pending: list[int] = []
        for idx, out in zip(pending, outs):
            attempt_n = len(history[idx]) + 1
            attempt_rec = {
                "n": attempt_n,
                "user": user_for[idx],
                "raw": out.raw,
                "thinking": out.thinking,
                "json_text": out.json_text,
                "parse_error": out.parse_error,
                "validation_error": None,
            }

            if out.parse_error is not None or out.parsed is None:
                attempt_rec["validation_error"] = None  # parse failed before validation
                history[idx].append(attempt_rec)
                if attempt_n < max_retries:
                    user_for[idx] = build_retry_user(
                        items[idx]["user"], f"JSON parse failed: {out.parse_error}", out.raw,
                    )
                    next_pending.append(idx)
                else:
                    results[idx] = {
                        "responses": items[idx]["fallback"](),
                        "status": "fallback",
                        "n_attempts": attempt_n,
                        "last_error": f"JSON parse failed: {out.parse_error}",
                        "attempts": history[idx],
                    }
                continue

            try:
                normalised = items[idx]["validate"](out.parsed.get("responses", []))
            except ValidationError as e:
                attempt_rec["validation_error"] = str(e)
                history[idx].append(attempt_rec)
                if attempt_n < max_retries:
                    user_for[idx] = build_retry_user(items[idx]["user"], str(e), out.raw)
                    next_pending.append(idx)
                else:
                    results[idx] = {
                        "responses": items[idx]["fallback"](),
                        "status": "fallback",
                        "n_attempts": attempt_n,
                        "last_error": str(e),
                        "attempts": history[idx],
                    }
                continue

            history[idx].append(attempt_rec)
            results[idx] = {
                "responses": normalised,
                "status": "ok",
                "n_attempts": attempt_n,
                "last_error": None,
                "attempts": history[idx],
            }
        pending = next_pending

    return results  # type: ignore[return-value]
