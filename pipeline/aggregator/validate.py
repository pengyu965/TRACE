"""Hard invariants for aggregator output (video-level coverage).

A response set is valid iff:
  - every response.text is a verbatim copy of some input claim (whitespace-normalised),
  - each citation is a known input video_id for this query/cluster,
  - citations within a single response are deduped,
  - every input video that contributed at least one claim appears in some
    response's citations (no video silently dropped),
  - total citations across responses do not exceed total input claims (no
    over-citation; soft cap, raised as a hard error so the model retries).

This matches the invariant used by the original MAGMaR Claude-era pipeline.
The prompt still asks the model for verbatim text and same-fact grouping; the
incentive to split distinct facts comes from the prompt, not from a per-text
coverage check (which would forbid legitimate same-fact merges across slightly
different surface forms).
"""
from __future__ import annotations


def _norm(s: str) -> str:
    return " ".join(s.split())


class ValidationError(Exception):
    pass


def _check_responses(
    responses: list[dict],
    input_pairs: list[tuple[str, str]],  # [(video_id, claim_text), ...]
) -> list[dict]:
    if not isinstance(responses, list) or not responses:
        raise ValidationError("responses must be a non-empty list")

    input_norm_texts: set[str] = {_norm(t) for _, t in input_pairs}
    input_videos: set[str] = {v for v, _ in input_pairs}
    n_input_claims = len(input_pairs)

    cited: set[str] = set()
    citations_total = 0
    normalised: list[dict] = []

    for i, r in enumerate(responses):
        if not isinstance(r, dict):
            raise ValidationError(f"response[{i}] must be an object")
        text = r.get("text", "")
        cits = r.get("citations", [])
        if not isinstance(text, str) or not text.strip():
            raise ValidationError(f"response[{i}].text is empty")
        if not isinstance(cits, list) or not cits:
            raise ValidationError(f"response[{i}].citations is empty")
        for c in cits:
            if not isinstance(c, str) or not c:
                raise ValidationError(f"response[{i}].citations has invalid entry {c!r}")

        if _norm(text) not in input_norm_texts:
            raise ValidationError(
                f"response[{i}].text is not a verbatim input claim: {text[:80]!r}"
            )

        for c in cits:
            if c not in input_videos:
                raise ValidationError(
                    f"response[{i}] cites unknown video_id {c!r}"
                )

        deduped = list(dict.fromkeys(cits))
        if len(deduped) != len(cits):
            raise ValidationError(
                f"response[{i}] citations contain duplicates: {cits}"
            )

        for c in cits:
            cited.add(c)
        citations_total += len(cits)
        normalised.append({"text": text, "citations": deduped})

    missing = input_videos - cited
    if missing:
        raise ValidationError(
            f"{len(missing)} contributing video(s) not cited by any response: "
            f"{sorted(missing)[:5]}"
        )

    if citations_total > n_input_claims:
        raise ValidationError(
            f"citations_total={citations_total} exceeds n_input_claims={n_input_claims} "
            "(model cited more videos than the input pool)"
        )
    return normalised


def check_method_a(responses, cluster_members: list[dict]) -> list[dict]:
    pairs = [(m["video_id"], m["claim"]) for m in cluster_members]
    return _check_responses(responses, pairs)


def check_method_b(responses, flat_claims: list[dict]) -> list[dict]:
    pairs = [(c["video_id"], c["claim"]) for c in flat_claims]
    return _check_responses(responses, pairs)
