"""Stage 2: Embed every claim with Qwen3-Embedding-8B (vLLM offline) and
greedy single-link cluster per query at cosine threshold tau.

Same algorithm as the validated MAGMaR pipeline. Embeds all claims in one
batched call across all queries for throughput, then slices per query.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from vllm import LLM

from . import data_io


def _greedy_cluster(embs: np.ndarray, tau: float):
    n = embs.shape[0]
    if n == 0:
        return [], np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    E = embs / norms
    sim = E @ E.T

    clusters: list[list[int]] = []
    assigned = [-1] * n
    for i in range(n):
        if assigned[i] >= 0:
            continue
        cid = len(clusters)
        clusters.append([i])
        assigned[i] = cid
        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue
            members = clusters[cid]
            max_sim = float(sim[j, members].max())
            if max_sim >= tau:
                clusters[cid].append(j)
                assigned[j] = cid
    return clusters, sim


def run(cfg):
    cfg.ensure_dirs()
    per_query_files = sorted(cfg.per_query_dir.glob("query_*.json"))
    per_query: list[dict] = []
    for p in per_query_files:
        with open(p) as f:
            d = json.load(f)
        flat = []
        for v in d["videos"]:
            for ci, c in enumerate(v["claims"]):
                flat.append((v["video_id"], ci, c))
        per_query.append({"query_id": d["query_id"], "claims": flat})

    flat_texts = [t for it in per_query for (_, _, t) in it["claims"]]
    print(f"[stage2] embedding {len(flat_texts)} claims with {cfg.embed_model}")

    llm = LLM(
        model=cfg.embed_model,
        runner="pooling",
        convert="embed",
        gpu_memory_utilization=cfg.embed_gpu_mem_util,
        max_model_len=cfg.embed_max_len,
        enforce_eager=False,
        trust_remote_code=True,
    )
    outs = llm.embed(flat_texts)
    vecs = [np.asarray(o.outputs.embedding, dtype=np.float32) for o in outs]
    embs_all = np.stack(vecs, axis=0) if vecs else np.zeros((0, 1), dtype=np.float32)
    print(f"[stage2] embeddings: {embs_all.shape}")

    cursor = 0
    rows = []
    for it in per_query:
        n = len(it["claims"])
        embs = embs_all[cursor:cursor + n]
        cursor += n
        clusters, sim = _greedy_cluster(embs, cfg.cluster_tau)

        out = {
            "query_id": it["query_id"],
            "tau": cfg.cluster_tau,
            "model": cfg.embed_model,
            "n_claims": n,
            "n_clusters": len(clusters),
            "clusters": [],
        }
        for cid, members in enumerate(clusters):
            entry = {
                "cluster_id": cid,
                "size": len(members),
                "members": [
                    {
                        "video_id": it["claims"][m][0],
                        "claim_idx": it["claims"][m][1],
                        "claim": it["claims"][m][2],
                    }
                    for m in members
                ],
            }
            if len(members) > 1:
                idxs = members
                sub = sim[np.ix_(idxs, idxs)]
                mask = ~np.eye(len(idxs), dtype=bool)
                entry["min_pairwise_sim"] = float(sub[mask].min())
                entry["mean_pairwise_sim"] = float(sub[mask].mean())
            out["clusters"].append(entry)

        data_io.write_json(
            cfg.clusters_dir / f"query_{it['query_id']}_clusters.json", out
        )
        rows.append((it["query_id"], n, len(clusters)))

    print(f"[stage2] wrote {len(rows)} cluster files to {cfg.clusters_dir}")
    print(f"{'qid':>6} {'claims':>7} {'clusters':>9} {'merged':>7}")
    for qid, nc, nk in rows:
        print(f"{qid:>6} {nc:>7} {nk:>9} {nc - nk:>7}")

    # Release the embedder so the 30B aggregator can claim the GPU in stage 3.
    try:
        import gc
        import torch
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
