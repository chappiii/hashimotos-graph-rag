import math

import numpy as np

from retriever.config.ret_config import (
    CERTAINTY_WEIGHTS,
    SECTION_WEIGHTS,
    QUALITY_PRE_FILTER_N,
    TOP_N_FINAL,
    EVIDENCE_PER_CLAIM,
    ENTITY_BUCKET_QUALITY,
    ENTITY_BUCKET_HYBRID,
)
from retriever.graph_ret.embed import embed_query, embed


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def _claim_quality(claim: dict) -> float:
    certainty_w = CERTAINTY_WEIGHTS.get(claim.get("certainty_max", "low"), 0.2)
    paper_w     = math.log1p(claim.get("paper_count", 0))
    study_w     = claim.get("study_weight_max") or 0.4

    ev_list = [e for e in claim.get("evidence_list", []) if e.get("section_type")]
    section_w = (
        max(SECTION_WEIGHTS.get(e["section_type"], 0.4) for e in ev_list)
        if ev_list else 0.4
    )
    return certainty_w * paper_w * study_w * section_w


def _path_quality(path: dict) -> float:
    """Product of per-claim qualities - chain decays multiplicatively."""
    score = 1.0
    for c in path["claims"]:
        score *= _claim_quality(c)
    return score


def _path_text(path: dict) -> str:
    """Concatenated chain text used for hybrid (semantic) re-ranking."""
    parts: list[str] = []
    for c in path["claims"]:
        parts.append(f"{c['source_name']} {c['relation_type']} {c['target_name']}")
    for c in path["claims"]:
        for ev in c.get("evidence_list", [])[:2]:
            if ev.get("evidence_text"):
                parts.append(ev["evidence_text"])
    return ". ".join(parts)


def _top_evidence(claim: dict, k: int) -> list[dict]:
    ev_list = [e for e in claim.get("evidence_list", []) if e.get("evidence_text")]
    if not ev_list:
        return []
    return sorted(
        ev_list,
        key=lambda e: SECTION_WEIGHTS.get(e.get("section_type", "OTHER"), 0.4),
        reverse=True,
    )[:k]


def rank_by_quality(
    paths: list[dict],
    specific_entities: list[str] | None = None,
    top_n: int = QUALITY_PRE_FILTER_N,
    entity_bucket_size: int = ENTITY_BUCKET_QUALITY,
) -> list[dict]:
    """
    Three-bucket pre-filter on paths (length-1 and length-2 co-ranked):
    1. Both-anchors bucket - every length-1 path with BOTH endpoints in matched specifics
                             (the most directly query-relevant direct claims, always kept)
    2. Entity bucket       - top entity_bucket_size paths per specific anchor entity
    3. Quality bucket      - fill remaining slots by global path-quality ranking
    """
    for p in paths:
        p["_quality_score"] = _path_quality(p)

    kept: dict[str, dict] = {}

    if specific_entities:
        specific_set = set(specific_entities)
        for p in paths:
            if p["length"] == 1 and set(p["entities"]).issubset(specific_set):
                kept[p["path_signature"]] = p

        for entity in specific_entities:
            entity_paths = [p for p in paths if entity in p["anchor_entities"]]
            entity_paths.sort(key=lambda p: p["_quality_score"], reverse=True)
            for p in entity_paths[:entity_bucket_size]:
                kept[p["path_signature"]] = p

    remaining = top_n - len(kept)
    if remaining > 0:
        global_ranked = sorted(paths, key=lambda p: p["_quality_score"], reverse=True)
        for p in global_ranked:
            if p["path_signature"] not in kept:
                kept[p["path_signature"]] = p
                remaining -= 1
                if remaining == 0:
                    break

    return list(kept.values())


def rank_by_hybrid(
    paths: list[dict],
    query: str,
    specific_entities: list[str] | None = None,
    top_n: int = TOP_N_FINAL,
    evidence_k: int = EVIDENCE_PER_CLAIM,
    entity_bucket_size: int = ENTITY_BUCKET_HYBRID,
) -> list[dict]:
    """Embed query + each path (concat chain text), rank by cosine, trim evidence."""
    query_vec = embed_query(query)
    print("  [embed] query embedded")

    for i, p in enumerate(paths):
        p["_cosine_score"] = _cosine(query_vec, embed(_path_text(p)))
        print(f"  [embed] {i+1}/{len(paths)} paths embedded", end="\r")
    print()

    for p in paths:
        p["_hybrid_score"] = p["_cosine_score"]

    ranked = sorted(paths, key=lambda p: p["_hybrid_score"], reverse=True)

    final: dict[str, dict] = {}
    if specific_entities:
        for entity in specific_entities:
            kept = 0
            for p in ranked:
                if kept >= entity_bucket_size:
                    break
                if entity in p["anchor_entities"] and p["path_signature"] not in final:
                    final[p["path_signature"]] = p
                    kept += 1

    remaining = top_n - len(final)
    for p in ranked:
        if remaining == 0:
            break
        if p["path_signature"] not in final:
            final[p["path_signature"]] = p
            remaining -= 1

    top = sorted(final.values(), key=lambda p: p["_hybrid_score"], reverse=True)[:top_n]

    print(f"  [embed] selecting top evidence for {len(top)} paths...")
    for p in top:
        for c in p["claims"]:
            c["evidence_list"] = _top_evidence(c, k=evidence_k)

    return top
