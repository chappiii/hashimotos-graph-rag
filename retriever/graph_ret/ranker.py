import math

import numpy as np

from retriever.config.ret_config import (
    CERTAINTY_WEIGHTS,
    SECTION_WEIGHTS,
    QUALITY_PRE_FILTER_N,
    TOP_N_FINAL,
    EVIDENCE_PER_CLAIM,
)
from retriever.graph_ret.embed import embed_query, embed


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def _quality_score(claim: dict) -> float:
    certainty_w = CERTAINTY_WEIGHTS.get(claim.get("certainty_max", "low"), 0.2)
    paper_w     = math.log1p(claim.get("paper_count", 0))
    study_w     = claim.get("study_weight_max") or 0.4

    ev_list = [e for e in claim.get("evidence_list", []) if e.get("section_type")]
    section_w = (
        max(SECTION_WEIGHTS.get(e["section_type"], 0.4) for e in ev_list)
        if ev_list else 0.4
    )
    return certainty_w * paper_w * study_w * section_w


def _claim_text(claim: dict) -> str:
    triple = f"{claim['source_name']} {claim['relation_type']} {claim['target_name']}"
    ev_texts = [e["evidence_text"] for e in claim.get("evidence_list", [])[:2] if e.get("evidence_text")]
    return triple + (". " + " ".join(ev_texts) if ev_texts else "")


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
    claims: list[dict],
    specific_entities: list[str] | None = None,
    top_n: int = QUALITY_PRE_FILTER_N,
    entity_bucket_size: int = 5,
) -> list[dict]:
    """
    Two-bucket pre-filter:
    1. Entity bucket  - top entity_bucket_size claims per specific entity
                        (guarantees niche entities survive regardless of paper count)
    2. Quality bucket - fill remaining slots with global quality ranking
    """
    for c in claims:
        c["_quality_score"] = _quality_score(c)

    kept: dict[str, dict] = {}

    if specific_entities:
        for entity in specific_entities:
            entity_claims = [
                c for c in claims
                if c["source_name"] == entity or c["target_name"] == entity
            ]
            entity_claims.sort(key=lambda c: c["_quality_score"], reverse=True)
            for c in entity_claims[:entity_bucket_size]:
                kept[c["claim_signature"]] = c

    remaining = top_n - len(kept)
    if remaining > 0:
        global_ranked = sorted(claims, key=lambda c: c["_quality_score"], reverse=True)
        for c in global_ranked:
            if c["claim_signature"] not in kept:
                kept[c["claim_signature"]] = c
                remaining -= 1
                if remaining == 0:
                    break

    return list(kept.values())


def rank_by_hybrid(
    claims: list[dict],
    query: str,
    specific_entities: list[str] | None = None,
    top_n: int = TOP_N_FINAL,
    evidence_k: int = EVIDENCE_PER_CLAIM,
) -> list[dict]:
    """
    Embed query + each claim text, compute cosine similarity, return top_n claims
    with evidence_list trimmed to the top-k highest-section-weight sentences.
    """
    query_vec = embed_query(query)
    print("  [embed] query embedded")

    for i, c in enumerate(claims):
        c["_cosine_score"] = _cosine(query_vec, embed(_claim_text(c)))
        print(f"  [embed] {i+1}/{len(claims)} claims embedded", end="\r")
    print()

    for c in claims:
        c["_hybrid_score"] = c["_cosine_score"]

    ranked = sorted(claims, key=lambda c: c["_hybrid_score"], reverse=True)

    final: dict[str, dict] = {}
    if specific_entities:
        for entity in specific_entities:
            for c in ranked:
                if c["source_name"] == entity or c["target_name"] == entity:
                    if c["claim_signature"] not in final:
                        final[c["claim_signature"]] = c
                        break

    remaining = top_n - len(final)
    for c in ranked:
        if remaining == 0:
            break
        if c["claim_signature"] not in final:
            final[c["claim_signature"]] = c
            remaining -= 1

    top = sorted(final.values(), key=lambda c: c["_hybrid_score"], reverse=True)[:top_n]

    print(f"  [embed] selecting top evidence for {len(top)} claims...")
    for c in top:
        c["evidence_list"] = _top_evidence(c, k=evidence_k)

    return top
