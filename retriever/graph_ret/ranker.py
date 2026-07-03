import math
import os

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

from retriever.config.ret_config import (
    CERTAINTY_WEIGHTS,
    SECTION_WEIGHTS,
    QUALITY_PRE_FILTER_N,
    TOP_N_FINAL,
    EVIDENCE_PER_CLAIM,
    ENTITY_BUCKET_QUALITY,
    RESERVED_1HOP_SLOTS,
    RESERVED_2HOP_SLOTS,
    RESERVED_2HOP_QUALITY,
    NEO4J_URI,
    NEO4J_DATABASE,
)
from retriever.graph_ret.embed import embed_query, embed

load_dotenv()
_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


def _has_anchored_endpoints(path: dict) -> bool:
    """True if both endpoints of this path were matched as query entities.

    For 1-hop: source and target must both be in anchor_entities.
    For 2-hop: source (entities[0]) and target (entities[2]) must both be in
    anchor_entities (the bridge entities[1] is irrelevant).

    A both-anchored path is the most directly query-relevant kind: the user
    actually mentioned both endpoints, so the path answers the question they
    asked rather than being a topologically incidental match.
    """
    anchors  = path.get("anchor_entities") or []
    entities = path.get("entities") or []
    if path["length"] == 1 and len(entities) == 2:
        return entities[0] in anchors and entities[1] in anchors
    if path["length"] == 2 and len(entities) == 3:
        return entities[0] in anchors and entities[2] in anchors
    return False


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


def _claim_text(claim: dict) -> str:
    """Per-claim text used for backfill and fallback (matches embed_claims.py format)."""
    parts = [f"{claim['source_name']} {claim['relation_type']} {claim['target_name']}"]
    for ev in claim.get("evidence_list", []):
        if ev.get("evidence_text"):
            parts.append(ev["evidence_text"])
            if len(parts) == 3:
                break
    return ". ".join(parts)


_WRITE_EMBEDDING_CYPHER = """
MATCH (c:Claim {claim_signature: $sig})
SET c.embedding_text = $text, c.embedding = $vec
"""


def _write_embedding(sig: str, text: str, vec: list[float]) -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=_AUTH)
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            session.run(_WRITE_EMBEDDING_CYPHER, sig=sig, text=text, vec=vec)
    finally:
        driver.close()


def _claim_embedding(claim: dict) -> list[float]:
    """Return the pre-computed embedding, or live-embed + persist as a fallback."""
    vec = claim.get("embedding")
    if vec is not None:
        return vec

    text = _claim_text(claim)
    vec  = embed(text)
    try:
        _write_embedding(claim["claim_signature"], text, vec)
    except Exception as e:
        print(f"  [embed] WARN: write-back failed for {claim['claim_signature']}: {e}")
    claim["embedding"] = vec
    return vec


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
    reserved_2hop: int = RESERVED_2HOP_QUALITY,
) -> list[dict]:
    """
    Four-bucket pre-filter on paths (length-1 and length-2 co-ranked):
    1. Both-anchors bucket - every length-1 path with BOTH endpoints in matched specifics
                             (the most directly query-relevant direct claims, always kept)
    2. Reserved 2-hop      - top reserved_2hop length-2 paths by quality
                             (guarantees 2-hop chains survive into the hybrid stage despite
                             multiplicative path-quality decay)
    3. Entity bucket       - top entity_bucket_size paths per specific anchor entity
    4. Quality bucket      - fill remaining slots by global path-quality ranking
    """
    for p in paths:
        p["_quality_score"] = _path_quality(p)

    kept: dict[str, dict] = {}

    two_hop_sorted = sorted(
        [p for p in paths if p["length"] == 2],
        key=lambda p: p["_quality_score"],
        reverse=True,
    )
    for p in two_hop_sorted[:reserved_2hop]:
        kept[p["path_signature"]] = p

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
    top_n: int = TOP_N_FINAL,
    evidence_k: int = EVIDENCE_PER_CLAIM,
    reserved_1hop: int = RESERVED_1HOP_SLOTS,
    reserved_2hop: int = RESERVED_2HOP_SLOTS,
) -> list[dict]:
    """Embed query, score paths via per-claim embeddings (mean cosine), then
    pick top reserved_1hop 1-hop paths + top reserved_2hop 2-hop paths.

    If one bucket is starved (fewer candidates than its reservation), the
    unused slots spill over to the other bucket so total top-N is preserved.
    """
    query_vec = embed_query(query)
    print("  [embed] query embedded")

    cache_hits = 0
    fallbacks  = 0
    for p in paths:
        cosines: list[float] = []
        for c in p["claims"]:
            if c.get("embedding") is not None:
                cache_hits += 1
            else:
                fallbacks += 1
            vec = _claim_embedding(c)
            cosines.append(_cosine(query_vec, vec))
        p["_cosine_score"] = sum(cosines) / len(cosines) if cosines else 0.0
    print(f"  [embed] scored {len(paths)} paths  (cache_hits={cache_hits}, fallbacks={fallbacks})")

    for p in paths:
        p["_hybrid_score"] = p["_cosine_score"]

    # Sort each pool by (both-endpoints-anchored, hybrid_score) descending.
    # Both-anchored paths rank above all others within their hop class — they answer
    # the query's actual entity pair, not just a topologically nearby cosine match.
    # Cosine breaks ties.
    one_hop = sorted(
        [p for p in paths if p["length"] == 1],
        key=lambda p: (_has_anchored_endpoints(p), p["_hybrid_score"]),
        reverse=True,
    )
    two_hop = sorted(
        [p for p in paths if p["length"] == 2],
        key=lambda p: (_has_anchored_endpoints(p), p["_hybrid_score"]),
        reverse=True,
    )

    picked_1h = one_hop[:reserved_1hop]
    picked_2h = two_hop[:reserved_2hop]

    unused_1h_slots = reserved_1hop - len(picked_1h)
    if unused_1h_slots > 0:
        already = {p["path_signature"] for p in picked_2h}
        spill = [p for p in two_hop if p["path_signature"] not in already][:unused_1h_slots]
        picked_2h.extend(spill)

    unused_2h_slots = reserved_2hop - len(picked_2h)
    if unused_2h_slots > 0:
        already = {p["path_signature"] for p in picked_1h}
        spill = [p for p in one_hop if p["path_signature"] not in already][:unused_2h_slots]
        picked_1h.extend(spill)

    final = sorted(
        picked_1h + picked_2h,
        key=lambda p: p["_hybrid_score"], reverse=True,
    )[:top_n]
    print(f"  [embed] picked 1-hop={len(picked_1h)}  2-hop={len(picked_2h)}  final={len(final)}")

    for p in final:
        for c in p["claims"]:
            c["evidence_list"] = _top_evidence(c, k=evidence_k)

    return final
