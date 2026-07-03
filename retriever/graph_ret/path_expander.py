import json
import os
from itertools import combinations

from dotenv import load_dotenv
from neo4j import GraphDatabase

from retriever.config.ret_config import NEO4J_URI, NEO4J_DATABASE, GENERIC_ENTITIES

load_dotenv()
_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))


_ANCHORED_2HOP_CYPHER = """
MATCH (a:Entity)-[:POSITIVE_CLAIM|NEGATIVE_CLAIM|HYPOTHETICAL_CLAIM]->(c1:Claim)
      -[:CLAIM_TARGET]->(mid:Entity)
      -[:POSITIVE_CLAIM|NEGATIVE_CLAIM|HYPOTHETICAL_CLAIM]->(c2:Claim)
      -[:CLAIM_TARGET]->(b:Entity)
WHERE ((a.canonical_name = $start AND b.canonical_name = $end)
    OR (a.canonical_name = $end AND b.canonical_name = $start))
  AND mid.canonical_name <> $start
  AND mid.canonical_name <> $end
  AND NOT mid.canonical_name IN $generic_block
WITH a, mid, b, c1, c2,
  [(c1)-[:HAS_EVIDENCE]->(ev:Evidence)-[:FROM_PAPER]->(p:Paper) | {
      evidence_text:   ev.evidence_text,
      paper_id:        p.paper_id,
      paper_title:     p.title,
      paper_year:      p.year,
      study_design:    p.study_design,
      section_type:    ev.section_type,
      claim_polarity:  ev.claim_polarity,
      claim_certainty: ev.claim_certainty
  }] AS c1_evidence,
  [(c2)-[:HAS_EVIDENCE]->(ev:Evidence)-[:FROM_PAPER]->(p:Paper) | {
      evidence_text:   ev.evidence_text,
      paper_id:        p.paper_id,
      paper_title:     p.title,
      paper_year:      p.year,
      study_design:    p.study_design,
      section_type:    ev.section_type,
      claim_polarity:  ev.claim_polarity,
      claim_certainty: ev.claim_certainty
  }] AS c2_evidence
RETURN
  a.canonical_name      AS source_name,
  mid.canonical_name    AS bridge_name,
  b.canonical_name      AS target_name,
  c1.claim_signature    AS c1_signature,
  c1.relation_type      AS c1_relation,
  c1.polarity_counts    AS c1_polarity,
  c1.certainty_max      AS c1_certainty,
  c1.paper_count        AS c1_papers,
  c1.study_weight_max   AS c1_study_weight,
  c1.embedding          AS c1_embedding,
  c1_evidence,
  c2.claim_signature    AS c2_signature,
  c2.relation_type      AS c2_relation,
  c2.polarity_counts    AS c2_polarity,
  c2.certainty_max      AS c2_certainty,
  c2.paper_count        AS c2_papers,
  c2.study_weight_max   AS c2_study_weight,
  c2.embedding          AS c2_embedding,
  c2_evidence
"""


def _row_to_claim(row: dict, prefix: str, source: str, target: str) -> dict:
    return {
        "claim_signature":  row[f"{prefix}_signature"],
        "source_name":      source,
        "target_name":      target,
        "relation_type":    row[f"{prefix}_relation"],
        "polarity_counts":  json.loads(row[f"{prefix}_polarity"] or "{}"),
        "certainty_max":    row.get(f"{prefix}_certainty") or "low",
        "paper_count":      row.get(f"{prefix}_papers") or 0,
        "study_weight_max": row.get(f"{prefix}_study_weight") or 0.4,
        "embedding":        row.get(f"{prefix}_embedding"),
        "evidence_list": [
            e for e in row[f"{prefix}_evidence"] if e.get("evidence_text")
        ],
    }


def _row_to_path(row: dict, matched_entities: list[str]) -> dict:
    source = row["source_name"]
    bridge = row["bridge_name"]
    target = row["target_name"]

    c1 = _row_to_claim(row, "c1", source, bridge)
    c2 = _row_to_claim(row, "c2", bridge, target)

    entities = [source, bridge, target]
    anchors  = [e for e in entities if e in matched_entities]

    return {
        "path_signature":  f"{c1['claim_signature']}>>{c2['claim_signature']}",
        "length":          2,
        "entities":        entities,
        "claims":          [c1, c2],
        "anchor_entities": anchors,
    }


def _build_pairs(matched: list[str]) -> list[tuple[str, str]]:
    specific = [e for e in matched if e not in GENERIC_ENTITIES]
    generic  = [e for e in matched if e in GENERIC_ENTITIES]

    pairs: list[tuple[str, str]] = []
    pairs.extend(combinations(specific, 2))            # specific x specific
    pairs.extend((s, g) for s in specific for g in generic)  # specific x generic
    return pairs


def expand_paths(matched_entities: list[str]) -> list[dict]:
    """Anchored 2-hop expansion between every pair of matched entities.

    Generic entities are allowed as endpoints but blocked as bridges (except
    when they are themselves an endpoint of the current pair).
    """
    pairs = _build_pairs(matched_entities)
    if not pairs:
        return []

    paths: dict[str, dict] = {}
    driver = GraphDatabase.driver(NEO4J_URI, auth=_AUTH)
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for start, end in pairs:
                generic_block = [g for g in GENERIC_ENTITIES if g not in (start, end)]
                rows = session.run(
                    _ANCHORED_2HOP_CYPHER,
                    start=start,
                    end=end,
                    generic_block=generic_block,
                ).data()
                for row in rows:
                    path = _row_to_path(row, matched_entities)
                    if path["path_signature"] not in paths:
                        paths[path["path_signature"]] = path
    finally:
        driver.close()

    return list(paths.values())


if __name__ == "__main__":
    from retriever.config.ret_config import SEP, SEP2
    from retriever.graph_ret.query_decomposer import decompose
    from retriever.graph_ret.entity_matcher import load_registry, match_entities

    query = "What is the effect of Vitamin D supplementation on TSH levels in Hashimoto's Thyroiditis patients?"

    registry = load_registry()
    spans    = decompose(query)
    matched  = match_entities(spans, registry)

    print(SEP)
    print(f"QUERY:   {query}")
    print(f"MATCHED: {matched}")
    print(f"PAIRS:   {_build_pairs(matched)}")
    print(SEP)

    paths = expand_paths(matched)

    print(f"\n  2-HOP PATHS  ({len(paths)})")
    print(SEP2)
    for i, p in enumerate(paths, 1):
        a, mid, b = p["entities"]
        r1 = p["claims"][0]["relation_type"]
        r2 = p["claims"][1]["relation_type"]
        print(f"  [{i}] {a} --[{r1}]--> {mid} --[{r2}]--> {b}")
        print(f"       anchors={p['anchor_entities']}")
        print(f"       c1: certainty={p['claims'][0]['certainty_max']}  papers={p['claims'][0]['paper_count']}")
        print(f"       c2: certainty={p['claims'][1]['certainty_max']}  papers={p['claims'][1]['paper_count']}")
    print(SEP)
