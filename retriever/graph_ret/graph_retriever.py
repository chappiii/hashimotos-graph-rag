import json
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

from retriever.config.ret_config import NEO4J_URI, NEO4J_DATABASE, GENERIC_ENTITIES

load_dotenv()
_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))

_EVIDENCE_FRAGMENT = """
  collect({
    evidence_text:   ev.evidence_text,
    paper_id:        p.paper_id,
    paper_title:     p.title,
    paper_year:      p.year,
    study_design:    p.study_design,
    section_type:    ev.section_type,
    claim_polarity:  ev.claim_polarity,
    claim_certainty: ev.claim_certainty
  }) AS evidence_list
"""

_BY_SPECIFIC_CYPHER = """
MATCH (source:Entity)-[:POSITIVE_CLAIM|NEGATIVE_CLAIM|HYPOTHETICAL_CLAIM]->(c:Claim)
      -[:CLAIM_TARGET]->(target:Entity)
WHERE source.canonical_name IN $names OR target.canonical_name IN $names
WITH DISTINCT source, target, c
OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(ev:Evidence)-[:FROM_PAPER]->(p:Paper)
RETURN
  source.canonical_name AS source_name,
  target.canonical_name AS target_name,
  c.claim_signature     AS claim_signature,
  c.relation_type       AS relation_type,
  c.polarity_counts     AS polarity_counts,
  c.certainty_max       AS certainty_max,
  c.paper_count         AS paper_count,
  c.study_weight_max    AS study_weight_max,
""" + _EVIDENCE_FRAGMENT

_BY_GENERIC_CYPHER = """
MATCH (source:Entity)-[:POSITIVE_CLAIM|NEGATIVE_CLAIM|HYPOTHETICAL_CLAIM]->(c:Claim)
      -[:CLAIM_TARGET]->(target:Entity)
WHERE (source.canonical_name IN $generic AND target.canonical_name IN $specific)
   OR (source.canonical_name IN $specific AND target.canonical_name IN $generic)
WITH DISTINCT source, target, c
OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(ev:Evidence)-[:FROM_PAPER]->(p:Paper)
RETURN
  source.canonical_name AS source_name,
  target.canonical_name AS target_name,
  c.claim_signature     AS claim_signature,
  c.relation_type       AS relation_type,
  c.polarity_counts     AS polarity_counts,
  c.certainty_max       AS certainty_max,
  c.paper_count         AS paper_count,
  c.study_weight_max    AS study_weight_max,
""" + _EVIDENCE_FRAGMENT


def _parse_rows(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for row in rows:
        sig = row["claim_signature"]
        if sig in seen:
            continue
        seen[sig] = {
            "claim_signature":  sig,
            "source_name":      row["source_name"],
            "target_name":      row["target_name"],
            "relation_type":    row["relation_type"],
            "polarity_counts":  json.loads(row["polarity_counts"] or "{}"),
            "certainty_max":    row.get("certainty_max") or "low",
            "paper_count":      row.get("paper_count") or 0,
            "study_weight_max": row.get("study_weight_max") or 0.4,
            "evidence_list": [
                e for e in row["evidence_list"] if e.get("evidence_text")
            ],
        }
    return list(seen.values())


def get_claims(entity_names: list[str]) -> list[dict]:
    if not entity_names:
        return []

    specific = [e for e in entity_names if e not in GENERIC_ENTITIES]
    generic  = [e for e in entity_names if e in GENERIC_ENTITIES]

    driver = GraphDatabase.driver(NEO4J_URI, auth=_AUTH)
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            rows = []
            if specific:
                rows += session.run(_BY_SPECIFIC_CYPHER, names=specific).data()
            if specific and generic:
                rows += session.run(
                    _BY_GENERIC_CYPHER, specific=specific, generic=generic
                ).data()
    finally:
        driver.close()

    return _parse_rows(rows)
