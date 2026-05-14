"""
Insert Evidence nodes and edges into Neo4j from claim_registry.json.

Three passes:
  1. MERGE all Evidence nodes
  2. Create Claim -[:HAS_EVIDENCE]-> Evidence edges
  3. Create Evidence -[:FROM_PAPER]-> Paper edges

Run: uv run -m kg_ingestion.neo4j.ingest_evidence
"""

import json
import sys

from neo4j import GraphDatabase

from kg_ingestion.config.kg_config import (
    NEO4J_URI,
    NEO4J_AUTH,
    NEO4J_DATABASE,
    CLAIM_REGISTRY_PATH,
)

BATCH_SIZE = 200

_MERGE_EVIDENCE = """
UNWIND $batch AS e
MERGE (n:Evidence {evidence_id: e.evidence_id})
SET
  n.evidence_text   = e.evidence_text,
  n.paper_id        = e.paper_id,
  n.section_type    = e.section_type,
  n.section_name    = e.section_name,
  n.claim_polarity  = e.claim_polarity,
  n.claim_certainty = e.claim_certainty,
  n.claim_signature = e.claim_signature
"""

_HAS_EVIDENCE = """
UNWIND $batch AS e
MATCH (c:Claim {claim_signature: e.claim_signature})
MATCH (ev:Evidence {evidence_id: e.evidence_id})
MERGE (c)-[:HAS_EVIDENCE]->(ev)
"""

_FROM_PAPER = """
UNWIND $batch AS e
MATCH (ev:Evidence {evidence_id: e.evidence_id})
MATCH (p:Paper {paper_id: e.paper_id})
MERGE (ev)-[:FROM_PAPER]->(p)
"""


def load_evidence(claim_registry_path) -> list[dict]:
    with open(claim_registry_path, encoding="utf-8") as f:
        cr = json.load(f)

    records = []
    for sig, data in cr.items():
        for ev in data.get("evidence") or []:
            records.append({
                "evidence_id":    ev["evidence_id"],
                "evidence_text":  ev.get("evidence_text") or "",
                "paper_id":       int(ev["paper_id"]),
                "section_type":   ev.get("section_type") or "OTHER",
                "section_name":   ev.get("section_name") or "",
                "claim_polarity": ev.get("claim_polarity") or "uncertain",
                "claim_certainty": ev.get("claim_certainty") or "low",
                "claim_signature": sig,
            })
    return records


def run_batches(session, records: list[dict], cypher: str, label: str) -> None:
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        session.run(cypher, batch=batch).consume()
        print(f"  {label}: {min(i + BATCH_SIZE, len(records))}/{len(records)}")


def main() -> None:
    records = load_evidence(CLAIM_REGISTRY_PATH)
    print(f"Loaded {len(records)} evidence items from {CLAIM_REGISTRY_PATH}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(database=NEO4J_DATABASE) as session:

        print("Pass 1: merging Evidence nodes...")
        run_batches(session, records, _MERGE_EVIDENCE, "evidence")

        print("Pass 2: creating HAS_EVIDENCE edges...")
        run_batches(session, records, _HAS_EVIDENCE, "HAS_EVIDENCE")

        print("Pass 3: creating FROM_PAPER edges...")
        run_batches(session, records, _FROM_PAPER, "FROM_PAPER")

    driver.close()
    print(f"Done - {len(records)} Evidence nodes and edges upserted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
