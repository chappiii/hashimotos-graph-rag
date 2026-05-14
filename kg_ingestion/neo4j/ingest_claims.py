"""
Insert Claim nodes and edges into Neo4j from claim_registry.json.

Three passes:
  1. MERGE all Claim nodes
  2. Create typed entity->claim edges (POSITIVE_CLAIM, NEGATIVE_CLAIM, HYPOTHETICAL_CLAIM)
  3. Create Claim->Entity CLAIM_TARGET edges

uncertain and mixed polarities are stored in polarity_counts on the Claim node
but do not get a typed edge - see design_decisions.md.

Run: uv run -m kg_ingestion.neo4j.ingest_claims
"""

import json
import sys

from neo4j import GraphDatabase

from kg_ingestion.config.kg_config import (
    NEO4J_URI,
    NEO4J_AUTH,
    NEO4J_DATABASE,
    CLAIM_REGISTRY_PATH,
    METADATA_PATH,
    STUDY_DESIGN_WEIGHTS,
)

BATCH_SIZE = 100

# polarity values that get a typed edge
_POLARITY_TO_EDGE = {
    "positive":    "POSITIVE_CLAIM",
    "negative":    "NEGATIVE_CLAIM",
    "hypothetical": "HYPOTHETICAL_CLAIM",
}

_MERGE_CLAIMS = """
UNWIND $batch AS c
MERGE (n:Claim {claim_signature: c.claim_signature})
SET
  n.relation_type    = c.relation_type,
  n.polarity_counts  = c.polarity_counts,
  n.certainty_max    = c.certainty_max,
  n.paper_count      = c.paper_count,
  n.paper_ids        = c.paper_ids,
  n.study_weight_max = c.study_weight_max
"""

_CLAIM_TARGET = """
UNWIND $batch AS c
MATCH (claim:Claim {claim_signature: c.claim_signature})
MATCH (target:Entity {canonical_name: c.target})
MERGE (claim)-[:CLAIM_TARGET]->(target)
"""

# one template per polarity - relationship type cannot be parameterized in Cypher
_POSITIVE_EDGE = """
UNWIND $batch AS c
MATCH (source:Entity {canonical_name: c.source})
MATCH (claim:Claim {claim_signature: c.claim_signature})
MERGE (source)-[:POSITIVE_CLAIM {relation_type: c.relation_type}]->(claim)
"""

_NEGATIVE_EDGE = """
UNWIND $batch AS c
MATCH (source:Entity {canonical_name: c.source})
MATCH (claim:Claim {claim_signature: c.claim_signature})
MERGE (source)-[:NEGATIVE_CLAIM {relation_type: c.relation_type}]->(claim)
"""

_HYPOTHETICAL_EDGE = """
UNWIND $batch AS c
MATCH (source:Entity {canonical_name: c.source})
MATCH (claim:Claim {claim_signature: c.claim_signature})
MERGE (source)-[:HYPOTHETICAL_CLAIM {relation_type: c.relation_type}]->(claim)
"""

_EDGE_CYPHER = {
    "POSITIVE_CLAIM":    _POSITIVE_EDGE,
    "NEGATIVE_CLAIM":    _NEGATIVE_EDGE,
    "HYPOTHETICAL_CLAIM": _HYPOTHETICAL_EDGE,
}


def load_paper_weights() -> dict[int, float]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        papers = json.load(f)["papers"]
    return {
        int(p["paper_id"]): STUDY_DESIGN_WEIGHTS.get(p.get("study_design") or "other", 0.4)
        for p in papers
    }


def load_claims() -> dict:
    with open(CLAIM_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _canonical(entity_field) -> str:
    if isinstance(entity_field, dict):
        return entity_field["canonical_name"]
    return entity_field


def build_claim_record(sig: str, data: dict, paper_weights: dict[int, float]) -> dict:
    paper_ids = data.get("paper_ids") or []
    study_weight_max = max(
        (paper_weights.get(pid, 0.4) for pid in paper_ids),
        default=0.4,
    )
    return {
        "claim_signature":  sig,
        "relation_type":    data["relation_type"],
        "source":           _canonical(data["source"]),
        "target":           _canonical(data["target"]),
        "polarity_counts":  json.dumps(data.get("polarity_counts") or {}),
        "certainty_max":    data.get("certainty_max") or "low",
        "paper_count":      data.get("paper_count") or 0,
        "paper_ids":        paper_ids,
        "study_weight_max": study_weight_max,
    }


def run_batches(session, records: list[dict], cypher: str, label: str) -> None:
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        session.run(cypher, batch=batch).consume()
        print(f"  {label}: {min(i + BATCH_SIZE, len(records))}/{len(records)}")


def main() -> None:
    paper_weights = load_paper_weights()
    raw_claims = load_claims()

    records = [
        build_claim_record(sig, data, paper_weights)
        for sig, data in raw_claims.items()
    ]
    print(f"Loaded {len(records)} claims from {CLAIM_REGISTRY_PATH}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(database=NEO4J_DATABASE) as session:

        print("Pass 1: merging Claim nodes...")
        run_batches(session, records, _MERGE_CLAIMS, "claims")

        print("Pass 2: creating CLAIM_TARGET edges...")
        run_batches(session, records, _CLAIM_TARGET, "edges")

        print("Pass 3: creating typed polarity edges...")
        for polarity, edge_type in _POLARITY_TO_EDGE.items():
            subset = [
                r for r in records
                if json.loads(r["polarity_counts"]).get(polarity, 0) > 0
            ]
            if subset:
                run_batches(session, subset, _EDGE_CYPHER[edge_type], edge_type)

    driver.close()
    print(f"Done - {len(records)} Claim nodes and edges upserted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
