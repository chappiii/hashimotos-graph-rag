"""
Insert Entity nodes into Neo4j from entity_registry.json.

Uses MERGE on canonical_name so re-runs are safe.
key_properties is stored as a JSON string because each entity_type
has a different schema - no flat property collision.

Run: uv run -m kg_ingestion.neo4j.ingest_entities
"""

import json
import sys

from neo4j import GraphDatabase

from kg_ingestion.config.kg_config import (
    NEO4J_URI,
    NEO4J_AUTH,
    NEO4J_DATABASE,
    ENTITY_REGISTRY_PATH,
)

BATCH_SIZE = 100

_CYPHER = """
UNWIND $batch AS e
MERGE (n:Entity {canonical_name: e.canonical_name})
SET
  n.entity_type    = e.entity_type,
  n.aliases        = e.aliases,
  n.key_properties = e.key_properties,
  n.paper_ids      = e.paper_ids
"""


def load_entities() -> dict:
    with open(ENTITY_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)["entities"]


def build_entity_record(canonical_name: str, data: dict) -> dict:
    return {
        "canonical_name": canonical_name,
        "entity_type":    data.get("entity_type") or "Unknown",
        "aliases":        data.get("aliases") or [],
        "key_properties": json.dumps(data.get("key_properties") or {}),
        "paper_ids":      data.get("paper_ids") or [],
    }


def ingest(session, records: list[dict]) -> None:
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        session.run(_CYPHER, batch=batch).consume()
        print(f"  inserted {min(i + BATCH_SIZE, len(records))}/{len(records)}")


def main() -> None:
    entities = load_entities()
    records = [build_entity_record(name, data) for name, data in entities.items()]
    print(f"Loaded {len(records)} entities from {ENTITY_REGISTRY_PATH}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(database=NEO4J_DATABASE) as session:
        ingest(session, records)
    driver.close()

    print(f"Done - {len(records)} Entity nodes upserted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
