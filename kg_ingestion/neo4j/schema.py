"""
Create all Neo4j constraints and indexes.

Idempotent: uses IF NOT EXISTS so safe to re-run.
Constraints also create a backing index automatically.

Run: uv run -m kg_ingestion.neo4j.schema
"""

import sys
from neo4j import GraphDatabase

from kg_ingestion.config.kg_config import NEO4J_URI, NEO4J_AUTH, NEO4J_DATABASE

# (label, property, type)
# "unique"  -> CONSTRAINT ... IS UNIQUE  (also creates an index)
# "index"   -> plain BTREE index for fast lookup / filtering
_CONSTRAINTS: list[tuple[str, str]] = [
    ("Paper",    "paper_id"),
    ("Entity",   "canonical_name"),
    ("Claim",    "claim_signature"),
    ("Evidence", "evidence_id"),
]

_INDEXES: list[tuple[str, str]] = [
    ("Paper",    "year"),
    ("Paper",    "study_design"),
    ("Entity",   "entity_type"),
    ("Claim",    "relation_type"),
    ("Claim",    "certainty_max"),
    ("Claim",    "paper_count"),
    ("Evidence", "paper_id"),
    ("Evidence", "section_type"),
    ("Evidence", "claim_polarity"),
    ("Evidence", "claim_certainty"),
]


def _constraint_name(label: str, prop: str) -> str:
    return f"constraint_{label.lower()}_{prop}"


def _index_name(label: str, prop: str) -> str:
    return f"index_{label.lower()}_{prop}"


def create_schema(session) -> None:
    for label, prop in _CONSTRAINTS:
        name = _constraint_name(label, prop)
        session.run(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )
        print(f"  constraint: {label}.{prop}")

    for label, prop in _INDEXES:
        name = _index_name(label, prop)
        session.run(
            f"CREATE INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{prop})"
        )
        print(f"  index:      {label}.{prop}")


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(database=NEO4J_DATABASE) as session:
        print("Creating constraints and indexes...")
        create_schema(session)
    driver.close()
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
