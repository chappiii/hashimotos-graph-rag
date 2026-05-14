"""
Insert Paper nodes into Neo4j from data/metadata.json.

Uses MERGE on paper_id so re-runs are safe (no duplicates).
study_design is read directly from the JSON; study_weight is looked up from STUDY_DESIGN_WEIGHTS.

Run: uv run -m kg_ingestion.neo4j.ingest_papers
"""

import json
import sys

from neo4j import GraphDatabase

from kg_ingestion.config.kg_config import (
    NEO4J_URI,
    NEO4J_AUTH,
    NEO4J_DATABASE,
    METADATA_PATH,
    STUDY_DESIGN_WEIGHTS,
)

BATCH_SIZE = 50

_CYPHER = """
UNWIND $batch AS p
MERGE (n:Paper {paper_id: p.paper_id})
SET
  n.title           = p.title,
  n.doi             = p.doi,
  n.year            = p.year,
  n.study_design    = p.study_design,
  n.study_weight    = p.study_weight,
  n.countries       = p.countries,
  n.keywords        = p.keywords,
  n.purpose_of_work = p.purpose_of_work
"""


def load_papers() -> list[dict]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)["papers"]


def build_paper_record(raw: dict) -> dict:
    study_design = raw.get("study_design") or "other"
    return {
        "paper_id":        int(raw["paper_id"]),
        "title":           raw.get("title") or "",
        "doi":             raw.get("doi"),
        "year":            raw.get("published_year"),
        "study_design":    study_design,
        "study_weight":    STUDY_DESIGN_WEIGHTS.get(study_design, 0.4),
        "countries":       raw.get("countries") or [],
        "keywords":        raw.get("keywords") or [],
        "purpose_of_work": raw.get("purpose_of_work") or "",
    }


def ingest(session, records: list[dict]) -> None:
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        session.run(_CYPHER, batch=batch).consume()
        print(f"  inserted {min(i + BATCH_SIZE, len(records))}/{len(records)}")


def main() -> None:
    raw_papers = load_papers()
    records = [build_paper_record(p) for p in raw_papers]
    print(f"Loaded {len(records)} papers from {METADATA_PATH}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    with driver.session(database=NEO4J_DATABASE) as session:
        ingest(session, records)
    driver.close()

    print(f"Done - {len(records)} Paper nodes upserted.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
