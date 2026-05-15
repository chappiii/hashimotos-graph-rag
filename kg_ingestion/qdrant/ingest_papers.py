"""
Embed purpose_of_work and upsert paper records into the `papers` Qdrant collection.
Skip logic: checks which paper_ids already exist before embedding.

Run: uv run -m kg_ingestion.qdrant.ingest_papers
"""

import json
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from kg_ingestion.config.kg_config import (
    QDRANT_URL,
    METADATA_PATH,
    STUDY_DESIGN_WEIGHTS,
)
from kg_ingestion.qdrant.embed import embed

COLLECTION = "papers"
BATCH_SIZE  = 20


def load_papers() -> list[dict]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)["papers"]


def build_payload(raw: dict) -> dict:
    study_design = raw.get("study_design") or "other"
    return {
        "paper_id":       int(raw["paper_id"]),
        "title":          raw.get("title") or "",
        "doi":            raw.get("doi") or "",
        "year":           raw.get("published_year"),
        "study_design":   study_design,
        "study_weight":   STUDY_DESIGN_WEIGHTS.get(study_design, 0.4),
        "countries":      raw.get("countries") or [],
        "keywords":       raw.get("keywords") or [],
        "purpose_of_work": raw.get("purpose_of_work") or "",
    }


def existing_ids(client: QdrantClient, ids: list[int]) -> set[int]:
    """Return the subset of ids that already exist in the collection."""
    found = client.retrieve(collection_name=COLLECTION, ids=ids, with_payload=False, with_vectors=False)
    return {p.id for p in found}


def ingest(client: QdrantClient, records: list[dict]) -> None:
    all_ids   = [r["paper_id"] for r in records]
    skip_ids  = existing_ids(client, all_ids)
    to_ingest = [r for r in records if r["paper_id"] not in skip_ids]

    print(f"  {len(skip_ids)} already ingested, {len(to_ingest)} to embed")

    for i in range(0, len(to_ingest), BATCH_SIZE):
        batch = to_ingest[i : i + BATCH_SIZE]
        points = []
        for r in batch:
            text = r["purpose_of_work"] or r["title"]
            vector = embed(text)
            points.append(PointStruct(id=r["paper_id"], vector=vector, payload=r))
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  upserted {min(i + BATCH_SIZE, len(to_ingest))}/{len(to_ingest)}")


def main() -> None:
    raw_papers = load_papers()
    records    = [build_payload(p) for p in raw_papers]
    print(f"Loaded {len(records)} papers from {METADATA_PATH}")

    client = QdrantClient(url=QDRANT_URL)
    ingest(client, records)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
