"""
Embed evidence sentences and upsert into the `evidence` Qdrant collection.
Skip logic: checks which evidence_ids already exist before embedding.

Point IDs are the UUID evidence_id strings from claim_registry.json.

Run: uv run -m kg_ingestion.qdrant.ingest_evidence
"""

import json
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from kg_ingestion.config.kg_config import QDRANT_URL, CLAIM_REGISTRY_PATH
from kg_ingestion.qdrant.embed import embed

COLLECTION = "evidence"
BATCH_SIZE  = 20


def load_evidence_records(registry_path) -> list[dict]:
    with open(registry_path, encoding="utf-8") as f:
        registry = json.load(f)

    records = []
    for claim_sig, claim in registry.items():
        source_entity = claim["source"]["canonical_name"]
        target_entity = claim["target"]["canonical_name"]
        relation_type = claim["relation_type"]

        for ev in claim.get("evidence", []):
            records.append({
                "evidence_id":    ev["evidence_id"],
                "claim_signature": claim_sig,
                "paper_id":       ev["paper_id"],
                "section_type":   ev.get("section_type", "OTHER"),
                "claim_polarity": ev.get("claim_polarity", "positive"),
                "claim_certainty": ev.get("claim_certainty", "low"),
                "relation_type":  relation_type,
                "source_entity":  source_entity,
                "target_entity":  target_entity,
                "evidence_text":  ev.get("evidence_text", ""),
            })
    return records


def existing_ids(client: QdrantClient, ids: list[str]) -> set[str]:
    found = client.retrieve(collection_name=COLLECTION, ids=ids, with_payload=False, with_vectors=False)
    return {p.id for p in found}


def ingest(client: QdrantClient, records: list[dict]) -> None:
    all_ids   = [r["evidence_id"] for r in records]
    skip_ids: set[str] = set()
    for i in range(0, len(all_ids), 500):
        skip_ids |= existing_ids(client, all_ids[i : i + 500])

    to_ingest = [r for r in records if r["evidence_id"] not in skip_ids]
    print(f"  {len(skip_ids)} already ingested, {len(to_ingest)} to embed")

    for i in range(0, len(to_ingest), BATCH_SIZE):
        batch = to_ingest[i : i + BATCH_SIZE]
        points = []
        for r in batch:
            vector = embed(r["evidence_text"])
            payload = {k: v for k, v in r.items() if k != "evidence_id"}
            points.append(PointStruct(id=r["evidence_id"], vector=vector, payload=payload))
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  upserted {min(i + BATCH_SIZE, len(to_ingest))}/{len(to_ingest)}")


def main() -> None:
    records = load_evidence_records(CLAIM_REGISTRY_PATH)
    print(f"Loaded {len(records)} evidence items from {CLAIM_REGISTRY_PATH}")

    client = QdrantClient(url=QDRANT_URL)
    ingest(client, records)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
