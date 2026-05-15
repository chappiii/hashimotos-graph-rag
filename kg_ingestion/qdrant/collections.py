"""
Create (or verify) all 3 Qdrant collections: papers, chunks, evidence.
Idempotent - safe to re-run.

Run: uv run -m kg_ingestion.qdrant.collections
"""

import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
)

from kg_ingestion.config.kg_config import QDRANT_URL, VECTOR_DIM

COLLECTIONS = {
    "papers": {
        "indexes": {
            "paper_id":     PayloadSchemaType.INTEGER,
            "keywords":     PayloadSchemaType.KEYWORD,
            "year":         PayloadSchemaType.INTEGER,
            "study_design": PayloadSchemaType.KEYWORD,
        }
    },
    "chunks": {
        "indexes": {
            "paper_id":    PayloadSchemaType.INTEGER,
            "section_type": PayloadSchemaType.KEYWORD,
        }
    },
    "evidence": {
        "indexes": {
            "paper_id":        PayloadSchemaType.INTEGER,
            "claim_polarity":  PayloadSchemaType.KEYWORD,
            "claim_certainty": PayloadSchemaType.KEYWORD,
            "relation_type":   PayloadSchemaType.KEYWORD,
        }
    },
}


def setup_collections(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}

    for name, cfg in COLLECTIONS.items():
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            print(f"  created collection: {name}")
        else:
            print(f"  collection exists:  {name}")

        for field, schema_type in cfg["indexes"].items():
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema_type,
            )

    print("Done.")


def main() -> None:
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    setup_collections(client)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
