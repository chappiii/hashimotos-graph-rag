import argparse

from kg_construction.config.kg_construction_config import (
    ENTITY_RELATION_DIR,
    METADATA_DIR,
    NEO4J_URI,
    NEO4J_AUTH,
    QDRANT_URL,
    QDRANT_COLLECTION_PARENT,
    QDRANT_COLLECTION_CHILDREN,
    VECTOR_DIMENSION,
    BATCH_SIZE,
)
from kg_construction.utils.connect import connect_neo4j, connect_qdrant
from kg_construction.utils.neo4j_ingest import ingest_to_neo4j
from kg_construction.utils.qdrant_ingest import (
    create_collection,
    ingest_parents,
    ingest_children,
)


def run_neo4j():
    driver = connect_neo4j(NEO4J_URI, NEO4J_AUTH)
    try:
        ingest_to_neo4j(ENTITY_RELATION_DIR, driver, batch_size=BATCH_SIZE)
    finally:
        driver.close()


def run_qdrant():
    client = connect_qdrant(QDRANT_URL)

    create_collection(client, QDRANT_COLLECTION_PARENT, VECTOR_DIMENSION)
    create_collection(client, QDRANT_COLLECTION_CHILDREN, VECTOR_DIMENSION)

    print("--- Ingesting parents ---")
    ingest_parents(client, QDRANT_COLLECTION_PARENT, METADATA_DIR)

    print("--- Ingesting children ---")
    ingest_children(client, QDRANT_COLLECTION_CHILDREN, ENTITY_RELATION_DIR, batch_size=BATCH_SIZE)


def main():
    parser = argparse.ArgumentParser(description="Knowledge graph construction pipeline")
    parser.add_argument("--neo4j", action="store_true", help="Ingest entities and relations into Neo4j")
    parser.add_argument("--qdrant", action="store_true", help="Ingest embeddings into Qdrant")
    args = parser.parse_args()

    if not args.neo4j and not args.qdrant:
        parser.print_help()
        return

    if args.neo4j:
        run_neo4j()

    if args.qdrant:
        run_qdrant()


if __name__ == "__main__":
    main()
