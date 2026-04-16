# Knowledge Graph Construction

Ingests extracted entities, relations, and metadata into **Neo4j** (graph database) and **Qdrant** (vector database).

---

## Pipeline

```
extract_entity_relation/extracted_entity_relations/gemini/
 │
 ▼
Neo4j Ingestion          # Entities → nodes (labeled by type), relations → edges
 │                       # Fuzzy dedup on entity names, merged edges per (source, target, type)
 ▼
Neo4j Graph DB

extract_metadata
 │
 ▼
Qdrant Ingestion         # Parents: paper metadata embedded via purpose_of_work
 │                       # Children: evidence sentences embedded via nomic-embed-text
 ▼
Qdrant Vector DB
```

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Neo4j instance 
- Qdrant instance
- Ollama with `nomic-embed-text` model

See [`.env.example`](../.env.example) for required environment variables.

### Dependencies

- `neo4j` — Neo4j Python driver
- `qdrant-client` — Qdrant Python client
- `rapidfuzz` — fuzzy string matching for entity dedup
- `requests` — Ollama embedding API calls
- `python-dotenv` — loads `.env` file

---

## Usage

```bash
# Neo4j only
uv run -m kg_construction.main --neo4j

# Qdrant only
uv run -m kg_construction.main --qdrant

# Both
uv run -m kg_construction.main --neo4j --qdrant
```
