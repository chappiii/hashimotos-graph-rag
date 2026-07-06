# Knowledge Graph Ingestion

Builds the knowledge graph from extracted entities, relations, and metadata. Writes structured knowledge into **Neo4j** (graph) and text vectors into **Qdrant** (semantic search).

The stage runs in three steps: **pre-ingestion** cleans and aggregates the raw extractions into two registries, then **Neo4j** and **Qdrant** ingestion load those registries (plus chunks and metadata) into the two databases.

---

## Pipeline

```
extract_entity_relation/.../gemini/   data/metadata.json   pdf_section_chunker/chunks/
 │                                     │                    │
 ▼                                     │                    │
Pre-ingestion                          │                    │
 ├─ entity_registry.py   # dedup entity names -> canonical  │
 └─ claim_registry.py    # dedup relations -> claims + evidence
 │                                     │                    │
 ▼                                     ▼                    ▼
Neo4j ingestion                        Qdrant ingestion
 ├─ Paper  nodes                        ├─ papers    (purpose_of_work embedded)
 ├─ Entity nodes                        ├─ chunks    (sliding-window section text)
 ├─ Claim  nodes + polarity edges       └─ evidence  (evidence sentences)
 └─ Evidence nodes + edges
 │                                     │
 ▼                                     ▼
Neo4j Graph DB                         Qdrant Vector DB
```

See [`pre_ingestion/README.md`](pre_ingestion/README.md) for the full detail on how deduplication and the registries work.

---

## Pre-ingestion (build the registries)

Extraction ran paper by paper, chunk by chunk, so the same concept shows up under slightly different names across the 115 papers (`Hashimoto's Thyroiditis` / `Hashimoto Thyroiditis` / `HT`). Loaded as-is, those become separate nodes and nothing aggregates. Pre-ingestion does one full pass over all extractions first and produces two clean registries the ingestion reads from.

- **`entity_registry.py`** — reads every `*_entities.json`, collapses near-duplicate entity names into one canonical entry (exact match first, then fuzzy match at threshold 88; clinically opposite terms like `Hypothyroidism`/`Hyperthyroidism` are protected from fuzzy merging). Emits `entity_registry.json` with a `forms_index` mapping every alias/surface form back to its canonical name.
- **`claim_registry.py`** — reads every `*_relations.json`, normalizes source/target names through the `forms_index`, then groups relations into unique claims keyed by a `claim_signature` (`source|relation_type|target`). Each claim aggregates polarity counts, max certainty, paper IDs, and every evidence sentence with provenance. Emits `claim_registry.json`. Requires the entity registry to exist first.

---

## Graph schema (Neo4j)

**Nodes:** `Paper`, `Entity`, `Claim`, `Evidence`

**Edges:**

| Edge | From → To | Meaning |
|---|---|---|
| `POSITIVE_CLAIM` / `NEGATIVE_CLAIM` / `HYPOTHETICAL_CLAIM` | Entity → Claim | source entity asserts the claim, typed by polarity |
| `CLAIM_TARGET` | Claim → Entity | claim's target entity |
| `HAS_EVIDENCE` | Claim → Evidence | supporting sentence for the claim |
| `FROM_PAPER` | Evidence → Paper | which paper the evidence came from |

`uncertain` / `mixed` polarities are kept in `polarity_counts` on the Claim node but get no typed edge.

All writes use `MERGE` on a unique key, so every script is idempotent and safe to re-run.

---

## Vector collections (Qdrant)

| Collection | Point | Embedded text |
|---|---|---|
| `papers` | one per paper | `purpose_of_work` (falls back to title) |
| `chunks` | 400-word window (80-word overlap) | section body text |
| `evidence` | one per evidence sentence | `evidence_text` |

Vectors are `gemini-embedding-2`, dim **3072**, cosine distance. Qdrant scripts skip point IDs that already exist, so re-runs only embed new data.

---

## Usage

Run in order. Each script is a standalone module and safe to re-run.

```bash
# 1. Pre-ingestion - build the registries (entity first, claim depends on it)
uv run -m kg_ingestion.pre_ingestion.entity_registry
uv run -m kg_ingestion.pre_ingestion.claim_registry

# 2. Neo4j - verify connection, create schema, then load nodes/edges
uv run -m kg_ingestion.neo4j.check_connection
uv run -m kg_ingestion.neo4j.schema
uv run -m kg_ingestion.neo4j.ingest_papers
uv run -m kg_ingestion.neo4j.ingest_entities
uv run -m kg_ingestion.neo4j.ingest_claims
uv run -m kg_ingestion.neo4j.ingest_evidence

# 3. Qdrant - create collections, then embed and load
uv run -m kg_ingestion.qdrant.collections

```

Neo4j `ingest_claims` and `ingest_evidence` require the registries (step 1) and `ingest_papers`/`ingest_entities` to have run first, since they `MATCH` existing nodes when creating edges.
