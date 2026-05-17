# Hybrid - Retriever

Hybrid retrieval system combining structured graph search and semantic vector search.
## How it works

```
Query
 ├── graph_ret ──► 7 structured claims with evidence
 └── vector_ret ──► 7 raw text chunks
```

### graph_ret

Structured retrieval through a Neo4j knowledge graph.

1. **Decompose**: extract n-gram candidate spans from the query
2. **Match**: fuzzy-match spans against the entity registry to find canonical entity names
3. **Retrieve**: Cypher query fetches all Claims where matched entities appear as source or target
4. **Pre-filter**: score claims by `certainty × log(papers) × study_weight × section_weight`, keep top 40
5. **Re-rank**: embed query and each claim, rank by cosine similarity, return top 7

Each result is a structured triple: `source → relation_type → target`, with supporting evidence sentences, paper count, certainty level, and study design metadata.

### vector_ret

Semantic retrieval over Qdrant.

- Embeds the query with Gemini `text-embedding-004` (`RETRIEVAL_QUERY` task type)
- Searches the `chunks` collection (400-word sliding windows over all paper sections)
- Returns top 7 chunks by cosine similarity with `paper_id`, `section_type`, and raw text

## Run

```bash
uv run -m retriever.main
```