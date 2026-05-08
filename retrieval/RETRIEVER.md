# Qdrant + Neo4j Retriever — How It Works

This document explains [qdrant_neo4j_retriever.py](qdrant_neo4j_retriever.py): how it pulls the right rows out of **Qdrant** (vector DB) and **Neo4j** (graph DB), and the order of operations end-to-end.

---

## 1. The two databases and what each one stores

| DB | Collection / Label | What's inside | Why it exists |
|----|---|---|---|
| Qdrant | `parent_collection` (e.g. `Parents`) | One point per *paper / section*. Payload has `parent_id`, `title`, `doi`, `keywords`, plus a 4096-d embedding | Coarse filter — "which papers are even worth looking at?" |
| Qdrant | `children_collection` | One point per *evidence sentence*. Payload has `parent_id`, `evidence` (raw text), embedding | Fine-grained semantic search inside the chosen papers |
| Neo4j | Entity nodes | Each node has `name` and `evidence_sentences` (a list of UUIDs pointing back to evidence text) | Graph context — entities/relations extracted from those same sentences |
| JSON file | `evidence_mapping.json` | `{ uuid: evidence_text }` | The **bridge** between Qdrant payloads (which store the *text*) and Neo4j nodes (which store *UUIDs*) |

Key insight: Qdrant and Neo4j don't share IDs directly. They are linked through the `evidence_mapping` JSON — same sentence, two representations.

---

## 2. The two functions

### `get_user_keywords(...)` — Parent-level keyword filter
1. **Format the prompt** with the user query.
2. **Call the LLM** ([test_ollama_llama](../extract_entity_relation/models/llama.py)) → returns a JSON list of keywords.
3. **Parse + lowercase** the keywords. If parsing fails, return `None`.
4. **Scroll** the parent collection (up to 100 points) and for each point:
   - Compare every extracted keyword against every payload keyword with `rapidfuzz.fuzz.ratio`.
   - Keep the **best score** across all pairs.
   - If `best_score >= threshold` (default 50), keep the point.
5. Return the list of matched parent points.

> **Why fuzzy and not vector here?** Parents are filtered cheaply by keyword overlap. The expensive vector search happens later, only on the children of these surviving parents.

### `retriever_search_with_parent(...)` — The orchestrator
This is the function actually called by the pipeline. It chains parent filtering → child vector search → Neo4j join.

---

## 3. End-to-end flow

```
user_query
    │
    ▼
[Step 1] llama_embeddings(query) ───────────► query_embedding (4096-d)
    │
    ▼
[Step 2] get_user_keywords()
    │   - LLM extracts keywords
    │   - Fuzzy match vs Qdrant parent payloads
    │   ▼
    │   parent_results
    │       └─ if empty → FALLBACK: pure vector search on parents
    │                     (score_threshold = 0.4, top 5)
    ▼
[Step 3] For each unique parent_id:
    │   qdrant.search(
    │       collection = children,
    │       query_vector = query_embedding,
    │       filter = {parent_id == pid},
    │       limit = 20,
    │       score_threshold = 0.3
    │   )
    │   ▼
    │   children_results  (list of evidence sentence points)
    ▼
[Step 4] Build neo4j_search_uuids:
    │   for each child:
    │       evidence_text = child.payload["evidence"]
    │   look up that text in evidence_mapping (text == text)
    │   collect the matching UUID
    ▼
[Step 5] Neo4j query:
    │   MATCH (n)
    │   WHERE n.evidence_sentences IS NOT NULL
    │     AND ANY(eid IN $point_ids WHERE eid IN n.evidence_sentences)
    │   RETURN n, n.name, n.evidence_sentences
    │   ▼
    │   graph_nodes
    ▼
[Step 6] Combine:
    │   - Build {evidence_text → [graph_nodes]} via evidence_mapping
    │   - Fetch parent payload for each parent_id (dummy-vector search + filter)
    │   - For each child:
    │       attach parent_payload + matching graph nodes (with their evidence sentences resolved back to text)
    ▼
combined_results  ── list of dicts ──►
{
  qdrant_score, qdrant_payload, qdrant_id,
  parent_payload,
  child_evidence_sentence,
  related_graph_nodes: [
      { name, node, evidence_sentences (UUIDs), evidence_sentences_text (resolved) }
  ]
}
```

---

## 4. Where each value comes from

| Field in final result | Source | How it's extracted |
|---|---|---|
| `qdrant_score` | Qdrant child search | Cosine similarity returned by `qdrant_client.search` |
| `qdrant_payload` | Qdrant `children_collection` | `child.payload` (whole payload dict) |
| `qdrant_id` | Qdrant child point | `str(child.id)` |
| `parent_payload` | Qdrant `parent_collection` | Looked up by `parent_id` filter (uses a dummy zero-vector since vector ranking is irrelevant when filtering exactly) |
| `child_evidence_sentence` | Qdrant child payload | `child.payload["evidence"]` |
| `related_graph_nodes[*].node` | Neo4j | `dict(record["n"])` |
| `related_graph_nodes[*].name` | Neo4j | `record["name"]` |
| `related_graph_nodes[*].evidence_sentences` | Neo4j | `record["evidence_sentences"]` (list of UUIDs) |
| `related_graph_nodes[*].evidence_sentences_text` | `evidence_mapping.json` | UUID → text resolved in [Step 6] |

---

## 5. The bridge logic (the trickiest part)

Qdrant stores *evidence text*. Neo4j stores *evidence UUIDs*. They are joined like this:

```
Qdrant child payload.evidence  ── (text == text) ──►  evidence_mapping  ── (UUID) ──►  Neo4j node.evidence_sentences
```

- **Qdrant → UUID** ([Step 4](qdrant_neo4j_retriever.py#L213-L223)): walk `evidence_mapping` and find the UUID whose mapped text equals the child's evidence text.
- **Neo4j → text** ([Step 6](qdrant_neo4j_retriever.py#L255-L264)): for each UUID in the node's `evidence_sentences`, look up the text in `evidence_mapping`.

Both directions go through `evidence_mapping.json`, which is why it's a required input.

---

## 6. Tunables (defaults)

| Parameter | Default | Where | Effect |
|---|---|---|---|
| Parent fuzzy threshold | 50 | [get_user_keywords](qdrant_neo4j_retriever.py#L15) | Higher = stricter parent match |
| Parent scroll limit | 100 | [qdrant_neo4j_retriever.py:60](qdrant_neo4j_retriever.py#L60) | How many parents are scanned for fuzzy match |
| Parent fallback `score_threshold` | 0.4 | [qdrant_neo4j_retriever.py:158](qdrant_neo4j_retriever.py#L158) | Used only if fuzzy match returned nothing |
| Children per parent | 20 | [qdrant_neo4j_retriever.py:183](qdrant_neo4j_retriever.py#L183) | Cap on evidence sentences per paper |
| Children `score_threshold` | 0.3 | [qdrant_neo4j_retriever.py:185](qdrant_neo4j_retriever.py#L185) | Minimum vector similarity for evidence |
| Embedding dim | 4096 | [qdrant_neo4j_retriever.py:272](qdrant_neo4j_retriever.py#L272) | Hardcoded in dummy vector for parent payload fetch |

---

## 7. Quick mental model

> *Filter papers by keyword → semantically search inside those papers for sentences → use those sentences to pull related entities from the graph → return everything stitched together.*

That's the whole retriever in one line.
