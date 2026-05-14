# Pre-Ingestion

These two scripts run **before** anything is written to Neo4j or Qdrant.
They read all the extracted data, clean it up, and produce two structured
files that the ingestion pipeline will build the graph from.

---

## Why pre-ingestion exists

The LLM extracted entities and relations paper by paper, chunk by chunk.
Each paper was processed in isolation, so the same concept can appear
with slightly different names across 115 papers:

```
Paper 4  -> "Hashimoto's Thyroiditis"
Paper 37 -> "Hashimoto Thyroiditis"
Paper 91 -> "Hashimoto thyroiditis"
```

If we push these straight into the graph, they become **three separate nodes**
that are never connected. Claims don't aggregate. Retrieval misses links.

Pre-ingestion solves this by doing one full pass over all the data first,
building a clean registry before a single node is written.

---

## Script 1 - `entity_registry.py`

**What it does:** Reads every `*_entities.json` file across all 115 papers
and collapses near-duplicate entity names into one canonical entry.

**How deduplication works:**
1. Exact match first (case-insensitive). "Hashimoto's Thyroiditis" and
   "hashimoto's thyroiditis" are the same thing, handled instantly.
2. Fuzzy match second. Catches variants like "Hashimoto Thyroiditis"
   (no apostrophe-s) that are clearly the same but not identical strings.
   Uses a similarity threshold of 88/100, high enough to avoid collapsing
   genuinely different entities.

**Protected terms:** Some entities look similar but are clinically opposite
or completely different things. These are never fuzzy-merged, no matter
how similar the names look:

- `Hypothyroidism` vs `Hyperthyroidism` - underactive vs overactive thyroid
- `Subclinical Hypothyroidism` - a distinct severity level, not the same as Hypothyroidism
- `Thyroglobulin Antibody` vs `Triglycerides` - both abbreviated "TG"
- Immune cell types: `T Cells`, `B Cells`, `Regulatory T Cells`
- Cancer subtypes: `MALT Lymphoma`, `Diffuse Large B-Cell Lymphoma`, etc.

**Alias collision resolution:** Some abbreviations (like "TG") point to
two different entities. After the full registry is built, collisions are
resolved by keeping the mapping to whichever entity appears in more papers.

**Output:** `output/entity_registry.json`
```
{
  "entities": {
    "Hashimoto's Thyroiditis": {
      "entity_type": "Diseases & Conditions",
      "aliases": ["HT", "chronic lymphocytic thyroiditis", ...],
      "key_properties": { ... },
      "paper_ids": [1, 2, 3, ...],
      "section_types": ["ABSTRACT", "RESULTS", ...],
      "surface_forms": ["Hashimoto's thyroiditis", "HT", ...]
    },
    ...
  },
  "forms_index": {
    "ht": "Hashimoto's Thyroiditis",
    "hashimoto thyroiditis": "Hashimoto's Thyroiditis",
    ...
  }
}
```

The `forms_index` is a flat lookup table: any surface form or alias maps
back to the one canonical name used in the graph.

**Results (115 papers):**
- 8137 raw entity mentions -> 1705 unique entities after dedup
- 620 entities have aliases (dedup working correctly)
- Hypothyroidism and Hyperthyroidism correctly separated as distinct nodes

---

## Script 2 - `claim_registry.py`

**What it does:** Reads every `*_relations.json` file across all 115 papers
and collapses all mentions of the same scientific claim into one entry with
all its evidence attached.

**Depends on:** `entity_registry.json` must exist first (run script 1 first).

**How it works:**
1. For each relation, normalize the source and target entity names through
   the `forms_index`, so "HT" and "Hashimoto's Thyroiditis" both produce
   the same claim signature.
2. Generate a `claim_signature`, a unique key for the claim:
   ```
   "Vitamin D|associated_with|Hashimoto's Thyroiditis"
   ```
3. Group all relations with the same signature from all papers and sections.
4. Aggregate everything: polarity counts, max certainty, paper IDs, and
   every individual evidence sentence with full provenance.

**Output:** `output/claim_registry.json`
```
{
  "Vitamin D|associated_with|Hashimoto's Thyroiditis": {
    "relation_type": "associated_with",
    "source": { "canonical_name": "Vitamin D", "entity_type": "..." },
    "target": { "canonical_name": "Hashimoto's Thyroiditis", "entity_type": "..." },
    "polarity_counts": { "positive": 8, "negative": 1, "hypothetical": 2, ... },
    "certainty_max": "high",
    "paper_ids": [2, 7, 23, 44],
    "paper_count": 4,
    "evidence": [
      {
        "evidence_id": "uuid",
        "evidence_text": "Vitamin D deficiency was found in 78% of HT patients.",
        "paper_id": 2,
        "section_type": "RESULTS",
        "claim_polarity": "positive",
        "claim_certainty": "high"
      },
      ...
    ]
  }
}
```

**Results (115 papers):**
- 5275 raw relations -> 3252 unique claims after dedup
- 433 claims supported by more than 1 paper
- 37 claims with both positive and negative evidence (real scientific contradictions)
- Top claim: TPO antibody is a marker of HT, confirmed by 68 papers

---

## How to run

```bash
# Step 1 - build entity registry
uv run python -m kg_ingestion.pre_ingestion.entity_registry

# Step 2 - build claim registry (needs step 1 output)
uv run python -m kg_ingestion.pre_ingestion.claim_registry

# Optional - analyze the outputs
uv run python -m kg_ingestion.pre_ingestion.analyze_registries
```

---

## Output files

| File | Description |
|---|---|
| `output/entity_registry.json` | 1705 unique entities with aliases, paper IDs, key properties |
| `output/claim_registry.json` | 3252 unique claims with aggregated evidence and polarity counts |
