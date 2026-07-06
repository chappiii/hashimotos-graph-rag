# graph_ret

Structured retrieval through a Neo4j knowledge graph. Transforms a raw query into ranked
structured claims in five stages.

```
query
  └── decompose ──► spans
        └── match ──► canonical entity names
              └── retrieve ──► raw claims from Neo4j
                    └── rank_by_quality ──► top 40 pre-filtered claims
                          └── rank_by_hybrid ──► top 7 final claims with evidence
```

---

## Stage 1 - Decompose (`query_decomposer.py`)

The query is tokenized with regex (`[A-Za-z0-9']+`), then every window of 1 to 5 tokens
is generated (all n-grams). Filters applied:

- Unigrams: dropped if shorter than 3 chars or in the stopword list (`"what"`, `"the"`, `"is"`, ...)
- Multi-word spans: dropped only if **every** token is a stopword (`"in the"` drops, `"Vitamin D"` survives)

For the query `"What is the effect of Vitamin D supplementation on TSH levels..."` this produces
spans like `Vitamin`, `TSH`, `Vitamin D`, `TSH levels`, `Vitamin D supplementation`, etc.

---

## Stage 2 - Match (`entity_matcher.py`)

The entity registry (`entity_registry.json`) stores each canonical entity name plus its aliases.
Every generated span is scored against each canonical name and all its aliases using `rapidfuzz`:

- Short names (< 5 chars): `fuzz.ratio` - strict, prevents `"T3"` matching `"T4"`
- Longer names: `fuzz.WRatio` - handles partial matches and word reordering

A coverage guard prevents short spans from matching long entity names: a 1-token span cannot
match a 5-token entity because `ceil(5 × 0.8) = 4` tokens are required as a minimum.

Only entities scoring >= 80 survive, ranked by score, top 10 returned.

---

## Stage 3 - Retrieve (`graph_retriever.py`)

Two Cypher queries run against Neo4j:

**`_BY_SPECIFIC_CYPHER`** - fetches claims where a specific matched entity is the source or
target. Example: `Vitamin D → reduces → TSH`.

**`_BY_GENERIC_CYPHER`** - fetches claims connecting a generic entity (Hashimoto's Thyroiditis,
Thyroid Gland, Autoimmune Thyroid Disease) to a specific matched entity. Generic entities are
separated because querying "anything connected to Hashimoto's" would return thousands of
irrelevant claims - so generic entities only participate when paired with a specific one.

Each result includes: the triple (`source → relation → target`), certainty level, paper count,
study design weight, and evidence sentences with paper metadata (year, study design, section).

---

## Stage 4 - Pre-filter by quality (`ranker.rank_by_quality`)

Every claim gets a `_quality_score`:

```
certainty_weight  (high=1.0, moderate=0.6, low=0.2)
× log(1 + paper_count)       -- log dampens outliers (50 papers is not 50x better than 1)
× study_weight_max            -- RCT > cohort > case report
× section_weight              -- RESULTS=1.0, ABSTRACT=0.9, DISCUSSION=0.6, METHODS=0.3
```

A two-bucket strategy prevents popular entities from crowding out niche ones:

1. **Entity bucket**: top 5 claims per specific matched entity are guaranteed in, regardless of score
2. **Quality bucket**: remaining slots (up to 40 total) filled by global quality ranking

Why this matters: if `Vitamin D` has 30 claims and `selenium` has 2, without the entity bucket
`selenium` might get zero representation in the top 40.

---

## Stage 5 - Re-rank by semantic similarity (`ranker.rank_by_hybrid`)

The query is embedded with Gemini (`RETRIEVAL_QUERY` task type). Each of the 40 pre-filtered
claims is converted to a text string:

```
"Vitamin D supplementation reduces TSH levels in HT patients. <evidence 1>. <evidence 2>."
```

...and embedded with `RETRIEVAL_DOCUMENT` task type. Cosine similarity between query vector
and each claim vector gives the final `_hybrid_score`.

The name is hybrid because it was designed to later combine quality score + cosine similarity -
currently it is cosine-only.

The entity-bucket guarantee applies again at final selection: at least 1 claim per specific
matched entity is in the final top 7.

Each surviving claim's `evidence_list` is trimmed to the top 2 sentences by section weight
(Results > Abstract > Discussion ...), keeping the context lean for the generator.
