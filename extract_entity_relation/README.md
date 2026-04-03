# Entity & Relation Extraction

Extracts biomedical entities and their relationships from per-section paper chunks using Gemini. Each section is processed independently so that outputs are traceable back to their source.

---

## Pipeline

```
chunks/{paper_id}/{section}.md
 |
 v
Entity Extraction Prompt       # Schema-guided: diseases, biomarkers, treatments, etc.
 |
 v
Gemini API
 |
 v
{section}_entities.json
 |
 v
Relation Extraction Prompt     # Uses extracted entities + section text
 |
 v
Gemini API
 |
 v
{section}_relations.json
```

---

## Output

Results are organized per paper, per section:

```
extracted_entity_relations/
└── 1/
    ├── 1-abstract_entities.json
    ├── 1-abstract_relations.json
    ├── 2-1._introduction_entities.json
    ├── 2-1._introduction_relations.json
    ├── 3-2._materials_and_methods_entities.json
    ├── 3-2._materials_and_methods_relations.json
    └── ...
```

---

## Running

```bash
uv run python -m extract_entity_relation.main
```

Requires `GEMINI_API_KEY` in `.env`.
