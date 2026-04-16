# Entity & Relation Extraction

Extracts biomedical entities and their relationships from per-section paper chunks. Each section is processed independently so outputs are traceable back to their source.

Supports three models: **Gemini**, **Claude**, and **GPT**.

---

## Pipeline

```
chunks/{paper_id}/{section}.md
 │
 ▼
Entity Extraction Prompt         # Schema-guided: 22 entity types (diseases, biomarkers, genes, etc.)
 │
 ▼
LLM API (gemini / claude / gpt)
 │
 ▼
{section}_entities.json
 │
 ▼
Relation Extraction Prompt       # Uses extracted entities + section text, 40+ relation types
 │
 ▼
LLM API (same model)
 │
 ▼
{section}_relations.json
```

Each model runs its own isolated pipeline, entity extraction feeds into relation extraction using the **same model's entities** (no cross-model dependency).

---

## Models

| Flag       | Model                    | Provider   |
| ---------- | ------------------------ | ---------- |
| `gemini`   | `gemini-3-flash-preview` | Google     |
| `claude`   | `claude-sonnet-4-6`      | Anthropic  |
| `gpt`      | `gpt-5.4-mini`           | OpenAI     |

---

## Output

Results are organized per model, per paper, per section:

```
extracted_entity_relations/
├── gemini/
│   └── 1/
│       ├── 1-abstract_entities.json
│       ├── 1-abstract_relations.json
│       ├── 2-1._introduction_entities.json
│       ├── 2-1._introduction_relations.json
│       └── ...
├── claude/
│   └── 1/
│       └── ...
└── gpt/
    └── 1/
        └── ...
```

---

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys in `.env`:

```env
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...    
OPENAI_API_KEY=...       
```

### Dependencies

- `google-genai` — Gemini API client
- `anthropic` — Claude API client
- `openai` — OpenAI API client
- `python-dotenv` — loads `.env` file

---

## Usage

```bash
# Run with a specific model (required)
uv run -m extract_entity_relation.main --model gemini
uv run -m extract_entity_relation.main --model claude
uv run -m extract_entity_relation.main --model gpt
```
