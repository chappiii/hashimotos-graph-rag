# Metadata Extraction

Extracts structured bibliographic metadata from academic PDFs using a two-stage local LLM pipeline — an extraction phase followed by a correction phase. Both stages run locally via [Ollama](https://ollama.com/).

---

## What It Extracts

For each PDF the pipeline produces a JSON record with the following fields:

| Field | Description |
|---|---|
| `paper_id` | PDF filename (Each paper was assigned a numeric ID and renamed for management and processing.)|
| `doi` | Digital Object Identifier |
| `title` | Paper title |
| `published_year` | Year of publication |
| `author_list` | Full author names as they appear |
| `countries` | Countries derived from author affiliations |
| `purpose_of_work` | 20–40 word summary of the research objective |
| `keywords` | Keywords section or key terms extracted from abstract |

---

## Pipeline

```
PDF
 |
 v
First Page Extraction        # PyPDF2 — only the first page is used
 |
 v
Extraction Prompt            # Instructs LLM to return structured JSON
 |
 v
Extraction LLM               # Configured model via Ollama API
 |
 v
Correction Prompt            # Instructs LLM to fix spelling/formatting only
 |
 v
Correction LLM               # Same or different model via Ollama API
 |
 v
JSON Output                  # Appended to extracted_metadata/{model}/run_{timestamp}/
```

**Why first page only?** Bibliographic metadata (title, authors, DOI, affiliations, keywords) almost always appears on the first page of a scientific paper. Limiting extraction to the first page keeps the context window small and reduces LLM latency.

**Two-stage design:** The extraction stage focuses on pulling out the fields. The correction stage is a lightweight pass that fixes spelling errors, removes duplicate entries, and cleans up formatting — without being allowed to change any values.

---

## Output

Results are written incrementally to avoid data loss on failure. Each batch of 10 PDFs produces a part file:

```
extracted_metadata/
  {extraction_model}_{correction_model}/
    run_{timestamp}/
      {model}_extracted_part_1.json
      {model}_extracted_part_2.json
      ...
      metrics.json     # per-document timing and memory usage
      outputs.json     # raw LLM responses
```

Each part file holds a `{"papers": [...]}` array where every entry follows the metadata template.

---

## Configuration

All settings live in `config/metadata_config.py` — models, batch size, paths, and API timeout.

---

## Model Experiments

The pipeline was evaluated across 9 model combinations (extraction → correction) on 115 papers:
LLaMA→LLaMA, Qwen(small)→LLaMA, Mistral→LLaMA, DeepSeek-R1→LLaMA, Qwen→Qwen, Mistral→Mistral, DeepSeek-R1→DeepSeek-R1, Gemma→LLaMA, Gemma→Gemma.

**Best accuracy:** Qwen → Qwen — 93.23% F1 in 1:51 over 115 papers

**Best efficiency:** Mistral → Mistral — 89.26% F1 in 0:30 over 115 papers

---

## Running

```bash
cd extract_metadata
python main.py
```

Requires Ollama running locally with the target models pulled.
