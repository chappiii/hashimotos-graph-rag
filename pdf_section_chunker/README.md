# PDF Section Chunker

Automatically extracts structured content from research PDFs using the Gemini. Each paper is split into per-section markdown files, ready for downstream use in knowledge graph construction.

---

## How it works

### 1. Structure extraction (Pass 1)

The pipeline uploads each PDF to the Gemini File API and prompts the model to extract the section hierarchy — title, headers, and subheaders. The paper title is injected from pre-extracted metadata (`extract_metadata`) so the model doesn't need to identify it — it uses the provided title as the first entry directly.

```
* Paper Title
* Abstract
* 1. Introduction
* 2. Materials and Methods
    * 2.1. Study Design and Studied Population
    * 2.2. Statistical Analysis
    * 2.3. Ethical Aspects
* 3. Results
* 4. Discussion
* 5. Conclusions
* References
```

### 2. Post-processing

Deterministic cleanup applied to the raw structure before verification:

- Normalizes numbering separators (e.g., `1 | Introduction` → `1. Introduction`) — some PDFs use pipes instead of dots in their section numbering.
- Strips appendix subtitles (e.g., `Appendix A Full Title` → `Appendix A`) — keeps naming consistent across papers.
- Removes excluded sections (supplementary/supporting information) — these are pointers to external files with no extractable content.

### 3. Structure correction (Pass 1.5)

The cleaned structure is sent back to the model alongside the PDF for comparison. The model checks for missing, extra, or misnamed sections and outputs a corrected structure. The injected title is anchored here as well — the correction prompt explicitly marks it as metadata-provided and not to be modified. Early testing showed that a single-shot extraction (Pass 1 alone) would miss sections or rename them — especially in longer, non-standard papers like clinical guidelines. A verification pass turns this into a self-correcting process.


### 4. Content extraction (Pass 2)

Using the verified structure, the pipeline iterates over each section and prompts the model to extract everything between that section header and the next one — preserving prose, subheadings, and formatting.

### 5. Chunked output (`chunks/<id>/`)

Each section is saved as a numbered markdown file under `chunks/<paper-id>/`:

```
chunks/
└── 1/
    ├── 1-abstract.md
    ├── 2-1._introduction.md
    ├── 3-2._materials_and_methods.md
    ├── 4-3._results.md
    ├── 5-4._discussion.md
    ├── 6-5._conclusions.md
    └── 7-references.md
```

### 4. Figure & Table extraction (`figs_tables/<id>/`)

Each paper's figures and tables are extracted into two JSON

```
figs_tables/
└── 1/
    ├── figures.json
    └── tables.json
```

**Table schema:**

```json
{
  "table_id": 1,
  "table_type": "baseline_characteristics",
  "section_label": "Results",
  "caption": "Demographic and clinical characteristics of study participants.",
  "population": "80 HT patients vs 60 healthy controls",
  "groups": ["HT group", "Control group"],
  "variables": ["Age", "BMI", "TSH", "TPO-Ab"],
  "key_findings": [
    "TSH higher in HT group: 5.8 vs 2.1 mIU/L, p<0.001"
  ],
  "model_adjustments": []
}
```

**Figure schema:**

```json
{
  "figure_id": 1,
  "figure_type": "bar",
  "section_label": "Results",
  "caption": "Serum TPO-Ab levels in HT patients and controls.",
  "population": "HT patients (n=80) vs controls (n=60)",
  "groups": ["HT group", "Control group"],
  "outcome": "TPO-Ab levels (IU/mL)",
  "key_findings": [
    "TPO-Ab significantly elevated in HT group vs controls, p<0.001"
  ]
}
```

---

## Why this approach

We experimented with GROBID, docling, and local Ollama models before settling on this method. The research papers in this corpus vary significantly in formatting and structure, and all of the automated tools failed to reliably capture the section boundaries — even when section headers were provided manually. Local Ollama models also lack native PDF reading capability, requiring a conversion step that introduced further quality loss. Gemini's File API allows uploading full PDFs and referencing them directly in prompts, avoiding lossy PDF-to-text conversion. Its ability to reason over the raw layout produced consistently better results across all paper formats.

---

## Usage

```bash
cd pdf_section_chunker

# Extract sections into chunks/
python main.py

# Extract figures and tables into figs_tables/
python extract_figs_tables.py
```

### Requirements

Add your Gemini API key to `.env`:

```
GEMINI_API_KEY=your_key_here
```


