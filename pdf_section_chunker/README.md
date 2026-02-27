# PDF Section Chunker

Extracts structured content from research PDFs using the Gemini 2.5-flash API. Each paper is split into per-section markdown files, ready for downstream use in a knowledge graph Construction.

---

## How it works

### 1. Human-defined section maps (`/paper-sections`)

For each paper, a human has pre-read the PDF and written a lightweight outline in `/paper-sections/<id>.md`. This file lists the top-level sections and their subheadings using a simple `*` indented format:

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

### 2. Model-powered extraction

The pipeline uploads the full PDF to the Gemini File API, then iterates over each section defined in the map. For each section it sends a prompt asking the model to extract everything between that section header and the next one — preserving prose, subheadings, and formatting all tables as Markdown.

This means the human only has to identify *what* the sections are. The model handles pulling out the full content, including nested subheadings, tables, figures references, and statistical results.

### 3. Chunked output (`chunks/<id>/`)

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
    ├── 7-references.md
    └── 8-appendix_a.md
```

---

## Why this approach

We experimented with GROBID, docling, and local Ollama models before settling on this method. The research papers in this corpus vary significantly in formatting and structure, and all of the automated tools failed to reliably capture the section boundaries — even when section headers were provided manually. Local Ollama models also lack native PDF reading capability, requiring a conversion step that introduced further quality loss. Gemini's ability to read the raw PDF directly and reason about its layout produced consistently better results across all paper formats.

---

## Usage

```bash
cd pdf_section_chunker
python main.py
```

The script processes all PDFs in `/pdfs` in numerical order.

### Requirements

Add your Gemini API key to `.env`:

```
GEMINI_API_KEY=your_key_here
```


