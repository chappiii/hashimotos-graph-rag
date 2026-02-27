# Graph RAG — Hashimoto's Thyroiditis

Building a Graph RAG system from scientific literature on Hashimoto's thyroiditis.
This repo documents experiments and progress as the pipeline is built incrementally.

---

## Pipeline Progress

**1. Metadata Extraction** — `done`
Extract title, authors, abstract, and other fields from PDFs using a local LLM.
→ [`extract_metadata/`](extract_metadata/README.md)

**2. PDF Section Chunking** — `experimenting`
Split PDFs into logical sections (intro, methods, results, etc.) before graph construction.
→ [`pdf_section_chunker/`](pdf_section_chunker/README.md)

**3. Graph Construction** — `pending`

**4. RAG Integration** — `pending`

---
