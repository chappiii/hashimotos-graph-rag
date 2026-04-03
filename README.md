# Graph RAG - Hashimoto's Thyroiditis

Building a Graph RAG system from scientific literature on Hashimoto's thyroiditis.
This repo documents experiments and progress as the pipeline is built incrementally.

---

## Pipeline Progress

**1. Metadata Extraction** - `done`
Extract title, authors, abstract, and other fields from PDFs using a local LLM.
→ [`extract_metadata/`](extract_metadata/README.md)

**2. PDF Section Chunking** - `done`
Split PDFs into logical sections (intro, methods, results, etc.) before graph construction.
→ [`pdf_section_chunker/`](pdf_section_chunker/README.md)

**3. Entity & Relation Extraction** - `in progress`
Extract biomedical entities and relationships from each section using Gemini.
→ [`extract_entity_relation/`](extract_entity_relation/README.md)

**4. Graph Construction** - `pending`

**5. RAG Integration** - `pending`

---
