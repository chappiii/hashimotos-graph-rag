# EDA

Exploratory analysis of the Extracted metadata and extracted chunks.

## Notebooks

### `metadat_eda.ipynb` - paper metadata
Reads [`data/pdf-info.json`](../data/pdf-info.json) (115 papers) into a
DataFrame and profiles the corpus:

- Overview: columns, dtypes, missing values
- Publication year distribution (2013-2025; ~57% from 2024-2025)
- Page count, table count, figure count distributions
- Keyword page location, DOI coverage (94.8% have a DOI)
- Country distribution and multi-country collaboration (39 countries; China-led)

### `chunks_eda.ipynb` - extracted chunks
Walks [`pdf_section_chunker/chunks/`](../pdf_section_chunker/chunks/), parses
each `{index}-{section}.md` file into a DataFrame (857 chunks, 115 papers), and
analyzes:

- Overall word/char totals and summary stats
- Sections per paper (mean ~7.5)
- Chunk word-count distribution and percentiles
- Long-section thresholds (informs a future sub-chunking cutoff)
- Section-name frequency (References, Abstract, Introduction most common)
- Outliers: stub chunks (<30 words) and very long chunks (>2500 words)
