# data.example/

The layout the pipeline expects under `data/`. Copy this to `data/`, then add
the PDFs before running the build.

`metadata.json` and `pdf-info.json` here are the real manifests for the full
115-paper corpus, not samples. The PDFs themselves are under publisher
copyright and are not redistributed, so `pdfs/` ships empty and
`paper-sections/` ships a single example. See the note on the corpus in the
[root README](../README.md#3-provide-the-corpus).

```
data/
  pdfs/                 the input corpus, one PDF per paper
    1.pdf
    2.pdf
    ...
    115.pdf
  metadata.json         per-paper metadata used across stages
  pdf-info.json         richer per-paper info (sections, tables, figures)
  paper-sections/       ground-truth section maps, one per paper
    1.md
    ...
```

## pdfs/

The raw corpus. Each file is named `{paper_id}.pdf`, numbered `1.pdf` through
`115.pdf`. The `paper_id` is the join key used by every downstream stage, so the
numbering must stay consistent across `pdfs/`, `metadata.json`, and
`pdf-info.json`. Only `.gitkeep` ships here; drop your own PDFs in.

## metadata.json

A single object `{ "papers": [ ... ] }`, one entry per paper for all 115.
Each entry carries the fields the graph build consumes (`paper_id`, `doi`,
title, authors, year, `study_design`, keywords).

This doubles as the corpus manifest: 109 of the 115 entries carry a DOI, and
the 6 that do not are identified by title. Use it to source the PDFs and name
them by `paper_id`.

## pdf-info.json

Same shape as `metadata.json` plus structural fields: `pages`, `keyword-page`,
`sections`, `tables`, `figures`. Used mainly by the EDA notebooks. Also covers
all 115 papers. See [`pdf-info.json`](pdf-info.json).

## paper-sections/

Human-written section maps, one `{paper_id}.md` per paper, used as ground truth
when validating the automatic section chunker. Top-level `*` entries are
sections; indented `*` entries are subsections. See [`paper-sections/1.md`](paper-sections/1.md).
