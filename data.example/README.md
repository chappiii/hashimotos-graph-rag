# data.example/

A template showing the layout the pipeline expects under `data/`. Copy this to
`data/` and populate it with the real corpus before running the build.

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

A single object `{ "papers": [ ... ] }`. Each entry carries the fields the graph
build consumes (title, authors, year, `study_design`, keywords). See
[`metadata.json`](metadata.json) for one trimmed sample paper.

## pdf-info.json

Same shape as `metadata.json` plus structural fields: `pages`, `keyword-page`,
`sections`, `tables`, `figures`. Used mainly by the EDA notebooks. See
[`pdf-info.json`](pdf-info.json).

## paper-sections/

Human-written section maps, one `{paper_id}.md` per paper, used as ground truth
when validating the automatic section chunker. Top-level `*` entries are
sections; indented `*` entries are subsections. See [`paper-sections/1.md`](paper-sections/1.md).
