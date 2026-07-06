# deval - RAG Evaluation

DeepEval-based evaluation of the Hashimoto's thyroiditis Graph RAG system. It scores
answer quality and retrieval quality across the competing RAG configurations, using an
LLM-as-judge tuned for biomedical literature.

Built on [DeepEval](https://deepeval.com/) - an open-source LLM evaluation framework.

## What gets compared

Each system answers the same fixed question set (20 questions). We evaluate three main
systems plus a set of ablations.

**Main systems**

| System    | Retrieval                                | Purpose                          |
| --------- | ---------------------------------------- | -------------------------------- |
| `hybrid`  | Vector chunks **+** knowledge-graph paths | The full Graph RAG system        |
| `vector`  | Vector chunks only                        | Vector-RAG baseline              |
| `vanilla` | None (LLM answers from parametric memory) | No-retrieval floor               |

**Ablations** (isolate one component of `hybrid`)

| System      | Change vs. hybrid                    |
| ----------- | ------------------------------------ |
| `no_graph`  | Graph paths removed (vector only)    |
| `no_vector` | Vector chunks removed (graph only)   |
| `no_cot`    | Chain-of-thought reasoning removed   |

Which systems run in a given pass is controlled by the `SYSTEMS` dict in
[eval.py](eval.py). Ablation results are written to `ab_results/`, main results to
`eval_results/`.

## Metrics

All metrics come from DeepEval. See the
[metrics reference](https://deepeval.com/docs/metrics-introduction) for full definitions.
Each is scored 0.0–1.0 by the judge; the pass `THRESHOLD` is `0.7`.

| Metric                  | Measures                                                                 | Applies to        |
| ----------------------- | ------------------------------------------------------------------------ | ----------------- |
| **Answer Relevancy**    | Does the answer actually address the question?                           | all systems       |
| **Faithfulness**        | Are the answer's claims grounded in the retrieved context (no hallucination)? | hybrid, vector |
| **Contextual Precision**| Is relevant retrieved context ranked above irrelevant context?           | hybrid, vector    |
| **Contextual Recall**   | Does the retrieved context cover what the expected answer needs?         | hybrid, vector    |
| **Contextual Relevancy**| What fraction of the retrieved context is actually relevant?             | hybrid, vector    |
| **RAGAS**               | Composite score: mean of the 4 metrics above (Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall) | hybrid, vector |

`vanilla` has no retrieval context, so it only gets **Answer Relevancy** - the
context-based metrics and RAGAS are `NA` for it.

Docs for the individual metrics:
[Answer Relevancy](https://deepeval.com/docs/metrics-answer-relevancy) ·
[Faithfulness](https://deepeval.com/docs/metrics-faithfulness) ·
[Contextual Precision](https://deepeval.com/docs/metrics-contextual-precision) ·
[Contextual Recall](https://deepeval.com/docs/metrics-contextual-recall) ·
[Contextual Relevancy](https://deepeval.com/docs/metrics-contextual-relevancy)

## The judge

DeepEval metrics are computed by an LLM judge. Instead of the default, we use a custom
judge in [judge.py](judge.py):

- **`BiomedicalOpenAIJudge`** - subclasses `DeepEvalBaseLLM`, backed by OpenAI
  `gpt-4o-mini` at `temperature=0` for reproducibility.
- Carries a **biomedical system prompt** so the judge scores like a domain expert:
  treats clinical synonyms as equal (`HT` = Hashimoto's thyroiditis, `TPOAb` =
  anti-TPO antibodies, ...), does not penalize appropriate hedging, understands
  `[Paper N]` / `[G#]` / `[V#]` citation anchors, and rewards mechanism-level detail.
- Tracks token usage per call so each run reports judge cost per system.

## Data

Inputs are gitignored (see repo `.gitignore`) - only the code lives in the repo.

**Ground truth** - `data/answers.json`

```json
{ "answers": [ { "id", "category", "question", "sources", "answer" } ] }
```

**Candidate outputs** - `data/{hybrid,vector,vanilla}_results.json` (list of items)

```json
{
  "id", "category", "question", "reasoning", "answer", "latency_seconds",
  "context": {
    "vector_chunks": [ { "paper_id", "section_name", "text" } ],
    "graph_paths":   [ { "claims": [ { "source_name", "relation_type",
                                       "target_name", "evidence_list": [...] } ] } ]
  }
}
```

`build_retrieval_context()` in [eval.py](eval.py) flattens `vector_chunks` and
`graph_paths` into the flat `retrieval_context` list DeepEval expects. `vanilla` items
have no `context` and produce an empty retrieval context.

## Outputs

Per run, one JSON per system is written to the output dir (`eval_results/` or
`ab_results/`):

```
{system}_deepeval.json
  ├─ summary        # mean of each metric across all questions
  ├─ per_question   # id, category, scores, judge reason per metric
  └─ judge_usage    # calls + prompt/completion tokens for this system
```

The console also prints a live per-question table, a mean-score summary table across
systems, and a judge token-usage table.

## Running

```bash
# from repo root
export OPENAI_API_KEY=...       # required by the judge
python -m deval.eval
```

To choose what runs, edit the `SYSTEMS` dict and `OUTPUT_DIR` at the top of
[eval.py](eval.py) (main systems vs. ablation set).

## Layout

```
deval/
├─ eval.py      # metric runner: loads data, builds test cases, scores, reports
├─ judge.py     # BiomedicalOpenAIJudge (custom DeepEval LLM)
├─ data/        # main-system inputs        (gitignored)
├─ ablation/    # ablation inputs           (gitignored)
├─ eval_results/, eval_results2/  # main results   (gitignored)
└─ ab_results/  # ablation results          (gitignored)
```