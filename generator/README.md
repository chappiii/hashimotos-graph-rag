# Answer Generation

The final stage of the Graph RAG pipeline. Takes a question plus the retrieved evidence (knowledge-graph paths + vector passages) and produces a grounded, citation-backed answer using **Chain-of-Thought (CoT) prompting** on Gemini.

Answers are grounded exclusively in the supplied evidence, no external knowledge is used.

---

## Main Idea: Chain-of-Thought Prompting

The generator forces the model to reason before it answers, in two explicit stages:

```
<reasoning>   # Stage 1 - work through the evidence
  - which graph claims / passages address the question
  - which sources corroborate, extend, or contradict each other
  - flag conflicts, note gaps, list excluded evidence
</reasoning>
<answer>      # Stage 2 - the final answer
  - single letter (A/B/C/D) for multiple-choice questions
  - 2-4 paragraphs of cited prose for open-ended questions
</answer>
```

Separating reasoning from the answer makes the model weigh conflicting evidence and cite deliberately, instead of pattern-matching a fluent-but-ungrounded response. The model's thinking budget (`thinking_budget=1024`) is enabled to support this.

---

## Pipeline

```
query + graph_results + vector_results
 |
 v
build_user_prompt()          # formats evidence into labelled, citable blocks
 |                           #   graph: [G1] direct (1-hop), [G4.1]/[G4.2] chain hops (2-hop)
 |                           #   vector: [V1], [V2] ...
 v
SYSTEM_PROMPT + user_prompt  # CoT instructions + citation rules
 |
 v
Gemini (gemini-3-flash-preview, thinking_budget=1024)
 |
 v
<reasoning> ... </reasoning> <answer> ... </answer>
```

---

## Citation Format

Every inline citation pairs the source-id with the paper number(s) backing it: `[<source-id> (<paper-list>)]`.

| Source | Example | Notes |
|---|---|---|
| Graph direct claim (1-hop) | `[G1 (P23)]` | one relation A -> B |
| Graph chain hop (2-hop) | `[G4.1 (P56)] [G4.2 (P12)]` | each hop cited separately |
| Vector passage | `[V2 (P44)]` | exactly one paper per chunk |

Paper IDs come only from the evidence block, the model is instructed never to invent them.

---

## Question Types

- **Multiple choice (A/B/C/D):** returns only the single best-supported letter, no prose.
- **Open-ended:** 2-4 paragraphs of cited prose, distinguishing claim strength (`evidence consistently shows` vs `one study suggests`) and stating conflicts explicitly.

---

## Configuration

All settings live in `config/gen_config.py`:

| Setting | Value |
|---|---|
| `GENERATOR_MODEL` | `gemini-3-flash-preview` |
| `GENERATOR_MAX_TOKENS` | `4096` |

Requires `GEMINI_API_KEY` in the environment (`.env`).

---

## Running

The generator is a library called by the retriever pipeline, not run standalone. `generate()` is invoked at the end of `retriever/main.py`:

```python
from generator.generator import generate

answer = generate(query, graph_results, vector_results)
```

To run the full retrieval + generation flow:

```bash
uv run python -m retriever.main
```
