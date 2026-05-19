from generator.config.gen_config import VECTOR_CHUNK_PREVIEW

SYSTEM_PROMPT = """\
You are a biomedical research assistant specialized in Hashimoto's thyroiditis and autoimmune thyroid diseases.
Your answers are grounded exclusively in the evidence provided in the context below.

## TASK
Answer the user's question in two stages:

### Stage 1 - Reasoning (inside <reasoning> tags)
Before writing your answer, work through the evidence:
1. Identify which graph claims [G1], [G2]... directly address the question.
2. Identify which passages [V1], [V2]... corroborate, extend, or contradict those claims.
3. Flag any conflicts between graph and passage evidence explicitly.
4. Note what the evidence cannot answer.
5. List any evidence items you are excluding and why.

### Stage 2 - Answer (inside <answer> tags)
**If the question includes OPTIONS (A/B/C/D):**
- State only the single letter of the best-supported option. Nothing else.
- Do not write prose, do not mention other options, do not explain.

**If the question is open-ended (no options):**
- Write 2-4 paragraphs of fluent prose synthesizing the evidence.
- After each factual claim, insert an inline citation (e.g. [P2, RCT, 2019] or [P44]).
- Distinguish claim strength: use "evidence consistently shows" for claims supported by
  multiple graph entries or papers, and "one study suggests" for single-paper or
  passage-only evidence.
- If the evidence contains contradictory findings, state the conflict explicitly
  (e.g. "evidence is mixed: ...").
- Do NOT invent facts or cite sources not listed in the context.
- If the context is insufficient to answer, say so plainly.

## OUTPUT FORMAT
<reasoning>
[your reasoning here]
</reasoning>
<answer>
[single letter A/B/C/D if multiple choice -- or prose if open-ended]
</answer>\
"""


def _format_graph_section(graph_results: list[dict]) -> str:
    lines = ["## KNOWLEDGE GRAPH EVIDENCE",
             "Structured relation claims extracted from 115 research papers.",
             ""]
    for i, claim in enumerate(graph_results, 1):
        label     = f"[G{i}]"
        triple    = f"{claim['source_name']} -> {claim['relation_type']} -> {claim['target_name']}"
        certainty = claim.get("certainty_max", "?")
        papers    = claim.get("paper_count", "?")
        lines.append(f"{label} {triple}")
        lines.append(f"     Certainty: {certainty} | Papers: {papers}")
        for ev in claim.get("evidence_list", []):
            pid    = ev.get("paper_id", "?")
            design = ev.get("study_design", "?")
            year   = ev.get("paper_year", "?")
            sec    = ev.get("section_type", "?")
            text   = ev.get("evidence_text", "").strip()
            cite   = f"[P{pid}, {design}, {year}]"
            lines.append(f"     - {cite} [{sec}] {text}")
        lines.append("")
    return "\n".join(lines)


def _format_vector_section(vector_results: list[dict]) -> str:
    lines = ["## RESEARCH PASSAGES",
             "Raw text chunks retrieved by semantic similarity.",
             ""]
    for i, chunk in enumerate(vector_results, 1):
        label    = f"[V{i}]"
        pid      = chunk.get("paper_id", "?")
        sec      = chunk.get("section_type", "?")
        sec_name = chunk.get("section_name", "")
        text     = chunk.get("text", "").strip()[:VECTOR_CHUNK_PREVIEW]
        cite     = f"[P{pid}]"
        lines.append(f"{label} Paper P{pid} | {sec} | {sec_name}")
        lines.append(f"     Cite as: {cite}")
        lines.append(f"     {text}")
        lines.append("")
    return "\n".join(lines)


def build_user_prompt(
    query: str,
    graph_results: list[dict],
    vector_results: list[dict],
    options: dict[str, str] | None = None,
) -> str:
    graph_section  = _format_graph_section(graph_results)
    vector_section = _format_vector_section(vector_results)

    if options:
        opts_text = "\n".join(f"{k}: {v}" for k, v in options.items())
        question_block = f"## QUESTION\n{query}\n\n## OPTIONS\n{opts_text}"
        answer_instruction = (
            "This is a multiple choice question. "
            "State the single letter (A, B, C, or D) best supported by the evidence "
            "inside <answer> tags. No prose needed."
        )
    else:
        question_block = f"## QUESTION\n{query}"
        answer_instruction = "Write your evidence-backed answer below, using inline citations as instructed."

    return (
        f"{question_block}\n\n"
        f"{graph_section}\n"
        f"{vector_section}\n"
        f"## ANSWER\n{answer_instruction}"
    )
