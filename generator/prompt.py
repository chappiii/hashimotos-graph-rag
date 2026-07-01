SYSTEM_PROMPT = """\
You are a biomedical research assistant specialized in Hashimoto's thyroiditis and autoimmune thyroid diseases.
Your answers are grounded exclusively in the evidence provided in the context below.

## GRAPH EVIDENCE FORMAT
Graph evidence comes in two forms:
- DIRECT claims (1-hop): A single relation A -> B. Cited as [G1], [G2]...
- CHAIN claims (2-hop): An indirect mechanism A -> B -> C linking two entities through a bridge.
  Each hop has its own citation: [G4.1] is the first hop (A -> B), [G4.2] is the second (B -> C).
  When supporting a claim from a chain, narrate the mechanism explicitly, e.g.
  "X reduces Z via the bridge entity Y [G4.1 (P56), G4.2 (P12)]", and cite the hop(s) you used.

## CITATION FORMAT (MANDATORY)
Every inline citation MUST pair the source-id with the paper number(s) backing it,
using this format: `[<source-id> (<paper-list>)]`.
- Graph direct claim: `[G1 (P23)]` -- or `[G1 (P23, P44, P56)]` if multiple papers back it.
- Graph chain hop:    `[G4.1 (P56)] [G4.2 (P12)]` -- cite each hop separately with its own paper(s).
- Vector passage:     `[V2 (P44)]` -- vector chunks have exactly one paper.
- Multiple sources in one claim: `[G1 (P23, P44)] [V2 (P56)]` (space-separated).

The paper number(s) for each source are listed directly under that source in the context
below (e.g. evidence lines `- [P23, ...]` under [G1], or `Paper P44` under [V2]).
Use ONLY those paper IDs. Do not invent paper numbers.

## TASK
Answer the user's question in two stages:

### Stage 1 - Reasoning (inside <reasoning> tags)
Before writing your answer, work through the evidence:
1. Identify which graph claims [G1], [G2]... (direct) or [G4.1], [G4.2]... (chain hops) directly address the question.
2. Identify which passages [V1], [V2]... corroborate, extend, or contradict those claims.
3. Flag any conflicts between graph and passage evidence explicitly.
4. Note what the evidence cannot answer.
5. List any evidence items you are excluding and why.

### Stage 2 - Answer (inside <answer> tags)
**If the question includes OPTIONS (A/B/C/D):**
- State only the single letter of the best-supported option. Nothing else.
- Do not write prose, do not mention other options, do not explain.

**If the question is open-ended (no options):**
- Write 2-4 paragraphs of fluent prose presenting the evidence from the context.
- After each factual claim, insert an inline citation using the MANDATORY format above
  (e.g. `[G1 (P23)]`, `[V2 (P44)]`, `[G4.1 (P56), G4.2 (P12)]`). Never cite without a paper ID.
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


def _format_evidence_lines(evidence_list: list[dict], indent: str) -> list[str]:
    lines: list[str] = []
    for ev in evidence_list:
        pid    = ev.get("paper_id", "?")
        design = ev.get("study_design", "?")
        year   = ev.get("paper_year", "?")
        sec    = ev.get("section_type", "?")
        text   = ev.get("evidence_text", "").strip()
        cite   = f"[P{pid}, {design}, {year}]"
        lines.append(f"{indent}- {cite} [{sec}] {text}")
    return lines


def _format_direct_path(path: dict, label: str) -> list[str]:
    claim     = path["claims"][0]
    triple    = f"{claim['source_name']} -> {claim['relation_type']} -> {claim['target_name']}"
    certainty = claim.get("certainty_max", "?")
    papers    = claim.get("paper_count", "?")
    lines = [
        f"{label} {triple}",
        f"     Certainty: {certainty} | Papers: {papers}",
    ]
    lines.extend(_format_evidence_lines(claim.get("evidence_list", []), indent="     "))
    lines.append("")
    return lines


def _format_chain_path(path: dict, chain_label: str) -> list[str]:
    c1, c2 = path["claims"]
    bridge = path["entities"][1]
    chain_text = (
        f"{c1['source_name']} -> {c1['relation_type']} -> {bridge} "
        f"-> {c2['relation_type']} -> {c2['target_name']}"
    )
    lines = [
        f"{chain_label} CHAIN: {chain_text}",
        f"     Bridge: {bridge}",
    ]

    for hop_idx, claim in enumerate(path["claims"], 1):
        hop_label = f"{chain_label[:-1]}.{hop_idx}]"
        triple    = f"{claim['source_name']} -> {claim['relation_type']} -> {claim['target_name']}"
        certainty = claim.get("certainty_max", "?")
        papers    = claim.get("paper_count", "?")
        lines.append(f"     {hop_label} {triple}")
        lines.append(f"          Certainty: {certainty} | Papers: {papers}")
        lines.extend(_format_evidence_lines(claim.get("evidence_list", []), indent="          "))

    lines.append("")
    return lines


def _format_graph_section(graph_results: list[dict]) -> str:
    """Format paths into prompt sections. Direct (1-hop) and chains (2-hop) are split.
    Citations: direct = [G1], chain hops = [G4.1], [G4.2] (chain itself referenced as G4).
    """
    direct_paths = [p for p in graph_results if p["length"] == 1]
    chain_paths  = [p for p in graph_results if p["length"] >= 2]

    sections: list[str] = []

    if direct_paths:
        lines = ["## KNOWLEDGE GRAPH EVIDENCE - DIRECT CLAIMS (1-hop)",
                 "Single-relation claims extracted from 115 research papers.",
                 ""]
        for i, path in enumerate(direct_paths, 1):
            lines.extend(_format_direct_path(path, label=f"[G{i}]"))
        sections.append("\n".join(lines))

    if chain_paths:
        lines = ["## KNOWLEDGE GRAPH EVIDENCE - INDIRECT MECHANISMS (2-hop chains)",
                 "Multi-hop relation chains showing how two entities are linked through a bridge entity.",
                 "Cite each hop separately as [Gx.1] (first hop) and [Gx.2] (second hop).",
                 ""]
        offset = len(direct_paths)
        for i, path in enumerate(chain_paths, 1):
            lines.extend(_format_chain_path(path, chain_label=f"[G{offset + i}]"))
        sections.append("\n".join(lines))

    return "\n".join(sections)


def _format_vector_section(vector_results: list[dict]) -> str:
    lines = ["## RESEARCH PASSAGES",
             "Raw text chunks retrieved by semantic similarity.",
             ""]
    for i, chunk in enumerate(vector_results, 1):
        label    = f"[V{i}]"
        pid      = chunk.get("paper_id", "?")
        sec      = chunk.get("section_type", "?")
        sec_name = chunk.get("section_name", "")
        text     = chunk.get("text", "").strip()
        cite     = f"[V{i} (P{pid})]"
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
