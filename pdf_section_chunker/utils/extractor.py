import time
from google.genai import types
from pdf_section_chunker.config.section_chunker_config import GEMINI_MODEL, SAFETY_SETTINGS
from pdf_section_chunker.utils.structure_parser import parse_structure_from_output
from pdf_section_chunker.utils.file_utils import sanitize_filename, save_chunk


_STRUCTURE_PROMPT_TEMPLATE = """\
The title of this paper is: "{title}"

Extract the section structure of this academic research paper.

### Instructions:

* Use the provided title above as the first entry in your output. Do not re-identify or modify it.
* Extract all main section headers and all subsections/sub-subsections.
* Preserve the original numbering and hierarchy exactly as they appear (e.g., 1., 2.1., 2.1.1.).
* If sections are not numbered, preserve their order and hierarchy based on formatting (e.g., font size, bolding, spacing, line breaks), without adding artificial numbering.

### Medical Paper Variability (IMPORTANT):

Scientific and medical papers may use different structures depending on study type (e.g., clinical trial, observational study, systematic review, meta-analysis, case report, narrative review).
You must extract ALL section headers as they appear, without forcing a standard structure. Valid section names may include (but are not limited to):

* Background (instead of Introduction)
* Introduction
* Materials and Methods / Methods / Patients and Methods
* Results / Findings
* Discussion / Interpretation
* Conclusion / Conclusions / Summary / Implications

Additional valid sections may include:

* Study Design
* Data Sources
* Participants / Population
* Inclusion Criteria / Exclusion Criteria
* Outcome Measures
* Statistical Analysis / Statistical Methods
* Ethics Approval / Ethical Considerations
* Trial Registration / Registration / Clinical Trial Registration
* Case Presentation
* Clinical Findings
* Intervention / Treatment
* Follow-up
* Search Strategy (systematic reviews)
* Data Extraction
* Quality Assessment / Risk of Bias

### Abstract Handling (STRICT):

* Abstract must always be a single top-level entry.
* If the abstract is structured (e.g., Background, Methods, Results, Conclusions), treat it as ONE section.
* Do NOT list structured abstract components as separate subsections, even if visually separated.

### Implicit Sections:

* Some papers may not explicitly label early sections (e.g., Abstract or Introduction).
* If unlabeled text clearly functions as Abstract or Introduction, include the appropriate section name.

### Appendix Handling:

* For appendices, include only the label (e.g., Appendix A, Appendix B).
* Do not include appendix subtitles.

### Robustness Rules (CRITICAL):

* Only extract section headers that are explicitly present in the document.
* Do NOT infer or invent section names.
* Do NOT normalize, rename, merge, or split section titles.
* Each section must remain exactly as it appears in the original document.
* Preserve the exact original wording, case, and numbering.
* Do not remove a section simply because it is uncommon or unexpected.
* If it clearly appears as a section header in the document, it must be preserved.
* Treat visually distinct lines (e.g., bold, uppercase, or isolated lines) as potential headers even if unnumbered.

### Sections to Exclude:

* Author contributions, funding statements, ethical statements, data availability, conflicts of interest, acknowledgements.
* Abbreviations, keywords, highlights, graphical abstracts.
* Figure captions, table titles.
* Running headers or footers (page numbers, journal names, author names).
* Supplementary information, supporting information.

### Additional Rules:

* References, Bibliography, or Works Cited must always be included if present as a top-level section.
* Do NOT treat figure captions or table titles as section headers, even if they appear in bold or large font.

### Formatting Rules:

* Every line must start with a `*` bullet (use `*` only).
* Use 4-space indentation for sub-levels.
* Do not use bold, italic, or markdown formatting.
* Do not include trailing punctuation (colons, periods).
* Do not include cross-references (e.g., "(see Appendix E)").
* Preserve original case and numbering exactly.

### Example Output (Order must follow the document exactly, even if uncommon):

* Main Title of the Paper
* Abstract
* 1. Introduction
* 2. Materials and Methods
    * 2.1. Study Population
    * 2.2. Experimental Procedure
        * 2.2.1. Phase 1
        * 2.2.2. Phase 2
    * 2.3. Statistical Analysis
* 3. Results
    * 3.1. Primary Outcomes
    * 3.2. Secondary Outcomes
* 4. Discussion
* 5. Conclusion
* References
* Appendix A

Extract the section headers now:
"""


_CORRECTION_PROMPT_TEMPLATE = """\
Below is a section structure extracted from this academic research paper. \
Compare it carefully against the actual PDF and check for missing or incorrect sections.

### Extracted structure:

{structure}

### Extraction rules (used to produce the structure above):

* The first entry is the paper title: "{title}" — injected from metadata. Do not modify or remove it.
* All main section headers and subsections are included, preserving original numbering and hierarchy.
* Abstract is always a single top-level entry — structured abstract sub-sections \
(Background, Objectives, Methods, Results, Conclusions) are part of Abstract and not listed separately.
* Some papers have unlabeled content at the beginning that functions as an Abstract or Introduction — \
these are intentionally included. Do not remove them.
* Appendices use only the label (e.g., Appendix A) without the full subtitle.

### Medical section variability:

Section names vary across medical papers depending on study type \
(e.g., clinical trial, systematic review, meta-analysis, case report).

Valid section names may include (but are not limited to): Background, Findings, \
Interpretation, Case Presentation, Statistical Analysis, Trial Registration, \
Study Design, Search Strategy, Quality Assessment, etc.

* Verify that no valid sections were mistakenly excluded due to naming variations.
* If such sections exist in the PDF, they must be added.

### Sections that must be excluded:

* Author contributions, funding statements, ethical statements, data availability, conflicts of interest, acknowledgements.
* Abbreviations, keywords, highlights, graphical abstracts.
* Figure captions, table titles.
* Running headers or footers (page numbers, journal names, author names).
* Supplementary information, supporting information.

### Task:

* Check if any sections in the PDF are missing from the structure above. If so, add them.
* Only remove a section if you are certain it does not exist anywhere in the PDF.
* Do not add any sections from the exclusion list above.
* Do NOT normalize, rename, merge, or split section titles. Preserve exact wording, case, and numbering.
* The paper title (first entry) is "{title}" — injected from metadata. Do not modify or remove it.

Output the corrected section structure using the exact same formatting rules: \
`*` bullets, 4-space indentation for sub-levels, no bold/italic, no trailing punctuation.

If the structure is already correct, output it unchanged.
"""


def extract_structure(pdf_file, client, title: str) -> tuple[str, float]:
    """Pass 1: Extract section structure from PDF. Returns (text, elapsed_seconds)."""
    prompt = _STRUCTURE_PROMPT_TEMPLATE.format(title=title)
    start = time.time()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[pdf_file, prompt],
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
    )
    elapsed = time.time() - start

    if response.text is not None:
        return response.text, elapsed

    reason = "UNKNOWN"
    if response.candidates:
        reason = str(response.candidates[0].finish_reason)
    raise ValueError(f"Structure extraction failed (finish_reason={reason})")


def correct_structure(pdf_file, cleaned_structure: str, client, title: str) -> tuple[str, float]:
    """Pass 1.5: Correct extracted structure against PDF. Returns (text, elapsed_seconds)."""
    prompt = _CORRECTION_PROMPT_TEMPLATE.format(structure=cleaned_structure, title=title)
    start = time.time()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[pdf_file, prompt],
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
    )
    elapsed = time.time() - start

    if response.text is not None:
        return response.text, elapsed

    reason = "UNKNOWN"
    if response.candidates:
        reason = str(response.candidates[0].finish_reason)
    raise ValueError(f"Structure correction failed (finish_reason={reason})")


def _build_primary_prompt(header_name: str, subheader_list: str, stop_instruction: str) -> str:
    """Build the primary content extraction prompt for a section."""
    return (
        f"You are an expert document analyst helping with academic research. "
        f"Your task is to read and write out the content of the section \"{header_name}\" "
        f"from this academic research paper. This is for academic research and analysis purposes.\n\n"
        f"Subheaders for this section (in order):\n{subheader_list}\n\n"
        f"Write out ALL the content that belongs under this section, including each subheader listed above "
        f"in order, and their content. Preserve the original structure as much as possible.\n\n"
        f"{stop_instruction}\n\n"
        "Sections may span multiple pages. Do not treat page numbers, running headers, or footers as section endings.\n\n"
        "Do not include content from other top-level sections.\n\n"
        "Output only the section text. Do not add explanations, reasoning, or meta-commentary."
    )


def _build_retry_prompt(header_name: str, subheader_list: str, stop_instruction: str) -> str:
    """Build the alternate prompt for RECITATION retry."""
    return (
        f"This is an academic research paper. For research analysis purposes, please transcribe the "
        f"\"{header_name}\" section of this paper.\n\n"
        f"Subheaders within this section:\n{subheader_list}\n\n"
        f"Include all text under this section and its subheaders in the order they appear. "
        f"Maintain the original structure.\n\n"
        f"{stop_instruction}\n\n"
        "Do not skip page headers/footers — they are not section boundaries.\n\n"
        "Output the section text only."
    )


def extract_content_for_header(pdf_file, header_name: str, subheaders: list, next_header: str | None, client) -> tuple[str, float]:
    """Extract content for a specific header section using Gemini."""
    subheader_list = '\n'.join(f'- {sh}' for sh in subheaders) if subheaders else 'None'

    if next_header:
        stop_instruction = (
            f'Stop when you reach the section titled "{next_header}". '
            f'Do not include any content from "{next_header}" or beyond.'
        )
    else:
        stop_instruction = 'Continue until the end of the document.'

    for attempt, build_prompt in enumerate([_build_primary_prompt, _build_retry_prompt], start=1):
        prompt = build_prompt(header_name, subheader_list, stop_instruction)
        start = time.time()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[pdf_file, prompt],
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
        )
        elapsed = time.time() - start

        if response.text is not None:
            return response.text, elapsed

        reason = "UNKNOWN"
        if response.candidates:
            reason = str(response.candidates[0].finish_reason)

        if attempt == 1 and "RECITATION" in reason:
            print(f"    RECITATION on attempt 1 for '{header_name}', retrying with alternate prompt...")
            continue

        raise ValueError(f"Gemini returned None (finish_reason={reason})")


def process_paper(pdf_file, headers_output: str, output_dir: str, client) -> list[dict]:
    """Extract and save all sections of a paper. Returns list of per-call timing dicts."""
    headers = parse_structure_from_output(headers_output)
    call_times = []

    # No content-bearing sections found — extract and save full text as a single chunk
    content_sections = [h for h in headers[1:] if h["header"].lower() not in {"references", "bibliography", "works cited"}]
    if len(content_sections) == 0:
        full_text_prompts = [
            (
                "You are an expert document analyst helping with academic research. "
                "Please read and write out the complete text content of this academic research paper. "
                "This is for academic research and analysis purposes.\n\n"
                "Output only the document text. Do not add explanations, reasoning, or meta-commentary. "
                "Preserve the original structure as much as possible."
            ),
            (
                "This is an academic research paper. For research analysis purposes, please transcribe "
                "the full text of this paper from beginning to end. Maintain the original structure.\n\n"
                "Output the paper text only."
            ),
        ]
        response = None
        for attempt, prompt in enumerate(full_text_prompts, start=1):
            start = time.time()
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[pdf_file, prompt],
                config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
            )
            elapsed = time.time() - start
            if response.text is not None:
                call_times.append({"section": "Full Text", "seconds": elapsed})
                break
            reason = "UNKNOWN"
            if response.candidates:
                reason = str(response.candidates[0].finish_reason)
            if attempt == 1 and "RECITATION" in reason:
                print(f"    RECITATION on attempt 1 for full-text, retrying...")
                continue
            raise ValueError(f"Gemini returned None for full-text extraction (finish_reason={reason})")

        if response.text is None:
            reason = "UNKNOWN"
            if response.candidates:
                reason = str(response.candidates[0].finish_reason)
            raise ValueError(f"Gemini returned None for full-text extraction (finish_reason={reason})")

        save_chunk(output_dir, "1-full_text.md", "Full Text", response.text)
        print("  No sections found — saved full text as single chunk")
        return call_times

    section_index = 1
    for i, header_info in enumerate(headers, 1):
        # Skip the title (first header)
        if i == 1:
            continue

        header_name = header_info["header"]
        subheaders = header_info["subheaders"]
        next_header = headers[i]["header"] if i < len(headers) else None

        try:
            print(f"  Extracting: {header_name}")
            content, elapsed = extract_content_for_header(pdf_file, header_name, subheaders, next_header, client)
            call_times.append({"section": header_name, "seconds": elapsed})
        except Exception as e:
            print(f"  Error extracting '{header_name}': {e}")
            content = f"Error extracting content: {e}"

        safe_header = sanitize_filename(header_name.lower().replace(' ', '_'))
        filename = f"{section_index}-{safe_header}.md"
        save_chunk(output_dir, filename, header_name, content)
        print(f"  Saved: {filename}")
        section_index += 1

    return call_times
