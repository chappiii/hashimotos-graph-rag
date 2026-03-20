from google.genai import types
from config.section_chunker_config import GEMINI_MODEL, SAFETY_SETTINGS
from utils.structure_parser import parse_structure_from_output
from utils.file_utils import sanitize_filename, save_chunk


STRUCTURE_PROMPT = """\
Extract the section structure of this academic research paper.

### Instructions:

* Identify the primary title of the paper as the first entry.
* Extract all main section headers (e.g., Abstract, Introduction, Methods, Results, Discussion, Conclusion, References, Appendix).
* Extract all subsections and sub-subsections, preserving their original numbering and hierarchy exactly as they appear in the paper (e.g., 1., 2.1., 2.1.1.).
* Abstract must always be a single top-level entry. Never split a structured abstract into sub-sections (e.g., Background, Objectives, Methods, Results, Conclusions are all part of Abstract — do not list them separately).
* Output must be a hierarchical bulleted list that precisely reflects the manuscript's organization.

### Formatting rules:

* Every line must start with a `*` bullet (use `*` only, never `-` or other markers).
* Use 4-space indentation for sub-levels.
* Do not use bold, italic, or any markdown formatting on section names.
* Do not include trailing punctuation (colons, periods) on section names.
* Do not include cross-references (e.g., "(see Appendix E)") in section names.
* Preserve the original case and numbering of each section name exactly as written in the paper.

### Sections to exclude:

* Author contributions, funding statements, ethical statements, data availability, conflicts of interest, acknowledgements.
* Abbreviations, keywords, highlights, graphical abstracts.
* Figure captions, table titles.
* Running headers or footers (page numbers, journal names, author names).

### Example output:

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


def extract_structure(pdf_file, client) -> str:
    """Pass 1: Extract section structure from PDF using Gemini."""
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[pdf_file, STRUCTURE_PROMPT],
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
    )

    if response.text is not None:
        return response.text

    reason = "UNKNOWN"
    if response.candidates:
        reason = str(response.candidates[0].finish_reason)
    raise ValueError(f"Structure extraction failed (finish_reason={reason})")


def _build_primary_prompt(header_name: str, subheader_list: str, stop_instruction: str) -> str:
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


def extract_content_for_header(pdf_file, header_name: str, subheaders: list, next_header: str | None, client) -> str:
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
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[pdf_file, prompt],
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
        )

        if response.text is not None:
            return response.text

        reason = "UNKNOWN"
        if response.candidates:
            reason = str(response.candidates[0].finish_reason)

        if attempt == 1 and "RECITATION" in reason:
            print(f"    RECITATION on attempt 1 for '{header_name}', retrying with alternate prompt...")
            continue

        raise ValueError(f"Gemini returned None (finish_reason={reason})")


def process_paper(pdf_file, headers_output: str, output_dir: str, client):
    """Extract and save all sections of a paper."""
    headers = parse_structure_from_output(headers_output)

    # No real sections found — extract and save full text as a single chunk
    if len(headers) <= 1:
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
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[pdf_file, prompt],
                config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
            )
            if response.text is not None:
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
        return

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
            content = extract_content_for_header(pdf_file, header_name, subheaders, next_header, client)
        except Exception as e:
            print(f"  Error extracting '{header_name}': {e}")
            content = f"Error extracting content: {e}"

        safe_header = sanitize_filename(header_name.lower().replace(' ', '_'))
        filename = f"{section_index}-{safe_header}.md"
        save_chunk(output_dir, filename, header_name, content)
        print(f"  Saved: {filename}")
        section_index += 1
