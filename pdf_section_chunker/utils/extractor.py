from google.genai import types
from config.section_chunker_config import GEMINI_MODEL, SAFETY_SETTINGS
from utils.structure_parser import parse_structure_from_output
from utils.file_utils import sanitize_filename, save_chunk


def extract_content_for_header(pdf_file, header_name: str, subheaders: list, next_header: str | None, client) -> str:
    """Extract content for a specific header section using Gemini."""
    subheader_list = '\n'.join(f'- {sh}' for sh in subheaders) if subheaders else 'None'

    if next_header:
        stop_instruction = (
            f'Stop extracting when you reach the section titled "{next_header}". '
            f'Do not include any content from "{next_header}" or any section after it.'
        )
    else:
        stop_instruction = 'Extract until the end of the document.'

    prompt = (
        f"You are an expert document analyst. Your task is to extract the content for the section \"{header_name}\" "
        f"and its subheaders from the provided PDF document.\n\n"
        f"Subheaders for this section (in order):\n{subheader_list}\n\n"
        "Extract ALL the content that belongs under this section, including each subheader listed above "
        "in order, and their content. Preserve the original structure as much as possible.\n\n"
        f"{stop_instruction}\n\n"
        "Sections may span multiple pages. Do not treat page numbers, running headers, or footers as section endings.\n\n"
        "Do not include content from other top-level sections.\n\n"
        "If the section contains tables, format each as a GitHub-Flavored Markdown table using these rules:\n"
        "- Separator row must use exactly three dashes per cell: | :--- | :--- | :--- |\n"
        "- Never repeat or extend dashes beyond three per cell in the separator row\n"
        "- Include ALL data rows from the PDF — do not stop after the separator row\n"
        "- Copy cell content verbatim from the PDF — never use placeholders\n"
        "Return only the document content. Do not include explanations, reasoning, or meta-commentary."
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[pdf_file, prompt],
        config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
    )
    return response.text


def process_paper(pdf_file, headers_output: str, output_dir: str, client):
    """Extract and save all sections of a paper."""
    headers = parse_structure_from_output(headers_output)

    # No real sections found — extract and save full text as a single chunk
    if len(headers) <= 1:
        prompt = (
            "Extract and return the complete text content of this PDF document.\n\n"
            "If the document contains tables, format each as a GitHub-Flavored Markdown table using these rules:\n"
            "- Separator row must use exactly three dashes per cell: | :--- | :--- | :--- |\n"
            "- Never repeat or extend dashes beyond three per cell in the separator row\n"
            "- Include ALL data rows from the PDF — do not stop after the separator row\n"
            "- Copy cell content verbatim from the PDF — never use placeholders\n"
            "Return only the document content. Do not include explanations, reasoning, or meta-commentary."
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[pdf_file, prompt],
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS)
        )
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
