from google.genai import types
from config.section_chunker_config import GEMINI_MODEL, SAFETY_SETTINGS
from utils.structure_parser import parse_structure_from_output
from utils.file_utils import sanitize_filename, save_chunk


def extract_content_for_header(pdf_file, header_name: str, subheaders: list, client) -> str:
    """Extract content for a specific header section using Gemini."""
    subheader_list = '\n'.join(f'- {sh}' for sh in subheaders) if subheaders else 'None'
    prompt = (
        f"You are an expert document analyst. Your task is to extract the content for the header \"{header_name}\" "
        f"and its subheaders from the provided PDF document.\n\n"
        f"Subheaders for this section (in order):\n{subheader_list}\n\n"
        "Extract ALL the content that belongs under this header, including each subheader in the list above, "
        "in order, and their content. Stop at the next top-level section. Preserve the original structure as much as possible. "
        "If the section contains any tables, format them as proper Markdown tables using | delimiters and a header separator row (|---|---|). "
        "Do not include content from other top-level sections."
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
            "Extract and return the complete text content of this PDF document. "
            "Format any tables as proper Markdown tables using | delimiters and a header separator row (|---|---|)."
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

        try:
            print(f"  Extracting: {header_name}")
            content = extract_content_for_header(pdf_file, header_name, subheaders, client)
        except Exception as e:
            print(f"  Error extracting '{header_name}': {e}")
            content = f"Error extracting content: {e}"

        safe_header = sanitize_filename(header_name.lower().replace(' ', '_'))
        filename = f"{section_index}-{safe_header}.md"
        save_chunk(output_dir, filename, header_name, content)
        print(f"  Saved: {filename}")
        section_index += 1
