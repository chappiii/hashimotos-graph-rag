import os
import re
from pdf_section_chunker.config.section_chunker_config import CHUNKS_DIR, AUTO_SECTIONS_DIR


def ensure_output_directory(base_name: str) -> str:
    """Create and return the chunks output directory for a paper."""
    run_dir = os.path.join(CHUNKS_DIR, base_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir


def is_already_processed(base_name: str) -> bool:
    """Check if a paper already has extracted chunks."""
    run_dir = os.path.join(CHUNKS_DIR, base_name)
    return os.path.isdir(run_dir) and len(os.listdir(run_dir)) > 0


def save_auto_section(base_name: str, structure_text: str) -> None:
    """Save the final corrected structure to auto-sections/{id}.md."""
    os.makedirs(AUTO_SECTIONS_DIR, exist_ok=True)
    out_path = os.path.join(AUTO_SECTIONS_DIR, f"{base_name}.md")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(structure_text)


def save_auto_section_raw(base_name: str, structure_text: str) -> None:
    """Save the raw Pass 1 output to auto-sections/{id}-raw.md for debugging."""
    os.makedirs(AUTO_SECTIONS_DIR, exist_ok=True)
    out_path = os.path.join(AUTO_SECTIONS_DIR, f"{base_name}-raw.md")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(structure_text)


def sanitize_filename(name: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    return re.sub(r'[\\/*?:"<>|]', '_', name)


def save_chunk(output_dir: str, filename: str, header_name: str, content: str) -> None:
    """Save a single section chunk with a Header: line and separator."""
    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Header: {header_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(content)
