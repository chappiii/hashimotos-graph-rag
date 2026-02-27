import os
import re
from config.section_chunker_config import CHUNKS_DIR


def ensure_output_directory(base_name: str) -> str:
    run_dir = os.path.join(CHUNKS_DIR, base_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    return run_dir


def is_already_processed(base_name: str) -> bool:
    run_dir = os.path.join(CHUNKS_DIR, base_name)
    return os.path.isdir(run_dir) and len(os.listdir(run_dir)) > 0


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name)


def save_chunk(output_dir: str, filename: str, header_name: str, content: str):
    out_path = os.path.join(output_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Header: {header_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(content)
