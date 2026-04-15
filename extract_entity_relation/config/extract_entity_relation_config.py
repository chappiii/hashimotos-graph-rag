import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _CONFIG_DIR.parent
_PROJECT_ROOT = _MODULE_DIR.parent

DATA_DIR = _PROJECT_ROOT / "pdf_section_chunker/chunks"
OUTPUT_DIR = _MODULE_DIR / "extracted_entity_relations"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-3-flash-preview"
