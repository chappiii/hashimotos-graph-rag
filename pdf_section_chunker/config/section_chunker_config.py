import os
from pathlib import Path
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

_CONFIG_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _CONFIG_DIR.parent
_PROJECT_ROOT = _MODULE_DIR.parent

PDFS = _PROJECT_ROOT / "pdfs"
SECTIONS_DIR = _PROJECT_ROOT / "paper-sections"
METADATA_DIR = _PROJECT_ROOT / "extract_metadata" / "GT_grouped"
AUTO_SECTIONS_DIR = _MODULE_DIR / "auto-sections"
CHUNKS_DIR = _MODULE_DIR / "chunks"
TIME_DIR = _MODULE_DIR / "time"
FIGS_TABLES_DIR = _MODULE_DIR / "figs_tables"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MODEL = "gemini-3-flash-preview"
UPLOAD_POLL_INTERVAL = 2

SKIP_EXISTING = True
SKIP_EXISTING_FIGS_TABLES = True

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

TABLE_SCHEMA = {
    "table_id":          int,
    "table_type":        str,
    "section_label":     str,
    "caption":           str,
    "population":        str,
    "groups":            [str],
    "variables":         [str],
    "key_findings":      [str],
    "model_adjustments": [str],
}

FIGURE_SCHEMA = {
    "figure_id":     int,
    "figure_type":   str,
    "section_label": str,
    "caption":       str,
    "population":    str,
    "groups":        [str],
    "outcome":       str,
    "key_findings":  [str],
}
