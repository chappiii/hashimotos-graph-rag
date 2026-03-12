import os
from pathlib import Path
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

PDFS = "../pdfs"
SECTIONS_DIR = "../paper-sections"
CHUNKS_DIR = "./chunks"
FIGS_TABLES_DIR = "./figs_tables"

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
    "table_id":      None,
    "table_type":    None,
    "section_label": None,
    "caption":       None,
    "population":    None,
    "groups":        [],
    "variables": [],
    "key_findings":  [],
    "model_adjustments": [],
}

FIGURE_SCHEMA = {
    "figure_id":     None,
    "figure_type":   None,
    "section_label": None,
    "caption":       None,
    "population":    None,
    "groups":        [],
    "outcome":       None,
    "key_findings":  [],
}