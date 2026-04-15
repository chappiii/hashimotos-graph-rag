import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _CONFIG_DIR.parent
_PROJECT_ROOT = _MODULE_DIR.parent

DATA_DIR = _PROJECT_ROOT / "pdf_section_chunker/chunks"
OUTPUT_DIR = _MODULE_DIR / "extracted_entity_relations"

# --- Models ---
GEMINI_MODEL = "gemini-3-flash-preview"
CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-5.4-mini"

# --- API keys ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Settings ---
MAX_OUTPUT_TOKENS = 32000
