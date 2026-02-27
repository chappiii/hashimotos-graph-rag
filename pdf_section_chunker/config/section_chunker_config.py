import os
from pathlib import Path
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

PDFS = "../pdfs"
SECTIONS_DIR = "../paper-sections"
CHUNKS_DIR = "./chunks"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
UPLOAD_POLL_INTERVAL = 2

SKIP_EXISTING = True

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]
