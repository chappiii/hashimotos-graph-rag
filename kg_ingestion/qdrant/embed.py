"""
Shared Gemini embedding helper with retry on rate-limit.
"""

import time

from google import genai
from google.genai import types as genai_types

from kg_ingestion.config.kg_config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

_MAX_RETRIES = 3
_RETRY_SLEEP  = 10  # seconds on first 429


def embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Return 768-d embedding vector for text. Retries up to 3× on 429."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = genai_types.EmbedContentConfig(task_type=task_type)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=text,
                config=config,
            )
            return response.embeddings[0].values
        except Exception as e:
            msg = str(e).lower()
            is_rate_limit = "429" in msg or "quota" in msg or "resource exhausted" in msg or "503" in msg or "unavailable" in msg
            if is_rate_limit and attempt < _MAX_RETRIES:
                wait = _RETRY_SLEEP * attempt
                print(f"    [embed] rate limit (attempt {attempt}), sleeping {wait}s…")
                time.sleep(wait)
                continue
            raise
