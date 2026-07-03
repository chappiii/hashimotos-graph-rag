import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()

from retriever.config.ret_config import EMBEDDING_MODEL, EMBED_MAX_RETRIES, EMBED_RETRY_SLEEP

_RETRY_MARKERS = ("429", "500", "502", "503", "504", "10060", "timeout", "timed out", "Connection")


def embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    config = genai_types.EmbedContentConfig(task_type=task_type)

    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=config,
            )
            return response.embeddings[0].values
        except Exception as e:
            msg = str(e)
            if any(m in msg for m in _RETRY_MARKERS) and attempt < EMBED_MAX_RETRIES:
                time.sleep(EMBED_RETRY_SLEEP * attempt)
                continue
            raise


def embed_query(text: str) -> list[float]:
    return embed(text, task_type="RETRIEVAL_QUERY")
