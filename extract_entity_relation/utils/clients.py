import asyncio

from google import genai
from google.genai import types as genai_types
import anthropic
from openai import AsyncOpenAI

from extract_entity_relation.config.extract_entity_relation_config import (
    GEMINI_MODEL, CLAUDE_MODEL, OPENAI_MODEL,
    GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
    MAX_OUTPUT_TOKENS, MAX_CONCURRENCY, MAX_RETRIES
)


# limit total in-flight API calls across all papers/chunks
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


async def _retry(fn, *args):
    """Retry with exponential backoff on rate-limit"""
    for attempt in range(MAX_RETRIES):
        try:
            return await fn(*args)
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "rate" in err_str or "resource" in err_str
            if not is_rate_limit or attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"    Rate limited, retrying in {wait}s... (attempt {attempt + 1})")
            await asyncio.sleep(wait)


async def call_gemini(prompt: str) -> str:
    async with _semaphore:
        return await _retry(_call_gemini_inner, prompt)


async def _call_gemini_inner(prompt: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = await client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0),
    )
    if response.text is not None:
        return response.text
    reason = "UNKNOWN"
    if response.candidates:
        reason = str(response.candidates[0].finish_reason)
    raise ValueError(f"Gemini generation failed (finish_reason={reason})")


async def call_claude(prompt: str) -> str:
    async with _semaphore:
        return await _retry(_call_claude_inner, prompt)


async def _call_claude_inner(prompt: str) -> str:
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    chunks = []
    async with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            chunks.append(text)
    return "".join(chunks)


async def call_openai(prompt: str) -> str:
    async with _semaphore:
        return await _retry(_call_openai_inner, prompt)


async def _call_openai_inner(prompt: str) -> str:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    r = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_completion_tokens=MAX_OUTPUT_TOKENS,
    )
    return r.choices[0].message.content


MODELS = {
    "gemini": (GEMINI_MODEL, call_gemini),
    "claude": (CLAUDE_MODEL, call_claude),
    "gpt": (OPENAI_MODEL, call_openai),
}
