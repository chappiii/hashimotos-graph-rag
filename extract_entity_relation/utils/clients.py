from google import genai
from google.genai import types as genai_types
import anthropic
from openai import OpenAI

from extract_entity_relation.config.extract_entity_relation_config import (
    GEMINI_MODEL, CLAUDE_MODEL, OPENAI_MODEL,
    GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
    MAX_OUTPUT_TOKENS,
)


def call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
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


def call_claude(prompt: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    chunks = []
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            chunks.append(text)
    return "".join(chunks)


def call_openai(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    r = client.chat.completions.create(
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
