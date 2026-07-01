import os

from google import genai
from google.genai import types as genai_types

from generator.config.gen_config import GENERATOR_MODEL, GENERATOR_MAX_TOKENS
from generator.prompt import SYSTEM_PROMPT, build_user_prompt


def generate(
    query: str,
    graph_results: list[dict],
    vector_results: list[dict],
    options: dict[str, str] | None = None,
    temperature: float = 0.2,
) -> str:
    user_prompt = build_user_prompt(query, graph_results, vector_results, options)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GENERATOR_MODEL,
        contents=full_prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=GENERATOR_MAX_TOKENS,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=1024),
        ),
    )

    finish_reason = None
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
    if getattr(response, "usage_metadata", None):
        um = response.usage_metadata
        print(
            f"  [generator] tokens: prompt={um.prompt_token_count}, "
            f"output={um.candidates_token_count}, total={um.total_token_count} | "
            f"finish={finish_reason}"
        )

    if response.text is not None:
        return response.text
    raise ValueError(f"Generator LLM call failed (finish_reason={finish_reason})")
