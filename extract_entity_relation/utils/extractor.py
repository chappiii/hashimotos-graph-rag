from google import genai
from google.genai import types
from extract_entity_relation.config.extract_entity_relation_config import GEMINI_MODEL
from extract_entity_relation.prompts.extract_entity_prompt import extract_entity_prompt
from extract_entity_relation.prompts.extract_relation import extract_relation_prompt


def generate_content(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    if response.text is not None:
        return response.text

    reason = "UNKNOWN"
    if response.candidates:
        reason = str(response.candidates[0].finish_reason)
    raise ValueError(f"Generation failed (finish_reason={reason})")


def build_entity_prompt(paper_text: str) -> str:
    return extract_entity_prompt + f'\n"""\n{paper_text}\n"""'


def build_relation_prompt(paper_text: str, entities: list[dict]) -> str:
    entities_formatted = "\n".join(
        [f"- {e['canonical_name']} ({e['entity_type']})" for e in entities]
    )
    return extract_relation_prompt.format(
        entities=f"**Pre-extracted Entities:**\n{entities_formatted}",
        text=paper_text,
    )
