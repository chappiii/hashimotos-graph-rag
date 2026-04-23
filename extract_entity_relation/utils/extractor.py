from extract_entity_relation.prompts.extract_entity_prompt import extract_entity_prompt
from extract_entity_relation.prompts.extract_relation import extract_relation_prompt
from extract_entity_relation.utils.clients import MODELS


async def generate_content(model_tag: str, prompt: str) -> str:
    _, fn = MODELS[model_tag]
    return await fn(prompt)


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
