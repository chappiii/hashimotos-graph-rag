from extract_entity_relation.prompts.extract_entity_prompt import extract_entity_prompt, ENTITY_SECTION_RULES
from extract_entity_relation.prompts.extract_relation import extract_relation_prompt, RELATION_SECTION_RULES
from extract_entity_relation.utils.clients import MODELS


def generate_content(model_tag: str, prompt: str) -> str:
    _, fn = MODELS[model_tag]
    return fn(prompt)


def build_entity_prompt(paper_text: str, section_type: str = "OTHER") -> str:
    return extract_entity_prompt.format(
        text=paper_text,
        section_type=section_type,
        entity_section_rules=ENTITY_SECTION_RULES.get(section_type, ENTITY_SECTION_RULES["OTHER"]),
    )


def build_relation_prompt(paper_text: str, entities: list[dict], section_type: str = "OTHER") -> str:
    entities_formatted = "\n".join(
        [f"- {e['canonical_name']} ({e['entity_type']})" for e in entities]
    )
    return extract_relation_prompt.format(
        entities=entities_formatted,
        text=paper_text,
        section_type=section_type,
        relation_section_rules=RELATION_SECTION_RULES.get(section_type, RELATION_SECTION_RULES["OTHER"]),
    )
