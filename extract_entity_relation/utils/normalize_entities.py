from extract_entity_relation.utils.ontology_lookup import lookup_ontology


def normalize_entity(entity: dict) -> dict:
    canonical = entity.get("canonical_name")
    entity_type = entity.get("entity_type")
    if canonical:
        entity["ontology"] = lookup_ontology(canonical, entity_type)
    else:
        entity["ontology"] = None
    return entity


def normalize_entities(parsed: dict) -> dict:
    """Enrich all entities in a parsed extraction result in-place."""
    entities = parsed.get("entities", [])
    parsed["entities"] = [normalize_entity(e) for e in entities]
    return parsed
