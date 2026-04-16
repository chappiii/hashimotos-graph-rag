import re


def parse_relation_filename(filename, paper_id):
    """Parse '1-abstract_relations.json' → {'paper_id': '1', 'section': 'abstract'}"""
    match = re.match(r"(\d+)-(.+)_relations\.json", filename)
    if not match:
        return None

    section_name = match.group(2).replace("_", " ").strip()
    return {
        "paper_id": paper_id,
        "section": section_name,
    }
