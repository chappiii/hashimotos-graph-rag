import re
import unicodedata

from rapidfuzz import fuzz, process


def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[\'\"‘’“”`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()

    if text == "ht":
        text = "hashimatos thyroiditis"

    return text


def find_similar(value, existing_list, threshold=85):
    if not existing_list:
        return None
    match, score, _ = process.extractOne(
        value, existing_list, scorer=fuzz.token_sort_ratio
    )
    return match if score >= threshold else None


def parse_chunk_filename(filename, paper_id):
    """Parse '1-abstract_entities.json' or '1-abstract_relations.json'."""
    match = re.match(r"(\d+)-(.+)_(entities|relations)\.json", filename)
    if not match:
        return None

    section = match.group(2).replace("_", " ").strip()
    return {
        "paper_id": paper_id,
        "section": section,
        "kind": match.group(3),
    }
