"""
Pre-ingestion pass 1: build a canonical entity registry across all 115 papers.

Reads all *_entities.json files, deduplicates by:
  1. Exact match on lowercased canonical_name (via forms_index)
  2. Fuzzy match with rapidfuzz token_sort_ratio >= FUZZY_THRESHOLD

Output: kg_ingestion/pre_ingestion/output/entity_registry.json
  {
    "entities": {
      "<canonical_name>": {
        "entity_type", "aliases", "key_properties",
        "paper_ids", "section_types", "surface_forms"
      }
    },
    "forms_index": { "<lowercased_form>": "<canonical_name>" }
  }

Run: uv run python -m kg_ingestion.pre_ingestion.entity_registry
"""

import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

from kg_ingestion.config.kg_config import (
    ENTITY_RELATION_DIR,
    ENTITY_REGISTRY_PATH,
    FUZZY_THRESHOLD,
    OUTPUT_DIR,
    PROTECTED_EXACT_ONLY,
)

_PROTECTED_LOWER: set[str] = {p.lower() for p in PROTECTED_EXACT_ONLY}

# Maps keywords in the section name to a canonical section type label.
_SECTION_MAP = {
    "abstract":     "ABSTRACT",
    "introduction": "INTRODUCTION",
    "background":   "INTRODUCTION",
    "method":       "METHODS",
    "material":     "METHODS",
    "result":       "RESULTS",
    "finding":      "RESULTS",
    "discussion":   "DISCUSSION",
    "conclusion":   "CONCLUSION",
}

_CERTAINTY_RANK = {"high": 2, "moderate": 1, "low": 0}


def _section_type(file_stem: str) -> str:
    """Infer ABSTRACT / INTRODUCTION / ... from a filename stem like '3-2._materials_and_methods_entities'."""
    s = re.sub(r"_(entities|relations)$", "", file_stem)
    s = re.sub(r"^\d+-", "", s)          # strip leading "{index}-"
    s = re.sub(r"^[\d.]+_?", "", s)      # strip leading "{number}._"
    s = s.lower()
    for keyword, label in _SECTION_MAP.items():
        if keyword in s:
            return label
    return "OTHER"


def _safe_aliases(aliases: list[str], canonical: str) -> list[str]:
    """Drop aliases that are protected terms for a *different* entity.

    Prevents e.g. 'hyperthyroidism' ending up as an alias of 'Hypothyroidism'
    because the LLM confused the two in a single chunk.
    """
    canonical_lower = canonical.lower()
    return [
        a for a in aliases
        if a and not (a.lower() in _PROTECTED_LOWER and a.lower() != canonical_lower)
    ]


def _new_record(entity: dict, paper_id: int, section_type: str, canonical: str) -> dict:
    return {
        "entity_type":    entity.get("entity_type", ""),
        "aliases":        list(set(_safe_aliases(entity.get("aliases", []), canonical))),
        "key_properties": entity.get("key_properties", {}),
        "paper_ids":      [paper_id],
        "section_types":  [section_type],
        "surface_forms":  [entity.get("surface_form", "")],
    }


def _merge_into(record: dict, entity: dict, paper_id: int, section_type: str, canonical: str) -> None:
    """Merge a new entity occurrence into an existing registry record."""
    for alias in _safe_aliases(entity.get("aliases", []), canonical):
        if alias not in record["aliases"]:
            record["aliases"].append(alias)

    sf = entity.get("surface_form", "")
    if sf and sf not in record["surface_forms"]:
        record["surface_forms"].append(sf)

    if paper_id not in record["paper_ids"]:
        record["paper_ids"].append(paper_id)

    if section_type not in record["section_types"]:
        record["section_types"].append(section_type)

    # Keep key_properties from the occurrence that has more non-null fields.
    existing_kp = record["key_properties"]
    incoming_kp = entity.get("key_properties", {})
    existing_filled = sum(1 for v in existing_kp.values() if v is not None)
    incoming_filled = sum(1 for v in incoming_kp.values() if v is not None)
    if incoming_filled > existing_filled:
        record["key_properties"] = incoming_kp


def _build_forms_index(registry: dict[str, dict]) -> dict[str, str]:
    """Build forms_index after registry is complete.

    Resolves alias collisions (same abbreviation → two entities) by keeping
    the mapping to whichever entity has more paper_ids. Logs every collision.
    """
    # alias_lower → {canonical → paper_count}
    candidates: dict[str, dict[str, int]] = {}

    for canonical, record in registry.items():
        for form in [canonical] + record["aliases"] + record["surface_forms"]:
            if not form:
                continue
            fl = form.lower()
            candidates.setdefault(fl, {})[canonical] = len(record["paper_ids"])

    forms_index: dict[str, str] = {}
    for form_lower, mapping in candidates.items():
        if len(mapping) == 1:
            forms_index[form_lower] = next(iter(mapping))
        else:
            # Collision: pick the canonical with the most papers.
            winner = max(mapping, key=mapping.get)
            losers = [c for c in mapping if c != winner]
            print(f"  [COLLISION] {form_lower!r} → kept {winner!r}, dropped {losers}")
            forms_index[form_lower] = winner

    return forms_index


def build_registry() -> dict:
    registry: dict[str, dict]   = {}   # canonical_name → record
    # forms_index built here only tracks canonical names (for fuzzy-cache hits).
    # Aliases are added in _build_forms_index after the registry is complete.
    forms_index_canonical: dict[str, str] = {}

    paper_dirs = sorted(
        (d for d in ENTITY_RELATION_DIR.iterdir() if d.is_dir()),
        key=lambda p: int(p.name),
    )

    total_seen = 0
    for paper_dir in paper_dirs:
        paper_id = int(paper_dir.name)
        for entity_file in sorted(paper_dir.glob("*_entities.json")):
            section_type = _section_type(entity_file.stem)
            try:
                data = json.loads(entity_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"  [WARN] bad JSON: {entity_file}")
                continue

            for ent in data.get("entities", []):
                canonical = ent.get("canonical_name", "").strip()
                if not canonical:
                    continue
                total_seen += 1
                canonical_lower = canonical.lower()

                # 1. Exact match via canonical forms_index.
                if canonical_lower in forms_index_canonical:
                    key = forms_index_canonical[canonical_lower]
                    _merge_into(registry[key], ent, paper_id, section_type, key)
                    continue

                # 2. Fuzzy match: blocked for protected terms on both sides.
                is_protected = canonical_lower in _PROTECTED_LOWER
                match_result = None
                if not is_protected and registry:
                    match_result = process.extractOne(
                        canonical,
                        [k for k in registry if k.lower() not in _PROTECTED_LOWER],
                        scorer=fuzz.token_sort_ratio,
                        score_cutoff=FUZZY_THRESHOLD,
                    )

                if match_result:
                    key = match_result[0]
                    _merge_into(registry[key], ent, paper_id, section_type, key)
                    forms_index_canonical[canonical_lower] = key
                else:
                    # New unique entity.
                    registry[canonical] = _new_record(ent, paper_id, section_type, canonical)
                    forms_index_canonical[canonical_lower] = canonical

    print(f"Seen {total_seen} raw entity mentions → {len(registry)} unique entities after dedup.")

    print("Resolving alias collisions …")
    forms_index = _build_forms_index(registry)
    return {"entities": registry, "forms_index": forms_index}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building entity registry …")
    registry = build_registry()
    ENTITY_REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved → {ENTITY_REGISTRY_PATH}")


if __name__ == "__main__":
    main()
