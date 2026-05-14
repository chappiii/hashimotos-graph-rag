"""
Pre-ingestion pass 2: build a canonical claim registry across all 115 papers.

Depends on entity_registry.json (run entity_registry.py first).

For each relation in every *_relations.json:
  - Normalizes source/target canonical names via the entity registry forms_index.
  - Generates claim_signature = "source_canonical|relation_type|target_canonical".
  - Aggregates polarity counts, max certainty, paper_ids, and all evidence instances.

Output: kg_ingestion/pre_ingestion/output/claim_registry.json
  {
    "<claim_signature>": {
      "relation_type", "source", "target",
      "polarity_counts", "certainty_max",
      "paper_ids", "paper_count",
      "evidence": [ { evidence_id, evidence_text, paper_id, section_type,
                      claim_polarity, claim_certainty } ]
    }
  }

Run: uv run python -m kg_ingestion.pre_ingestion.claim_registry
"""

import json
import re
import uuid
from pathlib import Path

from kg_ingestion.config.kg_config import (
    CLAIM_REGISTRY_PATH,
    ENTITY_REGISTRY_PATH,
    ENTITY_RELATION_DIR,
    OUTPUT_DIR,
)

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
    s = re.sub(r"_(entities|relations)$", "", file_stem)
    s = re.sub(r"^\d+-", "", s)
    s = re.sub(r"^[\d.]+_?", "", s).lower()
    for keyword, label in _SECTION_MAP.items():
        if keyword in s:
            return label
    return "OTHER"


def _resolve(canonical_name: str, forms_index: dict[str, str]) -> str:
    """Return the registry canonical form for a name, or the name itself if not found."""
    return forms_index.get(canonical_name.lower(), canonical_name)


def _update_certainty_max(current: str, incoming: str) -> str:
    if _CERTAINTY_RANK.get(incoming, -1) > _CERTAINTY_RANK.get(current, -1):
        return incoming
    return current


def build_registry(forms_index: dict[str, str], entities: dict) -> dict:
    registry: dict[str, dict] = {}

    paper_dirs = sorted(
        (d for d in ENTITY_RELATION_DIR.iterdir() if d.is_dir()),
        key=lambda p: int(p.name),
    )

    total_seen = 0
    skipped = 0
    for paper_dir in paper_dirs:
        paper_id = int(paper_dir.name)
        for rel_file in sorted(paper_dir.glob("*_relations.json")):
            section_type = _section_type(rel_file.stem)
            try:
                data = json.loads(rel_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"  [WARN] bad JSON: {rel_file}")
                continue

            for rel in data.get("relations", []):
                src = rel.get("source_entity", {})
                tgt = rel.get("target_entity", {})

                if not isinstance(src, dict) or not isinstance(tgt, dict):
                    skipped += 1
                    continue

                src_name = _resolve(src.get("canonical_name", "").strip(), forms_index)
                tgt_name = _resolve(tgt.get("canonical_name", "").strip(), forms_index)
                rel_type = rel.get("relation_type", "").strip()

                if not src_name or not tgt_name or not rel_type:
                    skipped += 1
                    continue

                total_seen += 1
                sig = f"{src_name}|{rel_type}|{tgt_name}"

                polarity  = rel.get("claim_polarity",  "positive")
                certainty = rel.get("claim_certainty", "moderate")
                evidence_text = rel.get("evidence", "").strip()

                if sig not in registry:
                    registry[sig] = {
                        "relation_type": rel_type,
                        "source": {
                            "canonical_name": src_name,
                            "entity_type":   src.get("entity_type", ""),
                        },
                        "target": {
                            "canonical_name": tgt_name,
                            "entity_type":   tgt.get("entity_type", ""),
                        },
                        "polarity_counts": {
                            "positive": 0, "negative": 0,
                            "hypothetical": 0, "uncertain": 0, "mixed": 0,
                        },
                        "certainty_max": certainty,
                        "paper_ids":   [],
                        "paper_count": 0,
                        "evidence":    [],
                    }

                claim = registry[sig]

                if polarity in claim["polarity_counts"]:
                    claim["polarity_counts"][polarity] += 1

                claim["certainty_max"] = _update_certainty_max(claim["certainty_max"], certainty)

                if paper_id not in claim["paper_ids"]:
                    claim["paper_ids"].append(paper_id)
                    claim["paper_count"] = len(claim["paper_ids"])

                if evidence_text:
                    claim["evidence"].append({
                        "evidence_id":    str(uuid.uuid4()),
                        "evidence_text":  evidence_text,
                        "paper_id":       paper_id,
                        "section_type":   section_type,
                        "claim_polarity": polarity,
                        "claim_certainty": certainty,
                    })

    print(
        f"Seen {total_seen} relations → {len(registry)} unique claims. "
        f"Skipped {skipped} malformed."
    )
    return registry


def main() -> None:
    if not ENTITY_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"entity_registry.json not found at {ENTITY_REGISTRY_PATH}. "
            "Run entity_registry.py first."
        )

    entity_data  = json.loads(ENTITY_REGISTRY_PATH.read_text(encoding="utf-8"))
    forms_index  = entity_data["forms_index"]
    entities     = entity_data["entities"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building claim registry …")
    registry = build_registry(forms_index, entities)
    CLAIM_REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved → {CLAIM_REGISTRY_PATH}")


if __name__ == "__main__":
    main()
