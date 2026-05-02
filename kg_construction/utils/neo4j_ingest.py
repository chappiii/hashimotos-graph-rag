import json
import re
from pathlib import Path
from collections import defaultdict

from kg_construction.utils.text_utils import (
    find_similar,
    normalize_text,
    parse_chunk_filename,
)
from kg_construction.config.kg_construction_config import ENTITY_LABELS


def _sanitize_label(text):
    """Cypher-safe identifier (used for relation types)."""
    return re.sub(r"\W+", "_", text.lower()).strip("_")


def setup_constraints(driver, entity_labels):
    """One uniqueness constraint per label on `name` — speeds MERGE, prevents dupes."""
    with driver.session() as session:
        for label in entity_labels.values():
            session.run(
                f"CREATE CONSTRAINT {label}_name_unique IF NOT EXISTS "
                f"FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
            )
    print(f"Setup {len(entity_labels)} uniqueness constraints.")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {path}")
            return None


def _collect_entities_and_relations(root_dir):
    entity_info = {}              # (label, name) -> {evidence, key_properties, paper_ids, sections}
    relation_info = {}            # (src_label, src_name, tgt_label, tgt_name, rel_type) -> {...}
    names_by_label = defaultdict(list)
    skipped_entity_types = set()

    root = Path(root_dir)
    paper_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
    )

    # ---- Pass 1: entities ----
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name

        for file in paper_dir.iterdir():
            if not file.name.endswith("_entities.json"):
                continue

            parsed = parse_chunk_filename(file.name, paper_id)
            if not parsed:
                print(f"Skipping unparseable filename: {file.name}")
                continue
            section = parsed["section"]

            data = _load_json(file)
            if data is None:
                continue

            for ent in data.get("entities", []):
                entity_type = (ent.get("entity_type") or "").strip()
                label = ENTITY_LABELS.get(entity_type)
                if not label:
                    if entity_type:
                        skipped_entity_types.add(entity_type)
                    continue

                raw_name = ent.get("canonical_name") or ""
                if not isinstance(raw_name, str):
                    continue
                name = normalize_text(raw_name.strip())
                if not name:
                    continue

                # fuzzy dedup within the same label only
                match = find_similar(name, names_by_label[label])
                key_name = match if match else name

                key = (label, key_name)
                if key not in entity_info:
                    entity_info[key] = {
                        "evidence": [],
                        "key_properties": [],
                        "paper_ids": set(),
                        "sections": set(),
                    }
                    names_by_label[label].append(key_name)

                evidence = ent.get("evidence")
                if isinstance(evidence, str) and evidence.strip():
                    entity_info[key]["evidence"].append(evidence.strip())

                key_props = ent.get("key_properties")
                if isinstance(key_props, dict) and key_props:
                    entity_info[key]["key_properties"].append(
                        json.dumps(key_props, ensure_ascii=False)
                    )

                entity_info[key]["paper_ids"].add(paper_id)
                entity_info[key]["sections"].add(section)

    if skipped_entity_types:
        print(f"Skipped unknown entity types: {sorted(skipped_entity_types)}")

    # name -> label index for relation matching (first label seen wins on collision)
    name_to_label = {}
    for (lbl, name) in entity_info.keys():
        name_to_label.setdefault(name, lbl)
    all_names = list(name_to_label.keys())

    # ---- Pass 2: relations ----
    unmatched_relations = 0
    for paper_dir in paper_dirs:
        paper_id = paper_dir.name

        for file in paper_dir.iterdir():
            if not file.name.endswith("_relations.json"):
                continue

            parsed = parse_chunk_filename(file.name, paper_id)
            if not parsed:
                print(f"Skipping unparseable filename: {file.name}")
                continue
            section = parsed["section"]

            data = _load_json(file)
            if data is None:
                continue

            for rel in data.get("relations", []):
                rel_type = (rel.get("relation_type") or "").strip()
                if not rel_type:
                    continue

                source = rel.get("source_entity")
                target = rel.get("target_entity")
                if not isinstance(source, dict) or not isinstance(target, dict):
                    continue

                src_raw = source.get("canonical_name") or ""
                tgt_raw = target.get("canonical_name") or ""
                if not isinstance(src_raw, str) or not isinstance(tgt_raw, str):
                    continue

                src_name = normalize_text(src_raw.strip())
                tgt_name = normalize_text(tgt_raw.strip())
                if not src_name or not tgt_name:
                    continue

                src_match = find_similar(src_name, all_names)
                tgt_match = find_similar(tgt_name, all_names)
                if not src_match or not tgt_match:
                    unmatched_relations += 1
                    continue

                src_label = name_to_label[src_match]
                tgt_label = name_to_label[tgt_match]

                rel_key = (src_label, src_match, tgt_label, tgt_match, rel_type)
                if rel_key not in relation_info:
                    relation_info[rel_key] = {
                        "evidence": [],
                        "key_properties": [],
                        "paper_ids": set(),
                        "sections": set(),
                    }

                evidence = rel.get("evidence")
                if isinstance(evidence, str) and evidence.strip():
                    relation_info[rel_key]["evidence"].append(evidence.strip())

                key_props = rel.get("key_properties")
                if isinstance(key_props, dict) and key_props:
                    relation_info[rel_key]["key_properties"].append(
                        json.dumps(key_props, ensure_ascii=False)
                    )

                relation_info[rel_key]["paper_ids"].add(paper_id)
                relation_info[rel_key]["sections"].add(section)

    if unmatched_relations:
        print(f"Skipped {unmatched_relations} relations with unmatched source/target.")

    return entity_info, relation_info


def _batch_create_entities(tx, batch):
    for label, name, evidence, key_properties, paper_ids, sections in batch:
        tx.run(
            f"MERGE (e:`{label}` {{name: $name}}) "
            f"SET e.evidence = $evidence, "
            f"e.key_properties = $key_properties, "
            f"e.paper_ids = $paper_ids, "
            f"e.sections = $sections",
            name=name,
            evidence=evidence,
            key_properties=key_properties,
            paper_ids=paper_ids,
            sections=sections,
        )


def _batch_create_relationships(tx, batch):
    for src_label, source, tgt_label, target, rel_type, evidence, key_properties, paper_ids, sections in batch:
        tx.run(
            f"MATCH (a:`{src_label}` {{name: $source}}) "
            f"MATCH (b:`{tgt_label}` {{name: $target}}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
            f"SET r.evidence = $evidence, "
            f"r.key_properties = $key_properties, "
            f"r.paper_ids = $paper_ids, "
            f"r.sections = $sections",
            source=source,
            target=target,
            evidence=evidence,
            key_properties=key_properties,
            paper_ids=paper_ids,
            sections=sections,
        )


def ingest_to_neo4j(root_dir, driver, batch_size=100):
    print("Collecting entities and relationships...")
    entity_info, relation_info = _collect_entities_and_relations(root_dir)
    print(f"Found {len(entity_info)} entities, {len(relation_info)} relationships.")

    entity_batches = [
        (
            label, name,
            info["evidence"],
            info["key_properties"],
            list(info["paper_ids"]),
            list(info["sections"]),
        )
        for (label, name), info in entity_info.items()
    ]

    rel_batches = [
        (
            src_label, source, tgt_label, target,
            _sanitize_label(rel_type),
            data["evidence"],
            data["key_properties"],
            list(data["paper_ids"]),
            list(data["sections"]),
        )
        for (src_label, source, tgt_label, target, rel_type), data in relation_info.items()
    ]

    with driver.session() as session:
        for i in range(0, len(entity_batches), batch_size):
            batch = entity_batches[i : i + batch_size]
            session.execute_write(_batch_create_entities, batch)
            print(f"Entities: {min(i + batch_size, len(entity_batches))}/{len(entity_batches)}")

        for i in range(0, len(rel_batches), batch_size):
            batch = rel_batches[i : i + batch_size]
            session.execute_write(_batch_create_relationships, batch)
            print(f"Relationships: {min(i + batch_size, len(rel_batches))}/{len(rel_batches)}")

    print("Neo4j ingestion complete.")
