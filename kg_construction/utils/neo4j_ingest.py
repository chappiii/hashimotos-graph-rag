import os
import json
import re
from pathlib import Path

from kg_construction.utils.normalize import normalize_text
from kg_construction.utils.fuzzy import find_similar
from kg_construction.utils.parse_filename import parse_relation_filename


def _sanitize_label(text):
    """'Diseases & Conditions' → 'diseases_conditions'"""
    label = re.sub(r"\W+", "_", text.lower()).strip("_")
    return label


def _collect_entities_and_relations(root_dir):
    """Walk relation JSONs, aggregate entities and merge relationships."""
    entity_info = {}       # canonical_name → {type, paper_ids, sections}
    relation_info = {}     # (source, target, rel_type) → {evidence, paper_ids, sections, key_properties}

    root = Path(root_dir)
    paper_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda d: int(d.name) if d.name.isdigit() else 0,
    )

    for paper_dir in paper_dirs:
        paper_id = paper_dir.name

        for file in paper_dir.iterdir():
            if not file.name.endswith("_relations.json"):
                continue

            parsed = parse_relation_filename(file.name, paper_id)
            if not parsed:
                print(f"Skipping unparseable filename: {file.name}")
                continue

            section = parsed["section"]

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            relations = data.get("relations", [])

            for rel in relations:
                evidence = rel.get("evidence", "").strip()
                rel_type = rel.get("relation_type", "").strip()
                key_props = rel.get("key_properties", {})
                if not rel_type:
                    continue

                source = rel.get("source_entity")
                target = rel.get("target_entity")
                if not isinstance(source, dict) or not isinstance(target, dict):
                    continue

                source_name = normalize_text(source.get("canonical_name", "").strip())
                source_type = source.get("entity_type", "").strip()
                target_name = normalize_text(target.get("canonical_name", "").strip())
                target_type = target.get("entity_type", "").strip()

                if not source_name or not source_type or not target_name or not target_type:
                    continue

                # fuzzy dedup against existing entity names
                source_match = find_similar(source_name, list(entity_info.keys()))
                source_key = source_match if source_match else source_name

                target_match = find_similar(target_name, list(entity_info.keys()))
                target_key = target_match if target_match else target_name

                # aggregate entity info
                for key, etype in [(source_key, source_type), (target_key, target_type)]:
                    if key not in entity_info:
                        entity_info[key] = {
                            "type": etype,
                            "paper_ids": set(),
                            "sections": set(),
                        }
                    entity_info[key]["paper_ids"].add(paper_id)
                    entity_info[key]["sections"].add(section)

                # merge relationships by (source, target, rel_type)
                rel_key = (source_key, target_key, rel_type)
                if rel_key not in relation_info:
                    relation_info[rel_key] = {
                        "evidence": [],
                        "paper_ids": set(),
                        "sections": set(),
                        "key_properties": [],
                    }

                if evidence:
                    relation_info[rel_key]["evidence"].append(evidence)
                relation_info[rel_key]["paper_ids"].add(paper_id)
                relation_info[rel_key]["sections"].add(section)
                if key_props:
                    relation_info[rel_key]["key_properties"].append(key_props)

    return entity_info, relation_info


def _batch_create_entities(tx, batch):
    for name, label, paper_ids, sections in batch:
        tx.run(
            f"MERGE (e:`{label}` {{name: $name}}) "
            f"SET e.paper_ids = $paper_ids, e.sections = $sections",
            name=name,
            paper_ids=paper_ids,
            sections=sections,
        )


def _batch_create_relationships(tx, batch):
    for source, target, rel_type, evidence, paper_ids, sections, key_properties in batch:
        tx.run(
            f"MATCH (a {{name: $source}}) "
            f"MATCH (b {{name: $target}}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
            f"SET r.evidence = $evidence, r.paper_ids = $paper_ids, "
            f"r.sections = $sections, r.key_properties = $key_properties",
            source=source,
            target=target,
            evidence=evidence,
            paper_ids=paper_ids,
            sections=sections,
            key_properties=[json.dumps(kp) for kp in key_properties],
        )


def ingest_to_neo4j(root_dir, driver, batch_size=100):
    print("Collecting entities and relationships...")
    entity_info, relation_info = _collect_entities_and_relations(root_dir)
    print(f"Found {len(entity_info)} entities, {len(relation_info)} relationships.")

    # prepare entity batches
    entity_batches = [
        (name, _sanitize_label(info["type"]), list(info["paper_ids"]), list(info["sections"]))
        for name, info in entity_info.items()
    ]

    # prepare relationship batches
    rel_batches = [
        (
            source, target, _sanitize_label(rel_type),
            data["evidence"],
            list(data["paper_ids"]),
            list(data["sections"]),
            data["key_properties"],
        )
        for (source, target, rel_type), data in relation_info.items()
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
