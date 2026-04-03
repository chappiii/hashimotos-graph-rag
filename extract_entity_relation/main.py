import os
import sys
import time

from extract_entity_relation.config.extract_entity_relation_config import (
    GEMINI_API_KEY, DATA_DIR, OUTPUT_DIR,
)
from extract_entity_relation.utils.gemini_client import configure_gemini
from extract_entity_relation.utils.extractor import (
    generate_content, build_entity_prompt, build_relation_prompt,
)
from extract_entity_relation.utils.file_utils import (
    read_content, get_sorted_chunks, load_entities_from_json, save_result,
)


def get_sorted_paper_dirs() -> list[str]:
    return sorted(
        [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))],
        key=lambda d: int(d),
    )


def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    client = configure_gemini(GEMINI_API_KEY)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    paper_dirs = get_sorted_paper_dirs()
    total = len(paper_dirs)
    print(f"\nEntity-Relation Extraction — {total} papers found\n")

    success, failed = 0, 0

    for idx, paper_id in enumerate(paper_dirs, 1):
        paper_path = os.path.join(DATA_DIR, paper_id)
        paper_output_dir = os.path.join(OUTPUT_DIR, paper_id)
        os.makedirs(paper_output_dir, exist_ok=True)

        chunk_files = get_sorted_chunks(paper_path)
        print(f"[{idx}/{total}] Paper {paper_id} ({len(chunk_files)} sections)")

        for chunk_file in chunk_files:
            chunk_name = chunk_file.rsplit(".", 1)[0]  # strip .md
            section_text = read_content(os.path.join(paper_path, chunk_file))

            if not section_text:
                continue

            try:
                # Entity extraction
                print(f"  Extracting entities: {chunk_name}")
                entity_prompt = build_entity_prompt(section_text)
                entity_response = generate_content(client, entity_prompt)
                entity_output = os.path.join(paper_output_dir, f"{chunk_name}_entities.json")
                save_result(entity_response, entity_output)
                time.sleep(5)

                # Load entities for relation extraction
                entities = load_entities_from_json(entity_output)
                if not entities:
                    print(f"    No entities found, skipping relations.")
                    continue

                # Relation extraction
                print(f"  Extracting relations: {chunk_name}")
                relation_prompt = build_relation_prompt(section_text, entities)
                relation_response = generate_content(client, relation_prompt)
                relation_output = os.path.join(paper_output_dir, f"{chunk_name}_relations.json")
                save_result(relation_response, relation_output)
                time.sleep(5)

            except Exception as e:
                print(f"    FAILED: {chunk_name} — {e}")
                failed += 1
                continue

        success += 1

    print(f"\nDone. {success} papers processed, {failed} section failures out of {total} papers.")


if __name__ == "__main__":
    main()
