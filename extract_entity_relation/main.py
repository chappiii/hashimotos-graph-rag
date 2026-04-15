import argparse
import os
import sys
import time

from extract_entity_relation.config.extract_entity_relation_config import (
    GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
    DATA_DIR, OUTPUT_DIR,
)
from extract_entity_relation.utils.clients import MODELS
from extract_entity_relation.utils.extractor import (
    generate_content, build_entity_prompt, build_relation_prompt,
)
from extract_entity_relation.utils.file_utils import (
    read_content, get_sorted_chunks, load_entities_from_json, save_result,
)


API_KEYS = {
    "gemini": ("GEMINI_API_KEY", GEMINI_API_KEY),
    "claude": ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
    "gpt": ("OPENAI_API_KEY", OPENAI_API_KEY),
}


def get_sorted_paper_dirs() -> list[str]:
    return sorted(
        [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))],
        key=lambda d: int(d),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Which model to run: gemini | claude | gpt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_tag = args.model

    env_name, api_key = API_KEYS[model_tag]
    if not api_key:
        print(f"Error: {env_name} not found in .env")
        sys.exit(1)

    model_output_dir = os.path.join(OUTPUT_DIR, model_tag)
    os.makedirs(model_output_dir, exist_ok=True)

    paper_dirs = get_sorted_paper_dirs()
    total = len(paper_dirs)
    print(f"\nEntity-Relation Extraction [{model_tag}] — {total} papers found\n")

    success, failed = 0, 0

    for idx, paper_id in enumerate(paper_dirs, 1):
        paper_path = os.path.join(DATA_DIR, paper_id)
        paper_output_dir = os.path.join(model_output_dir, paper_id)
        os.makedirs(paper_output_dir, exist_ok=True)

        chunk_files = get_sorted_chunks(paper_path)
        print(f"[{idx}/{total}] Paper {paper_id} ({len(chunk_files)} sections)")

        for chunk_file in chunk_files:
            chunk_name = chunk_file.rsplit(".", 1)[0]
            entity_output = os.path.join(paper_output_dir, f"{chunk_name}_entities.json")
            relation_output = os.path.join(paper_output_dir, f"{chunk_name}_relations.json")

            if os.path.exists(entity_output) and os.path.exists(relation_output):
                print(f"  Skipping (already processed): {chunk_name}")
                continue

            section_text = read_content(os.path.join(paper_path, chunk_file))

            if not section_text:
                continue

            try:
                print(f"  Extracting entities: {chunk_name}")
                entity_prompt = build_entity_prompt(section_text)
                entity_response = generate_content(model_tag, entity_prompt)
                save_result(entity_response, entity_output)
                # time.sleep(5)

                entities = load_entities_from_json(entity_output)
                if not entities:
                    print(f"    No entities found, skipping relations.")
                    continue

                print(f"  Extracting relations: {chunk_name}")
                relation_prompt = build_relation_prompt(section_text, entities)
                relation_response = generate_content(model_tag, relation_prompt)
                save_result(relation_response, relation_output)
                # time.sleep(5)

            except Exception as e:
                print(f"    FAILED: {chunk_name} — {e}")
                failed += 1
                continue

        success += 1

    print(f"\nDone. {success} papers processed, {failed} section failures out of {total} papers.")


if __name__ == "__main__":
    main()
