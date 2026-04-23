import argparse
import asyncio
import os
import sys
import time

from extract_entity_relation.config.extract_entity_relation_config import (
    GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
    DATA_DIR, OUTPUT_DIR, MAX_CONCURRENCY,
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


async def process_chunk(
    model_tag: str,
    paper_id: str,
    chunk_file: str,
    paper_path: str,
    paper_output_dir: str,
) -> bool:
    """Process a single chunk: extract entities, then relations. Returns True on success."""
    chunk_name = chunk_file.rsplit(".", 1)[0]
    entity_output = os.path.join(paper_output_dir, f"{chunk_name}_entities.json")
    relation_output = os.path.join(paper_output_dir, f"{chunk_name}_relations.json")

    if os.path.exists(entity_output) and os.path.exists(relation_output):
        print(f"  [Paper {paper_id}] Skipping (already processed): {chunk_name}")
        return True

    section_text = read_content(os.path.join(paper_path, chunk_file))
    if not section_text:
        return True

    try:
        # entities extraction
        print(f"  [Paper {paper_id}] Extracting entities: {chunk_name}")
        entity_prompt = build_entity_prompt(section_text)
        entity_response = await generate_content(model_tag, entity_prompt)
        save_result(entity_response, entity_output)

        entities = load_entities_from_json(entity_output)
        if not entities:
            print(f"    [Paper {paper_id}] No entities found, skipping relations.")
            return True

        # relation extraction
        print(f"  [Paper {paper_id}] Extracting relations: {chunk_name}")
        relation_prompt = build_relation_prompt(section_text, entities)
        relation_response = await generate_content(model_tag, relation_prompt)
        save_result(relation_response, relation_output)

        return True
    except Exception as e:
        print(f"    [Paper {paper_id}] FAILED: {chunk_name} — {e}")
        return False


async def process_paper(model_tag: str, paper_id: str, model_output_dir: str) -> tuple[int, int]:
    """Process all chunks in a paper. Returns (success_count, fail_count)."""
    paper_path = os.path.join(DATA_DIR, paper_id)
    paper_output_dir = os.path.join(model_output_dir, paper_id)
    os.makedirs(paper_output_dir, exist_ok=True)

    chunk_files = get_sorted_chunks(paper_path)
    print(f"  [Paper {paper_id}] {len(chunk_files)} sections")

    # chunks within a paper run concurrently
    results = await asyncio.gather(
        *[
            process_chunk(model_tag, paper_id, cf, paper_path, paper_output_dir)
            for cf in chunk_files
        ]
    )

    successes = sum(1 for r in results if r)
    failures = sum(1 for r in results if not r)
    return successes, failures


async def main_async():
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
    print(f"\nEntity-Relation Extraction [{model_tag}] — {total} papers found")
    print(f"Max concurrency: {MAX_CONCURRENCY}\n")

    start = time.time()

    results = await asyncio.gather(
        *[process_paper(model_tag, pid, model_output_dir) for pid in paper_dirs]
    )

    total_success = sum(s for s, _ in results)
    total_failed = sum(f for _, f in results)
    elapsed = time.time() - start

    print(f"\nDone in {elapsed:.1f}s. {total_success} sections succeeded, {total_failed} failed across {total} papers.")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
