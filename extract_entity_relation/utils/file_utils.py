import json
import os
import re

SKIP_SECTIONS = {"references", "appendix"}


def read_content(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"  File not found: {file_path}")
        return ""
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return ""


def get_sorted_chunks(paper_dir: str) -> list[str]:
    chunks = sorted(
        [f for f in os.listdir(paper_dir) if f.endswith(".md")],
        key=lambda f: int(f.split("-")[0]),
    )
    return [f for f in chunks if not _should_skip(f)]


def _should_skip(filename: str) -> bool:
    name = filename.rsplit(".", 1)[0]  # strip .md
    name = name.split("-", 1)[1] if "-" in name else name
    return name.lower() in SKIP_SECTIONS


def _remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_from_response(text: str) -> dict | list | None:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    json_string = match.group(1).strip() if match else text.strip()

    if not json_string:
        return None

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None


def load_entities_from_json(file_path: str) -> list[dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "entities" in data:
            return data["entities"]
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            if isinstance(first, dict) and "entities" in first:
                return first["entities"]

        print("  'entities' field not found.")
        return []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Error loading entities: {e}")
        return []


def save_result(response_text: str, output_path: str) -> str | None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cleaned = _remove_think_tags(response_text)
    parsed = _extract_json_from_response(cleaned)

    if parsed is None:
        print("  JSON parsing failed.")
        return None

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"  Error writing {output_path}: {e}")
        return None

    if isinstance(parsed, dict):
        entity_count = len(parsed.get("entities", []))
        relation_count = len(parsed.get("relations", []))
    elif isinstance(parsed, list):
        entity_count = len(parsed)
        relation_count = 0
    else:
        entity_count, relation_count = 0, 0

    summary = f"{entity_count} entities" if entity_count else f"{relation_count} relations"
    print(f"  Saved: {output_path} ({summary})")
    return output_path
