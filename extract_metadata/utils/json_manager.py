import os
import copy
import json
from pathlib import Path
from typing import Optional
from config.metadata_config import METADATA_TEMPLATE, OUTPUT_FOLDER, OUTPUT_FILENAME
from utils.llm_parser import parse_llm_output

def save_metadata_to_json(pdf_filename: str, llm_response: str, part_number: Optional[int] = None, run_dir: str = None) -> Optional[str]:

    output_dir = run_dir or OUTPUT_FOLDER
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving JSON to folder: {output_dir}")

    if part_number is not None:
        base_filename = OUTPUT_FILENAME.replace('.json', f'_part_{part_number}.json')
    else:
        base_filename = OUTPUT_FILENAME

    json_path = Path(output_dir) / base_filename

    metadata = copy.deepcopy(METADATA_TEMPLATE)
    metadata["paper_id"] = Path(pdf_filename).stem

    parsed = parse_llm_output(llm_response)

    if parsed:
        metadata.update(parsed)

    try:
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as fp:
                existing_data = json.load(fp)
                paper_list = existing_data.get("papers", [])
        else:
            paper_list = []

        paper_list.append(metadata)

        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump({"papers": paper_list}, fp, ensure_ascii=False, indent=2)
        print(f"JSON Saved -> {json_path}, (total {len(paper_list)} papers)")
        return str(json_path)
    except OSError as e:
        print(f"File Write Error: {e}")
        return None