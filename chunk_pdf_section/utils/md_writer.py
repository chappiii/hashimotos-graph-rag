import re
from pathlib import Path

def slugify(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:80] or "section"

def write_md_sections(doc_id, sections, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for idx, sec in enumerate(sections, start=1):
        if sec["number"]:
            filename = f"{doc_id}__{sec['number']}_{slugify(sec['title'])}.md"
        else:
            filename = f"{doc_id}__{idx:03d}_{slugify(sec['title'])}.md"

        file_path = out_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {sec['title']}\n\n")
            f.write(sec["text"])
            f.write("\n")