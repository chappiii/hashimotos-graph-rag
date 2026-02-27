def read_structure_from_md(md_path: str) -> str:
    with open(md_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_structure_from_output(headers_output: str) -> list:
    """Parse the structure markdown to get a list of {header, subheaders[]} dicts."""
    headers = []
    current = None

    for line in headers_output.split('\n'):
        if not line.strip():
            continue

        # Top-level header: starts with * but not indented with 4+ spaces
        if line.lstrip().startswith('*') and not line.startswith('    '):
            header = line.strip().lstrip('*').strip()
            if current:
                headers.append(current)
            current = {"header": header, "subheaders": []}
        # Subheader: starts with * and is indented with 4+ spaces
        elif line.strip().startswith('*') and line.startswith('    '):
            subheader = line.strip().lstrip('*').strip()
            if current:
                current["subheaders"].append(subheader)

    if current:
        headers.append(current)

    return headers
