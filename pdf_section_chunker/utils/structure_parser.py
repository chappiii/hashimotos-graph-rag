import re

_EXCLUDED_SECTIONS = {
    "supplementary information",
    "supporting information",
    "supplementary data",
    "supplementary material",
    "supplementary materials",
}

_APPENDIX_RE = re.compile(
    r'^(\s*\*\s*)(Appendix\s+[A-Z0-9]+)\s+.+',
    re.IGNORECASE
)


def clean_structure_output(headers_output: str) -> str:
    """Deterministic cleanup on raw structure before correction.

    Strips appendix subtitles, normalizes pipe separators,
    and removes excluded sections.
    """
    cleaned_lines = []
    for line in headers_output.split('\n'):
        stripped = line.strip().lstrip('*').strip().rstrip(':')

        if stripped.lower() in _EXCLUDED_SECTIONS:
            continue

        line = line.replace(' | ', '. ')

        appendix_match = _APPENDIX_RE.match(line)
        if appendix_match:
            line = appendix_match.group(1) + appendix_match.group(2)

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def parse_structure_from_output(headers_output: str) -> list[dict]:
    """Parse structure markdown into a list of {header, subheaders[]} dicts."""
    headers = []
    current = None

    for line in headers_output.split('\n'):
        if not line.strip():
            continue

        if line.lstrip().startswith('*') and not line.startswith('    '):
            header = line.strip().strip('*').strip().rstrip(':')
            if current:
                headers.append(current)
            current = {"header": header, "subheaders": []}
        elif line.strip().startswith('*') and line.startswith('    '):
            subheader = line.strip().strip('*').strip().rstrip(':')
            if current:
                current["subheaders"].append(subheader)

    if current:
        headers.append(current)

    return headers
