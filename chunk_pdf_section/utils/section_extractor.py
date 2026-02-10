import re
from lxml import etree
from config.chunk_config import TEI_NS, MAIN_SECTION_TITLES

def extract_sections(tei_path):

    tree = etree.parse(tei_path)
    body_divs = tree.xpath("//tei:text/tei:body/tei:div", namespaces=TEI_NS)

    sections = []

    for div in body_divs:
        head = div.find("tei:head", namespaces=TEI_NS)
        if head is None:
            continue

        title = clean_title(" ". join(head.itertext()))
        n = head.get("n")

        if not is_valid_title(title):
            continue

        if not is_main_section(title, n):
            continue

        content = collect_section_content(div)
        sections.append({
            "title": normalize_title(title),
            "number": extract_section_number(n),
            "text": content
        })

    return sections

# Helpers
def clean_title(title):
    title = re.sub(r"^\|?\s*\[?|\]?\s*$", "", title)
    return title.strip()

def normalize_title(title):
    return title.title() if title.isupper() else title

def extract_section_number(n):
    if n and re.match(r"^\d+\.?$", n):
        return n.rstrip(".")
    return None

def is_valid_title(title):
    if not title or len(title) < 2:
        return False
    if re.match(r"^[\d\.\s]+$", title):
        return False
    if re.match(r"^\[?Evidence level", title, re.I):
        return False
    return True

def is_main_section(title, n):
    if n and re.match(r"^\d+\.?$", n):
        return True
    if title.isupper() and len(title) >= 3 and not re.search(r"\d", title):
        return True
    if not n and title.lower() in (t.lower() for t in MAIN_SECTION_TITLES):
        return True
    return False

def collect_section_content(main_div):
    paragraphs = []

    for p in main_div.xpath(".//tei:p", namespaces=TEI_NS):
        txt = " ".join(p.itertext()).strip()
        if txt:
            paragraphs.append(txt)

    return "\n\n".join(paragraphs)