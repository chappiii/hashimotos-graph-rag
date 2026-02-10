from pathlib import Path

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
GROBID_TIMEOUT = 120
GROBID_RATE_LIMIT = 2

PDF_DIR = Path("../p")
TEI_DIR = Path("xml")
MD_DIR = Path("chunk_md")

TEI_DIR.mkdir(exist_ok=True)
MD_DIR.mkdir(exist_ok=True)

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

MAIN_SECTION_TITLES = (
    "Introduction",
    "Materials and Methods",
    "Methods",
    "Results",
    "Discussion",
    "Conclusion",
    "Conclusions",
    "Abstract",
    "Background",
    "Objectives",
    "Aims",
    "Purpose and Scope",
    "Identification",
    "Assessment",
    "Main Findings",
    "Strengths and Limitations",
    "Acknowledgements",
    "Conflicts of Interest",
    "References",
    "Appendix",
)