import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR   = Path(__file__).resolve().parent
_MODULE_DIR   = _CONFIG_DIR.parent
_PROJECT_ROOT = _MODULE_DIR.parent

# --- Input paths ---
ENTITY_RELATION_DIR = _PROJECT_ROOT / "extract_entity_relation" / "extracted_entity_relations" / "gemini"
METADATA_PATH       = _PROJECT_ROOT / "data" / "metadata.json"
CHUNKS_DIR          = _PROJECT_ROOT / "pdf_section_chunker" / "chunks"

# --- Output paths ---
OUTPUT_DIR          = _MODULE_DIR / "pre_ingestion" / "output"
ENTITY_REGISTRY_PATH = OUTPUT_DIR / "entity_registry.json"
CLAIM_REGISTRY_PATH  = OUTPUT_DIR / "claim_registry.json"

# --- Deduplication ---
FUZZY_THRESHOLD = 88  

# Entities that must NEVER be fuzzy-merged with anything else.
# They only match via exact (case-insensitive) lookup.
# Reason: these terms look similar to neighbours but are clinically distinct.
PROTECTED_EXACT_ONLY: set[str] = {
    # Thyroid function states — clinically opposite; fuzzy collapses them
    "Hypothyroidism",
    "Hyperthyroidism",
    "Subclinical Hypothyroidism",
    "Subclinical Hyperthyroidism",
    # Core HT-related diseases
    "Hashimoto's Thyroiditis",
    "Graves' Disease",
    # Same abbreviation (TG) — must stay separate
    "Thyroglobulin Antibody",
    "Triglycerides",
    # Immune cell types — distinct populations
    "T Cells",
    "B Cells",
    "Regulatory T Cells",
    "Natural Killer Cells",
    # Cancer subtypes — distinct histology
    "MALT Lymphoma",
    "Diffuse Large B-Cell Lymphoma",
    "Papillary Thyroid Carcinoma",
    "Thyroid Lymphoma",
}

# --- Gemini ---
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"


# --- Neo4j ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- Qdrant ---
QDRANT_URL = "http://localhost:6333"
VECTOR_DIM  = 3072

# --- Claim ranking weights ---
CERTAINTY_WEIGHTS = {"high": 1.0, "moderate": 0.6, "low": 0.2}
SECTION_WEIGHTS   = {
    "RESULTS":      1.0,
    "ABSTRACT":     0.9,
    "CONCLUSION":   0.85,
    "DISCUSSION":   0.6,
    "INTRODUCTION": 0.5,
    "METHODS":      0.3,
    "OTHER":        0.5,
}

# --- Study design weights ---
STUDY_DESIGN_WEIGHTS: dict[str, float] = {
    "meta_analysis": 1.0,
    "rct":           0.9,
    "cohort":        0.7,
    "case_control":  0.65,
    "review":        0.5,
    "case_report":   0.3,
    "other":         0.4,
}
