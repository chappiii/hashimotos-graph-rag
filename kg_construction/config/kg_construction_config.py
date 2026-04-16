import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _CONFIG_DIR.parent
_PROJECT_ROOT = _MODULE_DIR.parent

# --- Paths ---
ENTITY_RELATION_DIR = _PROJECT_ROOT / "extract_entity_relation" / "extracted_entity_relations-1"
METADATA_DIR = _PROJECT_ROOT / "extract_metadata" / "GT_grouped"
EVIDENCE_MAPPING_PATH = _MODULE_DIR / "evidence_mapping.json"

# --- Neo4j ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# --- Qdrant ---
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION_PARENT = os.getenv("QDRANT_COLLECTION_PARENT", "Parents")
QDRANT_COLLECTION_CHILDREN = os.getenv("QDRANT_COLLECTION_CHILDREN", "Children")

# --- Ollama (embeddings) ---
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "nomic-embed-text"
VECTOR_DIMENSION = 768

# --- Settings ---
EVIDENCE_COUNTER_START = 1
BATCH_SIZE = 100
FUZZY_THRESHOLD = 85