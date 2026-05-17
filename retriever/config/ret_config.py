# --- Paths ---
ENTITY_REGISTRY_PATH = "kg_ingestion/pre_ingestion/output/entity_registry.json"

# --- Display ---
SEP  = "=" * 72
SEP2 = "-" * 72

# --- Embedding ---
EMBEDDING_MODEL    = "gemini-embedding-2"
EMBED_MAX_RETRIES  = 3
EMBED_RETRY_SLEEP  = 5

# --- Qdrant ---
QDRANT_URL = "http://localhost:6333"

# --- Neo4j ---
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_DATABASE = "neo4j"

# --- Shared top-k ---
TOP_K = 7

# --- Domain: entities that appear in almost every claim ---
GENERIC_ENTITIES: set[str] = {
    "Hashimoto's Thyroiditis",
    "Autoimmune Thyroid Disease",
    "Thyroid Gland",
}

# --- Evidence quality signals ---
CERTAINTY_WEIGHTS: dict[str, float] = {"high": 1.0, "moderate": 0.6, "low": 0.2}
SECTION_WEIGHTS: dict[str, float] = {
    "RESULTS":      1.0,
    "ABSTRACT":     0.9,
    "CONCLUSION":   0.85,
    "DISCUSSION":   0.6,
    "INTRODUCTION": 0.5,
    "METHODS":      0.3,
    "OTHER":        0.5,
}

# --- graph_ret ranking ---
QUALITY_PRE_FILTER_N = 40
TOP_N_FINAL          = TOP_K
EVIDENCE_PER_CLAIM   = 2

# --- entity matching ---
ENTITY_MATCH_THRESHOLD = 80
ENTITY_TOP_K           = 10
MIN_MATCH_COVERAGE     = 0.8

# --- query decomposition ---
NGRAM_MAX_N       = 5
NGRAM_MIN_UNIGRAM = 3

STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "of", "to", "for", "by", "with", "from", "into",
    "and", "or", "but", "not", "no",
    "what", "which", "who", "how", "when", "where", "why",
    "does", "do", "did", "has", "have", "had",
    "this", "that", "these", "those", "its", "it",
    "can", "could", "may", "might", "would", "should", "will",
    "between", "among", "about", "after", "before",
    "patient", "patients", "study", "studies", "effect", "effects",
    "level", "levels", "role", "association", "relationship",
    "there", "link", "affect", "affects", "show", "shows",
    "increase", "decrease", "reduce", "cause", "causes",
    "high", "low", "more", "less", "change", "changes",
    "found", "seen", "observed", "reported", "significant",
    "compared", "compare", "used", "use", "treatment", "treatments",
}
