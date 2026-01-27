METADATA_TEMPLATE = {
    "paper_id": None,
    "doi": None,
    "title": None,
    "published_year": None,
    "author_list": [],
    "countries": [],
    "purpose_of_work": None,
    "keywords": []
}

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
PDFS = "./p"
OUTPUT_FOLDER = "./extracted_metadata"
EXPERIMENT_OUTPUT = "./experiments/results"

API_TIMEOUT = 1000
SLEEP_DURATION = 5
BATCH_SIZE = 10

EXTRACTION_MODEL = "gemma2:latest"
# EXTRACTION_MODEL = "llama3.1:latest"
# EXTRACTION_MODEL = "qwen3:latest"
# EXTRACTION_MODEL = "mistral:7b"
# EXTRACTION_MODEL = "deepseek-r1:latest"

CORRECTION_MODEL = "llama3.1:latest"
# CORRECTION_MODEL = "qwen3:latest"
# CORRECTION_MODEL = "mistral:7b"
# CORRECTION_MODEL = "deepseek-r1:latest"


def get_output_filename():
    model_name = EXTRACTION_MODEL.replace(":", "").replace("/", "_")
    return f"{model_name}_extracted.json"

OUTPUT_FILENAME = get_output_filename()