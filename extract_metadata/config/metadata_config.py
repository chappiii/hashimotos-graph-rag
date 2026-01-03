METADATA_TEMPLATE = {
    "paper_id": None,
    "doi": None,
    "title": None,
    "published_year": None,
    "author_list": [],
    "countries":[],
    "purpose_of_work": None,
    "keywords": []
}

OLLAMA_URL  = "http://127.0.0.1:11434/api/generate"
PDFS = "./pdfs"
OUTPUT_FOLDER = "./extracted_metadata"

API_TIMEOUT = 1000
SLEEP_DURATION = 5

EXTRACTION_MODEL = "llama3.1:latest"
CORRECTION_MODEL = "llama3.1:latest"

def get_output_filename():
    model_name = EXTRACTION_MODEL.replace(":", "").replace("/", "_")
    return f"{model_name}_extracted.json"

OUTPUT_FILENAME = get_output_filename()