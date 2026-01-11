# Extract Metadata

Automated metadata extraction from academic research papers using LLMs via Ollama.

## Purpose

Extracts structured bibliographic metadata from PDF research papers using a two-stage LLM pipeline:
1. **Extraction Stage**: Parses first page of PDF to extract metadata
2. **Correction Stage**: Fixes spelling errors and formatting issues

## Extracted Fields

- `paper_id`, `doi`, `title`, `published_year`
- `author_list`, `countries`, `purpose_of_work`, `keywords`

## Prerequisites

- Python 3.8+
- Ollama running locally with `qwen3:latest` model
- Dependencies: `PyPDF2`, `requests`

## Usage

1. Place PDF files in `pdfs/` directory
2. Run: `python main.py`
3. Output saved to `extracted_metadata/` as JSON files (batches of 10)

## Configuration

Edit `config/metadata_config.py`:

```python
EXTRACTION_MODEL = "qwen3:latest"
CORRECTION_MODEL = "qwen3:latest"
PDFS = "./pdfs"
OUTPUT_FOLDER = "./extracted_metadata"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
```

## Output Format

```json
{
  "papers": [
    {
      "paper_id": "example_paper",
      "doi": "10.1234/example.2024",
      "title": "Paper Title",
      "published_year": "2024",
      "author_list": ["Author 1", "Author 2"],
      "countries": ["USA", "Germany"],
      "purpose_of_work": "Research objective summary",
      "keywords": ["keyword1", "keyword2"]
    }
  ]
}
```
