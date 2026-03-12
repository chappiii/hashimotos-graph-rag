import sys
from pathlib import Path
from config.section_chunker_config import (
    PDFS, SECTIONS_DIR,
    GEMINI_API_KEY, UPLOAD_POLL_INTERVAL, SKIP_EXISTING
)
from utils.gemini_client import configure_gemini, upload_pdf, delete_pdf
from utils.structure_parser import read_structure_from_md
from utils.file_utils import ensure_output_directory, is_already_processed
from utils.extractor import process_paper


def get_sorted_pdfs() -> list:
    """Return all numeric PDFs in PDFS sorted by number."""
    pdfs = [f for f in Path(PDFS).glob("*.pdf") if f.stem.isdigit()]
    return sorted(pdfs, key=lambda f: int(f.stem))


def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    client = configure_gemini(GEMINI_API_KEY)
    pdfs = get_sorted_pdfs()
    total = len(pdfs)
    print(f"\nPDF Section Extraction — {total} papers found\n")

    success, skipped, failed = 0, 0, 0

    for idx, pdf_path in enumerate(pdfs, 1):
        base_name = pdf_path.stem
        md_path = Path(SECTIONS_DIR) / f"{base_name}.md"

        print(f"[{idx}/{total}] Paper {base_name}")

        # check if file is processed
        if SKIP_EXISTING and is_already_processed(base_name):
            print("  Skipping (already processed)")
            skipped += 1
            continue

        # Check if Section map exists 
        if not md_path.exists():
            print(f"  Skipping — no section file: {md_path.name}")
            skipped += 1
            continue

        output_dir = ensure_output_directory(base_name)
        headers_output = read_structure_from_md(str(md_path))

        pdf_file = None
        try:
            pdf_file = upload_pdf(str(pdf_path), client, UPLOAD_POLL_INTERVAL)
            process_paper(pdf_file, headers_output, output_dir, client)
            success += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        finally:
            if pdf_file:
                delete_pdf(pdf_file, client)

    print(f"\nDone. {success} processed, {skipped} skipped, {failed} failed out of {total} papers.\n")


if __name__ == "__main__":
    main()