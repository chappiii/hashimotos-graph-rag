import sys
import time
from datetime import datetime
from pathlib import Path
from config.section_chunker_config import (
    PDFS, GEMINI_API_KEY, UPLOAD_POLL_INTERVAL, SKIP_EXISTING, TIME_DIR
)
from utils.gemini_client import configure_gemini, upload_pdf, delete_pdf
from utils.file_utils import (
    ensure_output_directory, is_already_processed, save_auto_section
)
from utils.extractor import extract_structure, process_paper


def get_sorted_pdfs() -> list:
    """Return all numeric PDFs in PDFS sorted by number."""
    pdfs = [f for f in Path(PDFS).glob("*.pdf") if f.stem.isdigit()]
    return sorted(pdfs, key=lambda f: int(f.stem))


def init_time_report() -> Path:
    """Create time/ dir and a new timestamped report file. Returns the file path."""
    time_dir = Path(TIME_DIR)
    time_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    time_file = time_dir / f"{timestamp}.md"
    with open(time_file, "w", encoding="utf-8") as f:
        f.write(f"# Timing Report — {timestamp}\n\n")
    return time_file


def append_paper_time(time_file: Path, paper_id: str, structure_time: float, call_times: list, paper_total: float):
    """Append one paper's timing to the report."""
    with open(time_file, "a", encoding="utf-8") as f:
        f.write(f"## Paper {paper_id} — {paper_total:.1f}s\n\n")
        f.write(f"| # | Section | Time (s) |\n")
        f.write(f"| :--- | :--- | :--- |\n")
        f.write(f"| 1 | Structure extraction | {structure_time:.1f} |\n")
        for j, call in enumerate(call_times, 2):
            f.write(f"| {j} | {call['section']} | {call['seconds']:.1f} |\n")
        f.write(f"\n---\n\n")


def append_total_time(time_file: Path, total_elapsed: float, success: int, failed: int):
    """Append the final total time line."""
    with open(time_file, "a", encoding="utf-8") as f:
        f.write(f"**Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m) — "
                f"{success} processed, {failed} failed**\n")


def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    client = configure_gemini(GEMINI_API_KEY)
    pdfs = get_sorted_pdfs()
    total = len(pdfs)
    print(f"\nPDF Section Extraction — {total} papers found\n")

    success, skipped, failed = 0, 0, 0
    run_start = time.time()
    time_file = init_time_report()

    for idx, pdf_path in enumerate(pdfs, 1):
        base_name = pdf_path.stem
        print(f"[{idx}/{total}] Paper {base_name}")

        if SKIP_EXISTING and is_already_processed(base_name):
            print("  Skipping (already processed)")
            skipped += 1
            continue

        pdf_file = None
        paper_start = time.time()
        try:
            pdf_file = upload_pdf(str(pdf_path), client, UPLOAD_POLL_INTERVAL)

            # Pass 1: Extract section structure from PDF
            print("  Extracting section structure...")
            headers_output, structure_time = extract_structure(pdf_file, client)
            save_auto_section(base_name, headers_output)

            # Pass 2: Extract content per section
            output_dir = ensure_output_directory(base_name)
            call_times = process_paper(pdf_file, headers_output, output_dir, client)
            success += 1

            append_paper_time(time_file, base_name, structure_time, call_times, time.time() - paper_start)

        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
        finally:
            if pdf_file:
                delete_pdf(pdf_file, client)

    total_elapsed = time.time() - run_start
    append_total_time(time_file, total_elapsed, success, failed)

    print(f"\nDone. {success} processed, {skipped} skipped, {failed} failed out of {total} papers.")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)\n")


if __name__ == "__main__":
    main()
