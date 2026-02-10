import os
import time
import traceback
from config.chunk_config import TEI_DIR, MD_DIR, GROBID_RATE_LIMIT, PDF_DIR
from utils.grobid_client import pdf_to_tei
from utils.section_extractor import extract_sections
from utils.md_writer import write_md_sections

def process_pdf(pdf_path):
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    tei_path = TEI_DIR / f"{doc_id}.tei.xml"
    md_dir = MD_DIR / doc_id

    print(f"\n{'='*60}")
    print(f"\nProcessing: {doc_id}")
    print(f"\n{'='*60}")

    # 1. GROBID
    try:
        print(f" [1/3] Sending to GROBID...")
        pdf_to_tei(pdf_path, tei_path)
        print(f" XML Saved: {tei_path}")
    except Exception as e:
        print(f" GROBID processing failed: {e}")
        raise

    # 2. Extract sections
    try:
        print(f" [2/3] Extracting Sections...")
        sections = extract_sections(tei_path)
        if not sections:
            raise ValueError("No sections extracted")
        print(f" found {len(sections)} sectins")
    except Exception as e:
        print(f" Section extraction failed: {e}")
        raise

    # 3. Write markdown
    try:
        print(f"  [3/3] Writing markdown files...")
        write_md_sections(doc_id, sections, md_dir)
        print(f" Markdown files saved to: {md_dir}/")
    except Exception as e:
        print(f" Markdown writing failed: {e}")
        raise

    time.sleep(GROBID_RATE_LIMIT)
    print(f" SUCCESS: {doc_id} ({len(sections)} sections)")

    return {
        "doc_id": doc_id,
        "sections": len(sections),
        "status": "success"
    }

def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"), key=lambda p: int(p.stem))

    if not pdfs:
        print("No PDFs found")
        return

    results, failed = [], []

    for i, pdf in enumerate(pdfs, 1):
        try:
            print(f"[{i}/{len(pdfs)}]")
            results.append(process_pdf(pdf))
        except Exception as e:
            print(f"FAILED: {pdf.name}")
            print(f"   Error: {str(e)}")
            print(f"   Details:\n{traceback.format_exc()}")
            failed.append((pdf.name, str(e)))

    print("\nSUMMARY")
    print(f"Success: {len(results)}")
    print(f"Failed: {len(failed)}")

if __name__ == "__main__":
    main()