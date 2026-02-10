import os
import time
from chunk_pdf_section.config.chunk_config import TEI_DIR, MD_DIR, GROBID_RATE_LIMIT, PDF_DIR
from utils.grobid_client import pdf_to_tei
from utils.section_extractor import extract_sections
from utils.md_writer import write_md_sections

def process_pdf(pdf_path):
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    tei_path = TEI_DIR / f"{doc_id}.tei.xml"
    md_dir = MD_DIR / doc_id

    print(f"\nProcessing: {doc_id}")

    # 1. GROBID
    pdf_to_tei(pdf_path, tei_path)

    # 2. Extract sections
    sections = extract_sections(tei_path)
    if not sections:
        raise ValueError("No sections extracted")

    # 3. Write markdown
    write_md_sections(doc_id, sections, md_dir)

    time.sleep(GROBID_RATE_LIMIT)

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
            failed.append((pdf.name, str(e)))

    print("\nSUMMARY")
    print(f"Success: {len(results)}")
    print(f"Failed: {len(failed)}")

if __name__ == "__main__":
    main()