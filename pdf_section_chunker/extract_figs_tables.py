import json
import sys
from pathlib import Path

from google.genai import types

from pdf_section_chunker.config.section_chunker_config import (
    FIGS_TABLES_DIR,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    PDFS,
    SAFETY_SETTINGS,
    SKIP_EXISTING_FIGS_TABLES,
    FIGURE_SCHEMA,
    TABLE_SCHEMA,
    UPLOAD_POLL_INTERVAL,
)
from pdf_section_chunker.utils.gemini_client import configure_gemini, delete_pdf, upload_pdf

OUTPUT_DIR = Path(FIGS_TABLES_DIR)


def _type_serializer(obj):
    if obj is int:   return "integer"
    if obj is str:   return "string"
    if obj is float: return "number"
    if obj is bool:  return "boolean"
    raise TypeError(f"Not serializable: {obj!r}")

def get_sorted_pdfs() -> list:
    pdfs = [f for f in Path(PDFS).glob("*.pdf") if f.stem.isdigit()]
    return sorted(pdfs, key=lambda f: int(f.stem))


def is_extracted(paper_id: str, kind: str) -> bool:
    return (OUTPUT_DIR / paper_id / f"{kind}.json").exists()


def save_json(paper_id: str, kind: str, data: list):
    out_dir = OUTPUT_DIR / paper_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{kind}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")


def _call_gemini(pdf_file, prompt: str, client) -> list:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[pdf_file, prompt],
        config=types.GenerateContentConfig(
            safety_settings=SAFETY_SETTINGS,
            response_mime_type="application/json",
        ),
    )
    if response.text is None:
        reason = "UNKNOWN"
        if response.candidates:
            reason = str(response.candidates[0].finish_reason)
        raise ValueError(f"Gemini returned None (finish_reason={reason})")
    return json.loads(response.text) or []

def extract_tables(pdf_file, client) -> list:
    schema_str = json.dumps(TABLE_SCHEMA, indent=2, default=_type_serializer)
    prompt = (
        "You are an expert medical research analyst. "
        "Extract ALL tables from this PDF document.\n\n"
        "For each table, use these three sources of information:\n"
        "1. The table itself (all rows, columns, and cell values)\n"
        "2. The caption text directly above or below the table\n"
        "3. All paragraphs in the document that reference this table "
        "(e.g. sentences containing 'Table 1 shows...', 'as shown in Table 2')\n\n"
        "Return a JSON object per table using this exact schema:\n"
        f"{schema_str}\n\n"
        "Field instructions:\n"
        "- table_id: MUST be a JSON integer (e.g. 1, 2, 3). "
        "Never a string. Never 'Table 1'. Just the number. "
        "Assign sequentially by reading order starting at 1. "
        "The PDF label is already captured in caption\n"
        "- table_type: one of: baseline_characteristics, regression, outcomes\n"
        "- section_label: top-level section name where the table appears (e.g. 'Results')\n"
        "- caption: full caption text verbatim from the PDF\n"
        "- population: who was studied — use the table, caption, and referencing paragraphs "
        "(e.g. '80 HT patients vs 60 healthy controls')\n"
        "- groups: comparison column headers (e.g. ['HT group', 'Control group'])\n"
        "- variables: row variable names listed in the table\n"
        "- key_findings: plain strings, one per significant finding — include variable name, "
        "values, and p-value; draw from the table data AND referencing paragraphs "
        "(e.g. 'TSH higher in HT group: 5.8 vs 2.1 mIU/L, p<0.001')\n"
        "- model_adjustments: covariates adjusted for, if regression table — else []\n"
        "- Use null for strings that cannot be determined; [] for empty arrays\n\n"
        "Return a JSON array of table objects. No extra text or commentary."
    )
    return _call_gemini(pdf_file, prompt, client)

def extract_figures(pdf_file, client) -> list:
    schema_str = json.dumps(FIGURE_SCHEMA, indent=2, default=_type_serializer)
    prompt = (
        "You are an expert medical research analyst. "
        "Extract ALL figures from this PDF document.\n\n"
        "For each figure, use these three sources of information:\n"
        "1. The figure itself (the image, chart, or diagram — read it visually)\n"
        "2. The caption text directly above or below the figure\n"
        "3. All paragraphs in the document that reference this figure "
        "(e.g. sentences containing 'Figure 1 shows...', 'as illustrated in Fig. 2')\n\n"
        "Return a JSON object per figure using this exact schema:\n"
        f"{schema_str}\n\n"
        "Field instructions:\n"
        "- figure_id: MUST be a JSON integer (e.g. 1, 2, 3). "
        "Never a string. Never 'Figure 1'. Just the number. "
        "Assign sequentially by reading order starting at 1. "
        "The PDF label is already captured in caption\n"
        "- figure_type: one of: KM | forest | bar | box | ROC | scatter | flow | other\n"
        "- section_label: top-level section name where the figure appears (e.g. 'Results')\n"
        "- caption: full caption text verbatim from the PDF\n"
        "- population: who is shown — use the figure, caption, and referencing paragraphs "
        "(e.g. 'HT patients (n=80) vs controls (n=60)')\n"
        "- groups: comparison arms visible in the figure (e.g. ['HT group', 'Control group'])\n"
        "- outcome: what is being measured or shown (e.g. 'overall survival', 'TPO-Ab levels')\n"
        "- key_findings: plain strings — one per finding; draw from the visual data, caption, "
        "and referencing paragraphs; include values and p-value where stated\n"
        "- Use null for any field that cannot be determined; [] for empty arrays\n\n"
        "Return a JSON array of figure objects. No extra text or commentary."
    )
    return _call_gemini(pdf_file, prompt, client)

def main():
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    client = configure_gemini(GEMINI_API_KEY)
    pdfs = get_sorted_pdfs()
    total = len(pdfs)
    print(f"\nFigure & Table Extraction — {total} papers found\n")

    success, skipped, failed = 0, 0, 0

    for idx, pdf_path in enumerate(pdfs, 1):
        paper_id = pdf_path.stem
        tables_done  = is_extracted(paper_id, "tables")
        figures_done = is_extracted(paper_id, "figures")

        print(f"[{idx}/{total}] Paper {paper_id}")

        if SKIP_EXISTING_FIGS_TABLES and tables_done and figures_done:
            print("  Skipping (already extracted)")
            skipped += 1
            continue

        pdf_file = None
        try:
            pdf_file = upload_pdf(str(pdf_path), client, UPLOAD_POLL_INTERVAL)

            if not (SKIP_EXISTING_FIGS_TABLES and tables_done):
                print("  Extracting tables...")
                tables = extract_tables(pdf_file, client)
                save_json(paper_id, "tables", tables)
                print(f"  → {len(tables)} table(s) found")
            else:
                print("  Tables already extracted — skipping")

            if not (SKIP_EXISTING_FIGS_TABLES and figures_done):
                print("  Extracting figures...")
                figures = extract_figures(pdf_file, client)
                save_json(paper_id, "figures", figures)
                print(f"  → {len(figures)} figure(s) found")
            else:
                print("  Figures already extracted — skipping")

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
