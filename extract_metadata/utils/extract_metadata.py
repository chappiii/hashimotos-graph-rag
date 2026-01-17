import os 
import time
from config.metadata_config import SLEEP_DURATION, PDFS, EXTRACTION_MODEL, EXPERIMENT_OUTPUT, CORRECTION_MODEL, OUTPUT_FOLDER, MAX_WORKERS
from utils.pdf_utils import get_pdf_files
from experiments.metrics_logger import MetricsLogger
from utils.json_manager import save_metadata_to_json
from utils.first_page_extractor import extract_first_page
from prompts.correction_prompt import get_correction_prompt
from prompts.extraction_prompt import get_extraction_prompt
from utils.llm_client import extract_metadata_with_llm, correct_response_with_llm
from concurrent.futures import ThreadPoolExecutor
import threading

_print_lock = threading.Lock()

def safe_print(message):
    # print with thread safety
    with _print_lock:
        print(message)

def process_single_pdf(pdf_file: str, pdf_index: int, total_files: int, part_numbers: int, logger, run_dir: str):

    logger.start_documents(pdf_file.replace(".pdf", ""))

    pdf_path =  os.path.join(PDFS, pdf_file)
    safe_print(f"=== PDF {pdf_index}/{total_files}: {pdf_file} (Part {part_numbers}) ===")

    first_page  = extract_first_page(pdf_path)

    if first_page and "Error:" not in first_page:
        safe_print("Sending to LLM...")
        extraction_prompt = get_extraction_prompt(first_page)

        initial_response, extraction_duration  =  extract_metadata_with_llm(extraction_prompt)
        logger.log_extraction(extraction_duration, initial_response)

        if initial_response:
            safe_print("Initial LLM Response:")
            safe_print("-" * 50)
            safe_print(initial_response)
            safe_print("-" * 50)
            
            safe_print("Sending for correction...")
            correction_prompt = get_correction_prompt(initial_response)
            corrected_response, correction_duration = correct_response_with_llm(correction_prompt)
            logger.log_correction(correction_duration, corrected_response)

            logger.end_document(success=True)

            if corrected_response:
                safe_print("Corrected LLM Response:")
                safe_print("-" * 50)
                safe_print(corrected_response)
                safe_print("-" * 50)

                json_path = save_metadata_to_json(pdf_file, corrected_response, part_numbers, run_dir)
                if json_path is None:
                    safe_print("JSON could not be saved, LLM response should be checked.")
                time.sleep(SLEEP_DURATION)
                return True
            else:
                safe_print("Correction failed, using original response...")
                json_path = save_metadata_to_json(pdf_file, initial_response, part_numbers, run_dir)
                return True
        else:
            logger.end_document(success=False)
            safe_print("LLM processing failed.")
            return False
    else:
        logger.end_document(success=False)
        safe_print("PDF Reading Error, LLM processing skipped")
        return False


def process_pdf_worker(args):
    """
    Wrapper function for parallel processing.

    Args:
        args: Tuple of (pdf_file, pdf_index, total_files, part_number, logger, run_dir)

    Returns:
        dict: Result with success status and any error message
    """
    pdf_file, pdf_index, total_files, part_number, logger, run_dir = args

    try:
        success = process_single_pdf(pdf_file, pdf_index, total_files, part_number, logger, run_dir)
        return {
            "pdf": pdf_file,
            "success": success,
            "error": None
        }
    except Exception as e:
        safe_print(f"ERROR in worker processing {pdf_file}: {str(e)}")
        return {
            "pdf": pdf_file,
            "success": False,
            "error": str(e)
        }

def process_pdfs():
    safe_print("PDF Metadata Extractor (Batch Processing)")
    safe_print("=" * 50)

    logger = MetricsLogger(EXTRACTION_MODEL, EXPERIMENT_OUTPUT)
    logger.start_run()

    # Create run directory for extracted metadata
    extraction_name = EXTRACTION_MODEL.split(":")[0]
    correction_name = CORRECTION_MODEL.split(":")[0]
    run_dir = os.path.join(OUTPUT_FOLDER, f"{extraction_name}_{correction_name}", f"run_{logger.run_id}")
    os.makedirs(run_dir, exist_ok=True)

    pdf_files = get_pdf_files(PDFS)
    if not pdf_files:
        return
    
    total_files = len(pdf_files)
    batch_size = 10
    total_batches = (total_files + batch_size - 1) // batch_size
    
    safe_print(f"{total_files} PDF files found.")
    safe_print(f"Batch size: {batch_size} PDFs per part")
    safe_print(f"Total parts: {total_batches}")
    safe_print(f"Processing order: {', '.join(pdf_files[:5])}{'...' if total_files > 5 else ''}\n")

    for batch_number in range(total_batches):
        part_number = batch_number  + 1
        start_idx = batch_number * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_files = pdf_files[start_idx:end_idx]

        safe_print(f"\n PART {part_number}/{total_batches} - Processing PDFs {start_idx + 1}-{end_idx}")
        safe_print("=" * 60)

        # Prepare arguments for all PDFs in this batch
        worker_args = []
        for i, pdf_file in enumerate(batch_files):
            global_index = start_idx + i + 1
            worker_args.append((pdf_file, global_index, total_files, part_number, logger, run_dir))

        # Process batch in parallel
        safe_print(f"Processing {len(batch_files)} PDFs with {MAX_WORKERS} workers...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(process_pdf_worker, worker_args)

            # Check results
            for result in results:
                if not result["success"]:
                    safe_print(f"PDF {result['pdf']} could not be processed completely")
                    if result["error"]:
                        safe_print(f"  Error: {result['error']}")
        
        safe_print(f"\n Part {part_number} completed!")
        
        if part_number < total_batches:
            safe_print("Waiting 5 seconds before next part...")
            time.sleep(5)
    
    logger.save()
    safe_print(f"\n Complete: All {total_files} PDFs processed in {total_batches} parts!")