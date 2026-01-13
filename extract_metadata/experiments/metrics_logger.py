import time 
import json
import os
from datetime import datetime
import psutil

class MetricsLogger:
    def __init__(self, model_name, run_id=None):
        self.model_name = model_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.documents = []
        self.current_doc  = None
        self.run_start = None
        self.peak_ram_mb = 0
        
        
    def start_run(self):
        self.run_start = time.perf_counter()

    def start_documents(self, paper_id):
        self.current_doc = {
            "paper_id": paper_id,
            "start_time": time.perf_counter(),
            "extraction_time_s": None,
            "correction_time_s": None,
            "extraction_raw": None,
            "correction_raw": None,
            "ram_mb": None,
            "success": False
        }

    def log_extraction(self, duration, raw_response):
        self.current_doc["extraction_time_s"] = duration
        self.current_doc["extraction_raw"] = raw_response

    def log_correction(self, duration, raw_response):
        self.current_doc["correction_time_s"] = duration
        self.current_doc["correction_raw"] = raw_response

    def end_document(self, success=True):
        self.current_doc["total_time_s"] = time.perf_counter() - self.current_doc["start_time"]
        self.current_doc["ram_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
        self.current_doc["success"] = success
        self.peak_ram_mb = max(self.peak_ram_mb, self.current_doc["ram_mb"])
        del self.current_doc["start_time"]
        self.documents.append(self.current_doc)
        self.current_doc = None

    def save(self, output_dir):
        run_dir = os.path.join(output_dir, self.model_name.replace(":", "_"), f"run_{self.run_id}")
        os.makedirs(run_dir, exist_ok=True)

        # Save metrics
        metrics = {
            "model": self.model_name,
            "run_id": self.run_id,
            "total_documents": len(self.documents),
            "total_time_s": time.perf_counter() - self.run_start,
            "peak_ram_mb": self.peak_ram_mb,
            "documents": [{k: v for k, v in d.items() if not k.endswith("_raw")} for d in self.documents]
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save raw outputs
        outputs = {
            "documents": [{
                "paper_id": d["paper_id"],
                "extraction_raw": d["extraction_raw"],
                "correction_raw": d["correction_raw"]
            } for d in self.documents]
        }
        with open(os.path.join(run_dir, "outputs.json"), "w") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

        return run_dir