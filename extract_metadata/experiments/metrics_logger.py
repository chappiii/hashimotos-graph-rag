import os
import time 
import json
import psutil
import threading
from datetime import datetime

class MetricsLogger:
    def __init__(self, model_name, output_dir, run_id=None):
        self.model_name = model_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_doc = None
        self.run_start = None
        self.peak_ram_mb = 0
        self.doc_count = 0
        self._lock = threading.Lock()
        
        # Create run directory immediately
        self.run_dir = os.path.join(output_dir, self.model_name.replace(":", "_"), f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Initialize files
        self.metrics_path = os.path.join(self.run_dir, "metrics.json")
        self.outputs_path = os.path.join(self.run_dir, "outputs.json")
        
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
        
        # Save immediately to disk
        self._append_to_outputs()
        self._append_to_metrics()
        
        self.doc_count += 1
        self.current_doc = None

    def _append_to_outputs(self):
        """Append current doc's raw outputs to outputs.json"""
        with self._lock:
            output_entry = {
                "paper_id": self.current_doc["paper_id"],
                "extraction_raw": self.current_doc["extraction_raw"],
                "correction_raw": self.current_doc["correction_raw"]
            }
            
            # Load existing or create new
            if os.path.exists(self.outputs_path):
                with open(self.outputs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {"documents": []}
            
            data["documents"].append(output_entry)
            
            with open(self.outputs_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def _append_to_metrics(self):
        """Append current doc's metrics to metrics.json"""
        with self._lock:
            metric_entry = {k: v for k, v in self.current_doc.items() if not k.endswith("_raw")}
            
            # Load existing or create new
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {
                    "model": self.model_name,
                    "run_id": self.run_id,
                    "total_documents": 0,
                    "total_time_s": 0,
                    "peak_ram_mb": 0,
                    "documents": []
                }
            
            data["documents"].append(metric_entry)
            data["total_documents"] = len(data["documents"])
            data["total_time_s"] = time.perf_counter() - self.run_start
            data["peak_ram_mb"] = self.peak_ram_mb
            
            with open(self.metrics_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    def save(self):
        """Final save - just updates totals (data already saved incrementally)"""
        with self._lock:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["total_time_s"] = time.perf_counter() - self.run_start
                with open(self.metrics_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            return self.run_dir