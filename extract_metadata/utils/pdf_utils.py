import os
import re
from typing import List

def natural_sorting_key(filename: str) -> List[int]:
    numbers = re.findall(r'\d+', filename)
    return [int(x) for x in numbers] if numbers else [0]

def get_pdf_files(folder_path: str) -> List[str]:
    if not os.path.exists(folder_path):
        print(f"Error: '{folder_path}' folder not found")
        return []
    pdf_files = [f for f in  os.listdir(folder_path) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No pdf file found in '{folder_path}' folder")
        return []
    
    return sorted(pdf_files, key=natural_sorting_key)