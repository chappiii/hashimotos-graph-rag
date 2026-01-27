import requests
import time
from typing import Optional
from config.metadata_config import OLLAMA_URL, API_TIMEOUT, SLEEP_DURATION, EXTRACTION_MODEL, CORRECTION_MODEL


def make_llm_request(model: str, prompt: str) -> tuple[Optional[str], float]:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        start = time.perf_counter()
        response = requests.post(OLLAMA_URL, json=payload,  timeout=API_TIMEOUT)
        duration = time.perf_counter() - start


        if response.status_code == 200:
            result = response.json()
            return result.get('response', ''), duration
        else:
            print(f"API Error: {response.status_code}")
            return None, 0
        
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return None, 0
    
def extract_metadata_with_llm(prompt: str) -> tuple[Optional[str], float]:
    return make_llm_request(EXTRACTION_MODEL, prompt)

def correct_response_with_llm(prompt: str) -> tuple[Optional[str], float]:
    return make_llm_request(CORRECTION_MODEL, prompt)