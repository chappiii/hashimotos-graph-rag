import requests
import time
from typing import Optional
from config.metadata_config import OLLAMA_URL, API_TIMEOUT, SLEEP_DURATION, EXTRACTION_MODEL, CORRECTION_MODEL


def make_llm_request(model: str, prompt: str) -> Optional[str]:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(OLLAMA_URL, json=payload,  timeout=API_TIMEOUT)

        if response.status_code == 200:
            result = response.json()
            return result.get('response', '')
        else:
            print(f"API Error: {response.status_code}")
            return None
        
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        return None
    
def extract_metadata_with_llm(prompt: str) -> Optional[str]:
    return make_llm_request(EXTRACTION_MODEL, prompt)

def correct_response_with_llm(prompt: str) -> Optional[str]:
    return make_llm_request(CORRECTION_MODEL, prompt)

def process_with_correction(extraction_prompt: str, correction_prompt: str) -> Optional[str]:
    print("Sending to LLM...")

    initial_response = extract_metadata_with_llm(extraction_prompt)
    if not initial_response:
        return None
    
    print("Initial LLM Response:")
    print("-" * 50)
    print(initial_response)
    print("-" * 50)

    print("Sending for correction")

    corrected_response = correct_response_with_llm(correction_prompt)

    if corrected_response:
        print("Corrected LLM Response:")
        print("-" * 50)
        print(corrected_response)
        print("-" * 50)
        time.sleep(SLEEP_DURATION)
        return corrected_response
    else:
        print("Correction failed, using original response...")
        return initial_response 