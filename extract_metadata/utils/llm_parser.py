import re
import json
from typing import Optional

def parse_llm_output(raw: str) -> Optional[dict]:
    try:
        raw = re.sub(r"<thinking>.*? </thinking>", "", raw,flags=re.DOTALL | re.IGNORECASE)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()

        start_idx = raw.find('{')

        if start_idx == -1:
            print("NO Valid JSON object found (missing opening '{' )")
            return None
        
        raw_json = raw[start_idx:]
        data = json.loads(raw_json)

        if isinstance(data, dict) and "metadata" in data and isinstance(data["metadata"], dict):
            data = data["metadata"]

        print(f"LLM response successfully parsed: {len(data)} fields found")
        return data 
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None