from google import genai


def configure_gemini(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)
