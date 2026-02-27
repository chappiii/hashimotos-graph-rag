import time
from google import genai
from google.genai import types


def configure_gemini(api_key: str):
    return genai.Client(api_key=api_key)


def upload_pdf(pdf_path: str, client, poll_interval: int = 2):
    print(f"  Uploading {pdf_path}...")
    pdf_file = client.files.upload(
        file=str(pdf_path),
        config=types.UploadFileConfig(mime_type="application/pdf")
    )
    while pdf_file.state.name == "PROCESSING":
        time.sleep(poll_interval)
        pdf_file = client.files.get(name=pdf_file.name)
    if pdf_file.state.name == "FAILED":
        raise ValueError(f"File upload failed: {pdf_file.state}")
    print(f"  Uploaded: {pdf_file.name}")
    return pdf_file


def delete_pdf(pdf_file, client):
    client.files.delete(name=pdf_file.name)
    print(f"  Deleted uploaded file: {pdf_file.name}")
