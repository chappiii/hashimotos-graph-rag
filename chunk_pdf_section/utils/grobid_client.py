import requests
from config.chunk_config import GROBID_URL, GROBID_TIMEOUT

def pdf_to_tei(pdf_path, tei_path):
    """
    send PDF to GROBID and save TEI XML
    """

    with open(pdf_path, "rb") as f:
        response = requests.post(
            GROBID_URL,
            files={"input": f},
            timeout=GROBID_TIMEOUT
            )
        
    response.raise_for_status()

    with open(tei_path, "wb") as out:
        out.write(response.content)

    return tei_path