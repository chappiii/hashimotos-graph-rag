import PyPDF2
from typing import Optional

def extract_first_page(pdf_path: str) -> Optional[str]:
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            if len(pdf_reader.pages) > 0:
                # first_page = pdf_reader.pages[0]
                # text = first_page.extract_text()
                text = ""
                for i in range(min(2, len(pdf_reader.pages))):
                    text += pdf_reader.pages[i].extract_text()
                return text
            else:
                print(f"PDF empty - no pages: {pdf_path}")
                return None
            
    except Exception as e:
        print(f"PDF read error ({pdf_path}): {e}")
        return None