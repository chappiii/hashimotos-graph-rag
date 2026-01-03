import PyPDF2

def extract_first_page(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            if len(pdf_reader.pages) > 0:
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()
                return text
            else: 
                return "PDF empty - no pages"
            
    except Exception as e:
        return f"Error: {str(e)}"