from pathlib import Path
import pdfplumber

def extract_text(file_path):
    return Path(file_path).read_text(encoding="utf-8")

def extract_pdf(file_path):
    path = Path(file_path)
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append({'page': i, 'text': text, "source": path.name})
    return pages


