from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
from config import chunk_overlap, chunk_size

def load_txt(file_path):
    return Path(file_path).read_text(encoding="utf-8")

def load_pdf(file_path):
    path = Path(file_path)
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append({'page': i, 'text': text, "source": path.name})
    return pages

def split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", " ", ""],
    )
    return text_split.split_text(text)