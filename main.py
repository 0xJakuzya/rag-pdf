from pathlib import Path
from documents import extract_text
from embeddings import Embedder

def main():
    texts = Path(r'data\билеты.txt')
    texts = extract_text(texts)
    embedder = Embedder()
    embeddings = embedder.encode_texts(texts)
    print(f'embeddings: {embeddings}')

if __name__ == '__main__':
    main()