from pathlib import Path
from documents import load_txt, split_text
from embeddings import Embedder
from vector_store import VectorStore

def main():
    file_path = Path("data/билеты.txt")
    text = load_txt(file_path)

    chunks = split_text(text)
    chunk_ids = [f"{file_path.stem}_chunk_{number}" for number in range(1, len(chunks) + 1)]
    
    embedder = Embedder()
    embeddings = embedder.encode_texts(chunks)
    
    vector_store = VectorStore()
    vector_store.build_index(chunks=chunks, chunk_ids=chunk_ids, embeddings=embeddings)
    
    print(f"Создано чанков: {len(chunks)}")
    print(f"В индексе: {vector_store.total_chunks}")

if __name__ == "__main__":
    main()