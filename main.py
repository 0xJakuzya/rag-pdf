from pathlib import Path
from documents import load_txt, split_text
from embeddings import Embedder
from vector_store import VectorStore
from dotenv import load_dotenv
from chat_openai import OpenAIChat

load_dotenv()

def main():
    file_path = Path("data/вопросы.txt")
    text = load_txt(file_path)

    chunks = split_text(text)
    chunk_ids = [f"{file_path.stem}_chunk_{number}" for number in range(1, len(chunks) + 1)]
    
    embedder = Embedder()
    embeddings = embedder.encode_texts(chunks)
    
    vector_store = VectorStore()
    vector_store.clear()
    vector_store.build_index(chunks=chunks, chunk_ids=chunk_ids, embeddings=embeddings)
    print(f"Создано чанков: {len(chunks)}")
        
    chat = OpenAIChat(embedder, vector_store)
    question = input("Введите вопрос: ")
    answer, _ = chat.ask(question)
    print("\nОтвет:")
    print(answer)

if __name__ == "__main__":
    main()