import chromadb
import numpy as np
from pathlib import Path
from config import collection_name, db_path

class VectorStore:
    def __init__(self, db_path=db_path, collection_name=collection_name):
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.get_collection()

    @property
    def total_chunks(self) -> int:
        return self.collection.count()

    @property
    def is_ready(self) -> bool:
        return self.total_chunks > 0

    def get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def search(self, query_embedding, k):
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        result = self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=min(k, self.total_chunks),
            include=["documents", "distances"],
        )

        documents = result["documents"][0]
        ids = result["ids"][0]
        distances = result["distances"][0]
        return documents, ids, distances

    def clear(self):    
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        self.collection = self.get_collection()
        print("Векторный индекс очищен")

    def build_index(self, chunks, chunk_ids, embeddings):
        vectors = np.asarray(embeddings, dtype=np.float32)
        batch_size = 10
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self.collection.add(
                ids=chunk_ids[start:end],
                documents=chunks[start:end],
                embeddings=vectors[start:end].tolist(),
            )
        print(f"Создан индекс: {self.total_chunks} чанков")