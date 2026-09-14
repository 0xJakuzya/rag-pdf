from openai import OpenAI
from embeddings import Embedder
from vector_store import VectorStore
from config import prompt, base_url, chat_model
import os

class OpenAIChat:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.client = OpenAI(api_key=os.getenv("API_KEY"), base_url=base_url)
    
    def ask(self, question, k=4):
        query_embedding = self.embedder.encode_query(question)
        chunks, _, _ = self.vector_store.search(query_embedding, k=k)
        context = "\n\n".join(
            f"Фрагмент {number}:\n{chunk}"
            for number, chunk in enumerate(chunks, start=1)
        )
        response = self.client.responses.create(
            model=chat_model, 
            instructions=prompt,
            input=f"Контекст:\n{context}\n\nВопрос: {question}",
            reasoning={"effort": "none"},
        )
        return response.output_text, chunks