from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from config import model

class Embedder:
    def __init__(self, model=model):
        self.model = HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True}) 

    def encode_texts(self, texts):
        embeddings = self.model.embed_documents(texts)
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query):
        embedding = self.model.embed_query(query)
        return np.asarray([embedding], dtype=np.float32)