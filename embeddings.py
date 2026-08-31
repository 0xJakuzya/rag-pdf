from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

model = 'all-MiniLM-L6-v2'

class Embedder:
    def __init__(self, model=model):
        self.model = HuggingFaceEmbeddings(model_name=model) 

    def encode_texts(self, texts):
        embeddings = self.model.embed_documents(texts)
        return np.array(embeddings, dtype=np.float32)