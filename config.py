# embedding model
embedding_model = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# chunking
chunk_size=400
chunk_overlap=40

# chromadb
db_path='storage/chroma'
collection_name='uml_materials'

# llm
base_url = "https://api.groq.com/openai/v1"
chat_model = "qwen/qwen3.8-27b"

prompt = """Ты помощник по загруженным учебным материалам.
Отвечай только на основе переданного контекста.
Если в контексте нет ответа, честно сообщи об этом.
Не придумывай факты.Отвечай на русском языке."""