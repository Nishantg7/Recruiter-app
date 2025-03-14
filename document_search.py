from langchain_ollama import OllamaEmbeddings

def generate_embeddings(text_chunks, model="nomic-embed-text"):
    embedding = OllamaEmbeddings(model=model)
    return [embedding.embed_query(chunk) for chunk in text_chunks] 