import faiss
import numpy as np
from embedding_model import get_embeddings

def retrieve(query, index_path, top_k=5):
    index = faiss.read_index(index_path)
    query_embedding = get_embeddings([query])
    distances, indices = index.search(query_embedding, top_k)
    return indices

if __name__ == "__main__":
    query = "Sua consulta aqui"
    results = retrieve(query, 'index.faiss')
    print(results)
