import faiss
import numpy as np
import pickle

def create_index(embeddings, index_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)

if __name__ == "__main__":
    embeddings = np.load('embeddings.npy')
    create_index(embeddings, 'index.faiss')
