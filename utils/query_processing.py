import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# File paths
INDEX_PATH = "vector_store/faiss_index.index"


def load_index():
    index = faiss.read_index(INDEX_PATH)
    return index

def embed_query(query):
    index = load_index()

    query_embedding = model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")
    query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)

    return query_vector

