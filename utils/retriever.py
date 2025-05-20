import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = "vector_store/faiss_index.index"
METADATA_PATH = "vector_store/chunk_metadata.json"

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_index_and_metadata():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

def retrieve_similar_chunks(query: str, k: int = 5):
    index, metadata_list = load_index_and_metadata()

    # Encode and normalize the query embedding for cosine similarity
    query_embedding = model.encode([query])
    query_vector = np.array(query_embedding).astype("float32")
    query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)  

    # Search FAISS index
    D, I = index.search(query_vector, k)

    results = []
    for idx in I[0]:
        if idx < len(metadata_list):
            chunk = metadata_list[idx]
            chunk["score"] = round(float(D[0][list(I[0]).index(idx)]), 2)
            results.append(chunk)
    return results