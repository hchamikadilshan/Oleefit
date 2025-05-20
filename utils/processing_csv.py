import pandas as pd
import os
import re
import uuid
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import faiss

# Constants
VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "chunk_metadata.json")

# ───── Helpers ─────
def clean_sentences(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_placeholder_description(row):
    return f"The {row[1]} is an exercise targeting {row[4]} using {row[5]}."

# ───── Step 1: Process DataFrame and Generate Metadata ─────
def generate_meta_data(df):
    metadata_list = []
    for index, row in df.iterrows():
        if pd.isnull(row[2]):
            description = generate_placeholder_description(row)
            description_missing = True
        else:
            description = clean_sentences(row[2])
            description_missing = False

        metadata = {
            "id": str(uuid.uuid4()),
            "index": index,
            "description_missing": description_missing,
            "text": description,
            "type": row[3],
            "body_part": row[4],
            "equipment": row[5],
            "level": row[6],
        }

        metadata_list.append(metadata)
    return metadata_list

# ───── Step 2: Generate Embeddings and Save ─────
def store_embeddings_with_metadata(metadata_list, model_name='all-MiniLM-L6-v2', output_dir=VECTOR_STORE_DIR):
    model = SentenceTransformer(model_name)

    texts = [meta['text'] for meta in metadata_list]

    # Embed
    embeddings = model.encode(texts)
    embedding_matrix = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    normalized_embeddings = embedding_matrix / norms

    # Create or update FAISS index
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
    else:
        index = faiss.IndexFlatIP(normalized_embeddings.shape[1])

    index.add(normalized_embeddings)
    faiss.write_index(index, INDEX_PATH)

    # Save metadata
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2)

    print(f"✅ Saved {len(metadata_list)} embeddings and metadata to: {output_dir}")


def process_info_csv(df):
    metadata_list = generate_meta_data(df)
    store_embeddings_with_metadata(metadata_list, model_name='all-MiniLM-L6-v2', output_dir=VECTOR_STORE_DIR)

