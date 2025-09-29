import json
from txtai.embeddings import Embeddings

def index_chunks(chunks_file="chunks.json", index_path="index"):
    embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:
        embeddings.index((chunk["id"], chunk["text"], chunk["metadata"]))

    embeddings.save(index_path)
    print(f"✅ Indexed {len(chunks)} chunks and saved index to {index_path}")

if __name__ == "__main__":
    index_chunks()
