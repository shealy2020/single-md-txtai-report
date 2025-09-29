from txtai.embeddings import Embeddings
import json
import numpy as np
from collections import defaultdict

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def find_clusters(ids, vectors, threshold):
    """
    Build a similarity graph and find connected components (clusters).
    """
    n = len(ids)
    adjacency = defaultdict(set)

    # Build adjacency graph (link chunks if similarity > threshold)
    for i in range(n):
        for j in range(i + 1, n):
            score = cosine_similarity(vectors[i], vectors[j])
            if score > threshold:
                adjacency[ids[i]].add(ids[j])
                adjacency[ids[j]].add(ids[i])

    visited = set()
    clusters = []

    # DFS to find connected components
    def dfs(node, cluster):
        visited.add(node)
        cluster.add(node)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                dfs(neighbor, cluster)

    for node in ids:
        if node not in visited and node in adjacency:
            cluster = set()
            dfs(node, cluster)
            clusters.append(cluster)

    return clusters

def generate_similarity_report(index_path="index", report_file="similarity_report.md",
                               chunks_file="chunks.json", threshold=0.7):
    """
    Generate a Markdown report with clusters of semantically similar paragraphs.
    """
    # Load embeddings
    embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
    embeddings.load(index_path)

    # Load preprocessed chunks
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    texts = [chunk["text"] for chunk in chunks_data]
    ids = [chunk["id"] for chunk in chunks_data]
    metas = [chunk["metadata"] for chunk in chunks_data]

    # Encode paragraphs into vectors
    vectors = embeddings.model.encode(texts)

    # Find clusters of similar paragraphs
    clusters = find_clusters(ids, vectors, threshold)

    # Map ID → (text, metadata, vector)
    chunk_map = {
        cid: (text, meta, vec)
        for cid, text, meta, vec in zip(ids, texts, metas, vectors)
    }

    report_lines = ["# 📊 Paragraph-to-Paragraph Semantic Similarity Report (Clustered)", ""]

    for cluster_id, cluster in enumerate(clusters, 1):
        report_lines.append(f"---\n")
        report_lines.append(f"## 🔹 Similarity Cluster {cluster_id}")
        report_lines.append(f"**Cluster contains {len(cluster)} paragraphs**\n")

        # List all paragraphs in the cluster
        for cid in cluster:
            text, meta, _ = chunk_map[cid]
            preview = text.replace("\n", " ")[:200] + ("..." if len(text) > 200 else "")
            breadcrumb = meta.get("breadcrumb", "")
            source_file = meta.get("file", "N/A")

            report_lines.append(f"**ID:** {cid}")
            report_lines.append(f"**Source:** {source_file}")
            report_lines.append(f"**Breadcrumb:** {breadcrumb}")
            report_lines.append(f"**Preview (200 chars):** {preview}")
            report_lines.append("")

        # Pairwise similarity scores
        report_lines.append("### Pairwise Vector Similarities")
        cluster_ids = list(cluster)
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                id1, id2 = cluster_ids[i], cluster_ids[j]
                vec1, vec2 = chunk_map[id1][2], chunk_map[id2][2]
                score = cosine_similarity(vec1, vec2)

                report_lines.append(f"{id1} ↔ {id2}")
                report_lines.append(f"**Vector Score:** {score:.2f}")
                report_lines.append("")

    # Save Markdown report
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"✅ Clustered similarity report saved to {report_file}")

if __name__ == "__main__":
    generate_similarity_report()
