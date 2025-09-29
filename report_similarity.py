from txtai.embeddings import Embeddings
import json
import numpy as np

def cosine_similarity(vec1, vec2):
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def generate_similarity_report(index_path="index", report_file="similarity_report.md",
                               chunks_file="chunks.json", threshold=0.7, top_k=5):
    embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
    embeddings.load(index_path)

    # Load chunks JSON
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    texts = [chunk["text"] for chunk in chunks_data]
    ids = [chunk["id"] for chunk in chunks_data]
    metas = [chunk["metadata"] for chunk in chunks_data]

    # Encode all paragraphs
    vectors = embeddings.model.encode(texts)

    report_lines = ["# 📊 Paragraph-to-Paragraph Semantic Similarity Report", ""]

    for i, (uid, text, vec, meta) in enumerate(zip(ids, texts, vectors, metas)):
        preview = text.replace("\n", " ")[:200] + ("..." if len(text) > 200 else "")
        breadcrumb = meta.get("breadcrumb", "")

        sims = []
        for j, (other_id, other_vec, other_meta) in enumerate(zip(ids, vectors, metas)):
            if i == j:
                continue
            score = cosine_similarity(vec, other_vec)
            if score > threshold:
                other_text = texts[j]
                other_preview = other_text.replace("\n", " ")[:200] + ("..." if len(other_text) > 200 else "")
                sims.append((other_id, score, other_preview, other_meta))

        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]

        if sims:
            report_lines.append(f"---\n")
            report_lines.append(f"## Paragraph: {uid}")
            report_lines.append(f"**Breadcrumb:** {breadcrumb}")
            report_lines.append(f"**Preview (200 chars):**  \n*{preview}*")
            report_lines.append(f"\n**Most similar paragraphs (> {threshold}):**\n")

            for rank, (other_id, score, other_preview, other_meta) in enumerate(sims, 1):
                other_breadcrumb = other_meta.get("breadcrumb", "")
                report_lines.append(f"{rank}. **{other_id}** (Similarity: {score:.2f})  ")
                report_lines.append(f"   **Breadcrumb:** {other_breadcrumb}")
                report_lines.append(f"   *{other_preview}*\n")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"✅ Semantic similarity report with breadcrumbs saved to {report_file}")

if __name__ == "__main__":
    generate_similarity_report()
