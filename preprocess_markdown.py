import os
import json
import re
import tiktoken

def split_paragraphs(text):
    """Split text into paragraphs by blank lines."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def chunk_text(text, max_tokens=400, model="gpt-3.5-turbo"):
    """Token-aware chunking for long paragraphs."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = enc.decode(tokens[i:i+max_tokens])
        chunks.append(chunk.strip())
    return chunks

def extract_heading(line):
    """Return (level, text) if line is a Markdown heading."""
    match = re.match(r"^(#{1,6})\s+(.*)", line)
    if match:
        return len(match.group(1)), match.group(2).strip()
    return None

def markdown_to_chunks(filepath, max_tokens=400):
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    filename = os.path.basename(filepath)

    # Track hierarchy of headings
    breadcrumbs = {}
    paragraphs = []
    current_para = []
    lines = raw_text.splitlines()

    for line in lines:
        heading = extract_heading(line)
        if heading:
            level, heading_text = heading
            breadcrumbs[level] = heading_text
            # Remove deeper levels if we go up
            for deeper in list(breadcrumbs.keys()):
                if deeper > level:
                    del breadcrumbs[deeper]
            continue

        if line.strip() == "":
            if current_para:
                paragraphs.append(("\n".join(current_para), dict(breadcrumbs)))
                current_para = []
        else:
            current_para.append(line)

    # Add last paragraph
    if current_para:
        paragraphs.append(("\n".join(current_para), dict(breadcrumbs)))

    chunks = []
    order = 0
    for para, crumbs in paragraphs:
        breadcrumb_str = " > ".join([crumbs[k] for k in sorted(crumbs.keys())])
        for chunk in chunk_text(para, max_tokens=max_tokens):
            order += 1
            chunks.append({
                "id": f"{filename}-chunk-{order}",
                "text": chunk,
                "metadata": {
                    "file": filename,
                    "order": order,
                    "breadcrumb": breadcrumb_str
                }
            })
    return chunks

if __name__ == "__main__":
    filepath = "tigers-bears-1.md"  # Replace with your Markdown file
    chunks = markdown_to_chunks(filepath)

    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"✅ Preprocessing complete: {len(chunks)} chunks saved to chunks.json")
