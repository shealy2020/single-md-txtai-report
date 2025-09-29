# txtai Demo to Surface Semantically Similar Content in MD 
Find Semantic Similarities in Chunked MD via txtmd

A common problem with technical documentation management is that doc teams tend have a lot of stressed-out contributors working on shared content over an extended period of time that may be scattered across multiple departments. These factors often lead to content bloat and redundancy. 

To avoid junk in/out, my content needs some custodial refactoring like deduping and merging similar content if possible. Eventually, I'd like to contain our content in a local LLM for RAG purposes. In taking baby steps toward that goal, I've put together a prototype that surfaces semantically same or similar chunks in the form of a report, using a simple Markdown file as input.

The processing is straightforward:

1. **preprocess_markdown.py** - Chunks MD by heading to populate a JSON file.
2. **index_chunks.py** - txtai (sentence-transformers/all-MiniLM-L6-v2) indexes JSON file.
3. **report_similarity.py** - txtai compares paragraph vectors. Reports similarity clusters and vector scores.

# Windows Setup Guide for Markdown + txtai Pipeline

## 1. Install Python
- Download Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
- During installation, check "Add Python to PATH".

Verify install in PowerShell:
    python --version

## 2. Create Virtual Environment (recommended)
    python -m venv venv
    .\venv\Scripts\activate

## 3. Install Dependencies
    py -m pip install -r requirements.txt

This will install:
- txtai (semantic search, embeddings, FAISS backend)
- tiktoken (token counting for chunking)

## 4. Workflow
### Step 1: Preprocess Markdown into chunks.json
    python preprocess_markdown.py

### Step 2: Index chunks into txtai index/
    python index_chunks.py

### Step 3: Generate semantic similarity report
    python report_similarity.py

## 5. Outputs
- chunks.json: structured chunks from Markdown
- index/: txtai vector index
- similarity_report.md: human-readable semantic similarity report
