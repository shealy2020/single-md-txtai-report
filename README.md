# single-md-txtai-report
Find Semantic Similarities in Chunked MD

A common problem with technical documentation management is that doc teams tend have a lot of stressed-out contributors working on shared content over an extended period of time that may be scattered across multiple departments. These factors often lead to content bloat and redundancy. 

To avoid junk in/out, my content needs some custodial refactoring like deduping and merging similar content if possible. Eventually, I'd like to contain our content in a local LLM for RAG purposes. In taking baby steps toward that goal, I've put together a prototype that surfaces semantically same or similar chunks in the form of a report, using a simple Markdown file as input.

The processing is straightforward:

1. **preprocess_markdown.py** - Chunks MD by heading to populate a JSON file.
2. **index_chunks.py** - txtai (sentence-transformers/all-MiniLM-L6-v2) indexes JSON file.
3. **report_similarity.py** - txtai compares paragraph vectors. Reports similarity clusters and vector scores.