<!-- README.md -->
# APG 223 Hybrid Parent-Child + Neo4j Graph RAG

A regulatory retrieval-augmented generation pipeline for APRA Prudential Practice Guide **APG 223 (Residential Mortgage Lending)** combining:
- **Deterministic PyMuPDF Layout Parsing** (margin clipping, footnote marker linking, X-coordinate Glossary extraction, and bounded parent windows `<= 3200` chars)
- **Semantic + Recursive Child Chunking** (`BAAI/bge-large-en-v1.5`)
- **Hybrid Retrieval + RRF** (Pinecone dense vectors + in-memory BM25 sparse search fused via Reciprocal Rank Fusion)
- **Neo4j Knowledge Graph** (multi-hop traversal across Prudential Standards, whole-word regulatory concepts, and Glossary definitions)

## Quickstart
1. Copy `.env.example` to `.env` and fill in `PINECONE_API_KEY`, `NEO4J_URI`, `NEO4J_PASSWORD`, and `GEMINI_API_KEY`.
2. Place `APG223.pdf` at `data/raw/APG223.pdf`.
3. Install dependencies and run ingestion:
   ```bash
   uv sync
   uv run pytest
   uv run apg223-ingest