import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_PDF_PATH = ROOT_DIR / "data" / "raw" / "APG223.pdf"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Output Artifact Paths
TREE_JSON_PATH = PROCESSED_DIR / "document_tree.json"
PARENTS_JSON_PATH = PROCESSED_DIR / "parents.json"
TABLES_JSON_PATH = PROCESSED_DIR / "tables.json"
FOOTNOTES_JSON_PATH = PROCESSED_DIR / "footnotes.json"
CHILDREN_JSON_PATH = PROCESSED_DIR / "children.json"
GRAPH_PATH = PROCESSED_DIR / "knowledge_graph.graphml"
BM25_PICKLE_PATH = PROCESSED_DIR / "bm25_index.pkl"

# Models & Vector DB
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_DIMENSION = 1024
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_3XE99F_BDts6AwSpSMsL3H6Qz8MAHcJ6sypYCuosTDqK7a9URRUVLtnhfVr7PDapMogbjr")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "apg223-hybrid-v5")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# LLM Settings (Google Gemini)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")  

# Deterministic PDF Parsing Thresholds (APG 223 Typography)
START_CONTENT_PAGE = 3      # Skip Cover (1) and TOC (2)
HEADER_MARGIN_Y = 48.0      # Clip running headers above y=48
FOOTER_MARGIN_Y = 785.0     # Clip running footers below y=785
H1_MIN_FONT = 15.0
H2_MIN_FONT = 11.5
FOOTNOTE_MAX_FONT = 9.2

# Chunking & Retrieval Parameters
SEMANTIC_BREAK_PERCENTILE = 20
MAX_CHILD_CHARS = 1500
CHUNK_OVERLAP_CHARS = 200
RRF_K = 60

