import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_project_root() -> Path:
    if env_root := os.getenv("APG223_ROOT_DIR"):
        return Path(env_root).resolve()
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return cwd


ROOT_DIR = get_project_root()
RAW_PDF_PATH = ROOT_DIR / "data" / "raw" / "APG223.pdf"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

TREE_JSON_PATH = PROCESSED_DIR / "document_tree.json"
PARENTS_JSON_PATH = PROCESSED_DIR / "parents.json"
TABLES_JSON_PATH = PROCESSED_DIR / "tables.json"
GLOSSARY_JSON_PATH = PROCESSED_DIR / "glossary.json"
FOOTNOTES_JSON_PATH = PROCESSED_DIR / "footnotes.json"
CHILDREN_JSON_PATH = PROCESSED_DIR / "children.json"


@dataclass(frozen=True)
class LayoutConfig:
    start_content_page: int = 3
    end_content_page: int = 25      # Pages 26-27 are parsed explicitly as Glossary
    glossary_start_page: int = 26
    header_margin_y: float = 48.0
    footer_margin_y: float = 785.0
    h1_min_font: float = 15.0
    h2_min_font: float = 11.5
    footnote_max_font: float = 9.2


LAYOUT = LayoutConfig()

# Embedding & Pinecone
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")
EMBED_DIMENSION = 1024
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "apg223-hybrid-v6")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "apg223-v6")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Neo4j Settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Gemini LLM & Judge Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = list(dict.fromkeys([LLM_MODEL, "gemini-3.5-flash-lite", "gemini-3.8-flash"]))
LLM_TIMEOUT_MS = 45_000

# Bounded Parent & Child Chunking Hyperparameters
MAX_PARENT_CHARS = 3200
PARENT_OVERLAP_PARAS = 1
SEMANTIC_BREAK_PERCENTILE = 20
MAX_CHILD_CHARS = 1200
CHUNK_OVERLAP_CHARS = 150
RRF_K = 60