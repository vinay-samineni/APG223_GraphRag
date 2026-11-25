import json
import networkx as nx
from sentence_transformers import SentenceTransformer
from apg223_rag.config import (
    RAW_PDF_PATH,
    TREE_JSON_PATH,
    PARENTS_JSON_PATH,
    TABLES_JSON_PATH,
    FOOTNOTES_JSON_PATH,
    CHILDREN_JSON_PATH,
    GRAPH_PATH,
    BM25_PICKLE_PATH,
    EMBED_MODEL_NAME,
)
from apg223_rag.ingestion.table_extractor import extract_tables
from apg223_rag.ingestion.text_parser import parse_prose_and_footnotes
from apg223_rag.ingestion.tree_builder import build_document_tree_and_parents
from apg223_rag.chunking.semantic_chunker import build_child_chunks
from apg223_rag.graph.graph_builder import build_regulatory_graph
from apg223_rag.retrieval.vector_store import PineconeVectorStore
from apg223_rag.retrieval.keyword_search import BM25KeywordStore


def run_ingestion():
    print("1. Extracting isolated tables & bounding-box masks...")
    tables, table_bboxes = extract_tables(str(RAW_PDF_PATH))

    print("2. Parsing deterministic prose blocks & separating footnotes...")
    prose_blocks, footnotes = parse_prose_and_footnotes(str(RAW_PDF_PATH), table_bboxes)

    print("3. Building Document Tree & Parent Nodes...")
    doc_tree, parents, linked_tables = build_document_tree_and_parents(prose_blocks, tables, footnotes)

    print("4. Loading BGE-large-en-v1.5 & running Semantic + Recursive Chunking...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    children = build_child_chunks(parents, linked_tables, embed_model=embed_model)

    print("5. Building Regulatory Knowledge Graph...")
    kg = build_regulatory_graph(parents)

    # Save all intermediate JSON & Graph artifacts for Cursor debugging/inspection
    clean_parents = [{k: v for k, v in p.items() if k != "blocks"} for p in parents]
    TREE_JSON_PATH.write_text(json.dumps(doc_tree, indent=2))
    PARENTS_JSON_PATH.write_text(json.dumps(clean_parents, indent=2))
    TABLES_JSON_PATH.write_text(json.dumps(linked_tables, indent=2))
    FOOTNOTES_JSON_PATH.write_text(json.dumps(footnotes, indent=2))
    CHILDREN_JSON_PATH.write_text(json.dumps(children, indent=2))
    nx.write_graphml(kg, str(GRAPH_PATH))

    print("6. Indexing into BM25 (local) & Pinecone (dense)...")
    bm25_store = BM25KeywordStore(children)
    bm25_store.save(BM25_PICKLE_PATH)

    vector_store = PineconeVectorStore(embed_model=embed_model)
    vector_store.upsert_children(children)

    print(
        f"Done! Parents: {len(clean_parents)} | Tables: {len(linked_tables)} | "
        f"Children: {len(children)} | KG Nodes: {kg.number_of_nodes()} | KG Edges: {kg.number_of_edges()}"
    )


if __name__ == "__main__":
    run_ingestion()