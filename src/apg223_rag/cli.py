# src/apg223_rag/cli.py
import sys
import json
import time
import argparse
from sentence_transformers import SentenceTransformer
from apg223_rag.config import (
    PROCESSED_DIR,
    RAW_PDF_PATH,
    TREE_JSON_PATH,
    PARENTS_JSON_PATH,
    TABLES_JSON_PATH,
    GLOSSARY_JSON_PATH,
    FOOTNOTES_JSON_PATH,
    CHILDREN_JSON_PATH,
    EMBED_MODEL_NAME,
)
from apg223_rag.ingestion.table_extractor import extract_tables
from apg223_rag.ingestion.text_parser import parse_prose_and_footnotes, parse_glossary_terms
from apg223_rag.ingestion.tree_builder import build_document_tree_and_parents
from apg223_rag.chunking.semantic_chunker import build_child_chunks
from apg223_rag.graph.neo4j_graph import Neo4jGraphStore
from apg223_rag.retrieval.vector_store import PineconeVectorStore
from apg223_rag.retrieval.keyword_search import BM25KeywordStore
from apg223_rag.retrieval.hybrid_retriever import HybridGraphRetriever
from apg223_rag.generation.generator import generate_answer
from apg223_rag.generation.evaluator import evaluate_response


def ingest_main():
    parser = argparse.ArgumentParser(description="Ingest APG 223 PDF into Pinecone and Neo4j.")
    parser.add_argument("--pdf", type=str, default=str(RAW_PDF_PATH), help="Path to APG223.pdf")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Extracting verified data tables...")
    tables, table_bboxes = extract_tables(args.pdf)

    print("2. Parsing prose blocks, footnotes & two-column Glossary...")
    prose_blocks, footnotes = parse_prose_and_footnotes(args.pdf, table_bboxes)
    glossary = parse_glossary_terms(args.pdf)

    print("3. Building Bounded Parent Windows & Document Tree...")
    doc_tree, parents, linked_tables = build_document_tree_and_parents(prose_blocks, tables, footnotes)

    print("4. Running Semantic + Recursive Chunking...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    children = build_child_chunks(parents, linked_tables, embed_model=embed_model)

    clean_parents = [{k: v for k, v in p.items() if k != "block_spans"} for p in parents]
    TREE_JSON_PATH.write_text(json.dumps(doc_tree, indent=2))
    PARENTS_JSON_PATH.write_text(json.dumps(clean_parents, indent=2))
    TABLES_JSON_PATH.write_text(json.dumps(linked_tables, indent=2))
    GLOSSARY_JSON_PATH.write_text(json.dumps(glossary, indent=2))
    FOOTNOTES_JSON_PATH.write_text(json.dumps(footnotes, indent=2))
    CHILDREN_JSON_PATH.write_text(json.dumps(children, indent=2))

    print("5. Building Neo4j Knowledge Graph...")
    neo4j_store = Neo4jGraphStore()
    try:
        nodes, rels = neo4j_store.build_graph(clean_parents, glossary)
    finally:
        neo4j_store.close()

    print("6. Clearing stale vectors & upserting into Pinecone...")
    vector_store = PineconeVectorStore(embed_model=embed_model)
    vector_store.upsert_children(children)

    print(
        f"Ingestion Complete! Parents: {len(clean_parents)} | Glossary Terms: {len(glossary)} | "
        f"Tables: {len(linked_tables)} | Children: {len(children)} | Neo4j Nodes: {nodes} | Edges: {rels}"
    )


def query_main():
    parser = argparse.ArgumentParser(description="Query the APG 223 Hybrid + Neo4j Graph RAG pipeline.")
    parser.add_argument(
        "question",
        nargs="?",
        default="What interest rate buffer does APS 220 require for serviceability assessments, and when can HELP debt repayments be excluded via an override?",
        help="Regulatory question to ask",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip LLM-as-a-Judge evaluation")
    args = parser.parse_args()

    if not PARENTS_JSON_PATH.exists() or not CHILDREN_JSON_PATH.exists():
        print("Error: Processed JSON artifacts not found. Run `uv run apg223-ingest` first.", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    parents = json.loads(PARENTS_JSON_PATH.read_text())
    children = json.loads(CHILDREN_JSON_PATH.read_text())
    bm25_store = BM25KeywordStore(children)
    vector_store = PineconeVectorStore()
    neo4j_store = Neo4jGraphStore()
    retriever = HybridGraphRetriever(parents, children, vector_store, bm25_store, neo4j_store)
    print(f"[1/4] Loaded indexes & Neo4j connection in {time.perf_counter() - t0:.2f}s")

    try:
        t1 = time.perf_counter()
        bundle = retriever.retrieve(args.question)
        print(f"[2/4] Hybrid RRF + Neo4j traversal done in {time.perf_counter() - t1:.2f}s")

        t2 = time.perf_counter()
        answer = generate_answer(args.question, bundle)
        print(f"[3/4] Answer generated in {time.perf_counter() - t2:.2f}s")

        print("\n=== RETRIEVED PARENTS ===")
        for p in bundle["parents"]:
            print(f"- {p['parent_id']} ({p['char_length']} chars): {p['section']} > {p['subsection']} (pp. {p['page_start']}-{p['page_end']})")

        print("\n=== ACTIVE GRAPH TRIPLES ===")
        for t in bundle["graph_triples"]:
            print(f"- {t}")

        print("\n=== ANSWER ===\n", answer)

        if not args.no_eval:
            t3 = time.perf_counter()
            scores = evaluate_response(args.question, answer, bundle)
            print(f"\n[4/4] Evaluation completed in {time.perf_counter() - t3:.2f}s")
            print("\n=== EVALUATION ===\n", json.dumps(scores, indent=2))
    finally:
        neo4j_store.close()