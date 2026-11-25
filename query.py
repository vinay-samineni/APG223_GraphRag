import time
import json
import networkx as nx
from apg223_rag.config import PARENTS_JSON_PATH, CHILDREN_JSON_PATH, GRAPH_PATH, BM25_PICKLE_PATH
from apg223_rag.retrieval.vector_store import PineconeVectorStore
from apg223_rag.retrieval.keyword_search import BM25KeywordStore
from apg223_rag.retrieval.hybrid_retriever import HybridGraphRetriever
from apg223_rag.generation.generator import generate_answer
from apg223_rag.generation.evaluator import evaluate_response


def run_query(question: str):
    t0 = time.perf_counter()
    parents = json.loads(PARENTS_JSON_PATH.read_text())
    children = json.loads(CHILDREN_JSON_PATH.read_text())
    kg = nx.read_graphml(str(GRAPH_PATH))
    bm25_store = BM25KeywordStore.load(BM25_PICKLE_PATH)
    vector_store = PineconeVectorStore()
    retriever = HybridGraphRetriever(parents, children, vector_store, bm25_store, kg)
    print(f"[1/4] Loaded indexes & BGE model in {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    bundle = retriever.retrieve(question)
    print(f"[2/4] Hybrid RRF + Graph retrieval done in {time.perf_counter() - t1:.2f}s")

    t2 = time.perf_counter()
    answer = generate_answer(question, bundle)
    print(f"[3/4] LLM answer generated in {time.perf_counter() - t2:.2f}s")

    t3 = time.perf_counter()
    scores = evaluate_response(question, answer, bundle)
    print(f"[4/4] Evaluation completed in {time.perf_counter() - t3:.2f}s")

    print("\n=== RETRIEVED PARENTS ===")
    for p in bundle["parents"]:
        print(f"- {p['parent_id']}: {p['section']} > {p['subsection']} (pp. {p['page_start']}-{p['page_end']})")

    print("\n=== GRAPH TRIPLES ===")
    for t in bundle["graph_triples"]:
        print(f"- {t}")

    print("\n=== ANSWER ===\n", answer)
    print("\n=== EVALUATION ===\n", json.dumps(scores, indent=2))


if __name__ == "__main__":
    run_query(
        "What interest rate buffer does APS 220 require for serviceability assessments, "
        "and when can HELP debt repayments be excluded via an override?"
    )