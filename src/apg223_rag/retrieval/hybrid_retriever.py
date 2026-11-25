import networkx as nx
from typing import List, Dict, Any, Tuple
from apg223_rag.config import RRF_K
from apg223_rag.retrieval.vector_store import PineconeVectorStore
from apg223_rag.retrieval.keyword_search import BM25KeywordStore
from apg223_rag.graph.graph_builder import traverse_graph_for_query


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
    """Fuses multiple ranked chunk ID lists using RRF: sum(1 / (k + rank))."""
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridGraphRetriever:
    def __init__(
        self,
        parents: List[Dict[str, Any]],
        children: List[Dict[str, Any]],
        vector_store: PineconeVectorStore,
        keyword_store: BM25KeywordStore,
        graph: nx.DiGraph,
    ):
        self.parent_map = {p["parent_id"]: p for p in parents}
        self.child_map = {c["chunk_id"]: c for c in children}
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.graph = graph

    def retrieve(self, query: str, top_k_children: int = 6, max_parents: int = 3) -> Dict[str, Any]:
        # 1. Dense Semantic Search (Pinecone BGE)
        dense_ids = self.vector_store.search(query, top_k=10)

        # 2. Sparse Keyword Search (BM25)
        sparse_ids = self.keyword_store.search(query, top_k=10)

        # 3. Reciprocal Rank Fusion on Child Chunks
        fused_children = reciprocal_rank_fusion([dense_ids, sparse_ids])[:top_k_children]

        # 4. Expand Child Chunks -> Full Parent Subsections
        ordered_parent_ids: List[str] = []
        for cid, _ in fused_children:
            pid = self.child_map[cid]["parent_id"]
            if pid not in ordered_parent_ids:
                ordered_parent_ids.append(pid)

        # 5. Multi-hop Knowledge Graph Traversal
        graph_parent_ids, graph_triples = traverse_graph_for_query(self.graph, query, top_k_parents=2)
        for g_pid in graph_parent_ids:
            if g_pid not in ordered_parent_ids:
                ordered_parent_ids.append(g_pid)

        selected_parents = [
            self.parent_map[pid] for pid in ordered_parent_ids[:max_parents] if pid in self.parent_map
        ]

        return {
            "dense_hits": dense_ids[:5],
            "sparse_hits": sparse_ids[:5],
            "fused_children": fused_children,
            "parents": selected_parents,
            "graph_triples": graph_triples,
        }