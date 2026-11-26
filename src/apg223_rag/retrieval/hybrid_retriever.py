# src/apg223_rag/retrieval/hybrid_retriever.py
from typing import List, Dict, Any, Tuple
from apg223_rag.config import RRF_K
from apg223_rag.retrieval.vector_store import PineconeVectorStore
from apg223_rag.retrieval.keyword_search import BM25KeywordStore
from apg223_rag.graph.neo4j_graph import Neo4jGraphStore


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = RRF_K) -> List[Tuple[str, float]]:
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
        graph_store: Neo4jGraphStore,
    ):
        self.parent_map = {p["parent_id"]: p for p in parents}
        self.child_map = {c["chunk_id"]: c for c in children}
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.graph_store = graph_store

    def retrieve(
        self,
        query: str,
        top_k_children: int = 6,
        max_hybrid_parents: int = 2,
        max_graph_parents: int = 1,
    ) -> Dict[str, Any]:
        dense_ids = self.vector_store.search(query, top_k=10)
        sparse_ids = self.keyword_store.search(query, top_k=10)
        fused_children = reciprocal_rank_fusion([dense_ids, sparse_ids])[:top_k_children]

        hybrid_pids: List[str] = []
        for cid, _ in fused_children:
            child = self.child_map.get(cid)
            if not child:
                continue
            pid = child["parent_id"]
            if pid not in hybrid_pids and pid in self.parent_map:
                hybrid_pids.append(pid)
            if len(hybrid_pids) >= max_hybrid_parents:
                break

        graph_pids, all_triples, glossary_hits = self.graph_store.query_graph(
            query, top_k_parents=max_graph_parents + 1
        )
        final_pids = list(hybrid_pids)
        added_graph = 0
        for g_pid in graph_pids:
            if g_pid not in final_pids and g_pid in self.parent_map:
                final_pids.append(g_pid)
                added_graph += 1
            if added_graph >= max_graph_parents:
                break

        selected_parents = [self.parent_map[pid] for pid in final_pids]
        selected_pid_set = set(final_pids)

        active_triples = [
            t for t in all_triples
            if any(f"[{pid}]" in t for pid in selected_pid_set)
        ]

        return {
            "fused_children": fused_children,
            "parents": selected_parents,
            "graph_triples": active_triples,
            "glossary_hits": glossary_hits,
        }